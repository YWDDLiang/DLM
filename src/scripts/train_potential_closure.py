#!/usr/bin/env python3
"""Train equal-compute closure control and Potential-Closed crystal DLMs.

The trainer keeps the existing schedule-matched SPAD LoRA as both policy and
frozen reference.  Clean MP20 schedule CE and complete-transaction objectives
occupy separate optimizer updates; on-policy states are never trained with CE.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.potential_closure import (
    MAX_KL_BUDGET_NATS,
    build_potential_closure_posterior,
    potential_closure_loss,
)
from crystal_dlm.r5_dynamic_length import exact_body_token_count
from scripts import llada_d3po as D3PO
from scripts import llada_sft as SFT


LABEL_SCHEMA = "potential_closure_labelled_group_v1"
MANIFEST_SCHEMA = "potential_closure_label_manifest_v1"
RUN_SCHEMA = "potential_closure_train_v1"
PROBE_SCHEMA = "potential_closure_gradient_probe_v1"
MODES = ("closure_control", "potential_closed")
KINDS = ("cell", "site")
DOMAINS = ("mp20_clean", "on_policy")
EXPECTED_GROUPS = 2048
EXPECTED_GROUPS_PER_STRATUM = 512
EXPECTED_SFT_ROWS = 27136
TOTAL_UPDATES = 2048
CYCLES = 512
GRADIENT_ACCUMULATION = 6
POSTERIOR_DOMAIN_MICROBATCHES = 3
LEARNING_RATE = 5.0e-6
WARMUP_UPDATES = 100
MAX_LENGTH = 382
LOGGING_UPDATES = 10
PROBE_BATCHES = 5
GRADIENT_RATIO_MIN = 1.0e-2
GRADIENT_RATIO_MAX = 1.0e2
GRADIENT_COSINE_MIN = -0.5
EXPECTED_STRATA = tuple(
    f"{domain}_{kind}" for domain in DOMAINS for kind in KINDS
)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def optimizer_objective(update: int) -> str:
    """Return the frozen four-update objective for a one-based update."""

    if not 1 <= int(update) <= TOTAL_UPDATES:
        raise ValueError(f"update must lie in [1, {TOTAL_UPDATES}]")
    return ("clean_ce", "cell", "clean_ce", "site")[(int(update) - 1) % 4]


def learning_rate_for_update(update: int) -> float:
    """Linear warmup through update 100, followed by a constant LR."""

    if not 1 <= int(update) <= TOTAL_UPDATES:
        raise ValueError(f"update must lie in [1, {TOTAL_UPDATES}]")
    return LEARNING_RATE * min(1.0, float(update) / float(WARMUP_UPDATES))


class FiniteEpochIndexStream:
    """A deterministic concatenation of independently shuffled full passes."""

    def __init__(self, indices: Sequence[int], *, epochs: int, seed: int) -> None:
        values = [int(value) for value in indices]
        if not values or int(epochs) <= 0:
            raise ValueError("finite stream needs rows and positive epochs")
        self.indices = tuple(values)
        schedule: list[int] = []
        for epoch in range(int(epochs)):
            shuffled = list(values)
            random.Random(int(seed) + epoch * 1_000_003).shuffle(shuffled)
            schedule.extend(shuffled)
        self.schedule = tuple(schedule)
        self.cursor = 0

    def take(self, count: int) -> list[int]:
        end = self.cursor + int(count)
        if int(count) <= 0 or end > len(self.schedule):
            raise RuntimeError("finite stream exhausted or requested invalid count")
        result = list(self.schedule[self.cursor : end])
        self.cursor = end
        return result

    @property
    def exhausted(self) -> bool:
        return self.cursor == len(self.schedule)


class InfiniteEpochIndexStream:
    """Deterministic infinite shuffled passes for clean MP20 schedule CE."""

    def __init__(self, size: int, *, seed: int) -> None:
        if int(size) <= 0:
            raise ValueError("infinite stream needs a positive size")
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0
        self.buffer: list[int] = []

    def take(self, count: int) -> list[int]:
        if int(count) <= 0:
            raise ValueError("count must be positive")
        while len(self.buffer) < int(count):
            values = list(range(self.size))
            random.Random(self.seed + self.epoch * 1_000_003).shuffle(values)
            self.buffer.extend(values)
            self.epoch += 1
        result = self.buffer[: int(count)]
        del self.buffer[: int(count)]
        return result


@dataclass(frozen=True)
class TransactionBatchPlan:
    kind: str
    group_indices: tuple[int, ...]
    domains: tuple[str, ...]


class TransactionBatchStreams:
    """Construct the exact transaction microbatch ledger for one mode."""

    def __init__(
        self,
        indices_by_stratum: Mapping[str, Sequence[int]],
        *,
        mode: str,
        seed: int,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        for stratum in EXPECTED_STRATA:
            if len(indices_by_stratum.get(stratum, ())) != EXPECTED_GROUPS_PER_STRATUM:
                raise ValueError(f"{stratum} must contain exactly 512 groups")
        self.mode = mode
        self.streams: dict[tuple[str, str], FiniteEpochIndexStream] = {}
        for kind_index, kind in enumerate(KINDS):
            if mode == "potential_closed":
                for domain_index, domain in enumerate(DOMAINS):
                    stratum = f"{domain}_{kind}"
                    self.streams[(domain, kind)] = FiniteEpochIndexStream(
                        indices_by_stratum[stratum],
                        epochs=3,
                        seed=int(seed) + 100 * kind_index + 10 * domain_index,
                    )
            else:
                stratum = f"mp20_clean_{kind}"
                self.streams[("control", kind)] = FiniteEpochIndexStream(
                    indices_by_stratum[stratum],
                    epochs=6,
                    seed=int(seed) + 100 * kind_index,
                )

    def take(self, kind: str) -> TransactionBatchPlan:
        if kind not in KINDS:
            raise ValueError(f"unknown transaction kind {kind}")
        if self.mode == "potential_closed":
            clean = self.streams[("mp20_clean", kind)].take(
                POSTERIOR_DOMAIN_MICROBATCHES
            )
            on_policy = self.streams[("on_policy", kind)].take(
                POSTERIOR_DOMAIN_MICROBATCHES
            )
            indices: list[int] = []
            domains: list[str] = []
            for left, right in zip(clean, on_policy, strict=True):
                indices.extend((left, right))
                domains.extend(("mp20_clean", "on_policy"))
        else:
            indices = self.streams[("control", kind)].take(
                GRADIENT_ACCUMULATION
            )
            domains = ["mp20_clean"] * GRADIENT_ACCUMULATION
        if len(indices) != GRADIENT_ACCUMULATION:
            raise RuntimeError("transaction microbatch count changed")
        return TransactionBatchPlan(
            kind=kind,
            group_indices=tuple(indices),
            domains=tuple(domains),
        )

    def assert_exhausted(self) -> None:
        remaining = [key for key, stream in self.streams.items() if not stream.exhausted]
        if remaining:
            raise RuntimeError(f"transaction streams not exhausted: {remaining}")


def transaction_clean_ce(
    action_log_scores: torch.Tensor,
    *,
    target_index: int,
    transaction_length: int,
) -> torch.Tensor:
    """Complete-transaction teacher NLL normalized by 3 or 6 tokens."""

    if action_log_scores.ndim != 1 or not action_log_scores.numel():
        raise ValueError("action_log_scores must be a nonempty vector")
    if int(transaction_length) not in (3, 6):
        raise ValueError("transaction length must be three or six")
    if not 0 <= int(target_index) < int(action_log_scores.numel()):
        raise IndexError("target_index is outside the action vector")
    return -action_log_scores[int(target_index)] / float(transaction_length)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer(text, add_special_tokens=False)["input_ids"]
    ]


class PotentialClosureGroupDataset:
    """Validated 512x4 closure groups with lazy tokenizer conversion."""

    def __init__(self, path: Path, tokenizer: Any, *, max_length: int) -> None:
        self.path = Path(path).resolve()
        self.rows = list(iter_jsonl(self.path))
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        if len(self.rows) != EXPECTED_GROUPS:
            raise ValueError("potential closure requires exactly 2048 groups")
        indices_by_stratum: dict[str, list[int]] = {
            stratum: [] for stratum in EXPECTED_STRATA
        }
        informative_by_stratum = Counter()
        for index, row in enumerate(self.rows):
            if row.get("schema") != LABEL_SCHEMA:
                raise ValueError("labelled group schema changed")
            if int(row.get("group_idx", -1)) != index:
                raise ValueError("group indices must be contiguous and ordered")
            stratum = str(row.get("stratum"))
            if stratum not in indices_by_stratum:
                raise ValueError(f"unexpected stratum {stratum}")
            indices_by_stratum[stratum].append(index)
            expected_kind = stratum.rsplit("_", 1)[1]
            if str(row.get("transaction_kind")) != expected_kind:
                raise ValueError("stratum and transaction kind disagree")
            transaction_length = int(row.get("transaction_length", 0))
            if transaction_length != (6 if expected_kind == "cell" else 3):
                raise ValueError("transaction length and kind disagree")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
                raise ValueError("candidate K must lie in [1, 4]")
            if int(row.get("K", -1)) != len(candidates):
                raise ValueError("stored K disagrees with candidates")
            actions: list[tuple[int, ...]] = []
            for candidate_index, candidate in enumerate(candidates):
                if int(candidate.get("candidate_idx", -1)) != candidate_index:
                    raise ValueError("candidate indices are not contiguous")
                if candidate.get("valid_action") is not True:
                    raise ValueError("retained candidate is not legal")
                action = tuple(int(value) for value in candidate["action_token_ids"])
                if len(action) != transaction_length:
                    raise ValueError("candidate transaction length changed")
                actions.append(action)
            no_op_kind = candidates[0].get(
                "candidate_kind", candidates[0].get("candidate_source")
            )
            if no_op_kind != "noop":
                raise ValueError("candidate zero must be the no-op")
            if len(actions) != len(set(actions)):
                raise ValueError("candidate actions are not unique")
            informative = bool(row.get("informative"))
            if informative and len(candidates) < 2:
                raise ValueError("informative group has fewer than two actions")
            if informative:
                informative_by_stratum[stratum] += 1
        if any(
            len(indices_by_stratum[stratum]) != EXPECTED_GROUPS_PER_STRATUM
            for stratum in EXPECTED_STRATA
        ):
            raise ValueError("four-stratum 512-group balance changed")
        self.indices_by_stratum = {
            key: tuple(value) for key, value in indices_by_stratum.items()
        }
        self.informative_indices_by_stratum = {
            stratum: tuple(
                index
                for index in indices
                if bool(self.rows[index].get("informative"))
            )
            for stratum, indices in self.indices_by_stratum.items()
        }
        self.informative_by_stratum = {
            key: int(informative_by_stratum[key]) for key in EXPECTED_STRATA
        }
        if any(value < 256 for value in self.informative_by_stratum.values()):
            raise RuntimeError("a closure stratum has fewer than 256 informative groups")

    def __len__(self) -> int:
        return len(self.rows)

    def materialize(self, index: int, *, include_energy: bool) -> dict[str, Any]:
        row = self.rows[int(index)]
        prompt = str(row["prompt"]).rstrip() + "\n"
        source_answer = str(row["source_answer"])
        prompt_ids = _token_ids(self.tokenizer, prompt)
        source_ids = _token_ids(self.tokenizer, source_answer)
        num_atoms = int((row.get("plan_state") or {})["N"])
        if len(source_ids) != exact_body_token_count(num_atoms):
            raise ValueError("source answer is not exact 7+4N")
        if len(prompt_ids) + len(source_ids) > self.max_length:
            raise ValueError("closure group exceeds max sequence length")
        active_relative = tuple(int(value) for value in row["active_positions"])
        transaction_length = int(row["transaction_length"])
        if len(active_relative) != transaction_length:
            raise ValueError("active transaction length changed")
        active_absolute = torch.tensor(
            [len(prompt_ids) + value for value in active_relative],
            dtype=torch.long,
        )
        full = torch.tensor(prompt_ids + source_ids, dtype=torch.long)
        full[active_absolute] = int(MASK_TOKEN_ID)
        candidates = row["candidates"]
        actions = torch.tensor(
            [
                [int(value) for value in candidate["action_token_ids"]]
                for candidate in candidates
            ],
            dtype=torch.long,
        )
        no_op = [source_ids[position] for position in active_relative]
        if actions[0].tolist() != no_op:
            raise ValueError("candidate-zero action differs from source no-op")
        clean_teacher_index: int | None = None
        if str(row["source_domain"]) == "mp20_clean":
            clean_ids = _token_ids(self.tokenizer, str(row["clean_teacher_answer"]))
            if len(clean_ids) != len(source_ids):
                raise ValueError("clean teacher answer length changed")
            clean_action = tuple(clean_ids[position] for position in active_relative)
            for candidate_index, action in enumerate(actions.tolist()):
                if tuple(action) == clean_action:
                    clean_teacher_index = candidate_index
                    break
            if clean_teacher_index is None:
                raise ValueError("MP20-clean group does not retain its teacher action")
        metadata = {
            "group_idx": int(row["group_idx"]),
            "stratum": str(row["stratum"]),
            "source_sample_idx": int(row["source_sample_idx"]),
            "source_row_idx": int(row["source_row_idx"]),
            "plan_state": dict(row["plan_state"]),
            "source_answer": source_answer,
        }
        result = {
            "group_idx": int(row["group_idx"]),
            "stratum": str(row["stratum"]),
            "source_domain": str(row["source_domain"]),
            "transaction_kind": str(row["transaction_kind"]),
            "transaction_length": transaction_length,
            "input_ids": full,
            "attention_mask": torch.ones_like(full),
            "active_absolute": active_absolute,
            "active_relative": active_relative,
            "action_tokens": actions,
            "no_op_tokens": tuple(no_op),
            "differing_positions": tuple(
                tuple(int(value) for value in candidate["differing_positions"])
                for candidate in candidates
            ),
            "legal_mask": torch.ones((len(candidates),), dtype=torch.bool),
            "state_metadata_by_action": tuple(
                metadata for _ in range(len(candidates))
            ),
            "clean_teacher_index": clean_teacher_index,
            "informative": bool(row.get("informative")),
        }
        if include_energy:
            energies = []
            for candidate in candidates:
                value = candidate.get("raw_chgnet_energy_eV_per_atom")
                energies.append(math.nan if value is None else float(value))
            result["raw_energies"] = torch.tensor(energies, dtype=torch.float64)
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.materialize(index, include_energy=True)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def model_logits(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = getattr(output, "logits", None)
    if logits is None or logits.ndim != 3:
        raise RuntimeError("DLM did not return rank-three logits")
    return logits


def sequential_action_scores(
    runtime: Any,
    batch: Mapping[str, Any],
    *,
    reference: bool,
) -> torch.Tensor:
    """Score a complete 3/6-token action as sequential conditional log-probs."""

    if reference:
        runtime.activate_reference()
        gradient_context = torch.no_grad()
    else:
        runtime.activate_policy(trainable=True)
        gradient_context = nullcontext()
    source = batch["input_ids"].reshape(1, -1)
    attention = batch["attention_mask"].reshape(1, -1)
    positions = batch["active_absolute"]
    actions = batch["action_tokens"]
    if actions.ndim != 2 or int(actions.shape[1]) not in (3, 6):
        raise ValueError("actions must be K complete 3/6-token transactions")
    count = int(actions.shape[0])
    current = source.repeat(count, 1)
    expanded_attention = attention.repeat(count, 1)
    rows = torch.arange(count, device=source.device)
    score = torch.zeros((count,), dtype=torch.float32, device=source.device)
    with gradient_context, autocast_context(source.device):
        for transaction_offset, absolute_position in enumerate(positions.tolist()):
            logits = model_logits(runtime.model, current, expanded_attention)
            token_logp = F.log_softmax(
                logits[rows, int(absolute_position)].float(), dim=-1
            ).gather(
                1,
                actions[:, transaction_offset].unsqueeze(1),
            ).squeeze(1)
            score = score + token_logp
            current = current.clone()
            current[:, int(absolute_position)] = actions[:, transaction_offset]
    return score


def potential_group_loss(
    runtime: Any,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Potential posterior loss; uninformative groups retain zero weight."""

    reference_scores = sequential_action_scores(runtime, batch, reference=True)
    policy_scores = sequential_action_scores(runtime, batch, reference=False)
    if not bool(batch["informative"]):
        zero = policy_scores.sum() * 0.0
        return {
            "loss": zero,
            "teacher_kl_nats": 0.0,
            "informative": False,
            "action_count": int(batch["action_tokens"].shape[0]),
        }
    posterior = build_potential_closure_posterior(
        reference_scores,
        batch["raw_energies"],
        batch["legal_mask"],
        action_tokens=batch["action_tokens"].detach().cpu().tolist(),
        no_op_tokens=batch["no_op_tokens"],
        state_metadata_by_action=batch["state_metadata_by_action"],
        active_positions=batch["active_relative"],
        differing_positions_by_action=batch["differing_positions"],
        kl_budget_nats=MAX_KL_BUDGET_NATS,
    )
    if not posterior.informative:
        raise RuntimeError("labelled-informative group became uninformative")
    output = potential_closure_loss(policy_scores, posterior)
    return {
        "loss": output.loss,
        "teacher_kl_nats": float(posterior.kl_nats),
        "informative": True,
        "action_count": int(posterior.action_count),
        "transaction_length": int(posterior.transaction_length),
        "policy_kl": float(output.kl.detach().cpu()),
    }


def control_group_loss(runtime: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Equal-forward control using MP20 clean complete-transaction CE only."""

    if batch["source_domain"] != "mp20_clean":
        raise ValueError("closure control cannot consume on-policy states")
    target_index = batch["clean_teacher_index"]
    if target_index is None:
        raise ValueError("closure control lacks a clean teacher transaction")
    reference_scores = sequential_action_scores(runtime, batch, reference=True)
    policy_scores = sequential_action_scores(runtime, batch, reference=False)
    return {
        "loss": transaction_clean_ce(
            policy_scores,
            target_index=int(target_index),
            transaction_length=int(batch["transaction_length"]),
        ),
        "teacher_kl_nats": 0.0,
        "informative": True,
        "action_count": int(policy_scores.numel()),
        "step0_policy_reference_delta": float(
            torch.max(torch.abs(policy_scores.detach() - reference_scores.detach())).cpu()
        ),
    }


def step0_policy_reference_equality(
    runtime: Any,
    groups: PotentialClosureGroupDataset,
    device: torch.device,
) -> dict[str, Any]:
    deltas: dict[str, float] = {}
    for kind in KINDS:
        index = groups.informative_indices_by_stratum[f"mp20_clean_{kind}"][0]
        batch = move_to_device(
            groups.materialize(index, include_energy=False), device
        )
        reference = sequential_action_scores(runtime, batch, reference=True)
        policy = sequential_action_scores(runtime, batch, reference=False)
        delta = float(torch.max(torch.abs(policy.detach() - reference.detach())).cpu())
        if delta > 1.0e-6:
            raise RuntimeError(f"step0 policy/reference {kind} scores differ")
        deltas[kind] = delta
    return {"passed": True, "max_abs_score_delta_by_kind": deltas}


def clean_loss_config(tokenizer: Any, *, answer_token_count: int) -> dict[str, Any]:
    args = SimpleNamespace(
        representation="dynamic_v1",
        answer_token_count=int(answer_token_count),
        atom_count_loss_weight=1.0,
        slot_marker_loss_weight=1.0,
        empty_slot_loss_weight=1.0,
        nonempty_slot_loss_weight=1.0,
        late_slot_start=4,
        late_nonempty_slot_loss_weight=None,
        coordinate_loss_weight=1.0,
        pad_coordinate_loss_weight=1.0,
        physical_header_loss_weight=2.0,
        composition_module_loss_weight=2.0,
        lattice_module_loss_weight=1.0,
        sites_module_loss_weight=1.25,
        crysllmgen_lattice_loss_weight=1.0,
        crysllmgen_composition_loss_weight=2.5,
        crysllmgen_species_loss_weight=2.0,
        crysllmgen_coords_loss_weight=1.1,
        crysllmgen_site_coord_loss_weight=1.0,
        fixed_plain_count_loss_weight=3.0,
        fixed_plain_lattice_loss_weight=1.0,
        fixed_plain_elements_loss_weight=2.0,
        fixed_plain_coords_loss_weight=1.1,
        dynamic_lattice_length_loss_weight=1.0,
        dynamic_lattice_angle_loss_weight=1.0,
        dynamic_coord_loss_weight=1.0,
        dynamic_geometry_only=False,
        train_prefill_slot_tokens=False,
        periodic_metric_weight=0.0,
        periodic_pair_rdf_weight=0.0,
        periodic_overlap_weight=0.0,
        periodic_coordination_weight=0.0,
        periodic_exact_triclinic_pbc=False,
        periodic_image_radius=1,
        periodic_species_margin_scale=0.0,
        periodic_species_margin_floor=0.6,
        periodic_species_margin_ceiling=1.4,
        periodic_overlap_tail_temperature=0.1,
        periodic_overlap_tail_mix=0.0,
    )
    return SFT.build_loss_config(tokenizer, args)


def clean_ce_microbatch_loss(
    runtime: Any,
    dataset: Any,
    collator: Any,
    row_index: int,
    device: torch.device,
    loss_config: Mapping[str, Any],
) -> torch.Tensor:
    runtime.activate_policy(trainable=True)
    batch = move_to_device(collator([dataset[int(row_index)]]), device)
    loss = SFT.compute_loss(runtime.model, batch, dict(loss_config))
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("clean SPAD CE is nonfinite")
    return loss


def gradient_snapshot(
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor, ...]:
    values: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            values.append(torch.zeros_like(parameter, device="cpu", dtype=torch.float32))
        else:
            values.append(parameter.grad.detach().float().cpu().clone())
    return tuple(values)


def gradient_norm(snapshot: Sequence[torch.Tensor]) -> float:
    total = sum(float(torch.sum(value.double() * value.double()).item()) for value in snapshot)
    return math.sqrt(total)


def gradient_cosine(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> float:
    if len(left) != len(right):
        raise ValueError("gradient snapshots differ in length")
    left_norm = gradient_norm(left)
    right_norm = gradient_norm(right)
    if left_norm <= 0.0 or right_norm <= 0.0:
        return math.nan
    dot = sum(
        float(torch.sum(a.double() * b.double()).item())
        for a, b in zip(left, right, strict=True)
    )
    return dot / (left_norm * right_norm)


def probe_statistic_decision(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    if len(rows) != PROBE_BATCHES:
        raise ValueError("gradient probe requires exactly five rows")
    required = (
        "clean_grad_norm",
        "cell_grad_norm",
        "site_grad_norm",
        "cell_to_clean_ratio",
        "site_to_clean_ratio",
        "clean_cell_cosine",
        "clean_site_cosine",
        "max_teacher_kl_nats",
    )
    finite_nonzero_norms = all(
        math.isfinite(float(row[key])) and float(row[key]) > 0.0
        for row in rows
        for key in ("clean_grad_norm", "cell_grad_norm", "site_grad_norm")
    )
    all_finite = all(
        math.isfinite(float(row[key])) for row in rows for key in required
    )
    medians = {
        key: float(statistics.median(float(row[key]) for row in rows))
        for key in (
            "cell_to_clean_ratio",
            "site_to_clean_ratio",
            "clean_cell_cosine",
            "clean_site_cosine",
        )
    }
    ratio_ok = all(
        GRADIENT_RATIO_MIN <= medians[key] <= GRADIENT_RATIO_MAX
        for key in ("cell_to_clean_ratio", "site_to_clean_ratio")
    )
    cosine_ok = all(
        medians[key] > GRADIENT_COSINE_MIN
        for key in ("clean_cell_cosine", "clean_site_cosine")
    )
    kl_ok = all(
        float(row["max_teacher_kl_nats"]) <= MAX_KL_BUDGET_NATS + 1.0e-9
        for row in rows
    )
    gates = {
        "all_statistics_finite": all_finite,
        "all_gradient_norms_nonzero": finite_nonzero_norms,
        "posterior_to_ce_median_ratio_in_1e-2_1e2": ratio_ok,
        "clean_vs_posterior_median_cosine_gt_minus_0p5": cosine_ok,
        "all_teacher_kl_le_0p05_nat": kl_ok,
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "medians": medians,
    }


def verify_action_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("potential closure action manifest schema changed")
    if manifest.get("formal_action_pool_gate") is not True:
        raise RuntimeError("potential closure action manifest gate is not true")
    if int(manifest.get("groups", -1)) != EXPECTED_GROUPS:
        raise ValueError("potential closure action group denominator changed")
    return manifest


def verify_probe_report(path: Path) -> dict[str, Any]:
    report_path = Path(path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != PROBE_SCHEMA or report.get("passed") is not True:
        raise RuntimeError("gradient probe did not pass")
    if not (report_path.parent / "_SUCCESS").is_file():
        raise RuntimeError("gradient probe success marker is missing")
    return report


def load_clean_dataset(
    data_dir: Path, tokenizer: Any
) -> tuple[Any, Any, dict[str, Any]]:
    train_path = Path(data_dir) / "train.jsonl"
    dataset = SFT.JsonlSftDataset(
        train_path,
        tokenizer,
        MAX_LENGTH,
        fail_on_truncation=True,
    )
    if len(dataset) != EXPECTED_SFT_ROWS:
        raise ValueError("full SPAD schedule SFT train row count changed")
    for row in dataset.rows:
        if (
            row.get("schema") != "rollout_matched_transition_v1"
            or row.get("source_answer") is None
            or not row.get("forced_mask_positions")
            or not row.get("loss_positions")
        ):
            raise ValueError("clean CE data is not full SPAD schedule supervision")
    answer_token_count = SFT.infer_answer_token_count(Path(data_dir))
    return dataset, SFT.DataCollator(tokenizer), clean_loss_config(
        tokenizer, answer_token_count=answer_token_count
    )


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    loader_args = SimpleNamespace(
        model_path=Path(args.model_path),
        checkpoint_path=Path(args.checkpoint_path),
        data_dir=Path(args.sft_data_dir),
    )
    return D3PO.load_policy_and_reference_adapters(loader_args)


def backward_clean_batch(
    runtime: Any,
    dataset: Any,
    collator: Any,
    indices: Sequence[int],
    device: torch.device,
    loss_config: Mapping[str, Any],
) -> float:
    if len(indices) != GRADIENT_ACCUMULATION:
        raise ValueError("clean update must contain six microbatches")
    total = 0.0
    for index in indices:
        loss = clean_ce_microbatch_loss(
            runtime, dataset, collator, index, device, loss_config
        )
        (loss / GRADIENT_ACCUMULATION).backward()
        total += float(loss.detach().cpu())
    return total / GRADIENT_ACCUMULATION


def backward_transaction_batch(
    runtime: Any,
    groups: PotentialClosureGroupDataset,
    plan: TransactionBatchPlan,
    device: torch.device,
    *,
    mode: str,
) -> dict[str, float]:
    if len(plan.group_indices) != GRADIENT_ACCUMULATION:
        raise ValueError("transaction update must contain six microbatches")
    if mode == "potential_closed":
        if Counter(plan.domains) != Counter(
            {"mp20_clean": 3, "on_policy": 3}
        ):
            raise ValueError("potential update lost strict 3+3 domain balance")
    elif any(domain != "mp20_clean" for domain in plan.domains):
        raise ValueError("closure control consumed a non-clean group")
    else:
        if mode != "closure_control":
            raise ValueError(f"unknown mode {mode}")
    totals = Counter()
    max_teacher_kl = 0.0
    for index, expected_domain in zip(
        plan.group_indices, plan.domains, strict=True
    ):
        batch = move_to_device(
            groups.materialize(
                int(index), include_energy=mode == "potential_closed"
            ),
            device,
        )
        if batch["transaction_kind"] != plan.kind:
            raise ValueError("transaction batch kind changed")
        if batch["source_domain"] != expected_domain:
            raise ValueError("transaction batch domain changed")
        values = (
            potential_group_loss(runtime, batch)
            if mode == "potential_closed"
            else control_group_loss(runtime, batch)
        )
        loss = values["loss"]
        if not isinstance(loss, torch.Tensor) or not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("transaction loss is nonfinite")
        (loss / GRADIENT_ACCUMULATION).backward()
        totals["loss"] += float(loss.detach().cpu())
        totals["teacher_kl_nats"] += float(values["teacher_kl_nats"])
        max_teacher_kl = max(max_teacher_kl, float(values["teacher_kl_nats"]))
        totals["informative"] += float(bool(values["informative"]))
        totals["actions"] += float(values["action_count"])
    result = {
        key: float(value) / GRADIENT_ACCUMULATION
        for key, value in totals.items()
    }
    result["max_teacher_kl_nats"] = max_teacher_kl
    return result


def fixed_probe_plans(
    groups: PotentialClosureGroupDataset, *, seed: int
) -> dict[str, list[TransactionBatchPlan]]:
    result: dict[str, list[TransactionBatchPlan]] = {kind: [] for kind in KINDS}
    for kind_index, kind in enumerate(KINDS):
        clean = list(groups.informative_indices_by_stratum[f"mp20_clean_{kind}"])
        on_policy = list(groups.informative_indices_by_stratum[f"on_policy_{kind}"])
        if len(clean) < 15 or len(on_policy) < 15:
            raise RuntimeError("not enough informative groups for five fixed probes")
        random.Random(int(seed) + kind_index * 100).shuffle(clean)
        random.Random(int(seed) + kind_index * 100 + 1).shuffle(on_policy)
        for probe_index in range(PROBE_BATCHES):
            indices: list[int] = []
            domains: list[str] = []
            start = probe_index * POSTERIOR_DOMAIN_MICROBATCHES
            for left, right in zip(
                clean[start : start + 3],
                on_policy[start : start + 3],
                strict=True,
            ):
                indices.extend((left, right))
                domains.extend(("mp20_clean", "on_policy"))
            result[kind].append(
                TransactionBatchPlan(kind, tuple(indices), tuple(domains))
            )
    return result


def run_gradient_probe(
    args: argparse.Namespace,
    tokenizer: Any,
    runtime: Any,
    groups: PotentialClosureGroupDataset,
    clean_dataset: Any,
    clean_collator: Any,
    loss_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    parameters = tuple(
        parameter for parameter in runtime.policy_parameters if parameter.requires_grad
    )
    if not parameters:
        raise RuntimeError("policy LoRA has no trainable parameters")
    plans = fixed_probe_plans(groups, seed=int(args.seed) + 500)
    clean_stream = InfiniteEpochIndexStream(
        len(clean_dataset), seed=int(args.seed) + 600
    )
    rows: list[dict[str, float]] = []
    for probe_index in range(PROBE_BATCHES):
        runtime.model.zero_grad(set_to_none=True)
        backward_clean_batch(
            runtime,
            clean_dataset,
            clean_collator,
            clean_stream.take(GRADIENT_ACCUMULATION),
            device,
            loss_config,
        )
        clean_gradient = gradient_snapshot(parameters)
        clean_norm = gradient_norm(clean_gradient)

        runtime.model.zero_grad(set_to_none=True)
        cell_values = backward_transaction_batch(
            runtime,
            groups,
            plans["cell"][probe_index],
            device,
            mode="potential_closed",
        )
        cell_gradient = gradient_snapshot(parameters)
        cell_norm = gradient_norm(cell_gradient)

        runtime.model.zero_grad(set_to_none=True)
        site_values = backward_transaction_batch(
            runtime,
            groups,
            plans["site"][probe_index],
            device,
            mode="potential_closed",
        )
        site_gradient = gradient_snapshot(parameters)
        site_norm = gradient_norm(site_gradient)
        runtime.model.zero_grad(set_to_none=True)

        row = {
            "probe_batch": float(probe_index),
            "clean_grad_norm": clean_norm,
            "cell_grad_norm": cell_norm,
            "site_grad_norm": site_norm,
            "cell_to_clean_ratio": cell_norm / clean_norm if clean_norm else math.nan,
            "site_to_clean_ratio": site_norm / clean_norm if clean_norm else math.nan,
            "clean_cell_cosine": gradient_cosine(clean_gradient, cell_gradient),
            "clean_site_cosine": gradient_cosine(clean_gradient, site_gradient),
            "max_teacher_kl_nats": max(
                float(cell_values["max_teacher_kl_nats"]),
                float(site_values["max_teacher_kl_nats"]),
            ),
        }
        rows.append(row)
        append_jsonl(args.output_dir / "training_log.jsonl", {"event": "probe", **row})
    decision = probe_statistic_decision(rows)
    return {
        "schema": PROBE_SCHEMA,
        "status": "success" if decision["passed"] else "blocked",
        "passed": bool(decision["passed"]),
        "probe_batches": rows,
        **{key: value for key, value in decision.items() if key != "passed"},
        "diagnostic_only": True,
        "optimizer_updates": 0,
        "automatic_weight_lr_kl_or_epoch_changes": False,
    }


def save_endpoint(
    runtime: Any,
    tokenizer: Any,
    output_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    runtime.activate_policy(trainable=False)
    if D3PO.REFERENCE_ADAPTER not in runtime.model.peft_config:
        raise RuntimeError("reference adapter disappeared before save")
    runtime.model.delete_adapter(D3PO.REFERENCE_ADAPTER)
    runtime.model.set_adapter(D3PO.POLICY_ADAPTER)
    SFT.save_checkpoint(
        runtime.model,
        tokenizer,
        output_dir,
        TOTAL_UPDATES,
        save_embedding_layers="auto",
        data_dir=data_dir,
        is_main=True,
    )
    checkpoints = output_dir / "checkpoints"
    step_dirs = sorted(path.name for path in checkpoints.iterdir() if path.is_dir())
    if step_dirs != [f"step-{TOTAL_UPDATES}"]:
        raise RuntimeError(f"unexpected checkpoint directories: {step_dirs}")
    checkpoint = checkpoints / f"step-{TOTAL_UPDATES}"
    adapter_configs = list(checkpoint.rglob("adapter_config.json"))
    adapter_models = list(checkpoint.rglob("adapter_model.safetensors"))
    if len(adapter_configs) != 1 or len(adapter_models) != 1:
        raise RuntimeError("step2048 must contain exactly one policy adapter")
    if adapter_configs[0].parent != adapter_models[0].parent:
        raise RuntimeError("saved policy adapter files disagree")
    adapter_dir = adapter_configs[0].parent
    tokenizer.save_pretrained(adapter_dir)
    if set(runtime.model.peft_config) != {D3PO.POLICY_ADAPTER}:
        raise RuntimeError("reference adapter survived endpoint save")
    if (output_dir / "final").exists():
        raise RuntimeError("unexpected final checkpoint alias")
    return {
        "checkpoint_root": str(checkpoint.resolve()),
        "policy_adapter_path": str(adapter_dir.resolve()),
        "only_step2048": True,
        "reference_adapter_removed": True,
    }


def train(
    args: argparse.Namespace,
    tokenizer: Any,
    runtime: Any,
    groups: PotentialClosureGroupDataset,
    clean_dataset: Any,
    clean_collator: Any,
    loss_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    verify_probe_report(args.probe_report)
    streams = TransactionBatchStreams(
        groups.indices_by_stratum,
        mode=args.mode,
        seed=int(args.seed) + 1000,
    )
    clean_stream = InfiniteEpochIndexStream(
        len(clean_dataset), seed=int(args.seed) + 2000
    )
    parameters = tuple(
        parameter for parameter in runtime.policy_parameters if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.0)
    started = time.time()
    objective_counts = Counter()
    domain_exposures = Counter()
    for update in range(1, TOTAL_UPDATES + 1):
        objective = optimizer_objective(update)
        learning_rate = learning_rate_for_update(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        if objective == "clean_ce":
            metrics = {
                "loss": backward_clean_batch(
                    runtime,
                    clean_dataset,
                    clean_collator,
                    clean_stream.take(GRADIENT_ACCUMULATION),
                    device,
                    loss_config,
                )
            }
        else:
            plan = streams.take(objective)
            metrics = backward_transaction_batch(
                runtime,
                groups,
                plan,
                device,
                mode=args.mode,
            )
            for domain in plan.domains:
                domain_exposures[f"{domain}_{objective}"] += 1
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        gradient_value = float(torch.as_tensor(gradient).detach().cpu())
        if not math.isfinite(gradient_value) or gradient_value <= 0.0:
            raise FloatingPointError("optimizer gradient norm is nonfinite or zero")
        optimizer.step()
        objective_counts[objective] += 1
        if update == 1 or update % LOGGING_UPDATES == 0 or update == TOTAL_UPDATES:
            append_jsonl(
                args.output_dir / "training_log.jsonl",
                {
                    "event": "train",
                    "update": update,
                    "objective": objective,
                    "mode": args.mode,
                    "learning_rate": learning_rate,
                    "gradient_norm_before_clip": gradient_value,
                    "elapsed_sec": time.time() - started,
                    **metrics,
                },
            )
    streams.assert_exhausted()
    expected_objectives = Counter({"clean_ce": 1024, "cell": 512, "site": 512})
    if objective_counts != expected_objectives:
        raise RuntimeError("four-step objective accounting changed")
    if args.mode == "potential_closed":
        expected_domains = Counter(
            {
                "mp20_clean_cell": 1536,
                "on_policy_cell": 1536,
                "mp20_clean_site": 1536,
                "on_policy_site": 1536,
            }
        )
    else:
        expected_domains = Counter(
            {"mp20_clean_cell": 3072, "mp20_clean_site": 3072}
        )
    if domain_exposures != expected_domains:
        raise RuntimeError("transaction exposure accounting changed")
    checkpoint = save_endpoint(
        runtime, tokenizer, args.output_dir, Path(args.sft_data_dir)
    )
    return {
        "schema": RUN_SCHEMA,
        "status": "success",
        "mode": args.mode,
        "seed": int(args.seed),
        "optimizer_updates": TOTAL_UPDATES,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "objective_counts": dict(objective_counts),
        "transaction_exposures": dict(domain_exposures),
        "learning_rate": {
            "peak": LEARNING_RATE,
            "warmup_updates": WARMUP_UPDATES,
            "after_warmup": "constant",
        },
        "checkpoint": checkpoint,
        "on_policy_ce": False,
        "energy_labels_read": args.mode == "potential_closed",
        "elapsed_sec": time.time() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--labelled-groups", type=Path, required=True)
    parser.add_argument("--action-manifest", type=Path, required=True)
    parser.add_argument("--sft-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, default=99017)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-report", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("potential closure uses one process per visible GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("potential closure cell requires exactly one visible GPU")
    if args.probe_only and args.mode != "potential_closed":
        raise ValueError("gradient probe is defined for potential_closed mode")
    if not args.probe_only and args.probe_report is None:
        raise ValueError("training requires a passed --probe-report")
    args.output_dir = Path(args.output_dir).resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log_path = args.output_dir / "training_log.jsonl"
    config = {
        "schema": PROBE_SCHEMA if args.probe_only else RUN_SCHEMA,
        "mode": args.mode,
        "seed": int(args.seed),
        "probe_only": bool(args.probe_only),
        "model_path": str(Path(args.model_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "labelled_groups": str(Path(args.labelled_groups).resolve()),
        "action_manifest": str(Path(args.action_manifest).resolve()),
        "sft_data_dir": str(Path(args.sft_data_dir).resolve()),
        "updates": 0 if args.probe_only else TOTAL_UPDATES,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "four_step_cycle": ["clean_ce", "cell", "clean_ce", "site"],
        "warmup_updates": WARMUP_UPDATES,
        "learning_rate": LEARNING_RATE,
        "automatic_tuning": False,
    }
    write_json(args.output_dir / "RUN_CONFIG.json", config)
    append_jsonl(log_path, {"event": "start", **config})
    try:
        verify_action_manifest(Path(args.action_manifest))
        random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        tokenizer, runtime, adapter_report = load_runtime(args)
        device = torch.device("cuda", 0)
        runtime.model.to(device)
        groups = PotentialClosureGroupDataset(
            Path(args.labelled_groups), tokenizer, max_length=MAX_LENGTH
        )
        clean_dataset, clean_collator, loss_config = load_clean_dataset(
            Path(args.sft_data_dir), tokenizer
        )
        step0 = step0_policy_reference_equality(runtime, groups, device)
        config["adapter_report"] = adapter_report
        config["informative_by_stratum"] = groups.informative_by_stratum
        config["step0_policy_reference_equality"] = step0
        write_json(args.output_dir / "RUN_CONFIG.json", config)
        if args.probe_only:
            report = run_gradient_probe(
                args,
                tokenizer,
                runtime,
                groups,
                clean_dataset,
                clean_collator,
                loss_config,
                device,
            )
            write_json(args.output_dir / "PROBE_FINAL.json", report)
            append_jsonl(log_path, {"event": "probe_final", **report})
            if not report["passed"]:
                write_json(
                    args.output_dir / "_FAILED.json",
                    {"status": "blocked", "reason": "gradient_probe_gate"},
                )
                raise RuntimeError("gradient probe launch conditions did not pass")
        else:
            report = train(
                args,
                tokenizer,
                runtime,
                groups,
                clean_dataset,
                clean_collator,
                loss_config,
                device,
            )
            write_json(args.output_dir / "TRAIN_FINAL.json", report)
            append_jsonl(log_path, {"event": "success", **report})
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report, sort_keys=True))
    except Exception as exc:
        failure_path = args.output_dir / "_FAILED.json"
        if not failure_path.exists():
            write_json(
                failure_path,
                {
                    "schema": RUN_SCHEMA,
                    "status": "failed",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        append_jsonl(
            log_path,
            {
                "event": "failure",
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
