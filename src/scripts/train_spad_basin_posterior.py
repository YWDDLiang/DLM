#!/usr/bin/env python3
"""Pilot training for Llama-programmed SPAD basin transaction posteriors.

The trainer consumes exactly 128 labelled finite-action groups.  It scores
the supplied 3-token XYZ or 6-token lattice actions with the deployed SPAD
transaction process, builds a frozen-reference energy posterior inside a
0.05-nat KL ball, and alternates independent clean-CE and posterior optimizer
updates.  It intentionally contains no legacy stage resolver and never mixes
the two objectives in one scalar loss.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import hashlib
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
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GROUP_SCHEMA = "spad_basin_preflight_action_group_v1"
LABELLED_GROUP_SCHEMA = "spad_basin_preflight_labelled_group_v1"
EXPECTED_GROUPS = 128
WORLD_SIZE = 2
POSTERIOR_PASSES = 4
POSTERIOR_UPDATES = EXPECTED_GROUPS * POSTERIOR_PASSES // WORLD_SIZE
CLEAN_CE_UPDATES = POSTERIOR_UPDATES
TOTAL_UPDATES = POSTERIOR_UPDATES + CLEAN_CE_UPDATES
EXPECTED_CLEAN_ROWS = 27_136
SOURCE_ROLLOUT_BATCH_SIZE = 8
MAX_LENGTH = 382
ANSWER_TOKEN_COUNT = 87
LEARNING_RATE = 5.0e-6
WARMUP_UPDATES = 25
MIN_LR_RATIO = 0.1
GRADIENT_CLIP_NORM = 1.0
GRADIENT_PROBE_PAIRS = 5
MAX_KL_BUDGET_NATS = 0.05
DEPLOYED_TEMPERATURE = 0.7
DATA_ORDER_SEED = 20_260_904
ALLOWED_POSTERIOR_GRADIENT_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_VALUE_FIELD = "terminal_relax_k10_energy_eV_per_atom"
E0_VALUE_FIELD = "terminal_single_point_energy_eV_per_atom"
RUN_SCHEMA = "spad_basin_posterior_pilot_train_v1"


@lru_cache(maxsize=1)
def _runtime_modules() -> SimpleNamespace:
    """Import torch and model helpers only when a real training run starts."""

    import torch
    import torch.distributed as dist

    from crystal_dlm.deployed_transaction_scoring import (
        score_deployed_transaction_actions,
    )
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    from crystal_dlm.potential_closure import (
        build_potential_closure_posterior,
        potential_closure_loss,
    )
    from crystal_dlm.r5_dynamic_length import (
        exact_dynamic_schema_constraints,
        validate_dynamic_tokenizer_contract,
    )
    from scripts import llada_d3po, llada_sft
    from scripts.sample_llada_dynamic_crystals import (
        build_dynamic_lightweight_constraints,
    )

    return SimpleNamespace(
        torch=torch,
        dist=dist,
        score_actions=score_deployed_transaction_actions,
        mask_token_id=int(MASK_TOKEN_ID),
        build_posterior=build_potential_closure_posterior,
        posterior_loss=potential_closure_loss,
        exact_schema=exact_dynamic_schema_constraints,
        validate_tokenizer=validate_dynamic_tokenizer_contract,
        d3po=llada_d3po,
        sft=llada_sft,
        build_constraints=build_dynamic_lightweight_constraints,
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_supported_group_schema(schema: Any) -> bool:
    value = str(schema or "")
    return bool(
        value == GROUP_SCHEMA
        or value == LABELLED_GROUP_SCHEMA
        or value.startswith(GROUP_SCHEMA + "_")
        or (
            value.startswith("spad_basin_preflight_")
            and "action_group" in value
            and "label" in value
        )
    )


def _integer_sequence(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} must contain integers")
        result.append(int(item))
    return tuple(result)


def _candidate_action(candidate: Mapping[str, Any]) -> tuple[int, ...]:
    primary = candidate.get("action_token_ids")
    alternate = candidate.get("token_ids")
    if primary is None and alternate is None:
        raise ValueError("candidate lacks action_token_ids/token_ids")
    if primary is not None and alternate is not None:
        left = _integer_sequence(primary, name="action_token_ids")
        right = _integer_sequence(alternate, name="token_ids")
        if left != right:
            raise ValueError("candidate action_token_ids and token_ids disagree")
        return left
    return _integer_sequence(
        primary if primary is not None else alternate,
        name="candidate action token IDs",
    )


def _nested_value(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for component in str(path).split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise KeyError(path)
        value = value[component]
    return value


def candidate_scalar(candidate: Mapping[str, Any], field: str) -> float:
    """Read one candidate-nested scalar; absent/nonfinite labels are unknown."""

    locations: tuple[Mapping[str, Any], ...] = tuple(
        value
        for value in (
            candidate,
            candidate.get("terminal_values"),
            candidate.get("values"),
        )
        if isinstance(value, Mapping)
    )
    raw: Any = None
    found = False
    for location in locations:
        try:
            raw = _nested_value(location, field)
            found = True
            break
        except KeyError:
            if field in location:
                raw = location[field]
                found = True
                break
    if not found or raw is None:
        return math.nan
    if isinstance(raw, bool):
        raise TypeError(f"candidate value {field!r} cannot be boolean")
    value = float(raw)
    return value if math.isfinite(value) else math.nan


def normalize_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    transaction_width: int,
    value_field: str,
) -> tuple[dict[str, Any], ...]:
    """Return distinct K1..4 actions with the declared no-op first."""

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence")
    if not 1 <= len(candidates) <= 4:
        raise ValueError("retained candidate K must lie in [1,4]")
    normalized: list[dict[str, Any]] = []
    signatures: set[tuple[int, ...]] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("every candidate must be an object")
        action = _candidate_action(candidate)
        if len(action) != int(transaction_width):
            raise ValueError("candidate width differs from the active transaction")
        if action in signatures:
            raise ValueError("retained candidates must be unique")
        signatures.add(action)
        terminal_legal = candidate.get("terminal_legal")
        if type(terminal_legal) is not bool:
            raise TypeError("candidate terminal_legal must be boolean")
        source = str(candidate.get("source") or "")
        if not source:
            raise ValueError("candidate source is required")
        normalized.append(
            {
                "source": source,
                "action_token_ids": action,
                "terminal_legal": bool(terminal_legal),
                "value": candidate_scalar(candidate, value_field),
                "raw": dict(candidate),
            }
        )
    no_op_indices = [
        index for index, candidate in enumerate(normalized) if candidate["source"] == "no_op"
    ]
    if len(no_op_indices) != 1:
        raise ValueError("each group must contain exactly one source=no_op candidate")
    no_op_index = no_op_indices[0]
    return tuple(
        [normalized[no_op_index]]
        + [value for index, value in enumerate(normalized) if index != no_op_index]
    )


def left_pad_prompt_ids(
    token_ids: Sequence[int], *, target_length: int, pad_token_id: int
) -> tuple[list[int], list[int]]:
    """Reproduce the source rollout's left-padded prompt positions."""

    values = [int(value) for value in token_ids]
    padding = int(target_length) - len(values)
    if padding < 0:
        raise ValueError("target prompt length is shorter than the prompt")
    return (
        [int(pad_token_id)] * padding + values,
        [0] * padding + [1] * len(values),
    )


def zero_posterior_reason(candidates: Sequence[Mapping[str, Any]]) -> str | None:
    """K1 and groups with fewer than two known legal values are exact zeros."""

    if len(candidates) == 1:
        return "k1_retained"
    legal_known = sum(
        bool(candidate.get("terminal_legal"))
        and math.isfinite(float(candidate.get("value", math.nan)))
        for candidate in candidates
    )
    if legal_known < 2:
        return "fewer_than_two_known_legal_values"
    if not bool(candidates[0].get("terminal_legal")):
        return "no_op_terminal_illegal"
    return None


def _normalize_group(
    row: Mapping[str, Any], ordinal: int, *, value_field: str
) -> dict[str, Any]:
    if not _is_supported_group_schema(row.get("schema")):
        raise ValueError(f"group {ordinal} has unsupported schema {row.get('schema')!r}")
    state = row.get("state")
    if not isinstance(state, Mapping):
        raise ValueError(f"group {ordinal} lacks nested materialized state")
    required = (
        "sample_idx",
        "prompt",
        "state_body",
        "N",
        "plan_state",
        "species_program",
        "state_type",
        "active_generation_positions",
        "context_masked_generation_positions",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"group {ordinal} state lacks fields: {','.join(missing)}")
    sample_idx = int(state["sample_idx"])
    if sample_idx != int(ordinal):
        raise ValueError("128 groups must remain ordered by state.sample_idx 0..127")
    if row.get("sample_idx") is not None and int(row["sample_idx"]) != sample_idx:
        raise ValueError("row/state sample_idx disagree")
    num_atoms = int(state["N"])
    if not 1 <= num_atoms <= 20:
        raise ValueError("state N must lie in [1,20]")
    plan = state["plan_state"]
    if not isinstance(plan, Mapping):
        raise TypeError("state plan_state must be an object")
    if plan.get("N") is not None and int(plan["N"]) != num_atoms:
        raise ValueError("state and plan N disagree")
    program = state["species_program"]
    if isinstance(program, (str, bytes)) or not isinstance(program, Sequence) or not program:
        raise ValueError("state species_program must be a nonempty sequence")
    active = _integer_sequence(
        state["active_generation_positions"],
        name="active_generation_positions",
    )
    context = _integer_sequence(
        state["context_masked_generation_positions"],
        name="context_masked_generation_positions",
    )
    state_type = str(state["state_type"])
    expected_width = 6 if state_type == "cell" else 3 if state_type == "xyz" else 0
    if expected_width == 0 or len(active) != expected_width:
        raise ValueError("state_type and active transaction width disagree")
    if len(set(active)) != len(active) or len(set(context)) != len(context):
        raise ValueError("active/context positions must be unique")
    if set(active).intersection(context):
        raise ValueError("context masks overlap the active transaction")
    gen_length = 7 + 4 * num_atoms
    if any(position < 0 or position >= gen_length for position in (*active, *context)):
        raise ValueError("active/context position lies outside exact 7+4N body")
    if state_type == "cell" and active != tuple(range(1, 7)):
        raise ValueError("cell transaction must use native six-token lattice order")
    if state_type == "xyz":
        first = active[0]
        if first < 8 or (first - 8) % 4 or active != (first, first + 1, first + 2):
            raise ValueError("XYZ transaction must use one native coordinate triplet")
    prompt = state["prompt"]
    state_body = state["state_body"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("state prompt must be nonempty text")
    if not isinstance(state_body, str) or not state_body:
        raise ValueError("state_body must be nonempty text")
    candidates_raw = row.get("candidates")
    if candidates_raw is None:
        candidates_raw = state.get("candidates")
    candidates = normalize_candidates(
        candidates_raw,
        transaction_width=expected_width,
        value_field=value_field,
    )
    return {
        "schema": str(row["schema"]),
        "sample_idx": sample_idx,
        "prompt": prompt,
        "state_body": state_body,
        "N": num_atoms,
        "plan_state": dict(plan),
        "species_program": list(program),
        "state_type": state_type,
        "active_generation_positions": active,
        "context_masked_generation_positions": context,
        "candidates": candidates,
        "zero_posterior_reason": zero_posterior_reason(candidates),
    }


class BasinPosteriorGroupDataset:
    """Strict, ordered 128-group pilot dataset with dynamic K1..4."""

    def __init__(self, path: Path, *, value_field: str) -> None:
        raw = list(iter_jsonl(Path(path)))
        if len(raw) != EXPECTED_GROUPS:
            raise ValueError(f"labelled input must contain exactly {EXPECTED_GROUPS} groups")
        self.path = Path(path).resolve()
        self.value_field = str(value_field)
        self.rows = tuple(
            _normalize_group(row, ordinal, value_field=self.value_field)
            for ordinal, row in enumerate(raw)
        )
        self._allowed_cache: dict[tuple[int, int], Any] = {}
        self._constraints_cache: dict[int, Any] = {}
        self._prompt_batch_length_cache: dict[tuple[int, int], int] = {}

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def preinformative_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, row in enumerate(self.rows)
            if row["zero_posterior_reason"] is None
        )

    def summary(self) -> dict[str, Any]:
        k_hist = Counter(len(row["candidates"]) for row in self.rows)
        state_hist = Counter(row["state_type"] for row in self.rows)
        zero_hist = Counter(
            row["zero_posterior_reason"] or "preinformative" for row in self.rows
        )
        return {
            "schema": "spad_basin_posterior_pilot_dataset_v1",
            "groups": len(self.rows),
            "ordered_sample_idx": True,
            "candidate_k_histogram": dict(sorted(k_hist.items())),
            "state_type_counts": dict(sorted(state_hist.items())),
            "prelabel_status_counts": dict(sorted(zero_hist.items())),
            "value_field": self.value_field,
        }

    def materialize(
        self,
        index: int,
        tokenizer: Any,
        modules: SimpleNamespace,
    ) -> dict[str, Any]:
        row = self.rows[int(index)]
        prompt_text = str(row["prompt"]).rstrip() + "\n"
        unpadded_prompt_ids = [
            int(value)
            for value in tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        ]
        source_batch = int(index) // SOURCE_ROLLOUT_BATCH_SIZE
        prompt_cache_key = (id(tokenizer), source_batch)
        if prompt_cache_key not in self._prompt_batch_length_cache:
            start = source_batch * SOURCE_ROLLOUT_BATCH_SIZE
            stop = min(start + SOURCE_ROLLOUT_BATCH_SIZE, len(self.rows))
            self._prompt_batch_length_cache[prompt_cache_key] = max(
                len(
                    tokenizer(
                        str(self.rows[row_index]["prompt"]).rstrip() + "\n",
                        add_special_tokens=False,
                    )["input_ids"]
                )
                for row_index in range(start, stop)
            )
        prompt_ids, prompt_attention = left_pad_prompt_ids(
            unpadded_prompt_ids,
            target_length=self._prompt_batch_length_cache[prompt_cache_key],
            pad_token_id=int(tokenizer.pad_token_id),
        )
        body_ids = [
            int(value)
            for value in tokenizer(
                str(row["state_body"]), add_special_tokens=False
            )["input_ids"]
        ]
        gen_length = 7 + 4 * int(row["N"])
        if len(body_ids) != gen_length:
            raise ValueError(
                f"group {index} state_body is not exact 7+4N: {len(body_ids)} != {gen_length}"
            )
        if len(prompt_ids) + gen_length > MAX_LENGTH:
            raise ValueError(f"group {index} exceeds max_length={MAX_LENGTH}")
        context = tuple(int(value) for value in row["context_masked_generation_positions"])
        if any(body_ids[position] != modules.mask_token_id for position in context):
            raise ValueError("declared context masks do not contain the DLM mask token")
        cache_key = (id(tokenizer), int(row["N"]))
        if cache_key not in self._allowed_cache:
            self._allowed_cache[cache_key] = modules.exact_schema(
                tokenizer, int(row["N"])
            )
        tokenizer_key = id(tokenizer)
        if tokenizer_key not in self._constraints_cache:
            self._constraints_cache[tokenizer_key] = modules.build_constraints(
                tokenizer,
                duplicate_coordinate_mask=True,
                lattice_volume_mask=True,
                min_lattice_rad=1.0e-4,
                canonicalize_periodic_alias=True,
                pbc_min_distance_mask=True,
                pbc_min_distance_A=0.5,
                pbc_image_radius=2,
            )
        candidates = row["candidates"]
        actions = [list(candidate["action_token_ids"]) for candidate in candidates]
        no_op = tuple(int(value) for value in actions[0])
        active = tuple(int(value) for value in row["active_generation_positions"])
        differing = tuple(
            tuple(
                position
                for position, candidate_token, no_op_token in zip(
                    active, action, no_op, strict=True
                )
                if int(candidate_token) != int(no_op_token)
            )
            for action in actions
        )
        metadata = {
            "sample_idx": int(row["sample_idx"]),
            "state_type": str(row["state_type"]),
            "plan_state": dict(row["plan_state"]),
            "species_program": list(row["species_program"]),
        }
        torch = modules.torch
        complete = torch.tensor(prompt_ids + body_ids, dtype=torch.long)
        attention = torch.tensor(
            prompt_attention + [1] * len(body_ids), dtype=torch.long
        )
        return {
            "sample_idx": int(row["sample_idx"]),
            "complete_tokens": complete,
            "attention_mask": attention,
            "prompt_length": len(prompt_ids),
            "gen_length": gen_length,
            "N": int(row["N"]),
            "generation_positions": active,
            "context_masked_generation_positions": context,
            "action_token_ids": torch.tensor(actions, dtype=torch.long),
            "terminal_legal": torch.tensor(
                [bool(candidate["terminal_legal"]) for candidate in candidates],
                dtype=torch.bool,
            ),
            "energies": torch.tensor(
                [float(candidate["value"]) for candidate in candidates],
                dtype=torch.float64,
            ),
            "no_op_tokens": no_op,
            "differing_positions_by_action": differing,
            "state_metadata_by_action": tuple(metadata for _ in candidates),
            "allowed_token_ids_by_generation_pos": self._allowed_cache[cache_key],
            "lightweight_decoding_constraints": self._constraints_cache[tokenizer_key],
            "prelabel_zero_reason": row["zero_posterior_reason"],
            "action_sources": tuple(candidate["source"] for candidate in candidates),
        }


def optimizer_objective(update: int) -> str:
    if not 1 <= int(update) <= TOTAL_UPDATES:
        raise ValueError(f"update must lie in [1,{TOTAL_UPDATES}]")
    return "clean_ce" if int(update) % 2 else "transaction_posterior"


def deterministic_posterior_schedule(
    group_count: int = EXPECTED_GROUPS,
    *,
    passes: int = POSTERIOR_PASSES,
    world_size: int = WORLD_SIZE,
    seed: int = DATA_ORDER_SEED,
) -> tuple[tuple[int, ...], ...]:
    """Return global rank-tuples; every group occurs once in every pass."""

    if int(group_count) <= 0 or int(group_count) % int(world_size):
        raise ValueError("group_count must be positive and divisible by world_size")
    if int(passes) <= 0 or int(world_size) <= 0:
        raise ValueError("passes and world_size must be positive")
    result: list[tuple[int, ...]] = []
    for pass_index in range(int(passes)):
        order = list(range(int(group_count)))
        random.Random(int(seed) + pass_index).shuffle(order)
        for start in range(0, len(order), int(world_size)):
            result.append(tuple(order[start : start + int(world_size)]))
    return tuple(result)


def deterministic_clean_indices(
    row_count: int = EXPECTED_CLEAN_ROWS,
    *,
    updates: int = CLEAN_CE_UPDATES,
    world_size: int = WORLD_SIZE,
    seed: int = DATA_ORDER_SEED + 10_000,
) -> tuple[tuple[int, ...], ...]:
    required = int(updates) * int(world_size)
    if int(row_count) < required:
        raise ValueError("clean split is too small for unique pilot examples")
    order = list(range(int(row_count)))
    random.Random(int(seed)).shuffle(order)
    return tuple(
        tuple(order[start : start + int(world_size)])
        for start in range(0, required, int(world_size))
    )


def learning_rate_for_update(update: int) -> float:
    if not 1 <= int(update) <= TOTAL_UPDATES:
        raise ValueError(f"update must lie in [1,{TOTAL_UPDATES}]")
    if int(update) <= WARMUP_UPDATES:
        return LEARNING_RATE * float(update) / float(WARMUP_UPDATES)
    progress = float(update - WARMUP_UPDATES) / float(TOTAL_UPDATES - WARMUP_UPDATES)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * (MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine)


def select_posterior_gradient_scale(
    clean_norms: Sequence[float], posterior_norms: Sequence[float]
) -> dict[str, Any]:
    """Choose one frozen power-of-two multiplier nearest median balance."""

    if len(clean_norms) != GRADIENT_PROBE_PAIRS or len(posterior_norms) != GRADIENT_PROBE_PAIRS:
        raise ValueError("gradient scale selection requires exactly five pairs")
    values = [float(value) for value in (*clean_norms, *posterior_norms)]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise FloatingPointError("gradient probe contains nonfinite/zero norms")
    clean = float(statistics.median(float(value) for value in clean_norms))
    posterior = float(statistics.median(float(value) for value in posterior_norms))
    selected = min(
        ALLOWED_POSTERIOR_GRADIENT_SCALES,
        key=lambda scale: (
            abs(math.log((posterior * float(scale)) / clean)),
            abs(math.log2(float(scale))),
        ),
    )
    return {
        "allowed_scales": list(ALLOWED_POSTERIOR_GRADIENT_SCALES),
        "median_clean_gradient_norm": clean,
        "median_posterior_gradient_norm": posterior,
        "selected_posterior_gradient_multiplier": float(selected),
        "scaled_median_posterior_gradient_norm": posterior * float(selected),
        "frozen_for_all_posterior_updates": True,
        "per_batch_rescaling": False,
    }


def deployed_scoring_contract(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the exact arguments that bind training to deployed scoring."""

    return {
        "prompt_length": int(batch["prompt_length"]),
        "gen_length": int(batch["gen_length"]),
        "generation_positions": tuple(int(value) for value in batch["generation_positions"]),
        "context_masked_generation_positions": tuple(
            int(value) for value in batch["context_masked_generation_positions"]
        ),
        "temperature": DEPLOYED_TEMPERATURE,
        "mask_id": int(batch["mask_id"]),
        "atom_count_grammar": None,
        "allowed_token_ids_by_generation_pos": batch[
            "allowed_token_ids_by_generation_pos"
        ],
        "lightweight_decoding_constraints": batch[
            "lightweight_decoding_constraints"
        ],
    }


def move_to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to") and value.__class__.__module__.startswith("torch"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def _score_actions(
    runtime: Any,
    batch: Mapping[str, Any],
    modules: SimpleNamespace,
    *,
    reference: bool,
    require_grad: bool,
) -> Any:
    if reference:
        runtime.activate_reference()
    else:
        runtime.activate_policy(trainable=require_grad)
    context = nullcontext() if require_grad else modules.torch.no_grad()
    scoring_batch = dict(batch)
    scoring_batch["mask_id"] = modules.mask_token_id
    with context:
        return modules.score_actions(
            runtime.model,
            batch["complete_tokens"],
            action_token_ids=batch["action_token_ids"],
            attention_mask=batch["attention_mask"],
            **deployed_scoring_contract(scoring_batch),
        )


def transaction_posterior_objective(
    runtime: Any,
    batch: Mapping[str, Any],
    modules: SimpleNamespace,
    *,
    require_grad: bool = True,
) -> dict[str, Any]:
    """Compute one supplied-action posterior objective without terminal reranking."""

    reference = _score_actions(
        runtime, batch, modules, reference=True, require_grad=False
    )
    policy = _score_actions(
        runtime, batch, modules, reference=False, require_grad=require_grad
    )
    if not modules.torch.equal(reference.valid_mask, policy.valid_mask):
        raise RuntimeError("reference/policy deployed action-valid masks disagree")
    path_valid = reference.valid_mask & policy.valid_mask
    terminal_legal = batch["terminal_legal"].to(device=path_valid.device)
    legal = path_valid & terminal_legal
    energies = batch["energies"].to(device=path_valid.device)
    known_legal = legal & modules.torch.isfinite(energies)
    zero_reason = batch.get("prelabel_zero_reason")
    if int(batch["action_token_ids"].shape[0]) == 1:
        zero_reason = "k1_retained"
    elif int(known_legal.sum().detach().cpu()) < 2:
        zero_reason = "fewer_than_two_known_legal_values_after_deployed_masks"
    elif not bool(legal[0].detach().cpu()):
        zero_reason = "no_op_illegal_after_deployed_masks"
    finite_policy = modules.torch.where(
        modules.torch.isfinite(policy.action_logprobs),
        policy.action_logprobs,
        modules.torch.zeros_like(policy.action_logprobs),
    )
    if zero_reason is not None:
        return {
            "loss": finite_policy.sum() * 0.0,
            "informative": False,
            "zero_reason": str(zero_reason),
            "teacher_kl_nats": 0.0,
            "action_count": int(policy.action_logprobs.numel()),
            "unique_valid_action_count": int(reference.unique_valid_action_count),
        }
    finite_reference = modules.torch.where(
        path_valid,
        reference.action_logprobs,
        modules.torch.zeros_like(reference.action_logprobs),
    )
    finite_policy_scores = modules.torch.where(
        path_valid,
        policy.action_logprobs,
        modules.torch.zeros_like(policy.action_logprobs),
    )
    actions = batch["action_token_ids"].detach().cpu().tolist()
    posterior = modules.build_posterior(
        finite_reference,
        energies,
        legal,
        action_tokens=actions,
        no_op_tokens=batch["no_op_tokens"],
        state_metadata_by_action=batch["state_metadata_by_action"],
        active_positions=batch["generation_positions"],
        differing_positions_by_action=batch["differing_positions_by_action"],
        kl_budget_nats=MAX_KL_BUDGET_NATS,
    )
    if not posterior.informative:
        return {
            "loss": finite_policy.sum() * 0.0,
            "informative": False,
            "zero_reason": "energy_posterior_uninformative",
            "teacher_kl_nats": 0.0,
            "action_count": int(policy.action_logprobs.numel()),
            "unique_valid_action_count": int(reference.unique_valid_action_count),
        }
    output = modules.posterior_loss(finite_policy_scores, posterior)
    if not bool(modules.torch.isfinite(output.loss).detach().cpu()):
        raise FloatingPointError("transaction posterior loss is nonfinite")
    return {
        "loss": output.loss,
        "informative": True,
        "zero_reason": None,
        "teacher_kl_nats": float(posterior.kl_nats),
        "action_count": int(posterior.action_count),
        "unique_valid_action_count": int(reference.unique_valid_action_count),
    }


def _clean_loss_config(tokenizer: Any, modules: SimpleNamespace) -> dict[str, Any]:
    args = SimpleNamespace(
        representation="dynamic_v1",
        answer_token_count=ANSWER_TOKEN_COUNT,
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
        dynamic_geometry_only=True,
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
    return modules.sft.build_loss_config(tokenizer, args)


def load_clean_sft(
    data_dir: Path, tokenizer: Any, modules: SimpleNamespace
) -> tuple[Any, Any, dict[str, Any]]:
    train_path = Path(data_dir) / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    dataset = modules.sft.JsonlSftDataset(
        train_path,
        tokenizer,
        MAX_LENGTH,
        fail_on_truncation=True,
    )
    if len(dataset) != EXPECTED_CLEAN_ROWS:
        raise ValueError("clean closure SFT train split must contain 27,136 rows")
    return dataset, modules.sft.DataCollator(tokenizer), _clean_loss_config(
        tokenizer, modules
    )


def clean_ce_loss(
    runtime: Any,
    dataset: Any,
    collator: Any,
    row_index: int,
    device: Any,
    loss_config: Mapping[str, Any],
    modules: SimpleNamespace,
) -> Any:
    runtime.activate_policy(trainable=True)
    batch = move_to_device(collator([dataset[int(row_index)]]), device)
    loss = modules.sft.compute_loss(runtime.model, batch, dict(loss_config))
    if not bool(modules.torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("clean closure CE loss is nonfinite")
    return loss


def average_gradients_(
    parameters: Sequence[Any],
    *,
    world_size: int,
    reduce_fn: Callable[[Any], Any] | None = None,
) -> None:
    modules = _runtime_modules()
    reducer = reduce_fn or modules.dist.all_reduce
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = modules.torch.zeros_like(parameter)
        if parameter.grad.is_sparse:
            raise TypeError("sparse gradients are unsupported")
        reducer(parameter.grad)
        parameter.grad.div_(float(world_size))


def gradient_snapshot(parameters: Sequence[Any]) -> tuple[Any, ...]:
    modules = _runtime_modules()
    return tuple(
        modules.torch.zeros_like(parameter, device="cpu", dtype=modules.torch.float32)
        if parameter.grad is None
        else parameter.grad.detach().float().cpu().clone()
        for parameter in parameters
    )


def gradient_norm(snapshot: Sequence[Any]) -> float:
    total = sum(float(value.detach().double().square().sum()) for value in snapshot)
    return math.sqrt(total)


def gradient_cosine(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if len(left) != len(right):
        raise ValueError("gradient snapshots differ in length")
    left_norm = gradient_norm(left)
    right_norm = gradient_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    dot = sum(
        float((a.detach().double() * b.detach().double()).sum())
        for a, b in zip(left, right, strict=True)
    )
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def current_gradient_norm(parameters: Sequence[Any], modules: SimpleNamespace) -> float:
    total = modules.torch.zeros((), dtype=modules.torch.float64, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().double().square().sum()
    return math.sqrt(float(total.detach().cpu()))


def _reduce_scalar(value: float, device: Any, modules: SimpleNamespace, *, op: Any = None) -> float:
    tensor = modules.torch.tensor(float(value), dtype=modules.torch.float64, device=device)
    modules.dist.all_reduce(
        tensor,
        op=modules.dist.ReduceOp.SUM if op is None else op,
    )
    return float(tensor.detach().cpu())


def step0_policy_reference_equality(
    runtime: Any,
    dataset: BasinPosteriorGroupDataset,
    tokenizer: Any,
    device: Any,
    rank: int,
    modules: SimpleNamespace,
) -> dict[str, Any]:
    local_max = 0.0
    checked = 0
    for index in range(int(rank), len(dataset), WORLD_SIZE):
        batch = move_to_device(dataset.materialize(index, tokenizer, modules), device)
        reference = _score_actions(
            runtime, batch, modules, reference=True, require_grad=False
        )
        policy = _score_actions(
            runtime, batch, modules, reference=False, require_grad=False
        )
        if not modules.torch.equal(reference.valid_mask, policy.valid_mask):
            raise RuntimeError("step0 policy/reference valid masks differ")
        finite = reference.valid_mask & policy.valid_mask
        if bool(finite.any().detach().cpu()):
            delta = float(
                (reference.action_logprobs[finite] - policy.action_logprobs[finite])
                .abs()
                .max()
                .detach()
                .cpu()
            )
            local_max = max(local_max, delta)
        checked += 1
    max_tensor = modules.torch.tensor(local_max, dtype=modules.torch.float64, device=device)
    modules.dist.all_reduce(max_tensor, op=modules.dist.ReduceOp.MAX)
    global_checked = int(_reduce_scalar(checked, device, modules))
    maximum = float(max_tensor.detach().cpu())
    if global_checked != EXPECTED_GROUPS:
        raise RuntimeError("step0 equality did not cover all 128 groups")
    if not math.isfinite(maximum) or maximum > 1.0e-6:
        raise RuntimeError("step0 policy/reference supplied-action scores differ")
    return {
        "passed": True,
        "groups_checked": global_checked,
        "max_abs_supplied_action_score_delta": maximum,
        "tolerance": 1.0e-6,
    }


def run_gradient_probe(
    runtime: Any,
    group_dataset: BasinPosteriorGroupDataset,
    clean_dataset: Any,
    clean_collator: Any,
    loss_config: Mapping[str, Any],
    tokenizer: Any,
    device: Any,
    rank: int,
    modules: SimpleNamespace,
) -> dict[str, Any]:
    informative = group_dataset.preinformative_indices
    if len(informative) < GRADIENT_PROBE_PAIRS:
        raise RuntimeError("fewer than five preinformative groups for gradient probe")
    parameters = tuple(
        parameter for parameter in runtime.policy_parameters if parameter.requires_grad
    )
    python_state = random.getstate()
    cpu_state = modules.torch.get_rng_state()
    cuda_state = modules.torch.cuda.get_rng_state(device)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    try:
        for group_index in informative:
            if len(records) >= GRADIENT_PROBE_PAIRS:
                break
            runtime.model.zero_grad(set_to_none=True)
            group = move_to_device(
                group_dataset.materialize(group_index, tokenizer, modules), device
            )
            posterior = transaction_posterior_objective(
                runtime, group, modules, require_grad=True
            )
            posterior["loss"].backward()
            average_gradients_(parameters, world_size=WORLD_SIZE)
            post_snapshot = gradient_snapshot(parameters)
            post_norm = gradient_norm(post_snapshot)
            if not math.isfinite(post_norm):
                raise FloatingPointError("gradient probe produced nonfinite posterior gradient")
            if not posterior["informative"] or post_norm <= 0.0:
                skipped.append(
                    {
                        "posterior_group": int(group_index),
                        "reason": str(
                            posterior.get("zero_reason")
                            or "zero_gradient_after_deployed_masks"
                        ),
                        "posterior_gradient_norm": post_norm,
                    }
                )
                continue

            runtime.model.zero_grad(set_to_none=True)
            pair_index = len(records)
            clean_index = 2 * pair_index + int(rank)
            clean = clean_ce_loss(
                runtime,
                clean_dataset,
                clean_collator,
                clean_index,
                device,
                loss_config,
                modules,
            )
            clean.backward()
            average_gradients_(parameters, world_size=WORLD_SIZE)
            clean_snapshot = gradient_snapshot(parameters)
            clean_norm = gradient_norm(clean_snapshot)
            cosine = gradient_cosine(clean_snapshot, post_snapshot)
            if (
                not math.isfinite(clean_norm)
                or clean_norm <= 0.0
                or cosine is None
                or not math.isfinite(cosine)
            ):
                raise FloatingPointError("gradient probe produced nonfinite/zero gradients")
            records.append(
                {
                    "pair": pair_index + 1,
                    "clean_row_by_rank": [2 * pair_index, 2 * pair_index + 1],
                    "posterior_group": int(group_index),
                    "clean_gradient_norm": clean_norm,
                    "posterior_gradient_norm": post_norm,
                    "clean_posterior_cosine": float(cosine),
                    "clean_loss_rank0": float(clean.detach().cpu()),
                    "posterior_loss_rank0": float(posterior["loss"].detach().cpu()),
                }
            )
        if len(records) != GRADIENT_PROBE_PAIRS:
            raise RuntimeError(
                "fewer than five deployed-informative groups for gradient probe"
            )
    finally:
        runtime.model.zero_grad(set_to_none=True)
        random.setstate(python_state)
        modules.torch.set_rng_state(cpu_state)
        modules.torch.cuda.set_rng_state(cuda_state, device)
        runtime.activate_policy(trainable=True)
    clean_norms = [float(row["clean_gradient_norm"]) for row in records]
    post_norms = [float(row["posterior_gradient_norm"]) for row in records]
    cosines = [float(row["clean_posterior_cosine"]) for row in records]
    scale = select_posterior_gradient_scale(clean_norms, post_norms)
    median_cosine = float(statistics.median(cosines))
    if median_cosine < -0.8:
        raise RuntimeError("median clean/posterior gradient cosine is below -0.8")
    return {
        "schema": "spad_basin_posterior_gradient_probe_v1",
        "pairs": GRADIENT_PROBE_PAIRS,
        "records": records,
        "zero_information_groups_skipped_in_fixed_order": skipped,
        "median_clean_posterior_cosine": median_cosine,
        "abort_threshold": -0.8,
        "passed": True,
        **scale,
    }


def _load_runtime(
    args: argparse.Namespace, modules: SimpleNamespace
) -> tuple[Any, Any, dict[str, Any]]:
    loader_args = SimpleNamespace(
        model_path=Path(args.model_path),
        checkpoint_path=Path(args.checkpoint_path),
        data_dir=Path(args.sft_data_dir),
    )
    return modules.d3po.load_policy_and_reference_adapters(loader_args)


def init_distributed(modules: SimpleNamespace) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        raise RuntimeError(f"pilot training requires WORLD_SIZE={WORLD_SIZE}")
    if not modules.torch.cuda.is_available():
        raise RuntimeError("pilot training requires CUDA")
    modules.dist.init_process_group(backend="nccl")
    rank = int(modules.dist.get_rank())
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    modules.torch.cuda.set_device(local_rank)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "device": modules.torch.device("cuda", local_rank),
        "is_main": rank == 0,
    }


def validate_authorization(path: Path) -> dict[str, Any]:
    marker = Path(path).resolve()
    if not marker.is_file():
        raise FileNotFoundError(marker)
    text = marker.read_text(encoding="utf-8").strip()
    report: dict[str, Any] = {"marker": str(marker), "explicit": True}
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise TypeError("PRELIGHT authorization JSON must be an object")
        if payload.get("authorized") is False or payload.get("passed") is False:
            raise RuntimeError("PRELIGHT training report did not authorize training")
        report["report"] = payload
    return report


def _save_final_policy(
    runtime: Any,
    tokenizer: Any,
    output_dir: Path,
    args: argparse.Namespace,
    final_report: Mapping[str, Any],
    modules: SimpleNamespace,
) -> dict[str, Any]:
    inherited_path = Path(args.checkpoint_path) / "spad_basin_closure_capability.json"
    if not inherited_path.is_file():
        raise FileNotFoundError(inherited_path)
    inherited = json.loads(inherited_path.read_text(encoding="utf-8"))
    if inherited.get("spad_cell_closure_trained") is not True or inherited.get(
        "spad_species_block_closure_trained"
    ) is not True:
        raise RuntimeError("initialization checkpoint lacks trained SPAD closure capability")
    runtime.activate_policy(trainable=False)
    if modules.d3po.REFERENCE_ADAPTER not in runtime.model.peft_config:
        raise RuntimeError("reference adapter disappeared before final save")
    runtime.model.delete_adapter(modules.d3po.REFERENCE_ADAPTER)
    runtime.model.set_adapter(modules.d3po.POLICY_ADAPTER)
    final_root = Path(output_dir) / "final_policy"
    runtime.model.save_pretrained(
        final_root,
        selected_adapters=[modules.d3po.POLICY_ADAPTER],
        safe_serialization=True,
        save_embedding_layers="auto",
    )
    configs = list(final_root.rglob("adapter_config.json"))
    models = list(final_root.rglob("adapter_model.safetensors"))
    if len(configs) != 1 or len(models) != 1 or configs[0].parent != models[0].parent:
        raise RuntimeError("final save must contain exactly one policy adapter")
    adapter_dir = configs[0].parent
    tokenizer.save_pretrained(adapter_dir)
    capability = {
        "schema": "spad_basin_closure_capability_v1",
        "checkpoint_path": str(adapter_dir.resolve()),
        "initialization_checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "spad_cell_closure_trained": True,
        "spad_species_block_closure_trained": True,
        "closure_schedule_version": inherited.get("closure_schedule_version"),
        "basin_posterior_pilot": True,
        "value_field": str(args.value_field),
        "labelled_groups": EXPECTED_GROUPS,
        "posterior_passes": POSTERIOR_PASSES,
        "optimizer_updates": TOTAL_UPDATES,
        "training_seed": int(args.seed),
        "posterior_gradient_multiplier": float(
            final_report["gradient_probe"]["selected_posterior_gradient_multiplier"]
        ),
        "adapter_model_sha256": sha256_file(models[0]),
        "inherited_closure_capability": inherited,
    }
    capability_path = adapter_dir / "spad_basin_closure_capability.json"
    write_json(capability_path, capability)
    return {
        "policy_adapter_path": str(adapter_dir.resolve()),
        "capability_json": str(capability_path.resolve()),
        "only_final_policy_saved": True,
        "reference_adapter_saved": False,
    }


def train(
    args: argparse.Namespace,
    dist_info: Mapping[str, Any],
    tokenizer: Any,
    runtime: Any,
    group_dataset: BasinPosteriorGroupDataset,
    clean_dataset: Any,
    clean_collator: Any,
    loss_config: Mapping[str, Any],
    gradient_probe: Mapping[str, Any],
    modules: SimpleNamespace,
) -> dict[str, Any]:
    rank = int(dist_info["rank"])
    device = dist_info["device"]
    is_main = bool(dist_info["is_main"])
    parameters = tuple(
        parameter for parameter in runtime.policy_parameters if parameter.requires_grad
    )
    if not parameters:
        raise RuntimeError("policy adapter has no trainable LoRA parameters")
    multiplier = float(gradient_probe["selected_posterior_gradient_multiplier"])
    schedule = deterministic_posterior_schedule()
    clean_schedule = deterministic_clean_indices()
    if len(schedule) != POSTERIOR_UPDATES or len(clean_schedule) != CLEAN_CE_UPDATES:
        raise RuntimeError("pilot schedule accounting changed")
    optimizer = modules.torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=0.0
    )
    posterior_seen = modules.torch.zeros(
        EXPECTED_GROUPS, dtype=modules.torch.int16, device=device
    )
    objectives: Counter[str] = Counter()
    informative_local = 0
    zero_local = 0
    zero_reasons: Counter[str] = Counter()
    log_path = Path(args.output_dir) / "training_log.jsonl"
    started = time.time()

    for posterior_step in range(POSTERIOR_UPDATES):
        clean_update = 2 * posterior_step + 1
        if optimizer_objective(clean_update) != "clean_ce":
            raise RuntimeError("optimizer objective alternation changed")
        clean_lr = learning_rate_for_update(clean_update)
        for param_group in optimizer.param_groups:
            param_group["lr"] = clean_lr
        optimizer.zero_grad(set_to_none=True)
        clean_index = clean_schedule[posterior_step][rank]
        clean_loss_value = clean_ce_loss(
            runtime,
            clean_dataset,
            clean_collator,
            clean_index,
            device,
            loss_config,
            modules,
        )
        clean_loss_value.backward()
        average_gradients_(parameters, world_size=WORLD_SIZE)
        clean_grad = current_gradient_norm(parameters, modules)
        if not math.isfinite(clean_grad):
            raise FloatingPointError("clean CE gradient is nonfinite")
        modules.torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        objectives["clean_ce"] += 1

        posterior_update = clean_update + 1
        if optimizer_objective(posterior_update) != "transaction_posterior":
            raise RuntimeError("optimizer objective alternation changed")
        posterior_lr = learning_rate_for_update(posterior_update)
        for param_group in optimizer.param_groups:
            param_group["lr"] = posterior_lr
        optimizer.zero_grad(set_to_none=True)
        group_index = schedule[posterior_step][rank]
        batch = move_to_device(
            group_dataset.materialize(group_index, tokenizer, modules), device
        )
        posterior = transaction_posterior_objective(
            runtime, batch, modules, require_grad=True
        )
        raw_posterior_loss = posterior["loss"]
        scaled_posterior_loss = raw_posterior_loss * multiplier
        if not bool(modules.torch.isfinite(scaled_posterior_loss).detach().cpu()):
            raise FloatingPointError("scaled posterior loss is nonfinite")
        scaled_posterior_loss.backward()
        average_gradients_(parameters, world_size=WORLD_SIZE)
        posterior_grad = current_gradient_norm(parameters, modules)
        if not math.isfinite(posterior_grad):
            raise FloatingPointError("posterior gradient is nonfinite")
        modules.torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        posterior_seen[int(group_index)] += 1
        objectives["transaction_posterior"] += 1
        informative_local += int(bool(posterior["informative"]))
        zero_local += int(not bool(posterior["informative"]))
        if posterior.get("zero_reason"):
            zero_reasons[str(posterior["zero_reason"])] += 1

        if posterior_step == 0 or (posterior_step + 1) % 16 == 0:
            global_clean_loss = _reduce_scalar(
                float(clean_loss_value.detach().cpu()), device, modules
            ) / WORLD_SIZE
            global_posterior_loss = _reduce_scalar(
                float(raw_posterior_loss.detach().cpu()), device, modules
            ) / WORLD_SIZE
            global_informative = int(
                _reduce_scalar(int(bool(posterior["informative"])), device, modules)
            )
            max_kl = _reduce_scalar(
                float(posterior["teacher_kl_nats"]),
                device,
                modules,
                op=modules.dist.ReduceOp.MAX,
            )
            if is_main:
                append_jsonl(
                    log_path,
                    {
                        "event": "train",
                        "update": posterior_update,
                        "posterior_step": posterior_step + 1,
                        "clean_loss": global_clean_loss,
                        "posterior_loss": global_posterior_loss,
                        "posterior_gradient_multiplier": multiplier,
                        "clean_gradient_norm": clean_grad,
                        "posterior_gradient_norm_after_multiplier": posterior_grad,
                        "learning_rate": posterior_lr,
                        "informative_groups_in_global_step": global_informative,
                        "max_teacher_kl_nats": max_kl,
                        "elapsed_seconds": time.time() - started,
                    },
                )

    modules.dist.all_reduce(posterior_seen, op=modules.dist.ReduceOp.SUM)
    if not bool(modules.torch.all(posterior_seen == POSTERIOR_PASSES).detach().cpu()):
        raise RuntimeError("every posterior group must be seen exactly four times across ranks")
    expected_objectives = Counter(
        {"clean_ce": CLEAN_CE_UPDATES, "transaction_posterior": POSTERIOR_UPDATES}
    )
    if objectives != expected_objectives:
        raise RuntimeError("optimizer objective counts changed")
    informative_global = int(_reduce_scalar(informative_local, device, modules))
    zero_global = int(_reduce_scalar(zero_local, device, modules))
    if informative_global + zero_global != EXPECTED_GROUPS * POSTERIOR_PASSES:
        raise RuntimeError("posterior exposure denominator changed")
    modules.dist.barrier()
    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "success",
        "route_name": str(args.route_name),
        "value_field": str(args.value_field),
        "seed": int(args.seed),
        "world_size": WORLD_SIZE,
        "one_group_per_rank_per_posterior_step": True,
        "posterior_passes": POSTERIOR_PASSES,
        "posterior_group_exposures": EXPECTED_GROUPS * POSTERIOR_PASSES,
        "each_group_seen_exactly": POSTERIOR_PASSES,
        "clean_ce_updates": CLEAN_CE_UPDATES,
        "transaction_posterior_updates": POSTERIOR_UPDATES,
        "optimizer_updates": TOTAL_UPDATES,
        "objective_counts": dict(expected_objectives),
        "clean_examples_seen_across_ranks": CLEAN_CE_UPDATES * WORLD_SIZE,
        "clean_source_split_rows": EXPECTED_CLEAN_ROWS,
        "informative_posterior_exposures": informative_global,
        "zero_posterior_exposures_retained": zero_global,
        "zero_reason_counts_rank0": dict(sorted(zero_reasons.items())),
        "gradient_probe": dict(gradient_probe),
        "learning_rate": LEARNING_RATE,
        "warmup_updates": WARMUP_UPDATES,
        "lr_scheduler": "cosine",
        "min_lr_ratio": MIN_LR_RATIO,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "kl_budget_nats": MAX_KL_BUDGET_NATS,
        "temperature": DEPLOYED_TEMPERATURE,
        "same_step_scalar_mixture": False,
        "clean_ce_generated_states_used": False,
        "supplied_action_path_mass": True,
        "whole_terminal_probability": False,
        "source_rollout_left_padding_replayed": True,
        "source_rollout_batch_size": SOURCE_ROLLOUT_BATCH_SIZE,
        "elapsed_seconds": time.time() - started,
    }
    if is_main:
        report["checkpoint"] = _save_final_policy(
            runtime,
            tokenizer,
            Path(args.output_dir),
            args,
            report,
            modules,
        )
    modules.dist.barrier()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--labelled-groups", type=Path, required=True)
    parser.add_argument("--sft-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization-marker", type=Path, required=True)
    parser.add_argument("--value-field", default=DEFAULT_VALUE_FIELD)
    parser.add_argument("--route-name", choices=("e0", "k10"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    modules = _runtime_modules()
    dist_info = init_distributed(modules)
    rank = int(dist_info["rank"])
    is_main = bool(dist_info["is_main"])
    device = dist_info["device"]
    output_dir = Path(args.output_dir).resolve()
    args.output_dir = output_dir
    try:
        authorization = validate_authorization(Path(args.authorization_marker))
        if is_main:
            output_dir.mkdir(parents=True, exist_ok=False)
        modules.dist.barrier()
        random.seed(int(args.seed) + rank)
        modules.torch.manual_seed(int(args.seed) + rank)
        modules.torch.cuda.manual_seed_all(int(args.seed) + rank)
        tokenizer, runtime, adapter_report = _load_runtime(args, modules)
        modules.validate_tokenizer(tokenizer)
        runtime.model.to(device)
        groups = BasinPosteriorGroupDataset(
            Path(args.labelled_groups), value_field=str(args.value_field)
        )
        clean_dataset, clean_collator, loss_config = load_clean_sft(
            Path(args.sft_data_dir), tokenizer, modules
        )
        step0 = step0_policy_reference_equality(
            runtime, groups, tokenizer, device, rank, modules
        )
        gradient_probe = run_gradient_probe(
            runtime,
            groups,
            clean_dataset,
            clean_collator,
            loss_config,
            tokenizer,
            device,
            rank,
            modules,
        )
        config = {
            "schema": RUN_SCHEMA,
            "route_name": str(args.route_name),
            "value_field": str(args.value_field),
            "seed": int(args.seed),
            "model_path": str(Path(args.model_path).resolve()),
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "labelled_groups": str(Path(args.labelled_groups).resolve()),
            "sft_data_dir": str(Path(args.sft_data_dir).resolve()),
            "authorization": authorization,
            "dataset": groups.summary(),
            "adapter_load": adapter_report,
            "step0_policy_reference_equality": step0,
            "gradient_probe": gradient_probe,
            "training": {
                "world_size": WORLD_SIZE,
                "posterior_passes": POSTERIOR_PASSES,
                "posterior_updates": POSTERIOR_UPDATES,
                "clean_ce_updates": CLEAN_CE_UPDATES,
                "total_optimizer_updates": TOTAL_UPDATES,
                "objective_order": ["clean_ce", "transaction_posterior"],
                "one_group_per_rank_per_posterior_step": True,
                "learning_rate": LEARNING_RATE,
                "warmup_updates": WARMUP_UPDATES,
                "scheduler": "cosine",
                "min_lr_ratio": MIN_LR_RATIO,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "data_order_seed": DATA_ORDER_SEED,
            },
            "deployment_scoring": {
                "temperature": DEPLOYED_TEMPERATURE,
                "exact_dynamic_schema": True,
                "duplicate_coordinate_mask": True,
                "lattice_volume_mask": True,
                "pbc_min_distance_A": 0.5,
                "pbc_image_radius": 2,
                "pbc_images": 125,
                "context_masks_mandatory": True,
                "score": "supplied_action_path_mass",
            },
            "legacy_trainer_invoked": False,
            "automatic_route_selection": False,
        }
        if is_main:
            write_json(output_dir / "RUN_CONFIG.json", config)
            write_json(output_dir / "GRADIENT_PROBE.json", gradient_probe)
            append_jsonl(output_dir / "training_log.jsonl", {"event": "start", **config})
        report = train(
            args,
            dist_info,
            tokenizer,
            runtime,
            groups,
            clean_dataset,
            clean_collator,
            loss_config,
            gradient_probe,
            modules,
        )
        if is_main:
            report["step0_policy_reference_equality"] = step0
            report["dataset"] = groups.summary()
            write_json(output_dir / "TRAIN_FINAL.json", report)
            append_jsonl(output_dir / "training_log.jsonl", {"event": "success", **report})
            (output_dir / "_SUCCESS").touch()
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        modules.dist.barrier()
    except Exception as error:
        if is_main and output_dir.exists():
            write_json(
                output_dir / "_FAILED.json",
                {
                    "schema": RUN_SCHEMA,
                    "status": "failed",
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    finally:
        if modules.dist.is_initialized():
            modules.dist.destroy_process_group()


if __name__ == "__main__":
    main()
