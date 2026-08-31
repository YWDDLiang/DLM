#!/usr/bin/env python3
"""Train one stability-conditioned typed C3FD--Llama residual Planner.

The frozen C3FD-v2.5 model owns the calibrated base distribution and legal
support.  A fresh Meta-Llama-3 LoRA and a small typed residual module only
reweight that support through a unit-weight product of experts.  This file is
also intentionally factored into small pure helpers so the execution contract
can be tested with tiny fake C3FD and Llama modules on CPU.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from crystal_dlm.c3fd_calibration import StratumInteraction
from crystal_dlm.c3fd_llama_fused_plan import (
    STABILITY_CONDITIONS,
)
from crystal_dlm.c3fd_llama_typed_planner import (
    C3FDLlamaTypedPlannerConfig,
    C3FDLlamaTypedResidualPlanner,
    SOFT_FIELDS,
    joint_action_index,
    masked_log_softmax,
    row_balanced_typed_loss,
    unit_weight_poe_log_probs,
)
from crystal_dlm.c3fd_planner_model import C3FDPlannerConfig, C3FDPlannerModel
from crystal_dlm.ccfd_v2 import CCFDv2State, SetAtomCount
from crystal_dlm.composition_pair_prior import ValenceNode
from crystal_dlm.family_reachability import PaulingBitsetReachability
from crystal_dlm.semantic_composition_head import SemanticHeadFlags


TRAIN_SCHEMA = "c3fd_llama_typed_planner_train_v1"
DATASET_SCHEMA = "c3fd_llama_fused_typed_dataset_v1"
CHECKPOINT_SCHEMA = "h1a2_c3fd_planner_checkpoint_v1"
FINAL_CONFIG_SCHEMA = "c3fd_llama_typed_planner_final_config_v1"
FINAL_STATE_SCHEMA = "c3fd_llama_typed_planner_final_state_v1"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
SEED = 85017
IGNORE_INDEX = -100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model tree is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, *, label: str) -> str:
    expected = str(expected).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{label} expected SHA256 is malformed")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected}, observed {observed}")
    return observed


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            yield value


class FusedTypedDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = list(iter_jsonl(path))
        if not self.rows:
            raise ValueError(f"no fused typed rows in {path}")
        for row in self.rows:
            if row.get("schema") != DATASET_SCHEMA:
                raise ValueError("fused typed dataset schema changed")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


@dataclass(frozen=True)
class FrozenC3FDBundle:
    model: nn.Module
    context: Tensor
    interaction: StratumInteraction
    calibration: Mapping[str, Mapping[str, float]]
    vocabulary: Mapping[str, Any]
    proposal_legal_mask: Tensor
    stratum_to_index: Mapping[tuple[int, int, int], int]
    checkpoint_sha256: str
    vocabulary_sha256: str


def _temperature(calibration: Mapping[str, Mapping[str, float]], name: str) -> float:
    try:
        value = float(calibration[name]["temperature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"C3FD checkpoint lacks calibration for {name}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid C3FD temperature for {name}")
    return value


def _species_nodes(vocabulary: Mapping[str, Any]) -> tuple[ValenceNode, ...]:
    rows = sorted(vocabulary.get("species") or (), key=lambda row: int(row["id"]))
    if [int(row["id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("C3FD species vocabulary ids must be contiguous")
    return tuple(
        ValenceNode(int(row["atomic_number"]), int(row["oxidation_state"]))
        for row in rows
    )


def compile_proposal_legal_mask(
    interaction: StratumInteraction,
    vocabulary: Mapping[str, Any],
) -> Tensor:
    """Precompute the Pauling-bitset viability of every checkpoint stratum."""

    nodes = _species_nodes(vocabulary)
    reachability = PaulingBitsetReachability(nodes)
    soft = vocabulary.get("soft_vocabulary")
    if not isinstance(soft, Mapping):
        raise ValueError("C3FD vocabulary lacks soft_vocabulary")
    families = soft.get("anion_framework")
    if not isinstance(families, Sequence) or isinstance(families, (str, bytes)):
        raise ValueError("C3FD vocabulary lacks anion_framework labels")
    legal: list[bool] = []
    for family_id, target_n, arity in interaction.strata:
        if family_id < 0 or family_id >= len(families):
            raise ValueError("checkpoint proposal family is outside vocabulary")
        family = str(families[family_id])
        state = CCFDv2State.start().apply(SetAtomCount(int(target_n)))
        legal.append(
            bool(
                reachability.can_complete(
                    state,
                    family=family,
                    target_arity=int(arity),
                    max_species=7,
                )
            )
        )
    result = torch.tensor(legal, dtype=torch.bool)
    if result.numel() == 0 or not bool(result.any().item()):
        raise ValueError("checkpoint has no Pauling-bitset-completable proposal stratum")
    return result


def freeze_c3fd(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def assert_c3fd_frozen(model: nn.Module) -> None:
    if model.training:
        raise RuntimeError("C3FD must remain in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("C3FD contains a trainable parameter")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("C3FD accumulated gradients")


def load_frozen_c3fd(
    *,
    checkpoint_path: Path,
    vocabulary_path: Path,
    checkpoint_sha256: str,
    vocabulary_sha256: str,
) -> FrozenC3FDBundle:
    require_sha256(checkpoint_path, checkpoint_sha256, label="C3FD checkpoint")
    require_sha256(vocabulary_path, vocabulary_sha256, label="C3FD vocabulary")
    vocabulary_bytes = vocabulary_path.read_bytes()
    vocabulary = json.loads(vocabulary_bytes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected C3FD checkpoint schema")
    if checkpoint.get("vocabulary_sha256") != vocabulary_sha256:
        raise RuntimeError("C3FD checkpoint/vocabulary hash contract changed")
    calibration = checkpoint.get("calibration") or {}
    for name in ("family", "n", "arity", "species", "count"):
        _temperature(calibration, name)
    interaction = StratumInteraction.from_dict(checkpoint["stratum_interaction"])
    config = C3FDPlannerConfig(**checkpoint["config"])
    physics = torch.tensor(vocabulary["physics"]["matrix"], dtype=torch.float32)
    model = C3FDPlannerModel(config, physics_features=physics)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    freeze_c3fd(model)
    context = torch.as_tensor(checkpoint["context"], dtype=torch.float32)
    if context.ndim != 2 or context.shape[0] != 1:
        raise ValueError("C3FD checkpoint context must contain exactly one row")
    proposal_mask = compile_proposal_legal_mask(interaction, vocabulary)
    mapping = {tuple(value): index for index, value in enumerate(interaction.strata)}
    return FrozenC3FDBundle(
        model=model,
        context=context,
        interaction=interaction,
        calibration=calibration,
        vocabulary=vocabulary,
        proposal_legal_mask=proposal_mask,
        stratum_to_index=mapping,
        checkpoint_sha256=checkpoint_sha256,
        vocabulary_sha256=vocabulary_sha256,
    )


def _normalized_ledger(raw: Mapping[str, Any]) -> tuple[float, ...]:
    branch = str(raw.get("branch") or "unset")
    branch_vector = {
        "unset": (1.0, 0.0, 0.0),
        "ionic": (0.0, 1.0, 0.0),
        "alloy": (0.0, 0.0, 1.0),
    }.get(branch)
    if branch_vector is None:
        raise ValueError(f"unknown ledger branch {branch!r}")
    values = (
        float(raw["remaining_atoms"]) / 20.0,
        float(raw["net_charge"]) / 160.0,
        float(raw["remaining_species"]) / 7.0,
        *branch_vector,
    )
    if not all(math.isfinite(value) and abs(value) <= 1.0 + 1e-6 for value in values):
        raise ValueError("normalized ledger exceeds its six-value contract")
    return values


def _soft_legal_masks(vocabulary: Mapping[str, Any]) -> dict[str, Tensor]:
    soft = vocabulary.get("soft_vocabulary")
    if not isinstance(soft, Mapping):
        raise ValueError("C3FD vocabulary lacks soft_vocabulary")
    result: dict[str, Tensor] = {}
    for field in SOFT_FIELDS:
        values = soft.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"C3FD vocabulary lacks {field}")
        result[field] = torch.tensor(
            [str(value) != "<UNKNOWN>" for value in values], dtype=torch.bool
        )
        if not bool(result[field].any().item()):
            raise ValueError(f"{field} has no legal class")
    return result


def collate_typed_rows(
    rows: list[dict[str, Any]],
    *,
    bundle: FrozenC3FDBundle,
) -> dict[str, Tensor]:
    """Align proposal position 0, action queries, and terminal/EOS position."""

    if not rows:
        raise ValueError("cannot collate an empty typed batch")
    config = bundle.model.config
    num_species = int(config.num_species)
    max_count = int(config.max_count)
    eos_species = num_species
    num_actions = num_species * max_count + 1
    lengths = [len(row["species_ids"]) + 2 for row in rows]
    width = max(lengths)
    batch_size = len(rows)

    stability = torch.empty(batch_size, dtype=torch.long)
    proposal_states = torch.zeros(batch_size, width, dtype=torch.long)
    previous_species = torch.full((batch_size, width), -1, dtype=torch.long)
    previous_count = torch.zeros(batch_size, width, dtype=torch.long)
    previous_n = torch.zeros(batch_size, width, dtype=torch.long)
    ledger = torch.zeros(batch_size, width, 6, dtype=torch.float32)
    attention = torch.zeros(batch_size, width, dtype=torch.long)
    soft_positions = torch.empty(batch_size, dtype=torch.long)
    proposal_targets = torch.empty(batch_size, dtype=torch.long)
    proposal_masks = bundle.proposal_legal_mask.unsqueeze(0).expand(batch_size, -1).clone()
    action_targets = torch.full((batch_size, width - 1), IGNORE_INDEX, dtype=torch.long)
    action_masks = torch.zeros(batch_size, width - 1, num_actions, dtype=torch.bool)
    sample_weights = torch.empty(batch_size, dtype=torch.float32)
    soft_targets = {field: torch.empty(batch_size, dtype=torch.long) for field in SOFT_FIELDS}
    base_soft_masks = _soft_legal_masks(bundle.vocabulary)
    soft_masks = {
        field: mask.unsqueeze(0).expand(batch_size, -1).clone()
        for field, mask in base_soft_masks.items()
    }

    for row_index, (row, length) in enumerate(zip(rows, lengths)):
        if row.get("schema") != DATASET_SCHEMA:
            raise ValueError("fused typed dataset schema changed")
        condition = str(row.get("stability_condition"))
        if condition not in STABILITY_CONDITIONS:
            raise ValueError("unknown stability condition")
        stability[row_index] = STABILITY_CONDITIONS.index(condition)
        proposal = row.get("proposal_target")
        if not isinstance(proposal, Mapping):
            raise ValueError("row lacks proposal_target")
        stratum = (
            int(proposal["family_id"]),
            int(proposal["N"]),
            int(proposal["arity"]),
        )
        if stratum not in bundle.stratum_to_index:
            raise ValueError("teacher proposal is outside frozen checkpoint strata")
        stratum_index = int(bundle.stratum_to_index[stratum])
        if not bool(bundle.proposal_legal_mask[stratum_index].item()):
            raise ValueError("teacher proposal is illegal under Pauling-bitset mask")
        species = [int(value) for value in row["species_ids"]]
        counts = [int(value) for value in row["count_targets"]]
        if len(species) != len(counts) or len(species) != int(proposal["arity"]):
            raise ValueError("teacher action sequence does not match arity")
        legal_steps = row.get("legal_action_indices")
        if not isinstance(legal_steps, Sequence) or len(legal_steps) != len(species) + 1:
            raise ValueError("legal action sequence does not include explicit EOS")
        raw_ledger = row.get("ledger_steps")
        if not isinstance(raw_ledger, Sequence) or len(raw_ledger) != length:
            raise ValueError("ledger sequence does not align with typed queries")

        attention[row_index, :length] = 1
        # position 0 is the proposal query.  Positions 1..terminal all carry
        # the selected checkpoint stratum as id stratum+1.
        proposal_states[row_index, 1:length] = stratum_index + 1
        previous_n[row_index, 1] = int(proposal["N"])
        for action_index, (species_id, count) in enumerate(zip(species, counts)):
            previous_position = action_index + 2
            previous_species[row_index, previous_position] = species_id
            previous_count[row_index, previous_position] = count
        # Pre-proposal position has an unknown ledger and stays exactly zero.
        for position, value in enumerate(raw_ledger):
            if position > 0:
                ledger[row_index, position] = torch.tensor(
                    _normalized_ledger(value), dtype=torch.float32
                )

        teacher_actions = [
            joint_action_index(
                species_id,
                count,
                num_species=num_species,
                max_count=max_count,
            )
            for species_id, count in zip(species, counts)
        ]
        teacher_actions.append(
            joint_action_index(
                eos_species,
                0,
                num_species=num_species,
                max_count=max_count,
            )
        )
        for position, (teacher, legal_indices) in enumerate(
            zip(teacher_actions, legal_steps)
        ):
            indices = sorted({int(value) for value in legal_indices})
            if not indices or indices[0] < 0 or indices[-1] >= num_actions:
                raise ValueError("action legal mask contains an out-of-range class")
            action_masks[row_index, position, indices] = True
            if teacher not in indices:
                raise ValueError("teacher action is illegal under provided mask")
            action_targets[row_index, position] = teacher
        # Padded ignored positions still need a non-empty support for the
        # strict masked-logit validator; EOS is the neutral support.
        action_masks[row_index, len(teacher_actions) :, -1] = True

        soft_positions[row_index] = length - 1
        proposal_targets[row_index] = stratum_index
        for field in SOFT_FIELDS:
            soft_targets[field][row_index] = int(row["soft_targets"][field]["label"])
            if not bool(soft_masks[field][row_index, soft_targets[field][row_index]].item()):
                raise ValueError(f"teacher {field} is illegal")
        sample_weight = float(row.get("sample_weight", 1.0))
        if not math.isfinite(sample_weight) or sample_weight <= 0.0:
            raise ValueError("sample_weight must be finite and positive")
        sample_weights[row_index] = sample_weight

    result = {
        "stability_goal_ids": stability,
        "proposal_state_ids": proposal_states,
        "previous_species_indices": previous_species,
        "previous_count_values": previous_count,
        "previous_n_values": previous_n,
        "ledger_features": ledger,
        "attention_mask": attention,
        "soft_position_indices": soft_positions,
        "proposal_targets": proposal_targets,
        "proposal_legal_mask": proposal_masks,
        "action_targets": action_targets,
        "action_legal_mask": action_masks,
        "sample_weight": sample_weights,
    }
    result.update({f"soft_target:{field}": value for field, value in soft_targets.items()})
    result.update({f"soft_mask:{field}": value for field, value in soft_masks.items()})
    return result


def move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _last_hidden(outputs: Any) -> Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    last = getattr(outputs, "last_hidden_state", None)
    if last is None:
        raise ValueError("Llama output does not expose hidden states")
    return last


def _module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


@torch.no_grad()
def frozen_c3fd_logits(
    bundle: FrozenC3FDBundle,
    batch: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Return frozen calibrated proposal/action logits and terminal soft logits."""

    model = bundle.model
    assert_c3fd_frozen(model)
    device = next(model.parameters()).device
    batch_size = batch["proposal_targets"].shape[0]
    output = model(
        bundle.context.to(device).expand(batch_size, -1),
        previous_species_indices=batch["previous_species_indices"].to(device),
        previous_count_values=batch["previous_count_values"].to(device),
        previous_n_values=batch["previous_n_values"].to(device),
        ledger_features=batch["ledger_features"].to(device),
        flags=SemanticHeadFlags(use_physics=True),
    )
    proposals = []
    for row_index in range(batch_size):
        proposals.append(
            bundle.interaction.joint_scores(
                output.family_logits[row_index],
                output.n_logits[row_index],
                output.arity_logits[row_index],
                family_temperature=_temperature(bundle.calibration, "family"),
                n_temperature=_temperature(bundle.calibration, "n"),
                arity_temperature=_temperature(bundle.calibration, "arity"),
            )
        )
    proposal_logits = torch.stack(proposals)
    species = output.species_logits[:, 1:, :] / _temperature(
        bundle.calibration, "species"
    )
    counts = output.count_logits[:, 1:, :] / _temperature(
        bundle.calibration, "count"
    )
    action_logits = model.head.joint_action_scores(
        species,
        counts,
        flags=SemanticHeadFlags(use_pair_prior=False, use_hard_mask=False),
    )
    positions = batch["soft_position_indices"].to(device=device, dtype=torch.long)
    rows = torch.arange(batch_size, device=device)
    # C3FD-v2.5 has calibrated composition heads.  Its three independently
    # supervised soft heads have no fitted checkpoint temperature, so their
    # frozen deployed logits are used with the explicit identity temperature.
    soft_logits = {
        field: output.rich_logits[field][rows, positions]
        for field in SOFT_FIELDS
    }
    return proposal_logits, action_logits, soft_logits


def weighted_row_balanced_loss(
    *,
    proposal_log_probs: Tensor,
    action_log_probs: Tensor,
    soft_log_probs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
) -> Tensor:
    """Apply the core row-balanced objective, then source sample weights."""

    row_losses: list[Tensor] = []
    for row in range(proposal_log_probs.shape[0]):
        row_losses.append(
            row_balanced_typed_loss(
                proposal_log_probs=proposal_log_probs[row : row + 1],
                proposal_targets=batch["proposal_targets"][row : row + 1],
                proposal_legal_mask=batch["proposal_legal_mask"][row : row + 1],
                action_log_probs=action_log_probs[row : row + 1],
                action_targets=batch["action_targets"][row : row + 1],
                action_legal_mask=batch["action_legal_mask"][row : row + 1],
                soft_field_log_probs={
                    field: soft_log_probs[field][row : row + 1]
                    for field in SOFT_FIELDS
                },
                soft_field_targets={
                    field: batch[f"soft_target:{field}"][row : row + 1]
                    for field in SOFT_FIELDS
                },
                soft_field_legal_masks={
                    field: batch[f"soft_mask:{field}"][row : row + 1]
                    for field in SOFT_FIELDS
                },
            )
        )
    stacked = torch.stack(row_losses)
    weights = batch["sample_weight"].to(device=stacked.device, dtype=stacked.dtype)
    return (stacked * weights).sum() / weights.sum().clamp_min(1e-12)


def _masked_kl(fused: Tensor, base: Tensor, legal: Tensor) -> Tensor:
    terms = torch.where(legal, fused.exp() * (fused - base), torch.zeros_like(fused))
    return terms.sum(dim=-1)


def _teacher_rank(base_log_probs: Tensor, targets: Tensor, legal: Tensor) -> Tensor:
    selected = base_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    better = legal & (base_log_probs > selected.unsqueeze(-1))
    return better.sum(dim=-1).to(dtype=torch.float32) + 1.0


def forward_fused_batch(
    *,
    llama: nn.Module,
    residual: C3FDLlamaTypedResidualPlanner,
    bundle: FrozenC3FDBundle,
    batch: Mapping[str, Tensor],
) -> tuple[Tensor, dict[str, float]]:
    inputs_embeds = residual.typed_inputs_embeds(
        stability_goal_ids=batch["stability_goal_ids"],
        proposal_state_ids=batch["proposal_state_ids"],
        previous_species_indices=batch["previous_species_indices"],
        previous_count_values=batch["previous_count_values"],
        ledger_features=batch["ledger_features"],
    ).to(dtype=_module_dtype(llama))
    outputs = llama(
        inputs_embeds=inputs_embeds,
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = _last_hidden(outputs)
    residual_logits = residual(
        hidden.to(dtype=_module_dtype(residual)),
        soft_position_indices=batch["soft_position_indices"],
    )
    base_proposal, base_action, base_soft = frozen_c3fd_logits(bundle, batch)
    base_proposal = base_proposal.to(device=hidden.device, dtype=hidden.dtype)
    base_action = base_action.to(device=hidden.device, dtype=hidden.dtype)
    base_soft = {
        field: value.to(device=hidden.device, dtype=hidden.dtype)
        for field, value in base_soft.items()
    }
    proposal_mask = batch["proposal_legal_mask"]
    action_mask = batch["action_legal_mask"]
    proposal_fused = unit_weight_poe_log_probs(
        base_proposal, residual_logits.proposal, proposal_mask
    )
    action_fused = unit_weight_poe_log_probs(
        base_action, residual_logits.actions[:, 1:, :], action_mask
    )
    soft_fused = {
        field: unit_weight_poe_log_probs(
            base_soft[field], residual_logits.soft_fields[field], batch[f"soft_mask:{field}"]
        )
        for field in SOFT_FIELDS
    }
    loss = weighted_row_balanced_loss(
        proposal_log_probs=proposal_fused,
        action_log_probs=action_fused,
        soft_log_probs=soft_fused,
        batch=batch,
    )

    with torch.no_grad():
        base_proposal_lp = masked_log_softmax(base_proposal, proposal_mask)
        base_action_lp = masked_log_softmax(base_action, action_mask)
        supervised = batch["action_targets"] != IGNORE_INDEX
        kl_values = [_masked_kl(proposal_fused, base_proposal_lp, proposal_mask).mean()]
        action_kl = _masked_kl(action_fused, base_action_lp, action_mask)
        kl_values.append(action_kl[supervised].mean())
        ranks = [
            _teacher_rank(
                base_proposal_lp, batch["proposal_targets"], proposal_mask
            ).mean()
        ]
        safe_action_targets = batch["action_targets"].masked_fill(~supervised, 0)
        action_ranks = _teacher_rank(base_action_lp, safe_action_targets, action_mask)
        ranks.append(action_ranks[supervised].mean())
        for field in SOFT_FIELDS:
            base_lp = masked_log_softmax(base_soft[field], batch[f"soft_mask:{field}"])
            kl_values.append(
                _masked_kl(soft_fused[field], base_lp, batch[f"soft_mask:{field}"]).mean()
            )
            ranks.append(
                _teacher_rank(
                    base_lp,
                    batch[f"soft_target:{field}"],
                    batch[f"soft_mask:{field}"],
                ).mean()
            )
        diagnostics = {
            "fused_vs_base_kl": float(torch.stack(kl_values).mean().item()),
            "teacher_base_rank": float(torch.stack(ranks).mean().item()),
        }
    return loss, diagnostics


@torch.no_grad()
def audit_step0_equality(
    *,
    llama: nn.Module,
    residual: C3FDLlamaTypedResidualPlanner,
    bundle: FrozenC3FDBundle,
    batch: Mapping[str, Tensor],
    atol: float = 2e-5,
) -> dict[str, Any]:
    was_training = llama.training
    llama.eval()
    inputs = residual.typed_inputs_embeds(
        stability_goal_ids=batch["stability_goal_ids"],
        proposal_state_ids=batch["proposal_state_ids"],
        previous_species_indices=batch["previous_species_indices"],
        previous_count_values=batch["previous_count_values"],
        ledger_features=batch["ledger_features"],
    ).to(dtype=_module_dtype(llama))
    output = llama(
        inputs_embeds=inputs,
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    typed = residual(
        _last_hidden(output).to(dtype=_module_dtype(residual)),
        soft_position_indices=batch["soft_position_indices"],
    )
    base_proposal, base_action, base_soft = frozen_c3fd_logits(bundle, batch)
    base_proposal = base_proposal.to(typed.proposal)
    base_action = base_action.to(typed.actions)
    observed = {
        "proposal": unit_weight_poe_log_probs(
            base_proposal, typed.proposal, batch["proposal_legal_mask"]
        ),
        "actions": unit_weight_poe_log_probs(
            base_action, typed.actions[:, 1:, :], batch["action_legal_mask"]
        ),
    }
    expected = {
        "proposal": masked_log_softmax(base_proposal, batch["proposal_legal_mask"]),
        "actions": masked_log_softmax(base_action, batch["action_legal_mask"]),
    }
    for field in SOFT_FIELDS:
        value = base_soft[field].to(typed.soft_fields[field])
        observed[field] = unit_weight_poe_log_probs(
            value, typed.soft_fields[field], batch[f"soft_mask:{field}"]
        )
        expected[field] = masked_log_softmax(value, batch[f"soft_mask:{field}"])
    maximum = 0.0
    for name in expected:
        legal = (
            batch["proposal_legal_mask"]
            if name == "proposal"
            else batch["action_legal_mask"]
            if name == "actions"
            else batch[f"soft_mask:{name}"]
        )
        delta = (observed[name][legal] - expected[name][legal]).abs()
        maximum = max(maximum, float(delta.max().item()) if delta.numel() else 0.0)
    if maximum > float(atol):
        raise RuntimeError(f"step0 PoE equality failed: max_abs_delta={maximum}")
    if was_training:
        llama.train()
    return {"exact_zero_residual": True, "max_abs_log_probability_delta": maximum}


def named_gradient_norm(module: nn.Module, *, contains: str | None = None) -> float:
    squared = 0.0
    found = False
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or (contains is not None and contains not in name):
            continue
        found = True
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().item())
    if not found:
        raise RuntimeError(f"no trainable parameters matched {contains!r}")
    return math.sqrt(squared)


def residual_head_gradient_norm(residual: C3FDLlamaTypedResidualPlanner) -> float:
    heads = [residual.proposal_head, residual.action_head, *residual.soft_field_heads.values()]
    squared = 0.0
    for head in heads:
        for parameter in head.parameters():
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(squared)


def cosine_with_warmup_lambda(step: int, *, total_steps: int, warmup_steps: int) -> float:
    current = int(step)
    if current < int(warmup_steps):
        return float(current + 1) / float(max(1, int(warmup_steps)))
    progress = float(current - int(warmup_steps)) / float(
        max(1, int(total_steps) - int(warmup_steps))
    )
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(
    *,
    llama: nn.Module,
    residual: C3FDLlamaTypedResidualPlanner,
    bundle: FrozenC3FDBundle,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    llama_was_training = llama.training
    residual_was_training = residual.training
    llama.eval()
    residual.eval()
    total = 0.0
    count = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        loss, _ = forward_fused_batch(
            llama=llama, residual=residual, bundle=bundle, batch=batch
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite validation loss")
        total += float(loss.item())
        count += 1
    if llama_was_training:
        llama.train()
    if residual_was_training:
        residual.train()
    return total / max(1, count)


def save_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_real_llama(model_path: Path, *, device: torch.device, args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from crystal_dlm.h1_llm_planner import (
        disable_peft_bnb_autodetect,
        ensure_peft_cache_compat,
        load_llama3_compatible_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = load_llama3_compatible_config(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(LORA_TARGET_MODULES),
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    return model, tokenizer, int(config.hidden_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-tree-sha256", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--c3fd-checkpoint", type=Path, required=True)
    parser.add_argument("--c3fd-checkpoint-sha256", required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--vocabulary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--typed-embedding-size", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-max-batches", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    return parser.parse_args()


def validate_training_contract(args: argparse.Namespace) -> None:
    """Fail closed if the single-run, final-only scientific contract drifts."""

    if int(args.seed) != SEED:
        raise ValueError(f"training seed is frozen at {SEED}")
    if int(args.epochs) != 1:
        raise ValueError("typed Planner training is frozen at one epoch")
    if int(args.batch_size) * int(args.grad_accum) != 16:
        raise ValueError("typed Planner effective batch is frozen at 16")
    if float(args.lr) != 2e-5 or int(args.warmup_steps) != 100:
        raise ValueError("typed Planner LR schedule changed")


def main() -> None:
    args = parse_args()
    validate_training_contract(args)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sha256_tree(args.model_path) != str(args.model_tree_sha256).lower():
        raise RuntimeError("Meta-Llama model tree SHA256 mismatch")
    require_sha256(
        args.data_dir / "manifest.json",
        args.data_manifest_sha256,
        label="fused data manifest",
    )
    if not (args.data_dir / "_SUCCESS").is_file():
        raise RuntimeError("fused data lacks _SUCCESS")
    bundle = load_frozen_c3fd(
        checkpoint_path=args.c3fd_checkpoint,
        vocabulary_path=args.vocabulary,
        checkpoint_sha256=args.c3fd_checkpoint_sha256,
        vocabulary_sha256=args.vocabulary_sha256,
    )
    bundle.model.to(device)
    bundle = FrozenC3FDBundle(
        **{**bundle.__dict__, "context": bundle.context.to(device)}
    )
    assert_c3fd_frozen(bundle.model)

    llama, tokenizer, hidden_size = _load_real_llama(
        args.model_path, device=device, args=args
    )
    soft_dims = {
        field: len(bundle.vocabulary["soft_vocabulary"][field])
        for field in SOFT_FIELDS
    }
    typed_config = C3FDLlamaTypedPlannerConfig(
        llama_hidden_size=hidden_size,
        typed_embedding_size=int(args.typed_embedding_size),
        num_stability_goals=len(STABILITY_CONDITIONS),
        num_proposal_states=len(bundle.interaction.strata) + 1,
        num_proposal_strata=len(bundle.interaction.strata),
        num_species=int(bundle.model.config.num_species),
        max_count=int(bundle.model.config.max_count),
        ledger_feature_size=6,
        num_lattice_systems=soft_dims["lattice_system"],
        num_spacegroup_buckets=soft_dims["spacegroup_bucket"],
        num_volume_per_atom_bins=soft_dims["volume_per_atom_bin"],
        max_sequence_length=int(bundle.model.config.max_sequence_length),
    )
    residual = C3FDLlamaTypedResidualPlanner(typed_config).to(device)
    llama.train()
    residual.train()

    train_dataset = FusedTypedDataset(args.data_dir / "train.jsonl")
    val_dataset = FusedTypedDataset(args.data_dir / "val.jsonl")
    generator = torch.Generator().manual_seed(SEED)
    collate = lambda rows: collate_typed_rows(rows, bundle=bundle)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=int(args.num_workers),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate,
    )
    total_updates = math.ceil(len(train_loader) / int(args.grad_accum))
    if total_updates <= int(args.warmup_steps):
        raise ValueError("training epoch is not longer than warmup")

    first_batch = move_batch(next(iter(train_loader)), device)
    step0 = audit_step0_equality(
        llama=llama, residual=residual, bundle=bundle, batch=first_batch
    )
    save_json_exclusive(args.output_dir / "step0_poe_equality.json", step0)

    trainable = [parameter for parameter in llama.parameters() if parameter.requires_grad]
    trainable.extend(parameter for parameter in residual.parameters() if parameter.requires_grad)
    if not trainable:
        raise RuntimeError("typed Planner has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup_lambda(
            step,
            total_steps=total_updates,
            warmup_steps=int(args.warmup_steps),
        ),
    )
    run_config = {
        "schema": TRAIN_SCHEMA,
        "seed": SEED,
        "epochs": 1,
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "effective_batch": 16,
        "lr": float(args.lr),
        "scheduler": "cosine",
        "warmup_steps": int(args.warmup_steps),
        "total_updates": total_updates,
        "eligible_checkpoint": "final_only",
        "checkpoint_selection": "none",
        "validation_role": "monitoring_only",
        "c3fd_frozen": True,
        "c3fd_seed": 17,
        "unit_weight_poe": True,
        "lora": {
            "fresh": True,
            "r": 8,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "typed_config": asdict(typed_config),
        "input_sha256": {
            "model_tree": str(args.model_tree_sha256).lower(),
            "data_manifest": str(args.data_manifest_sha256).lower(),
            "data_train": sha256_file(args.data_dir / "train.jsonl"),
            "data_val": sha256_file(args.data_dir / "val.jsonl"),
            "c3fd_checkpoint": str(args.c3fd_checkpoint_sha256).lower(),
            "vocabulary": str(args.vocabulary_sha256).lower(),
        },
    }
    save_json_exclusive(args.output_dir / "run_config.json", run_config)

    log_path = args.output_dir / "training_log.jsonl"
    started = time.time()
    update = 0
    micro = 0
    recent_loss = 0.0
    diagnostics_accumulator: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            loss, diagnostics = forward_fused_batch(
                llama=llama, residual=residual, bundle=bundle, batch=batch
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite typed Planner loss")
            (loss / int(args.grad_accum)).backward()
            assert_c3fd_frozen(bundle.model)
            micro += 1
            recent_loss += float(loss.item())
            diagnostics_accumulator.append(diagnostics)
            is_last_micro = micro == len(train_loader)
            if micro % int(args.grad_accum) != 0 and not is_last_micro:
                continue
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
            if not math.isfinite(grad_norm):
                raise RuntimeError("non-finite typed Planner gradient")
            update += 1
            event = {
                "event": "train",
                "step": update,
                "loss_recent": recent_loss / len(diagnostics_accumulator),
                "grad_norm": grad_norm,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "fused_vs_base_kl": sum(
                    item["fused_vs_base_kl"] for item in diagnostics_accumulator
                )
                / len(diagnostics_accumulator),
                "teacher_base_rank": sum(
                    item["teacher_base_rank"] for item in diagnostics_accumulator
                )
                / len(diagnostics_accumulator),
                "elapsed_sec": time.time() - started,
            }
            if update <= 10:
                event["residual_head_grad_norm"] = residual_head_gradient_norm(residual)
                event["lora_grad_norm"] = named_gradient_norm(llama, contains="lora_")
            if not all(
                math.isfinite(float(event[name]))
                for name in ("loss_recent", "grad_norm", "lr", "fused_vs_base_kl", "teacher_base_rank")
            ):
                raise RuntimeError("non-finite training diagnostic")
            log.write(json.dumps(event, sort_keys=True) + "\n")
            log.flush()
            if update == 1 or update % int(args.logging_steps) == 0:
                print(json.dumps(event, sort_keys=True), flush=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            recent_loss = 0.0
            diagnostics_accumulator = []
    if update != total_updates:
        raise RuntimeError(f"one-epoch update mismatch: {update} != {total_updates}")

    validation_loss = evaluate(
        llama=llama,
        residual=residual,
        bundle=bundle,
        loader=val_loader,
        device=device,
        max_batches=int(args.eval_max_batches),
    )
    assert_c3fd_frozen(bundle.model)
    if not math.isfinite(validation_loss):
        raise RuntimeError("non-finite final validation loss")

    final_dir = args.output_dir / "final"
    final_dir.mkdir(exist_ok=False)
    adapter_dir = final_dir / "llama_adapter"
    adapter_dir.mkdir(exist_ok=False)
    llama.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(
        {
            "schema": FINAL_STATE_SCHEMA,
            "state_dict": residual.state_dict(),
        },
        final_dir / "typed_residual_state.pt",
    )
    save_json_exclusive(
        final_dir / "typed_residual_config.json",
        {
            "schema": FINAL_CONFIG_SCHEMA,
            "typed_planner_config": asdict(typed_config),
            "stability_goal_to_id": {
                value: index for index, value in enumerate(STABILITY_CONDITIONS)
            },
            "proposal_state_encoding": "zero_query_then_frozen_stratum_index_plus_one",
        },
    )
    (final_dir / "_SUCCESS").touch(exist_ok=False)
    metrics = {
        "schema": TRAIN_SCHEMA,
        "status": "complete",
        "global_step": update,
        "epochs_completed": 1,
        "final_validation_loss": validation_loss,
        "elapsed_sec": time.time() - started,
        "c3fd_frozen_verified": True,
        "step0": step0,
    }
    save_json_exclusive(args.output_dir / "train_metrics.json", metrics)
    output_files = sorted(
        path for path in args.output_dir.rglob("*") if path.is_file()
    )
    sums = args.output_dir / "SHA256SUMS"
    with sums.open("x", encoding="utf-8", newline="\n") as handle:
        for path in output_files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(args.output_dir).as_posix()}\n")
    (args.output_dir / "_SUCCESS").write_text(
        json.dumps(
            {
                "metrics_sha256": sha256_file(args.output_dir / "train_metrics.json"),
                "sha256sums_sha256": sha256_file(sums),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
