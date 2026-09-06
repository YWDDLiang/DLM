"""Two source views, with explicit prefix/dense measures, for periodic DLM v2.

No generated K4/K8 paths or physical outcome fields are read. The caller supplies
the original MP20 sources with its audited frozen Planner conditioning. All
numeric target positions follow the same compiled species program as runtime.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from crystal_dlm.fixed_slot import build_special_tokens
from crystal_dlm.periodic_base_training_data import (
    PreparedBaseSource,
    _config,
    _mask_id,
    prepare_periodic_base_source,
)
from crystal_dlm.periodic_v2_corruption import (
    V2_CORRUPTION_PROTOCOL,
    corrupt_periodic_v2_source,
)
from crystal_dlm.programmed_path_runtime import process_path_logits
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints


V2_PREFIX_PROBABILITY = .75
V2_MASK_PROBABILITY_RANGE = (.1, 1.)
V2_NOISE_METADATA_DROPOUT = .5
V2_TRAINING_DATA_PROTOCOL = {
    "schema": "periodic_v2_mp20_two_views_v1",
    "source": "original_MP20_with_audited_frozen_Planner_conditioning",
    "views_per_source_epoch": 2,
    "tasks": ["construction", "repair"],
    "prefix_probability_per_view": V2_PREFIX_PROBABILITY,
    "task_branch_probabilities": [3 / 8, 1 / 8, 3 / 8, 1 / 8],
    "task_branch_order": ["construction_prefix", "construction_dense", "repair_prefix", "repair_dense"],
    "prefix_position_probability": "1/2 lattice then uniform over 6; 1/2 coord then uniform over 3N",
    "dense_mask_probability": list(V2_MASK_PROBABILITY_RANGE),
    "dense_masks": "independent_Bernoulli_with_empty_masks_retained",
    "dense_position_weight": "1/(12p) for lattice; 1/(6Np) for coordinates",
    "prefix_admission": "all_teacher_actions_through_target_legal_and_repair_old_supported",
    "ineligible_prefix": "zero_legal_risk_without_removing_state_or_source",
    "dense_task": "structured_denoise",
    "noise_metadata_dropout_probability": V2_NOISE_METADATA_DROPOUT,
    "legacy_numeric_noise_level": "always_unknown_-1",
    "global_batch": 24,
    "corruption": V2_CORRUPTION_PROTOCOL,
}


class _CompactTokenizer:
    def __init__(self, vocabulary: Mapping[str, int]):
        self.vocabulary = dict(vocabulary)

    def get_vocab(self):
        return self.vocabulary

    def convert_tokens_to_ids(self, token):
        return self.vocabulary[token]


def _remap_constraints(constraints: Mapping[str, Any], to_compact: Mapping[int, int]) -> dict:
    """Relabel every token-valued field consumed by the shared runtime masks."""
    result = dict(constraints)
    for name in ("count_token_to_n",):
        if name in constraints:
            result[name] = {to_compact[int(key)]: value for key, value in constraints[name].items()}
    for name in ("length_token_to_bin", "angle_token_to_bin", "coord_token_to_bin"):
        if name in constraints:
            result[name] = {
                axis: {to_compact[int(key)]: value for key, value in table.items()}
                for axis, table in constraints[name].items()
            }
    for name in ("gamma_bin_to_token_id", "z_bin_to_token_id", "zero_length_token_ids_by_position"):
        if name in constraints:
            result[name] = {key: to_compact[int(value)] for key, value in constraints[name].items()}
    if "coord_bin_to_token_id" in constraints:
        result["coord_bin_to_token_id"] = {
            axis: {key: to_compact[int(value)] for key, value in table.items()}
            for axis, table in constraints["coord_bin_to_token_id"].items()
        }
    if "coordinate_alias_token_ids" in constraints:
        result["coordinate_alias_token_ids"] = {
            axis: tuple(to_compact[int(value)] for value in pair)
            for axis, pair in constraints["coordinate_alias_token_ids"].items()
        }
    return result


@dataclass(frozen=True)
class PrefixReachability:
    positions: tuple[int, ...]
    target_legal: tuple[bool, ...]
    reachable_through: tuple[bool, ...]
    failure_reasons: tuple[str | None, ...]
    first_failure_step: int | None


class PrefixSupportAuditor:
    """Exact teacher-path support replay using compact native vocabulary IDs.

    Relabeling removes the pretrained text-vocabulary dimension from CPU support
    work. It does not reimplement geometry rules: process_path_logits supplies
    exactly the same support/alias transform. Repair old admission is independent
    and must be checked for each newly corrupted old state.
    """
    def __init__(self, tokenizer: Any, constraints: Mapping[str, Any]):
        config = _config(constraints)
        vocabulary = tokenizer.get_vocab()
        names = build_special_tokens(config)
        compact = {name: index for index, name in enumerate(names)}
        self.mask_id = len(compact)
        compact["<V2_NATIVE_MASK>"] = self.mask_id
        self.tokenizer = _CompactTokenizer(compact)
        self.original_ids = tuple([int(vocabulary[name]) for name in names]
                                  + [_mask_id(tokenizer, vocabulary)])
        self.to_compact = {token: index for index, token in enumerate(self.original_ids)}
        if len(self.to_compact) != len(self.original_ids):
            raise ValueError("native crystal and mask token IDs must be distinct")
        self.constraints = _remap_constraints(constraints, self.to_compact)
        self.vocabulary_size = len(self.original_ids)
        self.allowed_cache: dict[int, torch.Tensor] = {}
        self.cache: dict[tuple[str, tuple[int, ...]], PrefixReachability] = {}

    def allowed(self, count: int) -> torch.Tensor:
        if count not in self.allowed_cache:
            support = exact_dynamic_schema_constraints(self.tokenizer, count)
            allowed = torch.zeros(7 + 4 * count, self.vocabulary_size, dtype=torch.bool)
            for position, ids in enumerate(support):
                allowed[position, ids] = True
            self.allowed_cache[count] = allowed
        return self.allowed_cache[count]

    def remap_body(self, body: Sequence[int]) -> torch.Tensor:
        return torch.tensor([[self.to_compact[int(token)] for token in body]], dtype=torch.long)

    def process_compact_logits(
        self, logits: torch.Tensor, compact_body: torch.Tensor, *, position: int, count: int,
    ) -> tuple[torch.Tensor, set[int]]:
        """Shared transform, retaining the input dtype and alias gradients."""
        return process_path_logits(
            logits, compact_body, prompt_length=0, gen_length=7 + 4 * count,
            allowed=self.allowed(count).to(logits.device), grammar=None,
            constraints=self.constraints, positions={0: int(position)}, mask_id=self.mask_id,
        )

    def audit(self, source: PreparedBaseSource) -> PrefixReachability:
        key = (source.canonical_answer_sha256, source.transaction_positions)
        if key in self.cache:
            return self.cache[key]
        count = int(source.arrays["num_atoms"])
        current = self.remap_body(source.clean_tokens)
        targets = current[0].clone()
        current[0, list(source.transaction_positions)] = self.mask_id
        raw = torch.zeros(1, current.shape[1], self.vocabulary_size)
        legal_actions, reachable, reasons = [], [], []
        path_legal, first_failure = True, None
        for step, position in enumerate(source.transaction_positions):
            processed, bad = self.process_compact_logits(raw, current, position=position, count=count)
            target = int(targets[position])
            legal = not bad and bool(processed[0, position, target] > torch.finfo(raw.dtype).min)
            reason = (None if legal else "teacher_state_has_no_legal_completion" if bad
                      else "teacher_target_outside_actual_support")
            if not legal and first_failure is None:
                first_failure = step
            path_legal = path_legal and legal
            legal_actions.append(legal)
            reachable.append(path_legal)
            reasons.append(reason)
            # Continue the teacher trace even after failure so a later individually
            # legal action cannot erase an unreachable predecessor.
            current[0, position] = target
        result = PrefixReachability(source.transaction_positions, tuple(legal_actions),
                                    tuple(reachable), tuple(reasons), first_failure)
        self.cache[key] = result
        return result


def _rng(source: PreparedBaseSource, *, seed: int, epoch: int, view: int, stream: str):
    key = (f"periodic-v2:{seed}:{source.metadata['source_split']}:"
           f"{source.metadata['source_row_idx']}:{epoch}:{view}:{stream}")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


def make_periodic_v2_training_example(
    row: Mapping[str, Any] | PreparedBaseSource,
    tokenizer: Any,
    constraints: dict,
    *,
    view: int,
    epoch: int,
    seed: int,
    is_padding: bool = False,
    vocabulary: Mapping[str, int] | None = None,
    prefix_auditor: PrefixSupportAuditor | None = None,
) -> dict[str, Any]:
    """Materialize exactly one prefix or dense state, with no extra branch weight."""
    view, epoch, seed = int(view), int(epoch), int(seed)
    if view not in (0, 1) or min(epoch, seed) < 0:
        raise ValueError("v2 view must be 0/1 with nonnegative epoch/seed")
    vocabulary = tokenizer.get_vocab() if vocabulary is None else vocabulary
    source = (row if isinstance(row, PreparedBaseSource) else prepare_periodic_base_source(
        row, tokenizer, constraints, vocabulary=vocabulary,
    ))
    clean = list(source.clean_tokens)
    transaction = list(source.transaction_positions)
    count = int(source.arrays["num_atoms"])
    mask_id = _mask_id(tokenizer, vocabulary)
    arguments = dict(seed=seed, epoch=epoch, view=view)
    choice = _rng(source, **arguments, stream="branch_and_position")
    branch = "prefix" if float(choice.random()) < V2_PREFIX_PROBABILITY else "dense"
    task = "construction" if view == 0 else "repair"
    old = clean.copy()
    sigmas = (0., 0., 0.)
    corruption_info: dict[str, Any] = {
        "target_source": "original_MP20_clean_native", "applied": False,
        "fallback_to_clean": False, "fallback_reason": None,
        "attempt_count": 0, "rejected_attempts": 0, "rejection_counts": {},
        "quantized_unchanged": True, "old_state_admitted": None,
        "requested_noise_components": [0., 0., 0.],
    }
    if view == 1:
        old, sigmas, corruption_info = corrupt_periodic_v2_source(
            source, _rng(source, **arguments, stream="corruption"), vocabulary, constraints,
        )
    current = old.copy()
    probability = None
    sampled_object = None
    step = None
    prefix_reachable = predecessor_reachable = target_legal = None
    first_failure_position = first_failure_reason = None
    legal_eligible = None
    if branch == "prefix":
        sampled_object = "lattice" if int(choice.integers(2)) == 0 else "coord"
        pool = [p for p in transaction if (p <= 6) == (sampled_object == "lattice")]
        position = int(choice.choice(pool))
        step = transaction.index(position)
        current = clean.copy()
        for p in transaction[step:]:
            current[p] = mask_id
        positions = [position]
        if view == 0:
            old = current.copy()
            active = transaction[step:]
            phase = "construct"
        else:
            active = transaction.copy()
            phase = "full_cell_repair"
        auditor = prefix_auditor or PrefixSupportAuditor(tokenizer, constraints)
        audit = auditor.audit(source)
        prefix_reachable = audit.reachable_through[step]
        predecessor_reachable = step == 0 or audit.reachable_through[step - 1]
        target_legal = audit.target_legal[step]
        if audit.first_failure_step is not None and audit.first_failure_step <= step:
            first_failure_position = transaction[audit.first_failure_step]
            first_failure_reason = audit.failure_reasons[audit.first_failure_step]
        legal_eligible = bool(prefix_reachable and (view == 0 or corruption_info["old_state_admitted"]))
    else:
        masks = _rng(source, **arguments, stream="mask")
        probability = float(masks.uniform(*V2_MASK_PROBABILITY_RANGE))
        selected = masks.random(len(transaction)) < probability
        positions = [p for p, take in zip(transaction, selected, strict=True) if take]
        for p in positions:
            current[p] = mask_id
        if view == 0:
            old = current.copy()
        active = positions.copy()
        phase = "structured_denoise"
    dropped = bool(_rng(source, **arguments, stream="noise_metadata").random() < V2_NOISE_METADATA_DROPOUT)
    components = [-1., -1., -1.] if dropped else list(sigmas)
    singular = positions[0] if positions else transaction[0]
    weight = float(source.metadata["sample_weight"])
    return {
        **deepcopy(source.metadata),
        "schema": V2_TRAINING_DATA_PROTOCOL["schema"],
        "input_body": current, "old_body": old,
        "num_atoms": count, "view": view, "epoch": epoch,
        "branch": branch, "branch_name": f"{task}_{branch}", "phase": phase,
        "positions": positions, "targets": [clean[p] for p in positions],
        "position": singular, "target_token": clean[singular],
        "transaction_positions": active, "full_transaction_positions": transaction,
        "transaction_step": step, "sampled_object": sampled_object,
        "mask_probability": probability,
        "inclusion_probabilities": [] if probability is None else [probability] * len(positions),
        "empty_supervision": not positions,
        "prefix_reachable": prefix_reachable,
        "prefix_predecessors_reachable": predecessor_reachable,
        "teacher_target_legal": target_legal,
        "first_illegal_teacher_position": first_failure_position,
        "first_illegal_teacher_reason": first_failure_reason,
        "old_admission_required": view == 1,
        "old_state_admitted": corruption_info["old_state_admitted"],
        "legal_eligible": legal_eligible,
        "numeric_noise_level": -1.,
        "numeric_noise_components": components,
        "declared_numeric_noise_components": list(sigmas),
        "noise_metadata_dropped": dropped,
        "source_sample_weight": weight,
        "sample_weight": 0. if is_padding else weight,
        "is_padding": bool(is_padding),
        "corruption_info": corruption_info,
        "structured_corruption_fallback": bool(corruption_info["fallback_to_clean"]),
        "alias_tokens_canonicalized": source.alias_tokens_canonicalized,
        "source_answer_sha256": source.source_answer_sha256,
        "canonical_answer_sha256": source.canonical_answer_sha256,
        "outcomes_read": False,
    }


class PeriodicV2TrainingDataset(Dataset):
    """Exactly two views per original source, plus zero-weight global24 padding."""
    def __init__(
        self,
        path_or_rows: str | Path | Sequence[Mapping[str, Any]],
        tokenizer: Any,
        constraints: dict,
        *,
        seed: int,
        expected_rows: int = 27136,
        expected_split: str = "train",
        effective_batch: int = 24,
    ):
        if expected_split not in ("train", "val") or int(effective_batch) < 1 or int(seed) < 0:
            raise ValueError("invalid v2 split, batch size, or data seed")
        if isinstance(path_or_rows, (str, Path)):
            with Path(path_or_rows).open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        else:
            rows = list(path_or_rows)
        if len(rows) != int(expected_rows) or not rows:
            raise ValueError("the complete registered MP20 source count changed")
        if any(row.get("source_split") != expected_split for row in rows):
            raise ValueError("v2 source file mixes original MP20 splits")
        identities = [int(row["source_row_idx"]) for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError("v2 source file contains duplicate original source identities")
        self.tokenizer, self.constraints = tokenizer, constraints
        self.vocabulary = tokenizer.get_vocab()
        self.sources = [prepare_periodic_base_source(
            row, tokenizer, constraints, vocabulary=self.vocabulary,
        ) for row in rows]
        self.rows = [source.metadata for source in self.sources]
        self.seed, self.epoch = int(seed), 0
        self.real_length = 2 * len(self.rows)
        self.effective_batch = int(effective_batch)
        self.padded_length = math.ceil(self.real_length / self.effective_batch) * self.effective_batch
        self.alias_rows = sum(source.alias_tokens_canonicalized > 0 for source in self.sources)
        self.alias_tokens = sum(source.alias_tokens_canonicalized for source in self.sources)
        self.prefix_auditor = PrefixSupportAuditor(tokenizer, constraints)

    def __len__(self):
        return self.padded_length

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < self.padded_length:
            raise IndexError(index)
        source_view = index % self.real_length
        return make_periodic_v2_training_example(
            self.sources[source_view // 2], self.tokenizer, self.constraints,
            view=source_view % 2, epoch=self.epoch, seed=self.seed,
            is_padding=index >= self.real_length, vocabulary=self.vocabulary,
            prefix_auditor=self.prefix_auditor,
        )


__all__ = [
    "V2_TRAINING_DATA_PROTOCOL", "V2_PREFIX_PROBABILITY", "V2_MASK_PROBABILITY_RANGE",
    "PrefixReachability", "PrefixSupportAuditor", "make_periodic_v2_training_example",
    "PeriodicV2TrainingDataset",
]
