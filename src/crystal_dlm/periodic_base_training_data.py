"""Two MP20-only views for a freshly initialized periodic crystal DLM.

Construction uses independent numeric masks and never supplies hidden clean
geometry through ``old_body``. Repair uses a quantized geometric corruption as
the full old snapshot and a reachable, teacher-forced transaction prefix. This
module reads no generated paths, energies, relaxation endpoints, or old masks.
Schema-valid sources remain present even when their geometry is outside the
deployment support; the loss separately accounts for that distinction.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens, parse_dynamic_answer
from crystal_dlm.fixed_slot import FixedSlotConfig
from crystal_dlm.llada_generation import lattice_angle_rad
from crystal_dlm.manifold_corruption import (
    lattice_matrix_from_parameters, lattice_parameters_from_matrix, symmetric_matrix_exp,
)
from crystal_dlm.programmed_path_runtime import (
    complete_geometry_supported, full_cell_transaction_positions,
)
from crystal_dlm.spad_program import program_from_element_order


# (symmetric log-strain operator-norm bound, Cartesian component half-width A).
STRUCTURED_NOISE_LEVELS = ((.01, .05), (.03, .15), (.06, .30))
MASK_PROBABILITY_RANGE = (.1, 1.)
BASE_TRAINING_DATA_PROTOCOL = {
    "schema": "periodic_base_mp20_two_views_v1",
    "source": "original_MP20_clean_native",
    "views_per_source_epoch": 2,
    "construction_mask_probability": list(MASK_PROBABILITY_RANGE),
    "construction_empty_mask": "retained_zero_contribution",
    "inverse_mask_probability_weighting": False,
    "repair_family": "uniform RNG keyed by seed, split, source_row_idx, epoch, and view",
    "noise_metadata_dropout_probability": .5,
    "repair_transaction": "full_cell_in_species_program_order",
    "structured_noise_levels": [list(level) for level in STRUCTURED_NOISE_LEVELS],
    "failed_corruption": "retain_source_with_clean_old_snapshot_and_explicit_reason",
    "source_filtering_by_deployment_support": False,
    "generated_paths_or_physics_read": False,
}

# Never copy a whole source row: the old rollout mask and any outcome fields
# have no role in the two newly sampled training views.
_METADATA_KEYS = (
    "prompt", "answer", "plan_state", "species_program", "species_program_source",
    "source_row_idx", "source_split", "source_id", "material_id", "mp_id",
    "sample_weight", "loss_profile", "pointer_semantics_available",
)


@dataclass(frozen=True)
class PreparedBaseSource:
    metadata: dict[str, Any]
    arrays: dict[str, Any]
    clean_tokens: tuple[int, ...]
    transaction_positions: tuple[int, ...]
    alias_tokens_canonicalized: int
    source_answer_sha256: str
    canonical_answer_sha256: str


def _config(constraints: Mapping[str, Any]) -> FixedSlotConfig:
    if int(constraints.get("body_offset", 0)) != 0:
        raise ValueError("base training requires body-relative constraints")
    config = FixedSlotConfig()
    if (float(constraints.get("length_step", config.length_step)) != config.length_step
            or int(constraints.get("coord_period", config.coord_max_bin)) != config.coord_max_bin):
        raise ValueError("base training keeps the original MP20 numeric codec")
    return config


def _canonical_tokens(tokens: Sequence[str]) -> tuple[list[str], int]:
    canonical, changed = [], 0
    for token in tokens:
        alias = token.startswith(("<X_", "<Y_", "<Z_")) and token.endswith("_100>")
        canonical.append(token.replace("_100>", "_000>") if alias else token)
        changed += int(alias)
    return canonical, changed


def _mask_id(tokenizer: Any, vocabulary: Mapping[str, int]) -> int:
    for attribute in ("mask_token_id", "mask_id"):
        value = getattr(tokenizer, attribute, None)
        if value is not None:
            return int(value)
    for token in ("<|mdm_mask|>", "<MASK>", "[MASK]"):
        if token in vocabulary:
            return int(vocabulary[token])
    raise ValueError("tokenizer must expose the native mask token")


def prepare_periodic_base_source(
    row: Mapping[str, Any], tokenizer: Any, constraints: Mapping[str, Any], *,
    vocabulary: Mapping[str, int] | None = None,
) -> PreparedBaseSource:
    """Compile one clean source once, without filtering physical outcomes."""
    config = _config(constraints)
    metadata = {key: deepcopy(row[key]) for key in _METADATA_KEYS if key in row}
    if metadata.get("source_split") not in ("train", "val"):
        raise ValueError("base training sources must identify the original train/val split")
    source_index = int(metadata["source_row_idx"])
    if source_index < 0:
        raise ValueError("source_row_idx must be nonnegative")
    metadata["source_row_idx"] = source_index
    if not isinstance(metadata.get("prompt"), str) or not metadata["prompt"].strip():
        raise ValueError("source requires its original nonempty Plan prompt")
    if not isinstance(metadata.get("answer"), str):
        raise ValueError("source requires its original native MP20 answer")
    weight = float(metadata.get("sample_weight", 1.))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("source sample_weight must be finite and nonnegative")
    metadata["sample_weight"] = weight
    original_answer = metadata["answer"]
    parsed = parse_dynamic_answer(original_answer, strict=True, config=config)
    tokens, alias_count = _canonical_tokens(parsed["tokens"])
    canonical_answer = " ".join(tokens)
    arrays = parse_dynamic_answer(canonical_answer, strict=True, config=config)
    metadata["answer"] = canonical_answer
    if not isinstance(metadata.get("species_program_source"), str) or not metadata["species_program_source"]:
        raise ValueError("source requires the recorded species program provenance")
    program = program_from_element_order(
        metadata["plan_state"], metadata["species_program"],
        order_source=metadata["species_program_source"],
    )
    if int(arrays["num_atoms"]) != program.num_atoms:
        raise ValueError("clean MP20 atom count differs from the exact Plan")
    for entry in program.entries:
        if any(arrays["species"][slot] != entry.symbol for slot in entry.slot_indices):
            raise ValueError("clean MP20 element slots differ from the canonical program")
    vocabulary = tokenizer.get_vocab() if vocabulary is None else vocabulary
    clean = tuple(int(vocabulary[token]) for token in tokens)
    return PreparedBaseSource(
        metadata=metadata, arrays=arrays, clean_tokens=clean,
        transaction_positions=tuple(full_cell_transaction_positions(program)),
        alias_tokens_canonicalized=alias_count,
        source_answer_sha256=hashlib.sha256(original_answer.encode("utf-8")).hexdigest(),
        canonical_answer_sha256=hashlib.sha256(canonical_answer.encode("utf-8")).hexdigest(),
    )


def _rng(source: PreparedBaseSource, *, seed: int, epoch: int, view: int):
    key = (f"periodic-base-views-v1:{seed}:{source.metadata['source_split']}:"
           f"{source.metadata['source_row_idx']}:{epoch}:{view}")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


def _family(position: int) -> int:
    return 0 if position <= 3 else 1 if position <= 6 else 2


def _structured_corruption(source, rng, vocabulary, constraints):
    """One bounded perturbation; geometric rejection retains the clean source."""
    level_index = int(rng.integers(len(STRUCTURED_NOISE_LEVELS)))
    strain_bound, displacement_bound = STRUCTURED_NOISE_LEVELS[level_index]
    clean = list(source.clean_tokens)
    info = {
        "target_source": "original_MP20_clean_native",
        "level_index": level_index,
        "requested_numeric_noise_level": (level_index + 1) / len(STRUCTURED_NOISE_LEVELS),
        "log_strain_operator_norm_bound": strain_bound,
        "cartesian_component_half_width_A": displacement_bound,
        "fallback_to_clean": False, "fallback_reason": None,
        "encoding": None, "alias_tokens_canonicalized": 0,
    }
    failure = None
    noisy = clean
    try:
        arrays = source.arrays
        lattice = lattice_matrix_from_parameters(arrays["lengths"], arrays["angles"])
        symmetric = rng.normal(size=(3, 3))
        symmetric = (symmetric + symmetric.T) / 2
        symmetric *= strain_bound / max(float(np.linalg.norm(symmetric, ord=2)), 1.)
        noisy_lattice = lattice @ symmetric_matrix_exp(symmetric)
        cartesian = rng.uniform(-displacement_bound, displacement_bound, size=(arrays["num_atoms"], 3))
        fractional = np.mod(
            np.asarray(arrays["frac_coords"]) + cartesian @ np.linalg.inv(noisy_lattice), 1.,
        )
        lengths, angles = lattice_parameters_from_matrix(noisy_lattice)
        tokens, diagnostics = arrays_to_dynamic_tokens(
            lengths, angles, arrays["species"], fractional, config=_config(constraints),
        )
        tokens, aliases = _canonical_tokens(tokens)
        info.update(
            log_strain_operator_norm=float(np.linalg.norm(symmetric, ord=2)),
            cartesian_max_displacement_A=float(np.linalg.norm(cartesian, axis=-1).max()),
            encoding=asdict(diagnostics), alias_tokens_canonicalized=aliases,
        )
        if diagnostics.length_clips or diagnostics.angle_clips or diagnostics.coord_clips:
            failure = "perturbation_tokenization_clipped"
        else:
            noisy = [int(vocabulary[token]) for token in tokens]
            quantized_angles = [float(int(tokens[p][-4:-1])) for p in (4, 5, 6)]
            if (constraints.get("lattice_volume_mask") and lattice_angle_rad(*quantized_angles)
                    <= float(constraints.get("min_lattice_rad", 1e-4))):
                failure = "perturbation_outside_lattice_logit_support"
            elif not complete_geometry_supported(torch.tensor(noisy, dtype=torch.long), constraints):
                failure = "perturbation_outside_native_geometry_support"
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        failure = f"perturbation_unavailable:{type(error).__name__}:{error}"
    if failure is not None:
        noisy = clean.copy()
        info.update(fallback_to_clean=True, fallback_reason=failure)
    protected = (0, *(7 + 4 * slot for slot in range(source.arrays["num_atoms"])))
    if len(noisy) != len(clean) or any(noisy[p] != clean[p] for p in protected):
        raise RuntimeError("structured corruption changed the exact native composition")
    changed = [p for p, (before, after) in enumerate(zip(clean, noisy)) if before != after]
    info.update(changed_positions=changed, applied=bool(changed), quantized_unchanged=not changed)
    noise_level = info["requested_numeric_noise_level"] if changed else 0.
    return noisy, float(noise_level), info


def make_periodic_base_training_example(
    row: Mapping[str, Any] | PreparedBaseSource, tokenizer: Any, constraints: dict, *,
    view: int, epoch: int, seed: int, is_padding: bool = False,
    vocabulary: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return dense construction targets or one full-cell conditional target.

    ``positions``/``targets`` are the actual supervised positions. They may be
    empty in construction. The singular ``position``/``target_token`` always
    exist for the shared batch materializer; their dummy value must not create
    a loss when the plural lists are empty. No inverse mask-probability weight
    is applied. All indices are relative to the exact ``7+4N`` body.
    """
    view, epoch, seed = int(view), int(epoch), int(seed)
    if view not in (0, 1) or min(epoch, seed) < 0:
        raise ValueError("view must be 0/1 and epoch/seed must be nonnegative")
    vocabulary = tokenizer.get_vocab() if vocabulary is None else vocabulary
    source = (row if isinstance(row, PreparedBaseSource) else
              prepare_periodic_base_source(row, tokenizer, constraints, vocabulary=vocabulary))
    rng = _rng(source, seed=seed, epoch=epoch, view=view)
    clean = list(source.clean_tokens)
    all_positions = list(source.transaction_positions)
    mask_id = _mask_id(tokenizer, vocabulary)
    probability, transaction_step, family = None, None, -1
    corruption_info = {
        "target_source": "original_MP20_clean_native", "applied": False,
        "fallback_to_clean": False, "fallback_reason": None,
    }
    if view == 0:
        probability = float(rng.uniform(*MASK_PROBABILITY_RANGE))
        positions = [p for p, selected in zip(all_positions, rng.random(len(all_positions)) < probability) if selected]
        current = clean.copy()
        for position in positions:
            current[position] = mask_id
        old = current.copy()  # Hidden clean coordinates must never enter geometry context.
        transaction = positions.copy()
        phase, noise_level = "construct", 0.
    else:
        family = int(rng.integers(3))
        candidates = [p for p in all_positions if _family(p) == family]
        position = int(rng.choice(candidates))
        old, noise_level, corruption_info = _structured_corruption(source, rng, vocabulary, constraints)
        current = old.copy()
        for p in all_positions:
            current[p] = mask_id
        transaction_step = all_positions.index(position)
        for p in all_positions[:transaction_step]:
            current[p] = clean[p]
        positions, transaction = [position], all_positions.copy()
        phase = "full_cell_repair"
    declared_noise_level = float(noise_level)
    noise_metadata_dropped = bool(rng.random() < .5)
    if noise_metadata_dropped:
        noise_level = -1.
    targets = [clean[p] for p in positions]
    if view == 0:
        probe_family = int(rng.integers(3))
        probe_pool = [p for p in positions if _family(p) == probe_family]
        legal_probe_position = int(rng.choice(probe_pool)) if probe_pool else None
    else:
        legal_probe_position = positions[0]
    singular_position = positions[0] if positions else all_positions[0]
    source_weight = float(source.metadata["sample_weight"])
    return {
        **deepcopy(source.metadata),
        "schema": BASE_TRAINING_DATA_PROTOCOL["schema"],
        "input_body": current, "old_body": old,
        "num_atoms": int(source.arrays["num_atoms"]),
        "positions": positions, "targets": targets,
        "legal_probe_position": legal_probe_position,
        "legal_probe_target": None if legal_probe_position is None else clean[legal_probe_position],
        "position": singular_position, "target_token": clean[singular_position],
        "target_families": [_family(p) for p in positions],
        "transaction_positions": transaction, "transaction_step": transaction_step,
        "phase": phase, "family": family, "view": view, "epoch": epoch,
        "numeric_noise_level": noise_level, "mask_probability": probability,
        "declared_numeric_noise_level": declared_noise_level,
        "noise_metadata_dropped": noise_metadata_dropped,
        "empty_supervision": not positions,
        "source_sample_weight": source_weight,
        "sample_weight": 0. if is_padding or not positions else source_weight,
        "is_padding": bool(is_padding),
        "alias_tokens_canonicalized": source.alias_tokens_canonicalized,
        "source_answer_sha256": source.source_answer_sha256,
        "canonical_answer_sha256": source.canonical_answer_sha256,
        "corruption_info": corruption_info,
        "structured_corruption_fallback": bool(corruption_info["fallback_to_clean"]),
        "outcomes_read": False, "runtime_support_checked": False,
        "requires_runtime_support_check": True,
    }


class PeriodicBaseTrainingDataset(Dataset):
    """Fixed two-view source coverage, including zero-weight global padding."""

    def __init__(
        self, path_or_rows: str | Path | Sequence[Mapping[str, Any]], tokenizer: Any,
        constraints: dict, *, seed: int, expected_rows: int = 27136,
        expected_split: str = "train", effective_batch: int = 16,
    ):
        if expected_split not in ("train", "val") or effective_batch < 1 or int(seed) < 0:
            raise ValueError("invalid source split, batch size, or seed")
        if isinstance(path_or_rows, (str, Path)):
            with Path(path_or_rows).open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        else:
            rows = list(path_or_rows)
        if len(rows) != expected_rows or not rows:
            raise ValueError("the complete registered MP20 source count changed")
        if any(row.get("source_split") != expected_split for row in rows):
            raise ValueError("source file mixes original MP20 splits")
        identities = [int(row["source_row_idx"]) for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate original MP20 source identity")
        self.tokenizer, self.constraints = tokenizer, constraints
        self.vocabulary = tokenizer.get_vocab()
        self.sources = [prepare_periodic_base_source(row, tokenizer, constraints, vocabulary=self.vocabulary)
                        for row in rows]
        self.rows = [source.metadata for source in self.sources]
        self.seed, self.epoch = int(seed), 0
        self.real_length = 2 * len(self.rows)
        self.padded_length = math.ceil(self.real_length / effective_batch) * effective_batch
        self.alias_rows = sum(row.alias_tokens_canonicalized > 0 for row in self.sources)
        self.alias_tokens = sum(row.alias_tokens_canonicalized for row in self.sources)

    def __len__(self):
        return self.padded_length

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < self.padded_length:
            raise IndexError(index)
        source_view = index % self.real_length
        return make_periodic_base_training_example(
            self.sources[source_view // 2], self.tokenizer, self.constraints,
            view=source_view % 2, epoch=self.epoch, seed=self.seed,
            is_padding=index >= self.real_length, vocabulary=self.vocabulary,
        )
