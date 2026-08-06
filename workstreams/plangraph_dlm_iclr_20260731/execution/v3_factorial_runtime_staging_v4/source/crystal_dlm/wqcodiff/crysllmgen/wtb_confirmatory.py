"""Frozen identities and paired statistics for WTB-256 confirmatory evaluation.

This module is deliberately MLIP-free.  It defines the method-independent
pairing, seed derivation, integrity signatures, and exact paired statistics
used by the confirmatory R/U/T comparison.  Generation, parent inference, and
CHGNet evaluation remain separate execution stages.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

from ..contracts import SeedDeriver


IDENTITY = "wq_wyckoff_chart_retraction_confirmatory256_v1"
PROTOCOL_NAME = "crysllmgen_wyckoff_georev_v4"
PAIRING_ID = f"{IDENTITY}-paired-source-and-noise"
SOURCE_EXPERIMENT_ID = f"{IDENTITY}-source"
ARM_EXPERIMENT_ID = f"{IDENTITY}-arms"

TRAINING_SEED = 11
SAMPLING_SEED = 101
START_ORDINAL = 512
ATTEMPTS = 256
END_ORDINAL_INCLUSIVE = START_ORDINAL + ATTEMPTS - 1
REVERSE_START_TIMESTEP = 800
REVERSE_STEPS = 32
DECODER_CALLS_PER_DIFFUSION_ARM = 64
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260726

SOURCE_METHOD = "WTB256-SOURCE-WQ"
RAW_METHOD = "WTB256-R-RAW-WQ"
UNCONSTRAINED_METHOD = "WTB256-U-PARENT-SCHEDULE32"
TANGENT_METHOD = "WTB256-T-EVERY-STEP-GLOBAL-CHART-RETRACTION"
ARM_METHODS = {
    "R": RAW_METHOD,
    "U": UNCONSTRAINED_METHOD,
    "T": TANGENT_METHOD,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_portable_seed(value: int) -> int:
    result = int(value)
    if not 0 <= result < (1 << 63):
        raise ValueError("seed must be a portable signed-63-bit integer")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class ConfirmatoryCell:
    """One method-independent WTB-256 source/noise identity."""

    ordinal: int
    pair_id: str
    source_attempt_id: str
    proposal_seed: int
    forward_noise_seed: int
    reverse_noise_seed: int
    arm_attempt_ids: Mapping[str, str]
    timestep: int = REVERSE_START_TIMESTEP

    def __post_init__(self) -> None:
        if not START_ORDINAL <= int(self.ordinal) <= END_ORDINAL_INCLUSIVE:
            raise ValueError("confirmatory ordinal is outside 512..767")
        if not self.pair_id or not self.source_attempt_id:
            raise ValueError("pair and source attempt identities are required")
        if set(self.arm_attempt_ids) != set(ARM_METHODS):
            raise ValueError("confirmatory cell must contain exact R/U/T arms")
        if len(set(self.arm_attempt_ids.values())) != len(ARM_METHODS):
            raise ValueError("arm attempt identities must be distinct")
        for seed in (
            self.proposal_seed,
            self.forward_noise_seed,
            self.reverse_noise_seed,
        ):
            _require_portable_seed(seed)
        if int(self.timestep) != REVERSE_START_TIMESTEP:
            raise ValueError("confirmatory reverse start must remain t=800")

    @property
    def panel_index(self) -> int:
        """Zero-based index used only by legacy duck-typed bridge helpers."""

        return int(self.ordinal) - START_ORDINAL

    @property
    def cell_id(self) -> str:
        payload = {
            "identity": IDENTITY,
            "ordinal": int(self.ordinal),
            "pair_id": self.pair_id,
            "timestep": int(self.timestep),
        }
        return f"c-{hashlib.sha256(_canonical_json(payload)).hexdigest()[:24]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": int(self.ordinal),
            "panel_index": self.panel_index,
            "cell_id": self.cell_id,
            "pair_id": self.pair_id,
            "source_attempt_id": self.source_attempt_id,
            "proposal_seed": int(self.proposal_seed),
            "forward_noise_seed": int(self.forward_noise_seed),
            "reverse_noise_seed": int(self.reverse_noise_seed),
            "arm_attempt_ids": dict(self.arm_attempt_ids),
            "timestep": int(self.timestep),
        }


def build_confirmatory_cells(
    *,
    training_seed: int = TRAINING_SEED,
    sampling_seed: int = SAMPLING_SEED,
    start_ordinal: int = START_ORDINAL,
    attempts: int = ATTEMPTS,
) -> tuple[ConfirmatoryCell, ...]:
    """Build the exact preregistered 256-cell panel.

    The strict range check intentionally prevents an apparently compatible
    development or replacement panel from entering the confirmatory identity.
    """

    if (
        int(training_seed) != TRAINING_SEED
        or int(sampling_seed) != SAMPLING_SEED
        or int(start_ordinal) != START_ORDINAL
        or int(attempts) != ATTEMPTS
    ):
        raise ValueError("WTB-256 panel differs from the frozen 11/101/512..767 panel")
    pairing = SeedDeriver(PROTOCOL_NAME, PAIRING_ID)
    source = SeedDeriver(PROTOCOL_NAME, SOURCE_EXPERIMENT_ID)
    arms = SeedDeriver(PROTOCOL_NAME, ARM_EXPERIMENT_ID)
    cells: list[ConfirmatoryCell] = []
    for ordinal in range(START_ORDINAL, END_ORDINAL_INCLUSIVE + 1):
        pair_id = pairing.pair_id(
            training_seed=TRAINING_SEED,
            sampling_seed=SAMPLING_SEED,
            ordinal=ordinal,
        )
        source_attempt_id = source.attempt_id(
            training_seed=TRAINING_SEED,
            sampling_seed=SAMPLING_SEED,
            ordinal=ordinal,
            method=SOURCE_METHOD,
        )
        cells.append(
            ConfirmatoryCell(
                ordinal=ordinal,
                pair_id=pair_id,
                source_attempt_id=source_attempt_id,
                proposal_seed=pairing.paired_derive(
                    training_seed=TRAINING_SEED,
                    sampling_seed=SAMPLING_SEED,
                    ordinal=ordinal,
                    stage="wq_proposal_generation",
                ),
                forward_noise_seed=pairing.paired_derive(
                    training_seed=TRAINING_SEED,
                    sampling_seed=SAMPLING_SEED,
                    ordinal=ordinal,
                    stage="parent_forward_noise",
                ),
                reverse_noise_seed=pairing.paired_derive(
                    training_seed=TRAINING_SEED,
                    sampling_seed=SAMPLING_SEED,
                    ordinal=ordinal,
                    stage="parent_reverse_noise",
                ),
                arm_attempt_ids={
                    arm: arms.attempt_id(
                        training_seed=TRAINING_SEED,
                        sampling_seed=SAMPLING_SEED,
                        ordinal=ordinal,
                        method=method,
                    )
                    for arm, method in ARM_METHODS.items()
                },
            )
        )
    if (
        len(cells) != ATTEMPTS
        or len({cell.cell_id for cell in cells}) != ATTEMPTS
        or len({cell.pair_id for cell in cells}) != ATTEMPTS
        or len({cell.source_attempt_id for cell in cells}) != ATTEMPTS
    ):
        raise RuntimeError("WTB-256 cell construction violated uniqueness")
    return tuple(cells)


def composition_signature(atomic_numbers: Sequence[int]) -> str:
    """Hash the ordered primitive composition without geometry."""

    values = tuple(int(value) for value in atomic_numbers)
    if not values or any(value <= 0 for value in values):
        raise ValueError("composition signature requires positive atomic numbers")
    return hashlib.sha256(
        _canonical_json(
            {
                "schema": "wtb256_ordered_primitive_composition_v1",
                "atomic_numbers": values,
            }
        )
    ).hexdigest()


def source_signature(
    *,
    proposal_state: Mapping[str, Any],
    topology_hash: str,
    atomic_numbers: Sequence[int],
) -> str:
    """Bind the discrete proposal, topology, and primitive composition."""

    if len(topology_hash) != 64:
        raise ValueError("source topology hash must be one SHA256")
    return hashlib.sha256(
        _canonical_json(
            {
                "schema": "wtb256_source_signature_v1",
                "proposal_state": dict(proposal_state),
                "topology_hash": topology_hash,
                "composition_signature": composition_signature(atomic_numbers),
            }
        )
    ).hexdigest()


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sample")
    position = (len(sorted_values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower]) * (1.0 - fraction) + float(
        sorted_values[upper]
    ) * fraction


def exact_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    """Two-sided exact McNemar/binomial test with explicit discordant counts."""

    if len(left) != len(right) or not left:
        raise ValueError("paired vectors must share one nonempty denominator")
    left_only = sum(bool(a) and not bool(b) for a, b in zip(left, right, strict=True))
    right_only = sum(not bool(a) and bool(b) for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "attempts": len(left),
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "two_sided_exact_p_value": p_value,
    }


def paired_binary_effect(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return paired percentage-point effect, CI, and exact discordance."""

    if len(left) != len(right) or not left:
        raise ValueError("paired vectors must share one nonempty denominator")
    if int(draws) != BOOTSTRAP_DRAWS:
        raise ValueError("WTB-256 paired bootstrap must use exactly 10,000 draws")
    differences = [
        float(bool(a)) - float(bool(b))
        for a, b in zip(left, right, strict=True)
    ]
    rng = random.Random(int(seed))
    size = len(differences)
    samples = [
        100.0
        * sum(differences[rng.randrange(size)] for _ in range(size))
        / size
        for _ in range(draws)
    ]
    samples.sort()
    return {
        "attempts": size,
        "left_count": sum(bool(value) for value in left),
        "right_count": sum(bool(value) for value in right),
        "left_rate": sum(bool(value) for value in left) / size,
        "right_rate": sum(bool(value) for value in right) / size,
        "difference_percentage_points": 100.0 * sum(differences) / size,
        "bootstrap": {
            "draws": draws,
            "seed": int(seed),
            "ci95_lower_percentage_points": _quantile(samples, 0.025),
            "ci95_upper_percentage_points": _quantile(samples, 0.975),
        },
        "exact_mcnemar": exact_mcnemar(left, right),
    }


def paired_numeric_effect(
    left: Sequence[float],
    right: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return a paired mean difference and percentile bootstrap interval.

    Callers must explicitly construct a common-observed pair set.  This
    function never silently drops nonfinite or missing values, and its output
    is descriptive only: all-attempt binary outcomes remain the promotion
    metrics.
    """

    if len(left) != len(right) or not left:
        raise ValueError("paired numeric vectors must share one nonempty denominator")
    if int(draws) != BOOTSTRAP_DRAWS:
        raise ValueError("WTB-256 paired bootstrap must use exactly 10,000 draws")
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if any(
        not math.isfinite(value)
        for value in (*left_values, *right_values)
    ):
        raise ValueError("paired numeric vectors must be entirely finite")
    differences = [
        a - b for a, b in zip(left_values, right_values, strict=True)
    ]
    rng = random.Random(int(seed))
    size = len(differences)
    samples = [
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(draws)
    ]
    samples.sort()
    return {
        "common_observed_pairs": size,
        "left_mean": sum(left_values) / size,
        "right_mean": sum(right_values) / size,
        "mean_difference": sum(differences) / size,
        "bootstrap": {
            "draws": draws,
            "seed": int(seed),
            "ci95_lower": _quantile(samples, 0.025),
            "ci95_upper": _quantile(samples, 0.975),
        },
        "selection_role": "descriptive_common_observed_pairs_only",
    }
