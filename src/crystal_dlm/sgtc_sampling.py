"""Matched sampling bookkeeping for SGTC-DLM screens."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from crystal_dlm.ctv_protocol import counter_seed


SGTC_SCREEN_DENOMINATORS = frozenset({256, 1000})


def validate_sgtc_denominator(value: int) -> int:
    denominator = int(value)
    if denominator not in SGTC_SCREEN_DENOMINATORS:
        raise ValueError(
            "SGTC sampling denominator must be the frozen L6=256 or L7=1000"
        )
    return denominator


def matched_base_noise_group(
    *, seed: int, composition_id: str, sample_idx: int
) -> int:
    if not composition_id:
        raise ValueError("SGTC composition identity must be non-empty")
    return counter_seed(
        "sgtc-l6-base-v1", int(seed), str(composition_id), int(sample_idx)
    )


def validate_sgtc_attempts(
    rows: Sequence[Mapping[str, Any]], *, expected: int
) -> dict[str, int]:
    denominator = int(expected)
    if denominator <= 0:
        raise ValueError("SGTC attempt denominator must be positive")
    ordinals = [int(row["ordinal"]) for row in rows]
    sample_indices = [int(row["sample_idx"]) for row in rows]
    if len(rows) != denominator or ordinals != list(range(denominator)):
        raise ValueError("SGTC attempts do not cover the requested ordinal denominator")
    if sample_indices != list(range(denominator)):
        raise ValueError("SGTC sample indices are not global ordinal aligned")
    return {
        "requested": denominator,
        "parsed": sum(row.get("parsed") is True for row in rows),
        "failed": sum(row.get("parsed") is not True for row in rows),
    }


def validate_sgtc_plan_rows(
    rows: Sequence[Mapping[str, Any]], *, expected: int
) -> dict[str, int]:
    denominator = int(expected)
    if denominator <= 0 or len(rows) != denominator:
        raise ValueError("SGTC Plans do not cover the requested denominator")
    sample_indices = [int(row["sample_idx"]) for row in rows]
    if sample_indices != list(range(denominator)):
        raise ValueError("SGTC Plan sample indices are not global ordinal aligned")
    identities = [
        str(row.get("reduced_composition_identity", "")).strip() for row in rows
    ]
    if any(not identity for identity in identities):
        raise ValueError("SGTC Plan composition identity must be non-empty")
    if any(not isinstance(row.get("plan_state"), Mapping) for row in rows):
        raise ValueError("SGTC Plan state must be a mapping")
    if any(not str(row.get("prompt", "")).strip() for row in rows):
        raise ValueError("SGTC Plan prompt must be non-empty")
    unique = len(set(identities))
    return {
        "plan_rows": denominator,
        "unique_composition_identities": unique,
        "duplicate_composition_attempts": denominator - unique,
    }


def validate_sgtc_plan_rows_with_missing(
    rows: Sequence[Mapping[str, Any]], *, expected: int
) -> dict[str, int]:
    """Validate a fixed ledger while retaining upstream Planner failures."""

    denominator = int(expected)
    if denominator <= 0 or len(rows) != denominator:
        raise ValueError("SGTC Plan ledger does not cover the requested denominator")
    if [int(row["sample_idx"]) for row in rows] != list(range(denominator)):
        raise ValueError("SGTC Plan ledger sample indices are not globally aligned")
    identities = [
        str(row.get("reduced_composition_identity", "")).strip() for row in rows
    ]
    if any(not identity for identity in identities):
        raise ValueError("SGTC Plan ledger composition identity must be non-empty")
    valid = 0
    failed = 0
    for row in rows:
        plan = row.get("plan_state")
        prompt = str(row.get("prompt") or "").strip()
        if isinstance(plan, Mapping):
            if not prompt:
                raise ValueError("valid SGTC Plan ledger row lacks prompt")
            valid += 1
        else:
            if row.get("planner_failed") is not True or prompt:
                raise ValueError("missing SGTC Plan row lacks explicit failure state")
            failed += 1
    return {
        "plan_rows": denominator,
        "plan_valid": valid,
        "plan_failed": failed,
        "unique_composition_identities": len(set(identities)),
        "duplicate_composition_attempts": denominator - len(set(identities)),
    }


__all__ = [
    "SGTC_SCREEN_DENOMINATORS",
    "matched_base_noise_group",
    "validate_sgtc_attempts",
    "validate_sgtc_denominator",
    "validate_sgtc_plan_rows",
    "validate_sgtc_plan_rows_with_missing",
]
