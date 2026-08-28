"""Matched sampling bookkeeping for SGTC-DLM screens."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from crystal_dlm.ctv_protocol import counter_seed


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


__all__ = ["matched_base_noise_group", "validate_sgtc_attempts"]
