"""Rank-independent identities and random seeds for registered attempts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


ORDINAL_SEED_SCHEMA = "h1a2_ordinal_seed_v1"
_MAX_TORCH_SEED = (1 << 63) - 1


def sha256_text(text: str) -> str:
    """Hash exact UTF-8 text without normalization."""

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def derive_ordinal_seed(
    base_seed: int,
    *,
    sample_idx: int,
    stage: str,
    role: str = "shared",
) -> int:
    """Derive one stable seed independent of rank, world size, and batch order."""

    ordinal = int(sample_idx)
    if ordinal < 0:
        raise ValueError("sample_idx must be non-negative")
    stage_value = str(stage).strip()
    role_value = str(role).strip()
    if not stage_value:
        raise ValueError("stage must be non-empty")
    if not role_value:
        raise ValueError("role must be non-empty")
    payload = {
        "base_seed": int(base_seed),
        "role": role_value,
        "sample_idx": ordinal,
        "schema": ORDINAL_SEED_SCHEMA,
        "stage": stage_value,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & _MAX_TORCH_SEED


def ordered_ordinal_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_count: int | None = None,
    require_complete: bool = False,
) -> list[Mapping[str, Any]]:
    """Sort records by ordinal and reject duplicate/out-of-range identities."""

    if require_complete and expected_count is None:
        raise ValueError("require_complete needs expected_count")
    limit = None if expected_count is None else int(expected_count)
    if limit is not None and limit < 0:
        raise ValueError("expected_count must be non-negative")

    by_ordinal: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if "sample_idx" not in record:
            raise ValueError("record is missing sample_idx")
        ordinal = int(record["sample_idx"])
        if ordinal < 0:
            raise ValueError(f"negative sample_idx {ordinal}")
        if limit is not None and ordinal >= limit:
            raise ValueError(
                f"sample_idx {ordinal} is outside registered range [0, {limit})"
            )
        if ordinal in by_ordinal:
            raise ValueError(f"duplicate sample_idx {ordinal}")
        by_ordinal[ordinal] = record

    if require_complete:
        observed = set(by_ordinal)
        expected = set(range(int(limit or 0)))
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(
                f"ordinal ledger mismatch: missing={missing[:16]}, extra={extra[:16]}"
            )
    return [by_ordinal[index] for index in sorted(by_ordinal)]


__all__ = [
    "ORDINAL_SEED_SCHEMA",
    "derive_ordinal_seed",
    "ordered_ordinal_records",
    "sha256_text",
]
