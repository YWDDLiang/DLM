"""Deterministic distributed index sharding without padding or duplication."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable, Sequence


def strided_shard_indices(
    length: int,
    *,
    num_replicas: int,
    rank: int,
) -> list[int]:
    """Return the rank's deterministic strided shard without padding."""

    length = int(length)
    num_replicas = int(num_replicas)
    rank = int(rank)
    if length < 0:
        raise ValueError("length must be non-negative")
    if num_replicas <= 0:
        raise ValueError("num_replicas must be positive")
    if rank < 0 or rank >= num_replicas:
        raise ValueError(
            f"rank must be in [0, {num_replicas}); received {rank}"
        )
    return list(range(rank, length, num_replicas))


def ordered_index_sha256(indices: Iterable[int]) -> str:
    """Hash an ordered index sequence with an unambiguous line encoding."""

    payload = "".join(f"{int(index)}\n" for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def audit_distributed_index_shards(
    length: int,
    shards: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Audit exact distributed coverage and the frozen strided rank mapping."""

    length = int(length)
    if length < 0:
        raise ValueError("length must be non-negative")
    if not shards:
        raise ValueError("at least one shard is required")

    normalized = [[int(index) for index in shard] for shard in shards]
    flattened = [index for shard in normalized for index in shard]
    counts = Counter(flattened)
    out_of_range = sorted(
        index for index in counts if index < 0 or index >= length
    )
    duplicate_indices = sorted(
        index for index, count in counts.items() if count > 1
    )
    missing_indices = sorted(index for index in range(length) if counts[index] == 0)
    expected = [
        strided_shard_indices(
            length,
            num_replicas=len(normalized),
            rank=rank,
        )
        for rank in range(len(normalized))
    ]
    per_rank = []
    for rank, indices in enumerate(normalized):
        per_rank.append(
            {
                "rank": rank,
                "count": len(indices),
                "first_index": indices[0] if indices else None,
                "last_index": indices[-1] if indices else None,
                "ordered_index_sha256": ordered_index_sha256(indices),
                "matches_expected_strided_shard": indices == expected[rank],
            }
        )

    exact_cover = (
        len(flattened) == length
        and not out_of_range
        and not duplicate_indices
        and not missing_indices
    )
    rank_mapping_exact = all(
        item["matches_expected_strided_shard"] for item in per_rank
    )
    return {
        "schema": "distributed_no_padding_index_audit_v1",
        "dataset_length": length,
        "world_size": len(normalized),
        "total_assigned": len(flattened),
        "unique_assigned": len(counts),
        "duplicate_count": sum(max(0, count - 1) for count in counts.values()),
        "duplicate_indices": duplicate_indices,
        "missing_count": len(missing_indices),
        "missing_indices": missing_indices,
        "out_of_range_indices": out_of_range,
        "ordered_global_index_sha256": ordered_index_sha256(range(length)),
        "per_rank": per_rank,
        "exact_cover": exact_cover,
        "rank_mapping_exact": rank_mapping_exact,
        "gate_passed": exact_cover and rank_mapping_exact,
    }
