#!/usr/bin/env python3
"""Dependency-light contract tests for the native post-refine supplement."""

from __future__ import annotations

from native_protocol import (
    NATIVE_DENOMINATOR,
    candidate_seed,
    first_success_ranks,
    ordered_candidate_rows,
)
from protocol import paired_seed


def main() -> None:
    for repeat in range(3):
        for ordinal in (0, 1, 17, 999):
            for channel in ("body", "refiner"):
                assert candidate_seed(repeat, ordinal, channel) == paired_seed(
                    repeat, ordinal, channel
                )

    rows = ordered_candidate_rows(
        {"candidate_rank": rank} for rank in range(1191)
    )
    assert len(rows) == 1191
    attempts = {
        rank: {"status": "failed" if rank in {3, 101, 999, 1002} else "succeeded"}
        for rank in range(1191)
    }
    selected = first_success_ranks(attempts, len(attempts))
    assert len(selected) == NATIVE_DENOMINATOR
    assert selected == sorted(selected)
    assert selected[-1] == 1003
    assert all(attempts[rank]["status"] == "succeeded" for rank in selected)
    print("native1000_self_test=PASS")


if __name__ == "__main__":
    main()
