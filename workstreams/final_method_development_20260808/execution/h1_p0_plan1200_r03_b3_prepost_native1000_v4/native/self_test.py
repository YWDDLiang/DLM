#!/usr/bin/env python3
"""Dependency-light contract tests for the native post-refine supplement."""

from __future__ import annotations

from native_protocol import (
    NATIVE_DENOMINATOR,
    candidate_seed,
    first_success_ranks,
    ordered_candidate_rows,
    sha256_text,
    canonical_sha256,
    validate_frozen_candidate_row,
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
    state = {
        "N": 2,
        "elements": ["Li", "O"],
        "counts": [1, 1],
        "charge_bucket": "neutral",
    }
    prompt = (
        "Generate only the exact-length dynamic crystal body for this fixed plan_state. "
        "The first token must match N and the element multiset must match elements/counts.\n"
        'plan_state: {"N":2,"charge_bucket":"neutral","counts":[1,1],'
        '"elements":["Li","O"]}\n'
        "dynamic_crystal_body:"
    )
    candidate = {
        "repeat": 0,
        "candidate_rank": 1000,
        "candidate_id": "p0-native-r0-1000",
        "planner_candidate_ordinal": 1007,
        "parsed_plan": {"elements": ["Li", "O"]},
        "plan_state": state,
        "plan_state_sha256": canonical_sha256(state),
        "body_prompt": prompt,
        "body_prompt_sha256": sha256_text(prompt),
        "body_prompt_contract": "historical_r5c_plan_state_json_exact_length",
        "raw_rich_seven_line_forwarded": False,
        "canonical_charge_bucket_visible": True,
        "body_noise_seed": candidate_seed(0, 1000, "body"),
        "refiner_noise_seed": candidate_seed(0, 1000, "refiner"),
        "candidate_partition": "frozen_reserve",
    }
    assert "parsed" not in candidate
    assert validate_frozen_candidate_row(
        candidate, repeat=0, candidate_rank=1000
    ) == state
    malformed = dict(candidate)
    malformed.pop("parsed_plan")
    try:
        validate_frozen_candidate_row(malformed, repeat=0, candidate_rank=1000)
    except ValueError:
        pass
    else:
        raise AssertionError("candidate without producer parse evidence passed")
    print("native1000_self_test=PASS")


if __name__ == "__main__":
    main()
