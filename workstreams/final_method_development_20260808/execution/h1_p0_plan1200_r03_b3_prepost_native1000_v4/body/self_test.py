#!/usr/bin/env python3
"""Pure self-tests for the V4 cohort-contract repair."""

from __future__ import annotations

from pathlib import Path

from protocol import (
    paired_seed,
    read_json,
    sha256_text,
    validate_config,
    validate_frozen_cohort_row,
)


def expect_failure(row: dict[str, object]) -> None:
    try:
        validate_frozen_cohort_row(row, repeat=0, ordinal=0)
    except ValueError:
        return
    raise AssertionError("malformed cohort row unexpectedly passed")


def main() -> None:
    source = Path(__file__).resolve().parent
    validate_config(read_json(source / "CONFIG.json"))
    plan_state = {
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
    row: dict[str, object] = {
        "repeat": 0,
        "cohort_ordinal": 0,
        "planner_candidate_ordinal": 7,
        "attempt_id": "p0-plan1200-r0-0000",
        "parsed_plan": {"elements": ["Li", "O"]},
        "plan_state": plan_state,
        "body_prompt": prompt,
        "body_prompt_sha256": sha256_text(prompt),
        "body_prompt_contract": "historical_r5c_plan_state_json_exact_length",
        "raw_rich_seven_line_forwarded": False,
        "canonical_charge_bucket_visible": True,
    }
    assert "parsed" not in row
    assert validate_frozen_cohort_row(row, repeat=0, ordinal=0) == plan_state
    missing_parse_evidence = dict(row)
    missing_parse_evidence.pop("parsed_plan")
    expect_failure(missing_parse_evidence)
    bad_prompt = dict(row)
    bad_prompt["body_prompt_sha256"] = "0" * 64
    expect_failure(bad_prompt)
    assert paired_seed(0, 0, "body") != paired_seed(0, 0, "refiner")
    print("V4 cohort-contract self-test PASS")


if __name__ == "__main__":
    main()
