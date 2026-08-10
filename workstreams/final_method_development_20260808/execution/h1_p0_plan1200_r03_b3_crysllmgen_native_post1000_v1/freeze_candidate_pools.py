#!/usr/bin/env python3
"""Freeze every parse-success P0 plan and prove its V3 prefix identity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from crystal_dlm.r5_plan_state import build_body_prompt

from native_protocol import (
    PREFIX_COUNT,
    RAW_ATTEMPTS,
    candidate_seed,
    canonical_sha256,
    identity,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    validate_repeat,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--planner-dir", type=Path, required=True)
    parser.add_argument("--v3-cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repeat = validate_repeat(args.repeat)
    planner = args.planner_dir.resolve()
    cohort_root = args.v3_cohort_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    raw_path = planner / "raw_generations.jsonl"
    plans_path = planner / "plans_for_dlm.jsonl"
    metrics_path = planner / "sample_metrics.json"
    cohort_path = cohort_root / "cohort1000.jsonl"
    cohort_manifest_path = cohort_root / "cohort_manifest.json"
    raw = sorted(read_jsonl(raw_path), key=lambda row: int(row["sample_idx"]))
    plans = sorted(read_jsonl(plans_path), key=lambda row: int(row["sample_idx"]))
    metrics = read_json(metrics_path)
    cohort = sorted(
        read_jsonl(cohort_path), key=lambda row: int(row["cohort_ordinal"])
    )
    cohort_manifest = read_json(cohort_manifest_path)
    if (
        len(raw) != RAW_ATTEMPTS
        or [int(row.get("sample_idx", -1)) for row in raw]
        != list(range(RAW_ATTEMPTS))
        or not PREFIX_COUNT <= len(plans) <= RAW_ATTEMPTS
        or len(cohort) != PREFIX_COUNT
        or [int(row.get("cohort_ordinal", -1)) for row in cohort]
        != list(range(PREFIX_COUNT))
        or int(metrics.get("requested_samples", -1)) != RAW_ATTEMPTS
        or int(metrics.get("plan_parse_success", -1)) != len(plans)
        or int(cohort_manifest.get("parse_successes", -1)) != len(plans)
        or int(cohort_manifest.get("reserve_parse_success_count", -1))
        != len(plans) - PREFIX_COUNT
    ):
        raise ValueError("frozen planner or V3 cohort denominator changed")

    recorded_cohort = (cohort_manifest.get("artifacts") or {}).get("cohort1000") or {}
    if (
        recorded_cohort.get("sha256") != sha256_file(cohort_path)
        or [int(row["sample_idx"]) for row in plans[:PREFIX_COUNT]]
        != [int(row["planner_candidate_ordinal"]) for row in cohort]
    ):
        raise ValueError("V3 prefix is not the first 1,000 parse-success plans")

    rows: list[dict[str, Any]] = []
    for candidate_rank, plan_record in enumerate(plans):
        state = plan_record.get("plan_state")
        if not isinstance(state, Mapping):
            raise ValueError(f"candidate {candidate_rank} lacks plan_state")
        prompt = build_body_prompt(dict(state))
        if "plan_state:" not in prompt or '"charge_bucket"' not in prompt:
            raise ValueError(f"candidate {candidate_rank} changed body prompt contract")
        planner_ordinal = int(plan_record["sample_idx"])
        row = {
            **plan_record,
            "repeat": repeat,
            "candidate_rank": candidate_rank,
            "candidate_id": f"p0-native-r{repeat}-{candidate_rank:04d}",
            "planner_candidate_ordinal": planner_ordinal,
            "plan_state_sha256": canonical_sha256(dict(state)),
            "body_prompt": prompt,
            "body_prompt_sha256": sha256_text(prompt),
            "body_prompt_contract": "historical_r5c_plan_state_json_exact_length",
            "raw_rich_seven_line_forwarded": False,
            "canonical_charge_bucket_visible": True,
            "body_noise_seed": candidate_seed(repeat, candidate_rank, "body"),
            "refiner_noise_seed": candidate_seed(repeat, candidate_rank, "refiner"),
            "candidate_partition": (
                "v3_prefix" if candidate_rank < PREFIX_COUNT else "frozen_reserve"
            ),
        }
        if candidate_rank < PREFIX_COUNT:
            prefix = cohort[candidate_rank]
            if (
                int(prefix["planner_candidate_ordinal"]) != planner_ordinal
                or prefix.get("plan_state") != state
                or prefix.get("body_prompt") != prompt
                or int(prefix.get("body_noise_seed", candidate_seed(repeat, candidate_rank, "body")))
                != candidate_seed(repeat, candidate_rank, "body")
            ):
                raise ValueError(f"V3 prefix identity changed at {candidate_rank}")
        rows.append(row)

    output.mkdir(parents=True)
    pool_path = output / "candidate_pool.jsonl"
    write_jsonl_exclusive(pool_path, rows)
    manifest = {
        "schema": "h1_plan1200_crysllmgen_native_candidate_pool_v1",
        "status": "complete",
        "repeat": repeat,
        "raw_attempts": RAW_ATTEMPTS,
        "parse_successes": len(rows),
        "parse_failures": RAW_ATTEMPTS - len(rows),
        "v3_prefix_count": PREFIX_COUNT,
        "reserve_count": len(rows) - PREFIX_COUNT,
        "candidate_order": "planner_parse_success_rank_by_planner_ordinal",
        "v3_prefix_byte_identity": True,
        "R03_B3_shared_candidate_pool": True,
        "body_success_selection_not_yet_applied": True,
        "same_plan_retry": False,
        "stochastic_replacement": False,
        "artifacts": {
            "raw_generations": identity(raw_path),
            "plans_for_dlm": identity(plans_path),
            "sample_metrics": identity(metrics_path),
            "v3_cohort1000": identity(cohort_path),
            "v3_cohort_manifest": identity(cohort_manifest_path),
            "candidate_pool": identity(pool_path),
        },
    }
    write_json_exclusive(output / "candidate_pool_manifest.json", manifest)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
