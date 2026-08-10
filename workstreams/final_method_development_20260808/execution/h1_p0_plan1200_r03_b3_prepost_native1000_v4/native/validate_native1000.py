#!/usr/bin/env python3
"""Validate one CrysLLMGen-native full-1,000 post-refine repeat."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from native_protocol import (
    NATIVE_DENOMINATOR,
    identity,
    read_json,
    read_jsonl,
    validate_arm,
    validate_repeat,
    write_json_exclusive,
)
from protocol import ordered_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    repeat = validate_repeat(args.repeat)
    root = args.repeat_root.resolve()
    selection = read_json(root / "selected_body1000/selection_report.json")
    reserve_refinement = read_json(
        root / "reserve_refinement/refinement_metrics.json"
    )
    generation_report = read_json(
        root / "post_model494/generation/generation_report.json"
    )
    generation = ordered_rows(
        read_jsonl(root / "post_model494/generation/generation.jsonl"),
        ordinal_field="ordinal",
    )
    stage = read_json(root / "post_model494/evaluation/stage_report.json")
    mapping = sorted(
        read_jsonl(root / "selected_body1000/selection_mapping.jsonl"),
        key=lambda row: int(row["native_ordinal"]),
    )
    ranks = [int(row["candidate_rank"]) for row in mapping]
    if (
        selection.get("status") != "complete"
        or int(selection.get("selected_body_successes", -1)) != NATIVE_DENOMINATOR
        or selection.get("selection")
        != "first_1000_body_successes_by_frozen_candidate_order"
        or selection.get("upstream_crysllmgen_semantics") is not True
        or reserve_refinement.get("status") != "complete"
        or reserve_refinement.get("all_selected_candidates_refined_after_merge")
        is not True
        or generation_report.get("ok") is not True
        or int(generation_report.get("attempts", -1)) != NATIVE_DENOMINATOR
        or int(generation_report.get("generation_succeeded", -1))
        != NATIVE_DENOMINATOR
        or generation_report.get("all_1000_diffusion_refined") is not True
        or len(mapping) != NATIVE_DENOMINATOR
        or [int(row.get("native_ordinal", -1)) for row in mapping]
        != list(range(NATIVE_DENOMINATOR))
        or ranks != sorted(ranks)
        or len(set(ranks)) != NATIVE_DENOMINATOR
        or any(row.get("status") != "succeeded" for row in generation)
        or any(row.get("diffusion_refinement_applied") is not True for row in generation)
        or any(row.get("retry_or_replacement_used") is not False for row in generation)
        or stage.get("status") != "complete"
        or stage.get("ok") is not True
        or int(stage.get("attempts", -1)) != NATIVE_DENOMINATOR
        or int(stage.get("generation_succeeded", -1)) != NATIVE_DENOMINATOR
        or str(stage.get("arm")) != arm
        or int(stage.get("repeat", -1)) != repeat
        or stage.get("stage") != "post_model494"
    ):
        raise ValueError("CrysLLMGen-native full-1000 contract changed")
    output = {
        "schema": "h1_plan1200_crysllmgen_native_repeat_report_v1",
        "status": "complete",
        "ok": True,
        "arm": arm,
        "repeat": repeat,
        "candidate_attempts_through_completion": int(
            selection["candidate_attempts_through_completion"]
        ),
        "body_failures_before_completion": int(
            selection["body_failures_before_completion"]
        ),
        "selected_prefix_count": int(selection["selected_prefix_count"]),
        "selected_reserve_count": int(selection["selected_reserve_count"]),
        "refined_structures": NATIVE_DENOMINATOR,
        "sampling_contract": "crysllmgen_native_first_1000_body_successes",
        "crysllmgen_denominator": NATIVE_DENOMINATOR,
        "sun_headline_denominator": "reconstructed_structures_exact_legacy",
        "sun_secondary_denominator": "all_1000_refined_structures",
        "direct_counts": stage["direct_counts"],
        "direct_rates_all_1000": stage["direct_rates_all_attempts"],
        "direct_native_report_complete": stage["direct_native_report_complete"],
        "sun_counts": stage["sun_counts"],
        "sun_rates_all_1000": stage["sun_rates_all_attempts"],
        "sun_exact_legacy_reconstructed_denominator": stage[
            "sun_exact_legacy_reconstructed_denominator"
        ],
        "sun_native_summary_complete": stage["sun_native_summary_complete"],
        "sun_diagnostics": stage["sun_diagnostics"],
        "same_plan_retry": False,
        "stochastic_replacement": False,
        "repair_filter_rerank": False,
        "artifacts": {
            "selection_report": identity(
                root / "selected_body1000/selection_report.json"
            ),
            "selection_mapping": identity(
                root / "selected_body1000/selection_mapping.jsonl"
            ),
            "generation": identity(
                root / "post_model494/generation/generation.jsonl"
            ),
            "generation_report": identity(
                root / "post_model494/generation/generation_report.json"
            ),
            "stage_report": identity(
                root / "post_model494/evaluation/stage_report.json"
            ),
        },
    }
    write_json_exclusive(root / "native_repeat_report.json", output)
    with (root / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
