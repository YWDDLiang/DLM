#!/usr/bin/env python3
"""Reuse the frozen V4 body implementation for a 256-row retrained cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import protocol
from protocol import DENOMINATOR, canonical_sha256, ordered_rows, paired_seed, read_jsonl, require_file, sha256_file, sha256_text


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--base-script", type=Path, required=True)
    parser.add_argument("--base-script-sha256", required=True)
    parser.add_argument("--import-self-test-only", action="store_true")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded and not args.import_self_test_only:
        raise ValueError("base body arguments are required")
    base = require_file(args.base_script, args.base_script_sha256, "frozen V4 body script")
    sys.modules["protocol"] = protocol
    if str(base.parent) not in sys.path:
        sys.path.append(str(base.parent))
    spec = importlib.util.spec_from_file_location("_frozen_v4_body", base)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen V4 body script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def load_tasks(config: Mapping[str, Any], arm: str, repeat: int, cohort_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from crystal_dlm.r5_plan_state import build_body_prompt
        from safe_axis_schedule import h1a2_safe_axis_generation_schedule, require_safe_axis_schedule

        rows = ordered_rows(read_jsonl(cohort_path.resolve()), ordinal_field="cohort_ordinal")
        tasks: list[dict[str, Any]] = []
        invariants: list[dict[str, Any]] = []
        for ordinal, row in enumerate(rows):
            base_task = {
                "ordinal": ordinal,
                "sample_idx": ordinal,
                "planner": "P0",
                "body": arm,
                "arm": arm,
                "repeat": repeat,
                "pair_id": f"h1a2-retrained:{ordinal:04d}",
                "planner_attempt_id": f"{row['cohort_id']}:{int(row['planner_sample_idx']):04d}",
                "planner_candidate_ordinal": int(row["planner_sample_idx"]),
                "planner_sampling_seed": int(row["planner_sampling_seed"]),
                "body_noise_seed": paired_seed(repeat, ordinal, "body"),
                "refiner_noise_seed": paired_seed(repeat, ordinal, "refiner"),
                "eligible": row.get("body_eligible") is True,
                "reason": str(row.get("ineligible_reason") or ""),
                "cohort_id": str(row["cohort_id"]),
            }
            if not base_task["eligible"]:
                tasks.append({**base_task, "plan_state_sha256": row.get("plan_state_sha256")})
                continue
            plan = row.get("plan_state")
            if not isinstance(plan, Mapping):
                raise ValueError(f"missing parsed plan at ordinal {ordinal}")
            plan = dict(plan)
            if canonical_sha256(plan) != row.get("plan_state_sha256"):
                raise ValueError(f"plan identity changed at ordinal {ordinal}")
            prompt = build_body_prompt(plan)
            if prompt != row.get("body_prompt") or sha256_text(prompt) != row.get("body_prompt_sha256") or '"charge_bucket"' not in prompt:
                raise ValueError(f"body prompt changed at ordinal {ordinal}")
            schedule = h1a2_safe_axis_generation_schedule(plan)
            invariant = require_safe_axis_schedule(schedule, num_atoms=int(plan["N"]))
            if invariant.get("gate_passed") is not True or int(invariant.get("z_before_xy_count", -1)) != 0 or invariant.get("all_xy_precede_all_z") is not True or int(invariant.get("mixed_axis_coordinate_groups", -1)) != 0:
                raise ValueError(f"safe-axis invariant failed at ordinal {ordinal}")
            plan_sha = canonical_sha256(plan)
            schedule_sha = canonical_sha256(schedule)
            tasks.append({**base_task, "plan_state": plan, "plan_state_sha256": plan_sha, "body_prompt": prompt, "body_prompt_sha256": sha256_text(prompt), "schedule": schedule, "schedule_sha256": schedule_sha, "schedule_invariant": invariant, "schedule_invariant_sha256": canonical_sha256(invariant)})
            invariants.append({"ordinal": ordinal, "plan_state_sha256": plan_sha, "schedule_sha256": schedule_sha, "invariant_sha256": canonical_sha256(invariant)})
        report = {
            "schema": "h1a2_retrained_body_input_report_v1",
            "arm": arm,
            "cohort_index": repeat,
            "cohort_id": str(rows[0]["cohort_id"]),
            "attempts": DENOMINATOR,
            "parsed": sum(task["eligible"] for task in tasks),
            "planner_ineligible": sum(not task["eligible"] for task in tasks),
            "ineligible_ordinals": [int(task["ordinal"]) for task in tasks if not task["eligible"]],
            "cohort256_sha256": sha256_file(cohort_path),
            "seed_ledger_sha256": protocol.REFERENCE_LEDGER_SHA256,
            "same_body_refiner_seeds_across_cohorts": True,
            "schedule_invariants": invariants,
            "all_safe_axis_invariants_passed": True,
        }
        return tasks, report

    module._load_tasks = load_tasks
    if module._load_tasks is not load_tasks:
        raise RuntimeError("frozen body task-loader override did not bind")
    if args.import_self_test_only:
        if module.DENOMINATOR != DENOMINATOR:
            raise RuntimeError("frozen body denominator did not bind")
        try:
            module.validate_frozen_cohort_row({}, repeat=0, ordinal=0)
        except RuntimeError as exc:
            if "forbidden" not in str(exc):
                raise
        else:
            raise RuntimeError("compatibility sentinel did not fail closed")
        print(json.dumps({"import_self_test": "PASS", "denominator": DENOMINATOR, "base_script": str(base), "task_loader": "retrained_override"}, sort_keys=True))
        return
    sys.argv = [str(base), *forwarded]
    module.main()


if __name__ == "__main__":
    main()
