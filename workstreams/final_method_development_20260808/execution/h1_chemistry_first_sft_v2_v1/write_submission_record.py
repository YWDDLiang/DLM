#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required submission environment {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("engineering_smoke", "planner64", "planner256"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common = {
        "schema": "h1_chemistry_first_submission_record_v1",
        "identity": "h1_chemistry_first_sft_v2_v1",
        "stage": args.stage,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "source_inventory_sha256": required("SOURCE_INVENTORY_SHA"),
        "archive_sha256": required("ARCHIVE_SHA"),
        "ledger64_sha256": required("LEDGER64_SHA"),
        "ledger256_sha256": required("LEDGER256_SHA"),
        "preflight_report_sha256": required("PREFLIGHT_SHA"),
        "sinfo_snapshot_sha256": required("SINFO_SHA"),
        "squeue_snapshot_sha256": required("SQUEUE_SHA"),
        "legacy_python": required("LEGACY_PYTHON"),
        "smact4_python": required("SMACT4_PYTHON"),
        "automatic_downstream": False,
        "automatic_rl": False,
    }
    if args.stage == "engineering_smoke":
        common["jobs"] = {
            "data": required("DATA_JOB_ID"),
            "smoke": required("SMOKE_JOB_ID"),
        }
        common["selection_role"] = "engineering_only_no_scientific_sampling"
        common["candidate_list"] = ["sft_v2", "sft_v2_c"]
    elif args.stage == "planner64":
        common["jobs"] = {
            "train": required("TRAIN_JOB_ID"),
            "planner64": required("PLANNER_JOB_ID"),
            "assemble64": required("ASSEMBLY_JOB_ID"),
        }
        common["prior_engineering_submission_sha256"] = required(
            "PRIOR_ENGINEERING_SUBMISSION_SHA"
        )
        common["candidate_list"] = ["sft_v2", "sft_v2_c"]
    else:
        common["jobs"] = {
            "planner256": required("PLANNER_JOB_ID"),
            "assemble256": required("ASSEMBLY_JOB_ID"),
        }
        common["candidate_list"] = required("EXPECTED_CANDIDATES").split(",")
        common["prior64_summary_sha256"] = required("PRIOR64_SUMMARY_SHA")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(common, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
