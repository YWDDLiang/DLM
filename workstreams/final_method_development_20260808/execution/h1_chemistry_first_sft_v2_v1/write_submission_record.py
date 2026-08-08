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
        choices=(
            "legacy_snapshot",
            "identity_probe",
            "engineering_smoke",
            "optimizer_smoke",
            "planner64_generation",
            "planner64_assembly",
            "planner256_generation",
            "planner256_assembly",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common = {
        "schema": "h1_chemistry_first_submission_record_v1",
        "identity": "h1_chemistry_first_sft_v2_smact_split_v2",
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
        "a800_smact_version": "3.1.0",
        "a800_smact4_execution": False,
        "smact4_execution_location": "local_windows_only",
        "automatic_downstream": False,
        "automatic_rl": False,
    }
    if args.stage == "legacy_snapshot":
        common["jobs"] = {"legacy_snapshot": required("SNAPSHOT_JOB_ID")}
        common["selection_role"] = "immutable_source_snapshot_only"
        common["candidate_list"] = []
    elif args.stage == "identity_probe":
        common["jobs"] = {"identity_probe": required("IDENTITY_PROBE_JOB_ID")}
        common["parent_v5_failure_report_sha256"] = required(
            "PARENT_V5_FAILURE_SHA"
        )
        common["selection_role"] = "no_forward_no_optimizer_engineering_identity_probe"
        common["candidate_list"] = []
    elif args.stage == "optimizer_smoke":
        common["jobs"] = {
            "optimizer_smoke": required("OPTIMIZER_SMOKE_JOB_ID"),
        }
        common["parent_v7_terminal_record_sha256"] = required(
            "PARENT_V7_TERMINAL_SHA"
        )
        common["selection_role"] = (
            "engineering_two_optimizer_updates_no_scientific_checkpoint"
        )
        common["optimizer_updates"] = 2
        common["full_training_total_updates"] = 4505
        common["full_training_warmup_steps"] = 135
        common["candidate_list"] = ["sft_v2", "sft_v2_c"]
    elif args.stage == "engineering_smoke":
        common["jobs"] = {
            "data": required("DATA_JOB_ID"),
            "smoke": required("SMOKE_JOB_ID"),
        }
        common["prior_snapshot_submission_sha256"] = required(
            "PRIOR_SNAPSHOT_SUBMISSION_SHA"
        )
        common["local_smact4_witness_root"] = required(
            "LOCAL_SMACT4_WITNESS_ROOT"
        )
        common["local_smact4_witness_manifest_sha256"] = required(
            "LOCAL_SMACT4_WITNESS_MANIFEST_SHA"
        )
        common["prior_identity_probe_submission_sha256"] = required(
            "PRIOR_IDENTITY_PROBE_SUBMISSION_SHA"
        )
        common["identity_probe_report_sha256"] = required(
            "IDENTITY_PROBE_REPORT_SHA"
        )
        common["identity_probe_gate_sha256"] = required(
            "IDENTITY_PROBE_GATE_SHA"
        )
        common["identity_probe_admission_sha256"] = required(
            "IDENTITY_PROBE_ADMISSION_SHA"
        )
        common["selection_role"] = "engineering_only_no_scientific_sampling"
        common["candidate_list"] = ["sft_v2", "sft_v2_c"]
    elif args.stage == "planner64_generation":
        common["jobs"] = {
            "train": required("TRAIN_JOB_ID"),
            "planner64": required("PLANNER_JOB_ID"),
        }
        common["prior_engineering_submission_sha256"] = required(
            "PRIOR_ENGINEERING_SUBMISSION_SHA"
        )
        if os.environ.get("OPTIMIZER_SMOKE_ADMISSION_SFT_V2_SHA"):
            common["optimizer_smoke_admission_sha256"] = {
                "sft_v2": required("OPTIMIZER_SMOKE_ADMISSION_SFT_V2_SHA"),
                "sft_v2_c": required("OPTIMIZER_SMOKE_ADMISSION_SFT_V2_C_SHA"),
            }
        else:
            common["identity_probe_admission_sha256"] = required(
                "IDENTITY_PROBE_ADMISSION_SHA"
            )
            common["smoke_admission_sha256"] = {
                "sft_v2": required("SMOKE_ADMISSION_SFT_V2_SHA"),
                "sft_v2_c": required("SMOKE_ADMISSION_SFT_V2_C_SHA"),
            }
        common["candidate_list"] = ["sft_v2", "sft_v2_c"]
    elif args.stage == "planner64_assembly":
        common["jobs"] = {"assemble64": required("ASSEMBLY_JOB_ID")}
        common["generation_sacct_sha256"] = required("GENERATION_SACCT_SHA")
        common["prior_generation_submission_sha256"] = required(
            "PRIOR_GENERATION_SUBMISSION_SHA"
        )
        common["local_smact4_audit_root"] = required("LOCAL_SMACT4_AUDIT_ROOT")
        common["local_smact4_audit_manifest_sha256"] = required(
            "LOCAL_SMACT4_AUDIT_MANIFEST_SHA"
        )
        common["audited_arms"] = required("AUDITED_ARMS").split(",")
        common["candidate_list"] = ["sft_v2", "sft_v2_c"]
    elif args.stage == "planner256_generation":
        common["jobs"] = {
            "planner256": required("PLANNER_JOB_ID"),
        }
        common["candidate_list"] = required("EXPECTED_CANDIDATES").split(",")
        common["prior64_summary_sha256"] = required("PRIOR64_SUMMARY_SHA")
        common["prior64_assembly_submission_sha256"] = required(
            "PRIOR64_ASSEMBLY_SUBMISSION_SHA"
        )
    else:
        common["jobs"] = {"assemble256": required("ASSEMBLY_JOB_ID")}
        common["generation_sacct_sha256"] = required("GENERATION_SACCT_SHA")
        common["candidate_list"] = required("EXPECTED_CANDIDATES").split(",")
        common["prior_generation_submission_sha256"] = required(
            "PRIOR_GENERATION_SUBMISSION_SHA"
        )
        common["local_smact4_audit_root"] = required("LOCAL_SMACT4_AUDIT_ROOT")
        common["local_smact4_audit_manifest_sha256"] = required(
            "LOCAL_SMACT4_AUDIT_MANIFEST_SHA"
        )
        common["audited_arms"] = required("AUDITED_ARMS").split(",")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(common, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
