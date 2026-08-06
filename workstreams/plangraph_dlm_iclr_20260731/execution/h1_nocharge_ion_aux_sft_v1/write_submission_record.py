#!/usr/bin/env python3
"""Write one immutable Slurm DAG submission record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("initial64", "planner256"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    common = {
        "schema": "h1_nocharge_ion_aux_submission_record_v1",
        "identity": "h1_nocharge_ion_aux_sft_v1",
        "stage": args.stage,
        "source_inventory_sha256": os.environ["SOURCE_INVENTORY_SHA"],
        "archive_sha256": os.environ["ARCHIVE_SHA"],
        "ledger64_sha256": os.environ["LEDGER64_SHA"],
        "ledger256_sha256": os.environ["LEDGER256_SHA"],
        "legacy_python": os.environ["LEGACY_PYTHON"],
        "smact4_python": os.environ["SMACT4_PYTHON"],
        "sinfo_snapshot_sha256": os.environ["SINFO_SHA"],
        "partitions": ["normal", "gpu"],
        "automatic_rl": False,
        "automatic_downstream": False,
        "dlm_model_changed": False,
    }
    if args.stage == "initial64":
        common["jobs"] = {
            "data": os.environ["DATA_JOB_ID"],
            "smoke_array": os.environ["SMOKE_JOB_ID"],
            "train_array": os.environ["TRAIN_JOB_ID"],
            "planner64_array": os.environ["PLANNER_JOB_ID"],
            "assemble64": os.environ["ASSEMBLY_JOB_ID"],
        }
        common["dependencies"] = {
            "smoke_array": "afterok:data",
            "train_array": "afterok:smoke_array",
            "planner64_array": "afterok:train_array",
            "assemble64": "afterany:planner64_array",
        }
        common["arrays"] = {
            "smoke": "0-1%2 (c0,c1)",
            "train": "0-1%2 (c0,c1)",
            "planner64": "0-2%2 (p0,c0,c1)",
        }
    else:
        common["jobs"] = {
            "planner256_array": os.environ["PLANNER_JOB_ID"],
            "assemble256": os.environ["ASSEMBLY_JOB_ID"],
        }
        common["dependencies"] = {"assemble256": "afterany:planner256_array"}
        common["arrays"] = {"planner256": "0-2%2 (p0,c0,c1)"}
        common["prior64_terminal_sha256"] = os.environ["PRIOR64_TERMINAL_SHA"]
    args.output.write_text(
        json.dumps(common, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
