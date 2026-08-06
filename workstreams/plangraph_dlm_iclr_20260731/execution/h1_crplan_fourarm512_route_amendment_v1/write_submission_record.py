#!/usr/bin/env python3
"""Write immutable Slurm submission identities for four-arm 512."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("array_submitted", "dag_submitted"),
        required=True,
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "h1_crplan_fourarm512_submission_record_v1",
        "identity": "h1_crplan_fourarm512_route_amendment_v1",
        "stage": args.stage,
        "array_job_id": os.environ["ARRAY_JOB_ID"],
        "assembly_job_id": os.environ.get("ASSEMBLY_JOB_ID"),
        "array_specification": "0-3%2",
        "array_partition": "gpu",
        "assembly_partition": "normal",
        "assembly_dependency": "afterany",
        "source_manifest_sha256": os.environ["SOURCE_SHA"],
        "science_ledger_sha256": os.environ["LEDGER_SHA"],
        "adapter_model_sha256": os.environ["ADAPTER_SHA"],
        "sinfo_snapshot_sha256": os.environ["SINFO_SHA"],
        "attempts_per_arm": 512,
        "modes": [
            "off",
            "grammar_only",
            "terminal_only",
            "full_prefix",
        ],
        "v4_state_gate_reused": False,
        "body_or_evaluator_downstream_submitted": False,
        "automatic_downstream": False,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
