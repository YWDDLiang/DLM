#!/usr/bin/env python3
"""Write the post-sbatch E1 submission identity without shell interpolation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema": "h1_crplan_e1_submission_record_v1",
        "job_id": os.environ["JOB_ID"],
        "partition": "gpu",
        "source_manifest_sha256": os.environ["SOURCE_SHA"],
        "adapter_model_sha256": os.environ["ADAPTER_SHA"],
        "attempts_per_mode": 18,
        "scalar_reference_ordinals": [2, 11],
        "four_arm_512_submitted": False,
        "automatic_downstream": False,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
