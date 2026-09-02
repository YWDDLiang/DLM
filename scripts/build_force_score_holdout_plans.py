#!/usr/bin/env python3
"""Freeze the 128 paired-source holdout prompts for matched raw generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-sft-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    holdout = read_jsonl(args.holdout_jsonl.resolve())
    teacher = {
        int(row["source_row_idx"]): row
        for row in read_jsonl(args.teacher_sft_jsonl.resolve())
    }
    if len(holdout) != 128:
        raise ValueError("force microstudent holdout must contain 128 rows")
    plans = []
    for sample_idx, row in enumerate(holdout):
        source_index = int(row["source_row_idx"])
        source = teacher[source_index]
        plans.append(
            {
                "schema": "projected_force_holdout_plan_v1",
                "sample_idx": sample_idx,
                "source_sample_idx": int(row["preflight_row_index"]),
                "base_index": int(row["base_index"]),
                "source_row_idx": source_index,
                "prompt": str(row["prompt"]),
                "plan_state": dict(source["plan_state"]),
                "reduced_composition_identity": str(
                    source["reduced_composition_identity"]
                ),
            }
        )
    if len({row["base_index"] for row in plans}) != 16:
        raise ValueError("holdout base-structure count changed")
    if [row["sample_idx"] for row in plans] != list(range(128)):
        raise ValueError("holdout sample order changed")
    output.mkdir(parents=True)
    (output / "plans.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in plans)
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "projected_force_holdout_plans_v1",
                "status": "complete",
                "rows": 128,
                "base_structures": 16,
                "trajectories_per_base": 8,
                "planner_resampled": False,
                "outcomes_read": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
