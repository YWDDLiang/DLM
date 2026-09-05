#!/usr/bin/env python3
"""Outcome-free, ineligible uniform weights for a bounded gradient check only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from crystal_dlm.programmed_path_data import read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if not (args.paths_jsonl.parent / "_SUCCESS").is_file():
        raise ValueError("sample accounting has not completed")
    paths = read_jsonl(args.paths_jsonl)
    if any(row.get("source_split") != "train" for row in paths):
        raise ValueError("engineering check uses train conditions only")
    if len({(r["checkpoint"], r["collection_round"]) for r in paths}) != 1:
        raise ValueError("mixed collection references")
    grouped = defaultdict(list)
    for row in paths:
        grouped[row["group_id"]].append(row)
    groups, usable = [], 0
    for key, rows in grouped.items():
        supported = sum(row["success"] for row in rows)
        usable += int(supported > 0)
        groups.append({"group_id": key, "candidates": [
            {"trajectory_id": r["trajectory_id"], "weight": 1 / supported if r["success"] else 0.,
             "verified": False, "raw_energy": None, "terminal_energy": None} for r in rows]})
    teacher = {"groups": groups,
        "summary": {"kind": "uniform_native_trace_engineering_check_only", "trainable_teacher": False,
                    "validated_groups": 0, "supervised_condition_count": usable,
                    "verified_energy_labels": 0, "outcomes_read": False, "diagnostic_only": True},
        "provenance": {"paths_jsonl": [str(args.paths_jsonl)], "checkpoint": paths[0]["checkpoint"],
                       "collection_round": paths[0]["collection_round"]}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "teacher.json").write_text(json.dumps(teacher) + "\n", encoding="utf-8")
    print(json.dumps(teacher["summary"]), flush=True)


if __name__ == "__main__":
    main()
