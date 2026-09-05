#!/usr/bin/env python3
"""One empirical A/B teacher for a complete, fixed train-condition pool."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.basin_path_objective import solve_basin_path_teacher
from crystal_dlm.programmed_path_data import read_jsonl, trace_summary
from crystal_dlm.programmed_path_training import join_terminal_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths-jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--labels-jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--expected-conditions", type=int, default=1024)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--diagnostic-only", action="store_true")
    args = p.parse_args()
    for path in [*args.paths_jsonl, *args.labels_jsonl]:
        if not (path.parent / "_SUCCESS").is_file():
            raise ValueError(f"input accounting has not completed: {path}")
    if not args.diagnostic_only and args.expected_conditions != 1024:
        raise ValueError("formal teacher requires the preregistered 1024 train conditions")
    paths = [r for path in args.paths_jsonl for r in read_jsonl(path)]
    labels = [r for path in args.labels_jsonl for r in read_jsonl(path)]
    groups = join_terminal_labels(paths, labels, expected_conditions=args.expected_conditions)
    teacher = solve_basin_path_teacher(groups)
    summary = teacher["summary"]
    summary["label_statuses"] = dict(Counter(r["status"] for r in labels))
    summary["verified_per_condition"] = dict(Counter(sum(c["verified"] is True for c in g["candidates"]) for g in groups))
    execution = [trace_summary(r["trace"]) for r in paths]
    summary["cooperative_attempted_paths"] = sum(bool(r["transactions_by_phase"].get("cooperative", 0)) for r in execution)
    summary["cooperative_accepted_paths"] = sum(bool(r["cooperative_accepted"]) for r in execution)
    summary["cooperative_changed_paths"] = sum(r["committed_changed_scalars_by_phase"].get("cooperative", 0) > 0 for r in execution)
    summary["successful_paths"] = sum(r["success"] for r in paths)
    summary["diagnostic_only"] = args.diagnostic_only
    summary["trainable_teacher"] = bool(not args.diagnostic_only and summary["solver_status"] == "optimal"
        and summary["rho_max"] > 0 and summary["primal_residual"] <= 1e-6)
    spans = []
    for group in groups:
        values = [c["terminal_energy"] for c in group["candidates"] if c["verified"] is True]
        if len(values) >= 2:
            spans.append(max(values) - min(values))
    summary["conditions_with_multiple_verified_paths"] = len(spans)
    summary["mean_within_condition_terminal_span_eV_atom"] = sum(spans) / len(spans) if spans else None
    teacher["provenance"] = {"paths_jsonl": [str(p) for p in args.paths_jsonl],
                             "labels_jsonl": [str(p) for p in args.labels_jsonl],
                             "checkpoint": paths[0]["checkpoint"], "collection_round": paths[0]["collection_round"],
                             "objective": "separate mean improvements in e0-eR and centered eR; no cross-composition ranking"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "teacher.json").write_text(json.dumps(teacher, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "TEACHER_FINAL.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()  # Completed solver/accounting, not a positive-gain claim.
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
