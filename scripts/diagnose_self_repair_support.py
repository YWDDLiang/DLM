#!/usr/bin/env python3
"""Training-only candidate support, never an evaluation or trained-policy result."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from crystal_dlm.programmed_path_data import read_jsonl, trace_terminal_body
from crystal_dlm.self_repair_data import repair_net_change
from crystal_dlm.terminal_energy_consistency import TERMINAL_VERIFICATION_PROTOCOL


def convex_support(points, tolerance):
    """2D convex-hull opportunity only; no KL or policy-realizability claim."""
    candidates = list(points)
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            for da, db in ((a[0] - tolerance, b[0] - tolerance),
                           (a[1] - tolerance, b[1] - tolerance),
                           (a[0] - a[1], b[0] - b[1])):
                if abs(da - db) < 1e-15:
                    continue
                weight = -db / (da - db)
                if 0 <= weight <= 1:
                    candidates.append(tuple(weight * x + (1 - weight) * y for x, y in zip(a, b)))
    if not candidates:
        return {"nonregression_feasible_without_KL": False, "improvement_feasible_without_KL": False,
                "max_common_gain_without_KL": None}
    good = lambda p: max(p) <= tolerance + 1e-12 and min(p) < -tolerance
    return {"nonregression_feasible_without_KL": any(max(p) <= tolerance + 1e-12 for p in candidates),
            "improvement_feasible_without_KL": any(good(p) for p in candidates),
            "max_common_gain_without_KL": -min(max(p) for p in candidates)}


def summarize(paths, labels, parent_labels, *, expected_roots, candidates=4, tolerance=.001):
    child = {r["trajectory_id"]: r for r in labels}
    parents = {r["trajectory_id"]: r for r in parent_labels}
    if len(child) != len(labels) or len(parents) != len(parent_labels):
        raise ValueError("duplicate label occurrence")
    if len(paths) != expected_roots * candidates or set(child) != {p["trajectory_id"] for p in paths}:
        raise ValueError("repair request/label accounting is incomplete")
    groups = defaultdict(list)
    cells = Counter()
    total_unchanged = 0
    for p in paths:
        if p.get("source_split") != "train" or p.get("path_mode") != "self_repair_support_check":
            raise ValueError("only recorded train self-repair paths are accepted")
        parent_id = p["repair_parent_trajectory_id"]
        if parent_id not in parents:
            raise ValueError("missing root label; unknown energy cannot be imputed")
        if p["trace"]["initial_body"] != p["repair_initial_body"]:
            raise ValueError("repair trace did not begin at its recorded complete root")
        if trace_terminal_body(p["trace"]) != p["final_body_token_ids"]:
            raise ValueError("repair trace endpoint differs")
        for event in p["trace"]["events"]:
            if event.get("phase") == "construct":
                raise ValueError("repair trace accidentally contains new construction")
            if event["op"] == "draw" and not math.isfinite(event["log_probability"]):
                raise ValueError("repair decision has nonfinite probability")
        parent, label = parents[parent_id], child[p["trajectory_id"]]
        if label["group_id"] != p["group_id"]:
            raise ValueError("repair label condition differs")
        pv, cv = bool(parent["verified"]), bool(label["verified"])
        cells[f"parent_{int(pv)}_child_{int(cv)}"] += 1
        change = repair_net_change(p["repair_initial_body"], p["final_body_token_ids"])
        total_unchanged += change["final_equals_initial"]
        item = {"trajectory_id": p["trajectory_id"], "candidate_index": p["candidate_index"],
                "success": bool(p["success"]), "parent_verified": pv, "child_verified": cv,
                "parent_status": parent["status"], "child_status": label["status"],
                "sampling_seed": p["sampling_seed"], **change,
                "delta_A": None, "delta_B": None, "delta_raw_energy": None,
                "jointly_better_with_tolerance": False}
        if pv and cv:
            values = [parent[k] for k in ("raw_energy", "terminal_energy", "gap")]
            values += [label[k] for k in ("raw_energy", "terminal_energy", "gap")]
            if not all(v is not None and math.isfinite(v) for v in values):
                raise ValueError("verified pair is missing finite energy data")
            da = label["gap"] - parent["gap"]
            db = label["terminal_energy"] - parent["terminal_energy"]
            de = label["raw_energy"] - parent["raw_energy"]
            if abs(de - da - db) > 1e-8:
                raise ValueError("delta energy != delta A + delta B")
            item.update(delta_A=da, delta_B=db, delta_raw_energy=de,
                        jointly_better_with_tolerance=max(da, db) <= tolerance and min(da, db) < -tolerance)
        groups[parent_id].append(item)
    if len(groups) != expected_roots:
        raise ValueError("root denominator differs")
    rows = []
    for parent_id, items in groups.items():
        if sorted(i["candidate_index"] for i in items) != list(range(candidates)):
            raise ValueError("missing or duplicate repair candidate")
        paired = [i for i in items if i["delta_A"] is not None]
        rows.append({"parent_trajectory_id": parent_id, "requests": len(items),
                     "comparable_candidates": len(paired),
                     "any_single_joint_improvement": any(i["jointly_better_with_tolerance"] for i in items),
                     "conditional_uniform_delta_A": statistics.fmean(i["delta_A"] for i in paired) if paired else None,
                     "conditional_uniform_delta_B": statistics.fmean(i["delta_B"] for i in paired) if paired else None,
                     **convex_support([(i["delta_A"], i["delta_B"]) for i in paired], tolerance),
                     "candidates": items})
    comparable = [r for r in rows if r["comparable_candidates"]]
    complete = [r for r in rows if r["comparable_candidates"] == candidates]
    def means(subset):
        return {axis: statistics.fmean(r[f"conditional_uniform_delta_{axis}"] for r in subset) if subset else None
                for axis in ("A", "B")}
    summary = {"role": "train_candidate_support_only", "trained_policy_result": False,
               "roots": expected_roots, "requests": len(paths), "candidates_per_root": candidates,
               "successful_requests": sum(bool(p["success"]) for p in paths),
               "unchanged_endpoint_requests": total_unchanged, "verification_four_cells": dict(cells),
               "comparable_roots": len(comparable), "complete_paired_roots": len(complete),
               "equal_root_conditional_uniform_mean_delta": means(comparable),
               "equal_root_complete_K_mean_delta": means(complete),
               "roots_with_single_joint_improvement": sum(r["any_single_joint_improvement"] for r in rows),
               "jointly_better_requests": sum(i["jointly_better_with_tolerance"] for r in rows for i in r["candidates"]),
               "roots_convex_improvement_feasible_without_KL": sum(r["improvement_feasible_without_KL"] for r in rows),
               "comparison_tolerance_eV_atom": tolerance,
               "criterion": "neither cost increases by > tolerance and at least one decreases by > tolerance",
               "scope": "conditional verified comparisons; no missing-energy imputation; oracle support is not deployment performance"}
    return {"summary": summary, "roots": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths-jsonl", type=Path, required=True)
    p.add_argument("--labels-jsonl", type=Path, required=True)
    p.add_argument("--parent-labels-jsonl", type=Path, required=True)
    p.add_argument("--expected-roots", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    for source in (args.paths_jsonl, args.labels_jsonl, args.parent_labels_jsonl):
        if not (source.parent / "_SUCCESS").is_file():
            raise ValueError("support check input is incomplete")
    reports = [json.loads((source.parent / "LABEL_FINAL.json").read_text())
               for source in (args.labels_jsonl, args.parent_labels_jsonl)]
    if any(r.get("purpose") != "train" or r.get("verification_protocol") != TERMINAL_VERIFICATION_PROTOCOL for r in reports):
        raise ValueError("support check needs train labels with uniform terminal verification")
    if reports[0]["protocol"] != reports[1]["protocol"]:
        raise ValueError("baseline and repairs have different physical evaluation protocols")
    sample = json.loads((args.paths_jsonl.parent / "SAMPLE_FINAL.json").read_text())
    if sample.get("path_mode") != "self_repair_support_check" or not sample.get("diagnostic_only"):
        raise ValueError("not a self-repair support-check collection")
    report = summarize(read_jsonl(args.paths_jsonl), read_jsonl(args.labels_jsonl),
                       read_jsonl(args.parent_labels_jsonl), expected_roots=args.expected_roots)
    report["provenance"] = {k: str(v) for k, v in vars(args).items()}
    report["collection"] = sample
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "SUPPORT_FINAL.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
