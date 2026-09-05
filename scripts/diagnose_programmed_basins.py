#!/usr/bin/env python3
"""Describe relaxed structure clusters; never use them to select or weight paths."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def teacher_reweighting_diagnostic(groups):
    """Expose dilution and cross-condition tradeoffs without changing weights."""
    rows = []
    for group in groups:
        candidates = []
        for candidate in group["candidates"]:
            weight = float(candidate["weight"])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("invalid teacher weight")
            raw, terminal = candidate.get("raw_energy"), candidate.get("terminal_energy")
            usable = (candidate.get("verified") is True and raw is not None and terminal is not None
                      and math.isfinite(float(raw)) and math.isfinite(float(terminal)))
            if not usable:
                if weight != 0:
                    raise ValueError("unverified energy cannot carry teacher mass")
                continue
            candidates.append((float(raw) - float(terminal), float(terminal), weight))
        if not candidates:
            continue
        count = len(candidates)
        if not math.isclose(math.fsum(c[2] for c in candidates), 1., rel_tol=0., abs_tol=1e-8):
            raise ValueError("condition teacher weights do not sum to one")
        reference = 1. / count
        deltas = [math.fsum((c[2] - reference) * (c[axis] - candidates[0][axis]) for c in candidates)
                  for axis in (0, 1)]
        rows.append({"group_id": group["group_id"], "verified_paths": count,
                     "total_variation_from_uniform": .5 * math.fsum(abs(c[2] - reference) for c in candidates),
                     "minimum_weight_ratio_to_uniform": min(c[2] * count for c in candidates),
                     "maximum_weight_ratio_to_uniform": max(c[2] * count for c in candidates),
                     "energy_labels_vary": any(abs(c[axis] - candidates[0][axis]) > 1e-9 for c in candidates for axis in (0, 1)),
                     "delta_A_eV_atom": deltas[0], "delta_B_eV_atom": deltas[1]})
    count = len(rows)
    summary = {"requested_conditions": len(groups), "verified_conditions": count,
               "unverified_conditions": len(groups) - count,
               "single_verified_path_conditions": sum(r["verified_paths"] == 1 for r in rows),
               "multiple_verified_path_conditions": sum(r["verified_paths"] > 1 for r in rows),
               "varying_energy_label_conditions": sum(r["energy_labels_vary"] for r in rows),
               "reweighted_conditions": sum(r["total_variation_from_uniform"] > 1e-9 for r in rows),
               "mean_total_variation_from_uniform": math.fsum(r["total_variation_from_uniform"] for r in rows) / count if count else None,
               "both_objectives_improve_conditions": sum(r["delta_A_eV_atom"] < -1e-9 and r["delta_B_eV_atom"] < -1e-9 for r in rows),
               "either_objective_worsens_conditions": sum(r["delta_A_eV_atom"] > 1e-9 or r["delta_B_eV_atom"] > 1e-9 for r in rows),
               "mean_definition": "equal verified-condition mean; not a guarantee for each condition or the student",
               "B_difference": "centered terminal-energy change; same-composition hull cancels",
               "single_path_interpretation": "q=u=1 gives no within-condition energy reweighting; it still contributes verified-path likelihood",
               "used_for_teacher_weights_or_selection": False}
    for axis in ("A", "B"):
        deltas = [r[f"delta_{axis}_eV_atom"] for r in rows]
        gains = sorted((max(-d, 0.) for d in deltas), reverse=True)
        positive_total = math.fsum(gains)
        top = max(1, math.ceil(.1 * count))
        summary[axis] = {"mean_delta_eV_atom": math.fsum(deltas) / count if count else None,
                         "improved_conditions": sum(d < -1e-9 for d in deltas),
                         "worsened_conditions": sum(d > 1e-9 for d in deltas),
                         "unchanged_conditions": sum(abs(d) <= 1e-9 for d in deltas),
                         "largest_positive_gain_share": gains[0] / positive_total if positive_total else None,
                         "top_10_percent_conditions_positive_gain_share": math.fsum(gains[:top]) / positive_total if positive_total else None,
                         "gain_share_denominator": "sum of positive improvements, before subtracting worsening conditions"}
    return {"summary": summary, "conditions": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-json", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    teacher = json.loads(args.teacher_json.read_text())
    matcher = StructureMatcher(ltol=.2, stol=.3, angle_tol=5)
    groups = []
    for group in teacher["groups"]:
        representatives, members = [], []
        for candidate in group["candidates"]:
            if candidate.get("verified") is not True:
                continue
            structure = Structure.from_dict(candidate["final_structure"])
            index = next((i for i, representative in enumerate(representatives) if matcher.fit(structure, representative)), None)
            if index is None:
                index = len(representatives)
                representatives.append(structure)
            members.append({"trajectory_id": candidate["trajectory_id"], "cluster": index,
                            "terminal_energy_eV_atom": candidate["terminal_energy"]})
        groups.append({"group_id": group["group_id"], "verified_occurrences": len(members),
                       "structure_clusters": len(representatives), "members": members})
    report = {"groups": groups, "cluster_count_histogram": dict(Counter(g["structure_clusters"] for g in groups)),
              "conditions_with_multiple_clusters": sum(g["structure_clusters"] > 1 for g in groups),
              "matcher": {"ltol": .2, "stol": .3, "angle_tol": 5, "scale": True},
              "interpretation": "approximate relaxed structure equivalence, not proven identity of potential-energy basins",
              "used_for_teacher_weights_or_selection": False}
    report["teacher_reweighting"] = teacher_reweighting_diagnostic(teacher["groups"])
    for axis, key in (("A", "mean_delta_gap"), ("B", "mean_delta_terminal")):
        actual = report["teacher_reweighting"]["summary"][axis]["mean_delta_eV_atom"]
        expected = teacher["summary"][key]
        if actual is not None and not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError("diagnostic and solver condition means differ")
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle)
        handle.write("\n")
    print(json.dumps({**{k:v for k,v in report.items() if k not in ("groups", "teacher_reweighting")},
                      "teacher_reweighting": report["teacher_reweighting"]["summary"]}), flush=True)


if __name__ == "__main__":
    main()
