#!/usr/bin/env python3
"""Describe relaxed structure clusters; never use them to select or weight paths."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


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
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle)
        handle.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k != "groups"}), flush=True)


if __name__ == "__main__":
    main()
