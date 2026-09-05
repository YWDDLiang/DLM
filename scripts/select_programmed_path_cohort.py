#!/usr/bin/env python3
"""Freeze a parser-only 1000-CIF prefix and retain the entire request ledger."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crystal_dlm.programmed_path_data import read_jsonl


def complete_requests(paths, planner_ledger):
    by_index = {int(r["sample_idx"]): r for r in paths}
    if len(by_index) != len(paths):
        raise ValueError("duplicate body request")
    if [int(r["sample_idx"]) for r in planner_ledger] != list(range(len(planner_ledger))):
        raise ValueError("Planner request ledger is not complete and ordered")
    if not set(by_index) <= set(range(len(planner_ledger))):
        raise ValueError("body lies outside the original request ledger")
    output = []
    for ledger in planner_ledger:
        index = int(ledger["sample_idx"])
        if ledger["planner_valid"]:
            if index not in by_index:
                raise ValueError("a valid Planner request has no accounted body attempt")
            row = dict(by_index[index])
        else:
            if index in by_index:
                raise ValueError("invalid Planner request unexpectedly has a body")
            row = {"trajectory_id": f"eval:{index}:0:0", "group_id": f"eval:{index}",
                   "sample_idx": index, "source_row_idx": index, "source_split": "evaluation",
                   "success": False, "body": "", "planner_failure": ledger.get("failure") or "planner_invalid"}
        row.update(evaluation_ordinal=index, source_request_ordinal=index, endpoint="native")
        output.append(row)
    return output


def select_prefix(records, *, target, parser):
    if target < 1:
        raise ValueError("target must be positive")
    selected, accounting = [], []
    cutoff = None
    for index, row in enumerate(records):
        entry = {"source_request_ordinal": index, "selected_ordinal": None}
        if cutoff is not None:
            entry["status"] = "not_examined_after_target"
        elif row.get("planner_failure"):
            entry["status"] = "planner_failure"
        else:
            try:
                parser(row)
            except Exception as error:
                entry.update(status="cif_parser_failure", error=f"{type(error).__name__}: {error}")
            else:
                selected_row = deepcopy(row)
                selected_row.update(evaluation_ordinal=len(selected), parser_only_selection=True)
                selected.append(selected_row)
                entry.update(status="selected_parseable_cif", selected_ordinal=len(selected) - 1)
                if len(selected) == target:
                    cutoff = index
        accounting.append(entry)
    if len(selected) != target:
        raise ValueError(f"only {len(selected)} parseable CIFs in the fixed source stream; requested {target}")
    return selected, accounting, {"source_requests_available": len(records), "source_requests_examined": cutoff + 1,
        "selected_parseable_cifs": len(selected), "discarded_before_target": cutoff + 1 - len(selected),
        "unexamined_after_target": len(records) - cutoff - 1, "selection_basis": "CIF_parser_only_in_source_order",
        "energy_or_stability_selection": False,
        "selected_source_ordinals": [r["source_request_ordinal"] for r in selected]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths-jsonl", type=Path, required=True)
    p.add_argument("--planner-ledger", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target", type=int, default=1000)
    args = p.parse_args()
    from pymatgen.core import Structure
    from crystal_dlm.dynamic_crystal import arrays_to_structure, parse_dynamic_answer
    def parser(row):
        # Deliberately independent of runtime success, graph creation, or energy.
        structure = arrays_to_structure(parse_dynamic_answer(row["body"], strict=True))
        parsed = Structure.from_str(structure.to(fmt="cif"), fmt="cif")
        if parsed.num_sites != structure.num_sites or parsed.composition != structure.composition:
            raise ValueError("CIF roundtrip changed exact composition")
    rows = complete_requests(read_jsonl(args.paths_jsonl), read_jsonl(args.planner_ledger))
    if any(r.get("source_split") != "evaluation" for r in rows):
        raise ValueError("training paths cannot become a main evaluation cohort")
    selected, accounting, report = select_prefix(rows, target=args.target, parser=parser)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, records in (("all_requests.jsonl", rows), ("selected_paths.jsonl", selected), ("selection_ledger.jsonl", accounting)):
        with (args.output_dir / name).open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "SELECTION_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps({k:v for k,v in report.items() if k != "selected_source_ordinals"}), flush=True)


if __name__ == "__main__":
    main()
