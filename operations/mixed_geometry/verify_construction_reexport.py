#!/usr/bin/env python3
"""Bind newly exported V2 construction graphs to the already evaluated endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path):
    values = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    mapping = {row["trajectory_id"]: row for row in values}
    if len(values) != 256 or len(mapping) != 256:
        raise ValueError("the construction comparison must retain all 256 unique requests")
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-paths", type=Path, required=True)
    p.add_argument("--original-labels", type=Path, required=True)
    p.add_argument("--exported-paths", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    original, labels, exported = map(rows, (args.original_paths, args.original_labels, args.exported_paths))
    if original.keys() != labels.keys() or original.keys() != exported.keys():
        raise ValueError("construction paths, labels and graph export have different request identities")
    fields = ("sample_idx", "group_id", "plan_state", "species_program", "species_program_source",
              "body", "final_body_token_ids", "sampling_seed", "candidate_index", "collection_round",
              "source_split", "prompt", "trace", "trajectory_id", "evaluation_ordinal")
    count = 0
    for key, old in original.items():
        new, label = exported[key], labels[key]
        if any(old.get(field) != new.get(field) for field in fields):
            raise ValueError(f"construction source identity changed: {key}")
        if old.get("success") != new.get("success") or old.get("parseable") != new.get("parseable"):
            raise ValueError(f"construction failure status changed: {key}")
        if old.get("structure") != new.get("structure") or (new.get("parseable") and old.get("structure") != new.get("native_structure")):
            raise ValueError(f"construction original float structure changed: {key}")
        geometry = json.dumps(new["structure"], sort_keys=True) if new.get("structure") is not None else str(new["body"])
        expected_key = hashlib.sha256(geometry.encode()).hexdigest() if new["success"] else key
        if label["endpoint_cache_key"] != expected_key:
            raise ValueError(f"saved raw label refers to another physical endpoint: {key}")
        count += bool(new["parseable"])
    report = {"status": "PASS", "requests": 256, "parseable": count,
              "structure_identity": "exact original/native dictionaries and label endpoint hash",
              "original_labels_reused": True, "new_raw_energy_calls": 0, "outcome_selection": False,
              "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (args.original_paths, args.original_labels, args.exported_paths)}}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
