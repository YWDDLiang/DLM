#!/usr/bin/env python3
"""Compare registered and optimized recovery artifacts attempt by attempt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


KEY_FIELDS = ("material_id", "ordinal", "pair_id", "paired_seed")
EXACT_FIELDS = (
    "status",
    "applicable",
    "reason",
    "source_topology_hash",
    "corrupt_topology_hash",
    "recovered_topology_hash",
    "exact_full_protostructure_recovery",
    "space_group_recovery",
    "topology_edit_distance_before",
    "topology_edit_distance_after",
    "orbit_count_error",
    "corruption_trace",
    "recovery_trace",
    "mechanism",
    "calls",
)


def _read(path: Path) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    records: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema") != "wqcodiff_recovery_attempt_v1":
                continue
            key = tuple(record.get(field) for field in KEY_FIELDS)
            if key in records:
                raise ValueError(f"duplicate recovery key at {path}:{line_number}: {key}")
            records[key] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordinate-atol", type=float, default=1.0e-7)
    args = parser.parse_args()
    registered = _read(args.registered)
    optimized = _read(args.optimized)
    keys_equal = registered.keys() == optimized.keys()
    mismatches: list[dict[str, Any]] = []
    for key in sorted(registered.keys() & optimized.keys(), key=repr):
        old = registered[key]
        new = optimized[key]
        for field in EXACT_FIELDS:
            if old.get(field) != new.get(field):
                mismatches.append(
                    {"key": key, "field": field, "registered": old.get(field), "optimized": new.get(field)}
                )
                break
        else:
            old_error = old.get("tangent_coordinate_error")
            new_error = new.get("tangent_coordinate_error")
            if old_error is None or new_error is None:
                coordinate_equal = old_error is new_error
            else:
                coordinate_equal = math.isclose(
                    float(old_error),
                    float(new_error),
                    rel_tol=0.0,
                    abs_tol=args.coordinate_atol,
                )
            if not coordinate_equal:
                mismatches.append(
                    {
                        "key": key,
                        "field": "tangent_coordinate_error",
                        "registered": old_error,
                        "optimized": new_error,
                    }
                )
        if len(mismatches) >= 25:
            break
    report = {
        "schema": "wqcodiff_recovery_equivalence_v1",
        "registered": str(args.registered.resolve()),
        "optimized": str(args.optimized.resolve()),
        "registered_attempts": len(registered),
        "optimized_attempts": len(optimized),
        "key_sets_equal": keys_equal,
        "coordinate_atol": args.coordinate_atol,
        "mismatch_count_capped": len(mismatches),
        "mismatches": mismatches,
        "passed": keys_equal and not mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
