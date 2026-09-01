#!/usr/bin/env python3
"""Freeze first-1000 and remainder Plan splits using MP-reference availability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "official_available_plan_splits_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chemsys_from_plan(row: Mapping[str, Any]) -> str:
    state = row.get("plan_state")
    if not isinstance(state, Mapping):
        raise ValueError("Plan row lacks plan_state")
    elements = state.get("elements")
    if not isinstance(elements, list) or not elements:
        raise ValueError("Plan row lacks elements")
    return "-".join(sorted({str(value) for value in elements}))


def freeze(
    records,
    plans,
    *,
    resolved: set[str],
    unresolved: set[str],
    expected_requested: int,
    primary_count: int,
):
    if len(records) != expected_requested:
        raise ValueError("Planner requested denominator changed")
    if [int(row["sample_idx"]) for row in records] != list(range(expected_requested)):
        raise ValueError("Planner record ordering changed")
    plans_by_idx = {int(row["sample_idx"]): row for row in plans}
    if len(plans_by_idx) != len(plans):
        raise ValueError("duplicate successful Plan sample_idx")

    accounting = []
    eligible = []
    for source_idx, record in enumerate(records):
        plan = plans_by_idx.get(source_idx)
        if record.get("comp_valid") is not True or plan is None:
            reason = "planner_failure"
            chemsys = None
        else:
            chemsys = chemsys_from_plan(plan)
            if chemsys in resolved:
                reason = "official_reference_available"
                eligible.append((source_idx, chemsys, plan))
            elif chemsys in unresolved:
                reason = "official_reference_unknown"
            else:
                reason = "official_cache_omission"
        accounting.append(
            {
                "schema": SCHEMA,
                "source_sample_idx": source_idx,
                "chemsys": chemsys,
                "eligible": reason == "official_reference_available",
                "reason": reason,
            }
        )
    if len(eligible) < primary_count:
        raise ValueError("fewer official-available Plans than primary_count")

    def materialize(values):
        output = []
        for execution_idx, (source_idx, chemsys, plan) in enumerate(values):
            row = dict(plan)
            row["source_sample_idx"] = source_idx
            row["sample_idx"] = execution_idx
            row["execution_sample_idx"] = execution_idx
            row["official_reference_chemsys"] = chemsys
            row["official_reference_available"] = True
            output.append(row)
        return output

    return (
        accounting,
        materialize(eligible[:primary_count]),
        materialize(eligible[primary_count:]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-records", type=Path, required=True)
    parser.add_argument("--planner-plans", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-requested", type=int, default=1200)
    parser.add_argument("--primary-count", type=int, default=1000)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    cache = args.official_run.resolve() / "official_mp_cache"
    if not (cache / "completion_SUCCESS").is_file():
        raise RuntimeError("official query is incomplete")
    resolved_rows = read_jsonl(cache / "official_slim_cache.jsonl")
    unresolved_rows = read_jsonl(cache / "unresolved_chemsys.jsonl")
    resolved = {str(row["chemsys"]) for row in resolved_rows}
    unresolved = {str(row["chemsys"]) for row in unresolved_rows}
    if resolved & unresolved:
        raise ValueError("official resolved/unresolved sets overlap")

    records = read_jsonl(args.planner_records.resolve())
    plans = read_jsonl(args.planner_plans.resolve())
    accounting, primary, remainder = freeze(
        records,
        plans,
        resolved=resolved,
        unresolved=unresolved,
        expected_requested=args.expected_requested,
        primary_count=args.primary_count,
    )
    output.mkdir(parents=True)
    (output / "main1000").mkdir()
    (output / "remainder").mkdir()
    write_jsonl(output / "all_requested_accounting.jsonl", accounting)
    write_jsonl(output / "main1000/plans_for_dlm.jsonl", primary)
    write_jsonl(output / "remainder/plans_for_dlm.jsonl", remainder)
    reason_counts: dict[str, int] = {}
    for row in accounting:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1
    outputs = (
        output / "all_requested_accounting.jsonl",
        output / "main1000/plans_for_dlm.jsonl",
        output / "remainder/plans_for_dlm.jsonl",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "requested": len(accounting),
        "official_reference_available": sum(row["eligible"] for row in accounting),
        "primary_count": len(primary),
        "remainder_count": len(remainder),
        "reason_counts": reason_counts,
        "selection": "first1000 official-reference-available source ordinals; remainder separate",
        "official_stability_values_read": False,
        "unknown_rows_preserved_in_accounting": True,
        "outputs": {path.relative_to(output).as_posix(): sha256_file(path) for path in outputs},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = (*outputs, output / "manifest.json")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
