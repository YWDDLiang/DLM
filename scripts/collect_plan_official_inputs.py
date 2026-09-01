#!/usr/bin/env python3
"""Collect composition-only official MP inputs before any DLM realization."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import protocol


def plan_chemsys(row) -> str:
    state = row.get("plan_state")
    if not isinstance(state, dict):
        raise protocol.ContractError("composition-valid Planner row lacks plan_state")
    elements = state.get("elements")
    counts = state.get("counts")
    if not isinstance(elements, list) or not isinstance(counts, list):
        raise protocol.ContractError("plan_state lacks elements/counts")
    if len(elements) == 0 or len(elements) != len(counts):
        raise protocol.ContractError("plan_state composition shape changed")
    if any(not isinstance(value, str) or not value for value in elements):
        raise protocol.ContractError("invalid Plan element symbol")
    if any(int(value) <= 0 for value in counts):
        raise protocol.ContractError("invalid Plan element count")
    return "-".join(sorted(set(elements)))


def collect(rows, *, expected_requested: int):
    if len(rows) != expected_requested:
        raise protocol.ContractError("Planner requested denominator changed")
    if [int(row["sample_idx"]) for row in rows] != list(range(expected_requested)):
        raise protocol.ContractError("Planner sample_idx accounting changed")
    wanted: set[str] = set()
    failures = 0
    for row in rows:
        if row.get("trajectory_attempts") != 1:
            raise protocol.ContractError("Planner trajectory count changed")
        if row.get("comp_valid") is True:
            wanted.add(plan_chemsys(row))
        else:
            failures += 1
    return wanted, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--planner-records", type=Path, required=True)
    parser.add_argument("--expected-requested", type=int, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    records_path = args.planner_records.resolve()
    rows = protocol.read_jsonl(records_path)
    wanted, failures = collect(rows, expected_requested=args.expected_requested)
    if len(wanted) == 0:
        raise protocol.ContractError("Planner produced no queryable chemsystem")

    final = run_root / "inputs"
    preparing = run_root / ".inputs.preparing"
    failed = run_root / ".inputs.FAILED"
    preparing.mkdir()
    try:
        wanted_rows = [
            {"query_index": index, "chemsys": chemsys, "elements": chemsys.split("-")}
            for index, chemsys in enumerate(sorted(wanted))
        ]
        protocol.write_jsonl_exclusive(preparing / "wanted_chemsys.jsonl", wanted_rows)
        manifest = {
            "schema": "h1a2_epoch2_exactplan_official_input_manifest_v1",
            "cell_count": 1,
            "cells": [
                {
                    "cell_id": "planner1200_before_dlm",
                    "attempts": len(rows),
                    "reconstructed": len(rows) - failures,
                    "unique_chemsys": len(wanted),
                    "labels": protocol.identity(records_path),
                }
            ],
            "wanted_chemsys_count": len(wanted_rows),
            "wanted_chemsys_sha256": protocol.canonical_sha256(wanted_rows),
            "stability_scope": "composition_reference_availability_before_DLM",
            "fresh_official_query": True,
            "historical_cache_reuse": False,
            "planner_failures_retained": failures,
            "generated_structure_or_energy_read": False,
        }
        protocol.write_json_exclusive(preparing / "input_manifest.json", manifest)
        (preparing / "inputs_SUCCESS").touch(exist_ok=False)
        protocol.write_source_manifest(
            preparing,
            ("input_manifest.json", "inputs_SUCCESS", "wanted_chemsys.jsonl"),
        )
        preparing.rename(final)
    except Exception:
        if preparing.exists():
            shutil.move(str(preparing), str(failed))
        raise
    print(
        protocol.canonical_json(
            {
                "requested": len(rows),
                "planner_failures": failures,
                "wanted_chemsys": len(wanted),
                "source_manifest_sha256": protocol.sha256_file(
                    final / "SOURCE_SHA256.txt"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
