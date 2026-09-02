#!/usr/bin/env python3
"""Freeze every requested actual SPAD Plan without filtering or replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import freeze_c3fd_native_prospective_cohort as identity  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_program(row: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    elements = [str(value) for value in plan["elements"]]
    program = [str(value) for value in row.get("species_program") or ()]
    indices = [int(value) for value in row.get("species_program_indices") or ()]
    if sorted(program) != sorted(elements):
        raise ValueError("species program is not a permutation of Plan elements")
    if sorted(indices) != list(range(len(elements))):
        raise ValueError("species program indices are not a permutation")
    if program != [elements[index] for index in indices]:
        raise ValueError("species program and permutation indices disagree")
    if str(row.get("species_program_source")) != "planner_llama_pointer":
        raise ValueError("Plan row does not carry the learned Llama pointer")
    if str(row.get("prompt_schema")) != "C3FD_NATIVE_PLAN_V2":
        raise ValueError("Plan row uses an unexpected prompt schema")


def freeze_requested(
    records: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    *,
    requested: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(records) != requested:
        raise ValueError("Planner records do not cover the requested denominator")
    record_by_index = {int(row["sample_idx"]): row for row in records}
    plan_by_index = {int(row["sample_idx"]): dict(row) for row in plans}
    if set(record_by_index) != set(range(requested)):
        raise ValueError("Planner request indices are not contiguous")
    if len(plan_by_index) != len(plans) or not set(plan_by_index) <= set(record_by_index):
        raise ValueError("successful Plan indices are malformed")

    frozen_plans: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    exact_seen: set[str] = set()
    for sample_idx in range(requested):
        record = record_by_index[sample_idx]
        parsed = record.get("parsed") is True
        comp_valid = record.get("comp_valid") is True
        plan_row = plan_by_index.get(sample_idx)
        if parsed != (plan_row is not None) or comp_valid != (plan_row is not None):
            raise ValueError("record/Plan success accounting differs")
        if plan_row is None:
            ledger.append(
                {
                    "sample_idx": sample_idx,
                    "planner_valid": False,
                    "failure": record.get("failure"),
                    "exact_composition_identity": None,
                    "reduced_composition_identity": None,
                    "chemsys": None,
                }
            )
            continue
        plan = identity.find_plan(plan_row)
        if plan is None:
            raise ValueError("successful Plan row lacks plan_state")
        validate_program(plan_row, plan)
        exact = identity.exact_identity(plan)
        reduced = identity.reduced_identity(plan)
        frozen_plans.append(plan_row)
        exact_seen.add(exact)
        ledger.append(
            {
                "sample_idx": sample_idx,
                "planner_valid": True,
                "failure": None,
                "exact_composition_identity": exact,
                "reduced_composition_identity": reduced,
                "chemsys": "-".join(element for element, _ in identity.canonical_counts(plan)),
            }
        )
    valid = len(frozen_plans)
    audit = {
        "requested": requested,
        "planner_valid": valid,
        "planner_invalid": requested - valid,
        "composition_valid_rate": valid / requested,
        "unique_exact_among_valid": len(exact_seen),
        "duplicate_exact_among_valid": valid - len(exact_seen),
    }
    return frozen_plans, ledger, audit


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--source-plans", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requested", type=int, default=256)
    parser.add_argument("--planner-sampling-seed", type=int, required=True)
    parser.add_argument("--minimum-comp-valid", type=float, default=0.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    records = read_jsonl(args.source_records)
    plans = read_jsonl(args.source_plans)
    frozen, ledger, audit = freeze_requested(
        records, plans, requested=int(args.requested)
    )
    target = float(args.minimum_comp_valid)
    if not 0.0 <= target <= 1.0:
        raise ValueError("minimum-comp-valid must be in [0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "plans_for_dlm.jsonl", frozen)
    write_jsonl(args.output_dir / "ledger.jsonl", ledger)
    manifest = {
        "schema": "spad_actual_plan_prospective_cohort_v2",
        "planner_sampling_seed": int(args.planner_sampling_seed),
        **audit,
        "selection": "none_all_requested_ordinals_retained",
        "replacement": False,
        "planner_resampled_after_freeze": False,
        "outcomes_read": False,
        "composition_valid_target": target,
        "composition_valid_target_met": audit["composition_valid_rate"] >= target,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
