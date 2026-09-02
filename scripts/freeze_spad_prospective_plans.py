#!/usr/bin/env python3
"""Freeze actual C3FD–Llama SPAD Plan rows before any DLM outcome exists."""

from __future__ import annotations

import argparse
from collections import Counter
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
    if str(row.get("species_program_source")) != "planner_llama_pointer":
        raise ValueError("Plan row does not carry the learned Llama pointer")


def freeze_rows(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    blocked: set[str],
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    exclusions: Counter[str] = Counter()
    for source_position, source in enumerate(source_rows):
        plan = identity.find_plan(source)
        if plan is None:
            exclusions["missing_plan"] += 1
            continue
        try:
            exact = identity.exact_identity(plan)
            reduced = identity.reduced_identity(plan)
            validate_program(source, plan)
        except Exception as exc:
            exclusions[f"invalid:{type(exc).__name__}"] += 1
            continue
        if exact in blocked:
            exclusions["blocked_exact"] += 1
            continue
        if exact in seen:
            exclusions["duplicate_exact"] += 1
            continue
        row = dict(source)
        source_sample_idx = int(source.get("sample_idx", source_position))
        sample_idx = len(selected)
        row["source_sample_idx"] = source_sample_idx
        row["sample_idx"] = sample_idx
        selected.append(row)
        seen.add(exact)
        ledger.append(
            {
                "sample_idx": sample_idx,
                "source_sample_idx": source_sample_idx,
                "source_position": source_position,
                "exact_composition_identity": exact,
                "reduced_composition_identity": reduced,
                "chemsys": "-".join(element for element, _ in identity.canonical_counts(plan)),
            }
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} eligible actual Planner rows")
    return selected, ledger, exclusions


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plans", type=Path, required=True)
    parser.add_argument("--mp20-train", type=Path, required=True)
    parser.add_argument("--exclude-cohort-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--planner-sampling-seed", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    source_rows = read_jsonl(args.source_plans)
    mp20_blocked = identity.blocked_from_file(args.mp20_train)
    cohort_blocked, cohort_files = identity.blocked_from_root(args.exclude_cohort_root)
    blocked = mp20_blocked | cohort_blocked
    selected, ledger, exclusions = freeze_rows(
        source_rows, blocked=blocked, count=int(args.count)
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "plans_for_dlm.jsonl", selected)
    write_jsonl(args.output_dir / "ledger.jsonl", ledger)
    manifest = {
        "schema": "spad_actual_plan_prospective_cohort_v1",
        "planner_sampling_seed": int(args.planner_sampling_seed),
        "source_plans": str(args.source_plans.resolve()),
        "source_rows": len(source_rows),
        "selected": len(selected),
        "selection": "first_eligible_actual_planner_row",
        "unique_exact": len({row["exact_composition_identity"] for row in ledger}),
        "unique_chemsys": len({row["chemsys"] for row in ledger}),
        "blocked_exact_identities": len(blocked),
        "blocked_cohort_files": len(cohort_files),
        "exclusions": dict(sorted(exclusions.items())),
        "outcomes_read": False,
        "planner_resampled_after_freeze": False,
        "gates": {
            "fixed_count": len(selected) == int(args.count),
            "sample_idx_contiguous": [row["sample_idx"] for row in selected]
            == list(range(int(args.count))),
            "exact_identity_unique": len(
                {row["exact_composition_identity"] for row in ledger}
            )
            == int(args.count),
            "blocked_overlap_zero": not bool(
                {row["exact_composition_identity"] for row in ledger} & blocked
            ),
            "pointer_program_complete": all(
                row.get("species_program_source") == "planner_llama_pointer"
                for row in selected
            ),
            "outcome_blind": True,
        },
    }
    if not all(manifest["gates"].values()):
        raise RuntimeError("SPAD prospective Plan gates failed")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
