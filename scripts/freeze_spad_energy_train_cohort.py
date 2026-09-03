#!/usr/bin/env python3
"""Freeze outcome-blind MP20-train rows for SPAD-E teacher rollouts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.fixed_slot import SYMBOL_TO_Z  # noqa: E402


SCHEMA = "spad_energy_train_cohort_v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def deterministic_key(row: Mapping[str, Any], seed: int) -> tuple[str, int]:
    source_idx = int(row["source_row_idx"])
    plan = row["plan_state"]
    identity = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{seed}|{source_idx}|{identity}".encode()).hexdigest()
    return digest, source_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-train", type=Path, required=True)
    parser.add_argument("--pointer-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requested", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if int(args.requested) != 2048:
        raise ValueError("SPAD-E first cohort is frozen at 2048 rows")

    teachers = {int(row["source_row_idx"]): row for row in iter_jsonl(args.teacher_train)}
    pointers = {int(row["source_row_idx"]): row for row in iter_jsonl(args.pointer_train)}
    if len(teachers) != 27136 or len(pointers) != 24558:
        raise ValueError("upstream MP20/pointer row count changed")
    eligible = []
    for source_idx in sorted(set(teachers) & set(pointers)):
        teacher = teachers[source_idx]
        pointer = pointers[source_idx]
        plan = teacher["plan_state"]
        if [int(SYMBOL_TO_Z[str(value)]) for value in plan["elements"]] != [
            int(value) for value in pointer["canonical_atomic_numbers"]
        ] or [int(value) for value in plan["counts"]] != [
            int(value) for value in pointer["canonical_element_counts"]
        ]:
            raise ValueError(f"teacher/pointer composition mismatch at {source_idx}")
        if teacher.get("source_split") != "train":
            raise ValueError("non-train row entered SPAD-E cohort")
        eligible.append(teacher)
    selected = sorted(eligible, key=lambda row: deterministic_key(row, int(args.seed)))[
        : int(args.requested)
    ]
    selected_indices = [int(row["source_row_idx"]) for row in selected]
    if len(selected_indices) != 2048 or len(set(selected_indices)) != 2048:
        raise RuntimeError("SPAD-E cohort selection changed denominator")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "teacher_rows.jsonl").open("x", encoding="utf-8") as handle:
        for sample_idx, source in enumerate(selected):
            row = {
                "schema": SCHEMA,
                "sample_idx": sample_idx,
                "source_row_idx": int(source["source_row_idx"]),
                "source_split": "train",
                "prompt": str(source["prompt"]),
                "answer": str(source["answer"]),
                "plan_state": dict(source["plan_state"]),
                "num_atoms": int(source["plan_state"]["N"]),
                "outcomes_read": False,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (args.output_dir / "pointer_rows.jsonl").open("x", encoding="utf-8") as handle:
        for sample_idx, source_idx in enumerate(selected_indices):
            row = dict(pointers[source_idx])
            row["sample_idx"] = sample_idx
            row["spad_energy_source_row_idx"] = source_idx
            row["outcomes_read"] = False
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    strata = Counter(
        (int(row["plan_state"]["N"]), len(row["plan_state"]["elements"]))
        for row in selected
    )
    manifest = {
        "schema": SCHEMA,
        "requested": 2048,
        "selected": 2048,
        "seed": int(args.seed),
        "selection": "outcome_blind_hash_order",
        "source_split": "MP20-train-only",
        "outcomes_read": False,
        "energy_hull_model494_chgnet_read": False,
        "pointer_supported_rows_only": True,
        "strata_N_arity": {
            f"N={n}|arity={arity}": count
            for (n, arity), count in sorted(strata.items())
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
