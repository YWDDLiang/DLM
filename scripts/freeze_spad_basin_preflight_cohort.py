#!/usr/bin/env python3
"""Freeze an outcome-blind MP20-train cohort for basin supervision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


STATE_TYPES = ("cell", "terminal_xyz")
DEFAULT_EXPECTED_GROUPS = 128


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def n_bin(value: int) -> str:
    if value <= 4:
        return "n01_04"
    if value <= 8:
        return "n05_08"
    if value <= 12:
        return "n09_12"
    return "n13_20"


def multiplicity_bin(value: int) -> str:
    if value <= 2:
        return "m01_02"
    if value <= 5:
        return "m03_05"
    return "m06_plus"


def stratum(row: Mapping[str, Any]) -> tuple[str, str, str]:
    plan = row["plan_state"]
    n = int(plan["N"])
    counts = [int(value) for value in plan["counts"]]
    return (
        n_bin(n),
        multiplicity_bin(max(counts)),
        "species3plus" if len(plan["elements"]) >= 3 else "species1_2",
    )


def deterministic_key(row: Mapping[str, Any], seed: int) -> str:
    payload = f"{int(seed)}:{int(row['source_row_idx'])}".encode()
    return hashlib.sha256(payload).hexdigest()


def validate_source(row: Mapping[str, Any]) -> None:
    if row.get("outcomes_read") is not False:
        raise ValueError("source is not outcome-blind")
    split = row.get("source_split", row.get("split"))
    if split is not None and str(split).lower() not in {
        "train",
        "mp20_train_only",
        "mp20-train-only",
    }:
        raise ValueError(f"source is not MP20 train: {split!r}")
    plan = row.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError("source lacks Plan")
    elements = [str(value) for value in plan.get("elements") or ()]
    counts = [int(value) for value in plan.get("counts") or ()]
    if (
        not elements
        or len(elements) != len(counts)
        or len(elements) != len(set(elements))
        or any(value <= 0 for value in counts)
        or sum(counts) != int(plan.get("N") or 0)
    ):
        raise ValueError("source Plan composition is malformed")
    program = [str(value) for value in row.get("species_program") or ()]
    if len(program) != len(elements) or set(program) != set(elements):
        raise ValueError("species_program does not permute Plan elements")
    if not row.get("prompt") or not row.get("teacher_answer"):
        raise ValueError("source lacks prompt or teacher body")


def select_round_robin(
    rows: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise ValueError("invalid selection count")
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        validate_source(row)
        buckets[stratum(row)].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: deterministic_key(row, seed))
    selected: list[dict[str, Any]] = []
    ordered = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in ordered:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            raise RuntimeError("stratified source pool exhausted")
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    expected_groups = int(
        getattr(args, "expected_groups", DEFAULT_EXPECTED_GROUPS)
    )
    if expected_groups <= 0:
        raise ValueError("--expected-groups must be positive")
    rows = list(iter_jsonl(args.plans_jsonl.resolve()))
    source_indices = [int(row["source_row_idx"]) for row in rows]
    if len(rows) != 27136 or len(set(source_indices)) != len(rows):
        raise ValueError("full MP20-train plan source changed")
    selected = select_round_robin(
        rows, count=expected_groups, seed=int(args.seed)
    )
    frozen: list[dict[str, Any]] = []
    for ordinal, source in enumerate(selected):
        row = dict(source)
        row["mp20_train_source_row_idx"] = int(source["source_row_idx"])
        row["sample_idx"] = ordinal
        row["preflight_idx"] = ordinal
        row["preflight_state_type"] = STATE_TYPES[ordinal % len(STATE_TYPES)]
        row["selection_uses_energy_force_stress_hull_or_generation"] = False
        frozen.append(row)
    state_counts = Counter(row["preflight_state_type"] for row in frozen)
    expected_state_counts = Counter(
        STATE_TYPES[ordinal % len(STATE_TYPES)]
        for ordinal in range(expected_groups)
    )
    if state_counts != expected_state_counts:
        raise RuntimeError("preflight state types do not match exact assignment")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plans_for_dlm.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for row in frozen:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    strata = Counter("|".join(stratum(row)) for row in frozen)
    manifest = {
        "schema": (
            "spad_basin_preflight_train128_manifest_v1"
            if expected_groups == DEFAULT_EXPECTED_GROUPS
            else f"spad_basin_preflight_train{expected_groups}_manifest_v1"
        ),
        "source": str(args.plans_jsonl.resolve()),
        "source_split": "MP20_train_only",
        "selection_seed": int(args.seed),
        "rows": len(frozen),
        "unique_source_rows": len({row["mp20_train_source_row_idx"] for row in frozen}),
        "state_types": dict(Counter(row["preflight_state_type"] for row in frozen)),
        "strata": dict(sorted(strata.items())),
        "high_N_rows": sum(int(row["plan_state"]["N"]) >= 13 for row in frozen),
        "high_multiplicity_rows": sum(
            max(int(value) for value in row["plan_state"]["counts"]) >= 6
            for row in frozen
        ),
        "outcomes_read": False,
        "selection_or_replacement": False,
    }
    if expected_groups != DEFAULT_EXPECTED_GROUPS:
        manifest["expected_groups"] = expected_groups
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--expected-groups", type=int, default=DEFAULT_EXPECTED_GROUPS
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
