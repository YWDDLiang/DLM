#!/usr/bin/env python3
"""Freeze one outcome-blind C3FD-native prospective 256-composition cohort."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_native_plan import (  # noqa: E402
    build_native_inference_prompt,
    native_plan_from_parts,
)


SCHEMA = "c3fd_native_prospective_cohort_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_plan(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("plan_state", "source_plan_state", "r5_plan_state", "identity"):
        value = row.get(key)
        if isinstance(value, Mapping) and {"N", "elements", "counts"} <= set(value):
            return value
    for key in ("source_row", "record", "candidate"):
        value = row.get(key)
        if isinstance(value, Mapping):
            nested = find_plan(value)
            if nested is not None:
                return nested
    return None


def canonical_counts(plan: Mapping[str, Any]) -> list[tuple[str, int]]:
    elements = [str(value) for value in plan.get("elements") or ()]
    counts = [int(value) for value in plan.get("counts") or ()]
    if not elements or len(elements) != len(counts):
        raise ValueError("plan lacks aligned composition")
    combined: Counter[str] = Counter()
    for element, count in zip(elements, counts):
        if count <= 0:
            raise ValueError("composition count must be positive")
        combined[element] += count
    if int(plan.get("N") or 0) != sum(combined.values()):
        raise ValueError("plan N/count conservation failed")
    return sorted(combined.items())


def exact_identity(plan: Mapping[str, Any]) -> str:
    return "|".join(f"{element}:{count}" for element, count in canonical_counts(plan))


def reduced_identity(plan: Mapping[str, Any]) -> str:
    values = canonical_counts(plan)
    divisor = 0
    for _element, count in values:
        divisor = math.gcd(divisor, count)
    return "|".join(f"{element}:{count // divisor}" for element, count in values)


def blocked_from_file(path: Path) -> set[str]:
    blocked: set[str] = set()
    for row in read_jsonl(path):
        plan = find_plan(row)
        if plan is not None:
            try:
                blocked.add(exact_identity(plan))
            except Exception:
                continue
    return blocked


def blocked_from_root(root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    blocked: set[str] = set()
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl")):
        identities = blocked_from_file(path)
        if not identities:
            continue
        blocked.update(identities)
        files.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "exact_identities": len(identities),
            }
        )
    return blocked, files


def freeze(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    blocked_exact: set[str],
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    exclusions: Counter[str] = Counter()
    for source_ordinal, row in enumerate(source_rows):
        plan = find_plan(row)
        if plan is None:
            exclusions["missing_plan"] += 1
            continue
        try:
            exact = exact_identity(plan)
            reduced = reduced_identity(plan)
            native = native_plan_from_parts(plan, plan)
            prompt = build_native_inference_prompt(plan, plan)
        except Exception as exc:
            exclusions[f"invalid:{type(exc).__name__}"] += 1
            continue
        if exact in blocked_exact:
            exclusions["blocked_exact"] += 1
            continue
        if exact in seen:
            exclusions["duplicate_exact"] += 1
            continue
        sample_idx = len(selected)
        seen.add(exact)
        source_sample_idx = int(row.get("sample_idx", source_ordinal))
        plan_row = {
            "schema": "c3fd_native_prospective_plan_row_v1",
            "sample_idx": sample_idx,
            "source_sample_idx": source_sample_idx,
            "source_ordinal": source_ordinal,
            "exact_composition_identity": exact,
            "reduced_composition_identity": reduced,
            "chemsys": "-".join(element for element, _count in canonical_counts(plan)),
            "plan_state": native,
            "prompt": prompt,
        }
        selected.append(plan_row)
        ledger.append(
            {
                key: plan_row[key]
                for key in (
                    "sample_idx",
                    "source_sample_idx",
                    "source_ordinal",
                    "exact_composition_identity",
                    "reduced_composition_identity",
                    "chemsys",
                )
            }
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} eligible prospective rows")
    return selected, ledger, {
        "selected": len(selected),
        "unique_exact": len(seen),
        "unique_chemsys": len({row["chemsys"] for row in ledger}),
        "exclusions": dict(sorted(exclusions.items())),
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plans", type=Path, required=True)
    parser.add_argument("--mp20-train", type=Path, required=True)
    parser.add_argument("--exclude-cohort-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_sha = sha256_file(args.source_plans)
    mp20_blocked = blocked_from_file(args.mp20_train)
    cohort_blocked, cohort_files = blocked_from_root(args.exclude_cohort_root)
    blocked = mp20_blocked | cohort_blocked
    source_rows = read_jsonl(args.source_plans)
    if sha256_file(args.source_plans) != source_sha:
        raise RuntimeError("prospective source changed while reading")
    plans, ledger, selection = freeze(
        source_rows=source_rows,
        blocked_exact=blocked,
        count=int(args.count),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "plans.jsonl", plans)
    write_jsonl(args.output_dir / "ledger.jsonl", ledger)
    (args.output_dir / "blocked_inputs.json").write_text(
        json.dumps(
            {
                "mp20_train": {
                    "path": str(args.mp20_train.resolve()),
                    "sha256": sha256_file(args.mp20_train),
                    "exact_identities": len(mp20_blocked),
                },
                "cohort_files": cohort_files,
                "union_exact_identities": len(blocked),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    hashes = {
        name: sha256_file(args.output_dir / name)
        for name in ("plans.jsonl", "ledger.jsonl", "blocked_inputs.json")
    }
    manifest = {
        "schema": SCHEMA,
        "planner_sampling_seed": 20,
        "source": {
            "path": str(args.source_plans.resolve()),
            "sha256": source_sha,
            "rows": len(source_rows),
        },
        "selected": len(plans),
        "selection": "first_eligible_source_ordinal",
        "selection_report": selection,
        "blocked_exact_identities": len(blocked),
        "outcomes_read": False,
        "policy_outcomes_exist_at_freeze": False,
        "output_sha256": hashes,
        "gates": {
            "fixed256": len(plans) == 256,
            "sample_idx_contiguous": [row["sample_idx"] for row in plans]
            == list(range(256)),
            "exact_identity_unique": len(
                {row["exact_composition_identity"] for row in plans}
            )
            == 256,
            "blocked_overlap_zero": not bool(
                {row["exact_composition_identity"] for row in plans} & blocked
            ),
            "outcome_blind": True,
        },
    }
    if not all(manifest["gates"].values()):
        raise RuntimeError("prospective cohort gates failed")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hashes["manifest.json"] = sha256_file(manifest_path)
    with (args.output_dir / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for name in sorted(hashes):
            handle.write(f"{hashes[name]}  {name}\n")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
