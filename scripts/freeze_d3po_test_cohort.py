#!/usr/bin/env python3
"""Freeze an outcome-blind, reduced-composition-disjoint D3PO test cohort."""

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

from crystal_dlm.composition_identity import (  # noqa: E402
    canonical_symbol_counts,
    identity_from_plan_state,
    identity_text,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = row.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError("cohort row lacks plan_state")
    return plan


def reduced_identity(row: Mapping[str, Any]) -> str:
    return identity_text(identity_from_plan_state(plan_from_row(row)))


def exact_identity(row: Mapping[str, Any]) -> str:
    plan = plan_from_row(row)
    counts = canonical_symbol_counts(
        [str(value) for value in (plan.get("elements") or ())],
        [int(value) for value in (plan.get("counts") or ())],
    )
    n_value = int(plan.get("N") or 0)
    if n_value != sum(count for _symbol, count in counts):
        raise ValueError("Plan N/count conservation failed")
    return "|".join(f"{symbol}:{count}" for symbol, count in counts)


def freeze_rows(
    source_rows: Iterable[Mapping[str, Any]],
    blocked_rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    blocked = {reduced_identity(row) for row in blocked_rows}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    for row in sorted(source_rows, key=lambda value: int(value["sample_idx"])):
        identity = reduced_identity(row)
        if identity in blocked:
            rejected["blocked_reduced_identity"] += 1
            continue
        if identity in seen:
            rejected["duplicate_source_reduced_identity"] += 1
            continue
        output = dict(row)
        output["d3po_test_ordinal"] = len(selected)
        output["d3po_reduced_composition_identity"] = identity
        output["d3po_exact_composition_identity"] = exact_identity(row)
        selected.append(output)
        seen.add(identity)
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise RuntimeError(f"insufficient disjoint test rows: {len(selected)}/{count}")
    return selected, rejected


def distribution_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    n_counts: Counter[str] = Counter()
    arity_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    element_counts: Counter[str] = Counter()
    row_count = 0
    for row in rows:
        row_count += 1
        plan = plan_from_row(row)
        elements = [str(value) for value in (plan.get("elements") or ())]
        n_counts[str(int(plan["N"]))] += 1
        arity_counts[str(len(elements))] += 1
        family_counts[str(plan.get("anion_framework") or "missing")] += 1
        element_counts.update(elements)
    return {
        "rows": row_count,
        "N": dict(sorted(n_counts.items())),
        "arity": dict(sorted(arity_counts.items())),
        "family": dict(sorted(family_counts.items())),
        "elements": dict(sorted(element_counts.items())),
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed17-source", type=Path, required=True)
    parser.add_argument("--blocked-cohort", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    source = list(iter_jsonl(args.seed17_source))
    if len(source) != 1000 or {int(row["sample_idx"]) for row in source} != set(
        range(1000)
    ):
        raise ValueError("seed17 source must preserve sample_idx0..999")
    blocked_by_path = {
        str(path.resolve()): list(iter_jsonl(path)) for path in args.blocked_cohort
    }
    selected, rejected = freeze_rows(
        source,
        (
            row
            for rows in blocked_by_path.values()
            for row in rows
        ),
        count=int(args.count),
    )
    blocked_identities = {
        reduced_identity(row)
        for rows in blocked_by_path.values()
        for row in rows
    }
    selected_identities = {reduced_identity(row) for row in selected}
    gate = {
        "requested_count_exact": len(selected) == int(args.count),
        "selected_reduced_identities_unique": len(selected_identities) == len(selected),
        "blocked_overlap_zero": not (selected_identities & blocked_identities),
        "source_ledger_complete": len(source) == 1000,
        "outcome_labels_unused": True,
        "selection_order_sample_idx_only": True,
    }
    gate["d3po_test_cohort_authorized"] = all(gate.values())

    args.output_dir.mkdir(parents=True)
    cohort_path = args.output_dir / "D3PO_TEST_PLANS.jsonl"
    cohort_hash = write_jsonl(cohort_path, selected)
    manifest = {
        "schema": "h1a2_d3po_test_cohort_v1",
        "sources": {
            "seed17": {
                "path": str(args.seed17_source.resolve()),
                "sha256": sha256_file(args.seed17_source),
            },
            "blocked": {
                path: {
                    "rows": len(rows),
                    "sha256": sha256_file(Path(path)),
                }
                for path, rows in sorted(blocked_by_path.items())
            },
        },
        "selection": {
            "requested": int(args.count),
            "selected": len(selected),
            "rejected": dict(sorted(rejected.items())),
            "blocked_reduced_identities": len(blocked_identities),
            "rule": "first sample_idx after reduced-identity exclusion; no outcomes",
        },
        "distribution": distribution_summary(selected),
        "hashes": {"D3PO_TEST_PLANS.jsonl": cohort_hash},
        "gate": gate,
        "gpu_jobs_used": 0,
    }
    (args.output_dir / "D3PO_TEST_COHORT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
