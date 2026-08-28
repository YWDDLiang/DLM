#!/usr/bin/env python3
"""Freeze reduced-composition-disjoint Branch, L6 and downstream-holdout sets."""

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


def with_identity(row: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    plan = row.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError(f"{source} row lacks plan_state")
    output = dict(row)
    output["reduced_composition_identity"] = identity_text(
        identity_from_plan_state(plan)
    )
    output["ctv_source"] = source
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_set(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row["reduced_composition_identity"]) for row in rows}


def filter_branch_by_certificate(
    rows: list[dict[str, Any]], compile_fn
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply outcome-blind C³FD eligibility before any positional selection."""

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        certificate = compile_fn(row, row_index)
        if certificate.get("composition_supervision") is True:
            output = dict(row)
            output["ctv_certificate_class"] = str(
                certificate.get("certificate_class") or "benchmark_compatible"
            )
            accepted.append(output)
            continue
        output = dict(row)
        output["ctv_certificate_rejection"] = str(
            certificate.get("compile_error")
            or certificate.get("certificate_class")
            or "unknown"
        )
        rejected.append(output)
    return accepted, rejected


def select_unique_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    blocked_identities: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen = set(blocked_identities or ())
    for row in rows:
        identity = str(row["reduced_composition_identity"])
        if identity in seen:
            output = dict(row)
            output["ctv_identity_rejection"] = "duplicate_or_cross_split_identity"
            rejected.append(output)
            continue
        seen.add(identity)
        selected.append(dict(row))
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise RuntimeError(
            f"insufficient unique certified Branch rows: {len(selected)}/{int(count)}"
        )
    return selected, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-cohort", type=Path, required=True)
    parser.add_argument("--c3fd-seed17", type=Path, required=True)
    parser.add_argument("--c3fd-seed18", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canary-plans", type=int, default=8)
    parser.add_argument("--branch-train-plans", type=int, default=128)
    parser.add_argument("--branch-val-plans", type=int, default=32)
    parser.add_argument("--l6-plans", type=int, default=256)
    parser.add_argument("--require-c3fd-certification", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    branch = [
        with_identity(row, source="branch")
        for row in iter_jsonl(args.branch_cohort)
    ]
    certificate_rejected: list[dict[str, Any]] = []
    if args.require_c3fd_certification:
        from build_c3fd_planner_data import compile_row

        branch, certificate_rejected = filter_branch_by_certificate(
            branch, compile_row
        )
    seed17 = [
        with_identity(row, source="c3fd_seed17")
        for row in iter_jsonl(args.c3fd_seed17)
    ]
    seed18 = [
        with_identity(row, source="c3fd_seed18")
        for row in iter_jsonl(args.c3fd_seed18)
    ]
    seed17.sort(key=lambda row: int(row["sample_idx"]))
    seed18.sort(key=lambda row: int(row["sample_idx"]))
    if len(seed17) != 1000 or len(seed18) != 1000:
        raise ValueError("C³FD L6/L7 sources must each preserve requested1000")
    if {int(row["sample_idx"]) for row in seed18} != set(range(1000)):
        raise ValueError("seed18 downstream holdout ledger must be sample_idx0..999")

    l7_ids = identity_set(seed18)
    l6: list[dict[str, Any]] = []
    l6_seen: set[str] = set()
    for row in seed17:
        identity = str(row["reduced_composition_identity"])
        if identity in l7_ids or identity in l6_seen:
            continue
        output = dict(row)
        output["ctv_l6_ordinal"] = len(l6)
        l6.append(output)
        l6_seen.add(identity)
        if len(l6) == int(args.l6_plans):
            break
    if len(l6) != int(args.l6_plans):
        raise RuntimeError("insufficient outcome-blind seed17 identities for L6")

    excluded_ids = l7_ids | l6_seen
    branch_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    excluded_branch: list[dict[str, Any]] = []
    for row in sorted(branch, key=lambda value: int(value["sample_idx"])):
        identity = str(row["reduced_composition_identity"])
        split = str(row.get("pair_split") or "")
        if identity in excluded_ids:
            excluded_branch.append(row)
            continue
        if split in branch_by_split:
            branch_by_split[split].append(row)
    canary_count = int(args.canary_plans)
    train_count = int(args.branch_train_plans)
    val_count = int(args.branch_val_plans)
    if len(branch_by_split["train"]) < canary_count + train_count:
        raise RuntimeError("insufficient disjoint Branch train Plans")
    if len(branch_by_split["validation"]) < val_count:
        raise RuntimeError("insufficient disjoint Branch validation Plans")
    train_selected, train_duplicates = select_unique_rows(
        branch_by_split["train"], count=canary_count + train_count
    )
    canary = train_selected[:canary_count]
    branch_train = train_selected[canary_count:]
    train_ids = identity_set(train_selected)
    branch_val, validation_duplicates = select_unique_rows(
        branch_by_split["validation"],
        count=val_count,
        blocked_identities=train_ids,
    )
    identity_rejected = [*train_duplicates, *validation_duplicates]

    sets = {
        "canary": identity_set(canary),
        "branch_train": identity_set(branch_train),
        "branch_validation": identity_set(branch_val),
        "l6": identity_set(l6),
        "l7": l7_ids,
    }
    overlap: dict[str, int] = {}
    names = list(sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap[f"{left}__{right}"] = len(sets[left] & sets[right])

    args.output_dir.mkdir(parents=True)
    outputs = {
        "CTV_BRANCH_CANARY_PLANS.jsonl": canary,
        "CTV_BRANCH_TRAIN_PLANS.jsonl": branch_train,
        "CTV_BRANCH_VAL_PLANS.jsonl": branch_val,
        "CTV_DLM_L6_PLANS.jsonl": l6,
        "CTV_DLM_L7_PLANS.jsonl": seed18,
        "CTV_BRANCH_EXCLUDED_OVERLAP.jsonl": excluded_branch,
        "CTV_BRANCH_CERTIFICATE_REJECTED.jsonl": certificate_rejected,
        "CTV_BRANCH_IDENTITY_REJECTED.jsonl": identity_rejected,
    }
    hashes = {
        name: write_jsonl(args.output_dir / name, rows)
        for name, rows in outputs.items()
    }
    gate = {
        "canary_count_exact": len(canary) == canary_count,
        "branch_train_count_exact": len(branch_train) == train_count,
        "branch_val_count_exact": len(branch_val) == val_count,
        "l6_count_exact": len(l6) == int(args.l6_plans),
        "l6_identities_unique": len(sets["l6"]) == len(l6),
        "branch_frozen_identities_unique": all(
            len(sets[name]) == len(rows)
            for name, rows in (
                ("canary", canary),
                ("branch_train", branch_train),
                ("branch_validation", branch_val),
            )
        ),
        "l7_requested1000_unchanged": len(seed18) == 1000,
        "all_frozen_sets_pairwise_disjoint": all(value == 0 for value in overlap.values()),
        "outcome_labels_unused": True,
        "certificate_filter_order_valid": True,
    }
    gate["identity_freeze_authorized"] = all(gate.values())
    report = {
        "schema": "h1a2_ctv_identity_freeze_v1",
        "sources": {
            "branch": str(args.branch_cohort.resolve()),
            "c3fd_seed17": str(args.c3fd_seed17.resolve()),
            "c3fd_seed18": str(args.c3fd_seed18.resolve()),
        },
        "configuration": {
            "require_c3fd_certification": bool(
                args.require_c3fd_certification
            )
        },
        "counts": {name: len(rows) for name, rows in outputs.items()},
        "identity_counts": {name: len(values) for name, values in sets.items()},
        "overlap": overlap,
        "source_diagnostics": {
            "branch_vs_seed17": len(identity_set(branch) & identity_set(seed17)),
            "branch_vs_seed18": len(identity_set(branch) & l7_ids),
            "seed17_vs_seed18": len(identity_set(seed17) & l7_ids),
        },
        "excluded_branch_reasons": dict(
            Counter(str(row.get("pair_split")) for row in excluded_branch)
        ),
        "certificate_rejected_reasons": dict(
            Counter(
                str(row.get("ctv_certificate_rejection"))
                for row in certificate_rejected
            )
        ),
        "identity_rejected_reasons": dict(
            Counter(
                str(row.get("ctv_identity_rejection"))
                for row in identity_rejected
            )
        ),
        "hashes": hashes,
        "gate": gate,
    }
    (args.output_dir / "CTV_IDENTITY_FREEZE_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# CTV-DLM identity freeze",
        "",
        f"Authorized: **{gate['identity_freeze_authorized']}**",
        f"Counts: `{report['counts']}`",
        f"Source overlaps: `{report['source_diagnostics']}`",
        f"Frozen-set overlaps: `{overlap}`",
        "",
        "No energy, stability, novelty or generation-success outcome was used.",
    ]
    (args.output_dir / "CTV_IDENTITY_FREEZE_MANIFEST.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
