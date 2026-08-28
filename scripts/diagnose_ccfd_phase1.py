#!/usr/bin/env python3
"""Diagnose why CCFD representability gains did not transfer to legacy comp-valid."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import reduce
import json
from math import gcd
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crystal_dlm.composition_validity import classify_smact_validity  # noqa: E402
from crystal_dlm.fixed_slot import SYMBOL_TO_Z  # noqa: E402
from crystal_dlm.valence_assignment import assign_crysvcd_valences  # noqa: E402
from finalize_ccfd_phase1 import extract_formula  # noqa: E402


SEEDS = (17, 18)
ARMS = ("f0", "f1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reduced(counts: list[int]) -> list[int]:
    divisor = reduce(gcd, counts) if counts else 1
    return [int(value) // max(1, int(divisor)) for value in counts]


def annotate(row: dict[str, Any]) -> dict[str, Any]:
    extracted = extract_formula(row)
    if extracted is None:
        return {
            "parsed": False,
            "assignment": False,
            "legacy": False,
            "legacy_reason": "formula_parse_failed",
            "mode": "none",
            "tier": "none",
            "N": None,
            "formula": None,
            "plan_parsed": row.get("parsed") is True,
            "raw_reason": str(row.get("reason") or ""),
            "raw_message": str(row.get("message") or ""),
        }
    formula, symbols, counts = extracted
    assignment = assign_crysvcd_valences(symbols, counts, max_species=7)
    legacy = classify_smact_validity(
        [int(SYMBOL_TO_Z[symbol]) for symbol in symbols], reduced(counts)
    )
    return {
        "parsed": True,
        "assignment": assignment.get("assigned") is True,
        "legacy": legacy.get("valid") is True,
        "legacy_reason": str(legacy.get("reason") or "unknown"),
        "mode": str(assignment.get("mode") or "unassigned"),
        "tier": str(assignment.get("state_catalog_tier") or "none"),
        "N": sum(counts),
        "formula": formula,
        "plan_parsed": row.get("parsed") is True,
        "raw_reason": str(row.get("reason") or ""),
        "raw_message": str(row.get("message") or ""),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cross: Counter[str] = Counter()
    legacy_reasons: Counter[str] = Counter()
    assignment_only_legacy_reasons: Counter[str] = Counter()
    assignment_only_modes: Counter[str] = Counter()
    assignment_only_tiers: Counter[str] = Counter()
    n_counts: Counter[str] = Counter()
    parse_failures: Counter[str] = Counter()
    for row in rows:
        if not row["parsed"]:
            cross["formula_parse_failed"] += 1
            parse_failures[f"{row['raw_reason']}|{row['raw_message']}"] += 1
            n_counts["failed"] += 1
            continue
        key = (
            "both_valid"
            if row["assignment"] and row["legacy"]
            else "assignment_only"
            if row["assignment"]
            else "legacy_only"
            if row["legacy"]
            else "neither"
        )
        cross[key] += 1
        legacy_reasons[row["legacy_reason"]] += 1
        n_counts[str(row["N"])] += 1
        if key == "assignment_only":
            assignment_only_legacy_reasons[row["legacy_reason"]] += 1
            assignment_only_modes[row["mode"]] += 1
            assignment_only_tiers[row["tier"]] += 1
    return {
        "rows": len(rows),
        "cross_tab": dict(sorted(cross.items())),
        "legacy_reasons": dict(sorted(legacy_reasons.items())),
        "assignment_only_legacy_reasons": dict(
            sorted(assignment_only_legacy_reasons.items())
        ),
        "assignment_only_modes": dict(sorted(assignment_only_modes.items())),
        "assignment_only_tiers": dict(sorted(assignment_only_tiers.items())),
        "N": dict(sorted(n_counts.items())),
        "parse_failures": dict(parse_failures.most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    cell_reports: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            raw = read_jsonl(args.run_dir / f"{arm}_seed{seed}/raw_generations.jsonl")
            ordered = sorted(raw, key=lambda row: int(row["sample_idx"]))
            if len(ordered) != 1000 or [int(row["sample_idx"]) for row in ordered] != list(range(1000)):
                raise RuntimeError(f"{arm} seed{seed} denominator changed")
            annotated = [annotate(row) for row in ordered]
            cells[(arm, seed)] = annotated
            cell_reports.append({"arm": arm, "seed": seed, **summarize(annotated)})
    pooled = {
        arm: summarize([row for seed in SEEDS for row in cells[(arm, seed)]])
        for arm in ARMS
    }
    transitions: dict[str, Any] = {}
    for seed in SEEDS:
        left = cells[("f0", seed)]
        right = cells[("f1", seed)]
        transitions[str(seed)] = {
            "legacy": dict(
                sorted(
                    Counter(
                        f"{int(a['legacy'])}->{int(b['legacy'])}"
                        for a, b in zip(left, right)
                    ).items()
                )
            ),
            "assignment": dict(
                sorted(
                    Counter(
                        f"{int(a['assignment'])}->{int(b['assignment'])}"
                        for a, b in zip(left, right)
                    ).items()
                )
            ),
            "same_formula": sum(a["formula"] == b["formula"] and a["formula"] is not None for a, b in zip(left, right)),
            "N_delta": dict(
                sorted(
                    Counter(
                        str(int(b["N"]) - int(a["N"]))
                        for a, b in zip(left, right)
                        if a["N"] is not None and b["N"] is not None
                    ).items(),
                    key=lambda item: int(item[0]),
                )
            ),
        }
    f0_n = pooled["f0"]["N"]
    f1_n = pooled["f1"]["N"]
    n_contributors = sorted(
        (
            {
                "N": key,
                "f0": int(f0_n.get(key, 0)),
                "f1": int(f1_n.get(key, 0)),
                "delta": int(f1_n.get(key, 0)) - int(f0_n.get(key, 0)),
                "abs_rate_delta": abs(int(f1_n.get(key, 0)) - int(f0_n.get(key, 0))) / 2000,
            }
            for key in set(f0_n) | set(f1_n)
        ),
        key=lambda row: (-row["abs_rate_delta"], str(row["N"])),
    )
    report = {
        "schema": "h1a2_ccfd_phase1_diagnostic_v1",
        "cells": cell_reports,
        "pooled": pooled,
        "paired_transitions": transitions,
        "N_drift_contributors": n_contributors,
        "interpretation": {
            "constraint_target": "existence under the expanded valence catalog, including mixed valence and catalog extensions",
            "independent_target": "legacy SMACT one-state-per-element charge neutrality plus Pauling test",
            "main_question": "whether internal representability transfers to the independent validator without distribution shift",
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stem = "CCFD_PHASE1_DIAGNOSTIC"
    (output / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# CCFD Phase1 diagnostic",
        "",
        "| Arm | Both valid | Assignment only | Legacy only | Neither | Formula fail |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        cross = pooled[arm]["cross_tab"]
        lines.append(
            f"| {arm} | {cross.get('both_valid', 0)} | {cross.get('assignment_only', 0)} | "
            f"{cross.get('legacy_only', 0)} | {cross.get('neither', 0)} | {cross.get('formula_parse_failed', 0)} |"
        )
    lines.extend(
        [
            "",
            f"F1 all-parsed legacy reasons: `{pooled['f1']['legacy_reasons']}`",
            "F1 assignment-only legacy reasons: "
            f"`{pooled['f1']['assignment_only_legacy_reasons']}`",
            f"F1 assignment-only modes: `{pooled['f1']['assignment_only_modes']}`",
            f"F1 assignment-only tiers: `{pooled['f1']['assignment_only_tiers']}`",
            f"Largest N drift contributors: `{n_contributors[:10]}`",
            f"Paired transitions: `{transitions}`",
        ]
    )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
