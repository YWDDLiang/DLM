#!/usr/bin/env python3
"""Finalize the same-checkpoint free-vs-CCFD Planner comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import reduce
import json
from math import comb, gcd, sqrt
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.composition_validity import classify_smact_validity  # noqa: E402
from crystal_dlm.fixed_slot import SYMBOL_TO_Z  # noqa: E402
from crystal_dlm.r5_plan_body import formula_from_symbol_counts, symbol_counts_from_formula  # noqa: E402
from crystal_dlm.r5_plan_state import anion_framework_from_symbols  # noqa: E402
from crystal_dlm.valence_assignment import assign_crysvcd_valences  # noqa: E402


SEEDS = (17, 18)
ARMS = ("f0", "f1")
FORMULA_LINE = re.compile(r"(?im)^\s*formula\s*:\s*([^\s]+)\s*$")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def reduced_counts(counts: list[int]) -> list[int]:
    divisor = reduce(gcd, counts) if counts else 1
    return [int(value) // max(1, int(divisor)) for value in counts]


def tvd(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    left_total = sum(int(value) for value in left.values())
    right_total = sum(int(value) for value in right.values())
    keys = set(left) | set(right)
    if left_total == 0 or right_total == 0:
        return 1.0
    return 0.5 * sum(
        abs(int(left.get(key, 0)) / left_total - int(right.get(key, 0)) / right_total)
        for key in keys
    )


def paired_bootstrap_ci(
    left: list[bool], right: list[bool], *, draws: int = 10_000, seed: int = 20260828
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equally sized nonempty vectors")
    differences = [int(r) - int(l) for l, r in zip(left, right)]
    observed = sum(differences) / len(differences)
    # Use a deterministic normal approximation for large formal cohorts and a
    # deterministic percentile bootstrap for smaller engineering pilots.
    if len(differences) >= 512:
        variance = sum((value - observed) ** 2 for value in differences) / (len(differences) - 1)
        half = 1.959963984540054 * sqrt(variance / len(differences))
        return {"delta": observed, "low": observed - half, "high": observed + half, "method": "paired_normal"}
    rng = random.Random(int(seed))
    samples = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(int(draws))
    )
    return {
        "delta": observed,
        "low": samples[int(0.025 * (len(samples) - 1))],
        "high": samples[int(0.975 * (len(samples) - 1))],
        "method": "paired_percentile_bootstrap",
    }


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict[str, Any]:
    b = sum(l and not r for l, r in zip(left, right))
    c = sum((not l) and r for l, r in zip(left, right))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2.0 * sum(comb(n, index) for index in range(min(b, c) + 1)) / (2**n))
    return {"f0_only": b, "f1_only": c, "discordant": n, "exact_two_sided_p": p}


def extract_formula(row: Mapping[str, Any]) -> tuple[str, list[str], list[int]] | None:
    text = str(row.get("raw_plan_text") or row.get("plan_text") or "")
    match = FORMULA_LINE.search(text)
    if match is None:
        return None
    try:
        symbols, counts = symbol_counts_from_formula(match.group(1))
    except Exception:  # noqa: BLE001
        return None
    if not symbols or sum(counts) > 20:
        return None
    formula = formula_from_symbol_counts(symbols, counts)
    return formula, symbols, counts


def train_formula_set(path: Path) -> set[str]:
    formulas: set[str] = set()
    for row in iter_jsonl(path):
        plan = row.get("plan_state") or row.get("r5_plan_state")
        if isinstance(plan, Mapping) and plan.get("formula"):
            try:
                symbols, counts = symbol_counts_from_formula(str(plan["formula"]))
            except Exception:  # noqa: BLE001
                continue
            formulas.add(formula_from_symbol_counts(symbols, counts))
    return formulas


def summarize_cell(
    arm: str,
    seed: int,
    path: Path,
    *,
    requested: int,
    known_train_formulas: set[str],
    allow_requested_prefix: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_rows = list(iter_jsonl(path))
    if allow_requested_prefix:
        raw_rows = [
            row for row in raw_rows if 0 <= int(row.get("sample_idx", -1)) < requested
        ]
    by_idx = {int(row.get("sample_idx", -1)): row for row in raw_rows}
    if len(by_idx) != requested or sorted(by_idx) != list(range(requested)):
        raise RuntimeError(f"{arm} seed{seed} attempt coverage changed")
    records: list[dict[str, Any]] = []
    formula_counts: Counter[str] = Counter()
    for ordinal in range(requested):
        raw = by_idx[ordinal]
        extracted = extract_formula(raw)
        record: dict[str, Any] = {
            "ordinal": ordinal,
            "formula_valid": False,
            "plan_parsed": raw.get("parsed") is True,
            "assignable": False,
            "legacy_comp_valid": False,
            "novel": False,
            "unique": False,
            "novel_unique": False,
            "formula": None,
            "family": "failed",
            "arity": "failed",
            "n": "failed",
            "all_metal": False,
        }
        if extracted is not None:
            formula, symbols, counts = extracted
            record.update(
                {
                    "formula_valid": True,
                    "formula": formula,
                    "family": anion_framework_from_symbols(symbols),
                    "arity": str(len(symbols)),
                    "n": str(sum(counts)),
                    "novel": formula not in known_train_formulas,
                }
            )
            assignment = assign_crysvcd_valences(symbols, counts, max_species=7)
            record["assignable"] = assignment.get("assigned") is True
            atomic_numbers = [int(SYMBOL_TO_Z[symbol]) for symbol in symbols]
            legacy = classify_smact_validity(atomic_numbers, reduced_counts(counts))
            record["legacy_comp_valid"] = legacy.get("valid") is True
            try:
                import smact

                record["all_metal"] = all(symbol in smact.metals for symbol in symbols)
            except Exception:  # noqa: BLE001
                plan = raw.get("plan_state") or raw.get("parsed_plan") or {}
                record["all_metal"] = str(plan.get("charge_bucket")) == "all_metal"
            formula_counts[formula] += 1
        records.append(record)
    seen: set[str] = set()
    for record in records:
        formula = record.get("formula")
        if isinstance(formula, str) and formula not in seen:
            record["unique"] = True
            seen.add(formula)
        record["novel_unique"] = bool(record["novel"] and record["unique"])

    counts = {
        "requested": requested,
        "formula_valid": sum(row["formula_valid"] for row in records),
        "plan_parsed": sum(row["plan_parsed"] for row in records),
        "assignable": sum(row["assignable"] for row in records),
        "legacy_comp_valid": sum(row["legacy_comp_valid"] for row in records),
        "novel": sum(row["novel"] for row in records),
        "unique": sum(row["unique"] for row in records),
        "novel_unique": sum(row["novel_unique"] for row in records),
        "all_metal": sum(row["all_metal"] for row in records),
    }
    distributions = {
        key: dict(sorted(Counter(str(row[key]) for row in records).items()))
        for key in ("family", "arity", "n")
    }
    summary = {
        "arm": arm,
        "seed": seed,
        **counts,
        **{f"{key}_rate": value / requested for key, value in counts.items() if key != "requested"},
        "unique_formula_count": len(formula_counts),
        "distributions": distributions,
    }
    return summary, records


def pool(cells: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    chosen = [row for row in cells if row["arm"] == arm]
    requested = sum(int(row["requested"]) for row in chosen)
    count_keys = (
        "formula_valid",
        "plan_parsed",
        "assignable",
        "legacy_comp_valid",
        "novel",
        "unique",
        "novel_unique",
        "all_metal",
    )
    result: dict[str, Any] = {"arm": arm, "seed": "pooled", "requested": requested}
    for key in count_keys:
        result[key] = sum(int(row[key]) for row in chosen)
        result[f"{key}_rate"] = result[key] / requested
    result["distributions"] = {}
    for key in ("family", "arity", "n"):
        merged: Counter[str] = Counter()
        for row in chosen:
            merged.update(row["distributions"][key])
        result["distributions"][key] = dict(sorted(merged.items()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--requested", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    known = train_formula_set(args.train_jsonl)
    cells: list[dict[str, Any]] = []
    records: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            summary, rows = summarize_cell(
                arm,
                seed,
                args.run_dir / f"{arm}_seed{seed}/raw_generations.jsonl",
                requested=int(args.requested),
                known_train_formulas=known,
            )
            cells.append(summary)
            records[(arm, seed)] = rows
    pooled = [pool(cells, arm) for arm in ARMS]
    by_arm = {row["arm"]: row for row in pooled}
    f0, f1 = by_arm["f0"], by_arm["f1"]
    drift = {
        "all_metal_abs_pp": abs(f1["all_metal_rate"] - f0["all_metal_rate"]),
        "family_tvd": tvd(f0["distributions"]["family"], f1["distributions"]["family"]),
        "arity_tvd": tvd(f0["distributions"]["arity"], f1["distributions"]["arity"]),
        "n_tvd": tvd(f0["distributions"]["n"], f1["distributions"]["n"]),
        "max_family_abs_pp": max(
            abs(
                f1["distributions"]["family"].get(key, 0) / f1["requested"]
                - f0["distributions"]["family"].get(key, 0) / f0["requested"]
            )
            for key in set(f0["distributions"]["family"]) | set(f1["distributions"]["family"])
        ),
    }
    left = [bool(row["legacy_comp_valid"]) for seed in SEEDS for row in records[("f0", seed)]]
    right = [bool(row["legacy_comp_valid"]) for seed in SEEDS for row in records[("f1", seed)]]
    ci = paired_bootstrap_ci(left, right)
    seed_deltas = {
        str(seed): next(row for row in cells if row["arm"] == "f1" and row["seed"] == seed)["legacy_comp_valid_rate"]
        - next(row for row in cells if row["arm"] == "f0" and row["seed"] == seed)["legacy_comp_valid_rate"]
        for seed in SEEDS
    }
    formal = int(args.requested) == 1000
    gates = {
        "formal_requested1000": formal,
        "f1_assignment_at_least_99pct": f1["assignable_rate"] >= 0.99,
        "independent_comp_strictly_positive": f1["legacy_comp_valid_rate"] > f0["legacy_comp_valid_rate"],
        "all_metal_drift_at_most_3pp": drift["all_metal_abs_pp"] <= 0.03,
        "max_family_drift_at_most_3pp": drift["max_family_abs_pp"] <= 0.03,
        "arity_tvd_at_most_0p05": drift["arity_tvd"] <= 0.05,
        "n_tvd_at_most_0p05": drift["n_tvd"] <= 0.05,
        "novel_noninferior_1pp": f1["novel_rate"] - f0["novel_rate"] >= -0.01,
        "unique_noninferior_1pp": f1["unique_rate"] - f0["unique_rate"] >= -0.01,
        "both_seeds_positive": all(value > 0.0 for value in seed_deltas.values()),
        "paired_ci_lower_positive": ci["low"] > 0.0,
    }
    gates["phase1_pass"] = all(gates.values())
    report = {
        "schema": "h1a2_ccfd_phase1_final_v1",
        "formal": formal,
        "same_checkpoint_and_tokenizer": True,
        "cells": cells,
        "pooled": pooled,
        "drift": drift,
        "seed_independent_comp_deltas": seed_deltas,
        "paired_independent_comp_ci95": ci,
        "mcnemar_independent_comp": exact_mcnemar(left, right),
        "gates": gates,
        "claim_boundary": "composition correctness only; no stability attribution",
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stem = "CCFD_PHASE1_FINAL"
    (output / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [key for key in pooled[0] if not isinstance(pooled[0][key], dict)]
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in pooled)
    lines = [
        "# CCFD Phase 1 same-checkpoint factorial",
        "",
        f"Formal requested1000 run: **{formal}**. Phase 1 pass: **{gates['phase1_pass']}**.",
        "",
        "| Arm | Requested | Formula | Plan parsed | Assignable | Legacy comp-valid | Novel/Unique/NU | All-metal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['arm']} | {row['requested']} | {row['formula_valid']} | {row['plan_parsed']} | "
            f"{row['assignable']} | {row['legacy_comp_valid']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | {row['all_metal']} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
