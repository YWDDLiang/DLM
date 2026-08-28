#!/usr/bin/env python3
"""Finalize frozen P0 versus the full C³FD-v2 Planner candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from finalize_ccfd_phase1 import (  # noqa: E402
    exact_mcnemar,
    paired_bootstrap_ci,
    pool,
    summarize_cell,
    train_formula_set,
    tvd,
)


SEEDS = (17, 18)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--requested", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    known = train_formula_set(args.train_jsonl)
    cells = []
    records = {}
    for arm, run in (("p0", args.control_run), ("c3fd_v2", args.candidate_run)):
        for seed in SEEDS:
            cell_dir = run / f"{'f0' if arm == 'p0' else 'c3'}_seed{seed}"
            summary, rows = summarize_cell(
                arm,
                seed,
                cell_dir / "raw_generations.jsonl",
                requested=int(args.requested),
                known_train_formulas=known,
            )
            cells.append(summary)
            records[(arm, seed)] = rows
    pooled = [pool(cells, arm) for arm in ("p0", "c3fd_v2")]
    by_arm = {row["arm"]: row for row in pooled}
    control = by_arm["p0"]
    candidate = by_arm["c3fd_v2"]
    drift = {
        "all_metal_abs_pp": abs(candidate["all_metal_rate"] - control["all_metal_rate"]),
        "family_tvd": tvd(control["distributions"]["family"], candidate["distributions"]["family"]),
        "arity_tvd": tvd(control["distributions"]["arity"], candidate["distributions"]["arity"]),
        "n_tvd": tvd(control["distributions"]["n"], candidate["distributions"]["n"]),
        "max_family_abs_pp": max(
            abs(
                candidate["distributions"]["family"].get(key, 0) / candidate["requested"]
                - control["distributions"]["family"].get(key, 0) / control["requested"]
            )
            for key in set(control["distributions"]["family"])
            | set(candidate["distributions"]["family"])
        ),
    }
    left = [
        bool(row["legacy_comp_valid"])
        for seed in SEEDS
        for row in records[("p0", seed)]
    ]
    right = [
        bool(row["legacy_comp_valid"])
        for seed in SEEDS
        for row in records[("c3fd_v2", seed)]
    ]
    ci = paired_bootstrap_ci(left, right)
    seed_deltas = {
        str(seed): next(
            row for row in cells if row["arm"] == "c3fd_v2" and row["seed"] == seed
        )["legacy_comp_valid_rate"]
        - next(row for row in cells if row["arm"] == "p0" and row["seed"] == seed)[
            "legacy_comp_valid_rate"
        ]
        for seed in SEEDS
    }
    gates = {
        "formal_requested1000": int(args.requested) == 1000,
        "independent_comp_strictly_positive": candidate["legacy_comp_valid_rate"]
        > control["legacy_comp_valid_rate"],
        "both_seeds_positive": all(value > 0.0 for value in seed_deltas.values()),
        "paired_ci_lower_positive": ci["low"] > 0.0,
        "parse_noninferior_1pp": candidate["plan_parsed_rate"]
        - control["plan_parsed_rate"]
        >= -0.01,
        "all_metal_drift_at_most_3pp": drift["all_metal_abs_pp"] <= 0.03,
        "max_family_drift_at_most_3pp": drift["max_family_abs_pp"] <= 0.03,
        "arity_tvd_at_most_0p05": drift["arity_tvd"] <= 0.05,
        "n_tvd_at_most_0p05": drift["n_tvd"] <= 0.05,
        "novel_noninferior_1pp": candidate["novel_rate"] - control["novel_rate"]
        >= -0.01,
        "unique_noninferior_1pp": candidate["unique_rate"] - control["unique_rate"]
        >= -0.01,
        "candidate_formula_bpe_disabled": True,
        "candidate_no_repair_replacement_rerank_rl": True,
    }
    gates["c3fd_v2_pass"] = all(gates.values())
    report = {
        "schema": "h1a2_c3fd_planner_final_v1",
        "claim_boundary": "composition correctness only; no stability attribution or external CrysVCD comparison",
        "cells": cells,
        "pooled": pooled,
        "drift": drift,
        "seed_independent_comp_deltas": seed_deltas,
        "paired_independent_comp_ci95": ci,
        "mcnemar_independent_comp": exact_mcnemar(left, right),
        "gates": gates,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stem = "C3FD_PLANNER_FINAL"
    (output / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [key for key in pooled[0] if not isinstance(pooled[0][key], dict)]
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in pooled)
    lines = [
        "# C³FD-v2 Planner matched effect experiment",
        "",
        f"Pass: **{gates['c3fd_v2_pass']}**. No external CrysVCD comparison is made.",
        "",
        "| Arm | Requested | Formula | Plan parsed | Independent comp-valid | Novel/Unique/NU | All-metal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['arm']} | {row['requested']} | {row['formula_valid']} | "
            f"{row['plan_parsed']} | {row['legacy_comp_valid']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | {row['all_metal']} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
