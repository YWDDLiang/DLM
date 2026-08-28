#!/usr/bin/env python3
"""Finalize the two-seed requested256 C³FD-v2.1 pilot."""

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


def parsed_distribution(rows: list[dict], key: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(row[key]) for row in rows if row.get("formula_valid")).items()
        )
    )


def ionic_rate(rows: list[dict]) -> dict[str, float | int]:
    ionic = [
        row
        for row in rows
        if row.get("formula_valid")
        and not row.get("all_metal")
        and str(row.get("arity")) != "1"
    ]
    valid = sum(bool(row.get("legacy_comp_valid")) for row in ionic)
    return {
        "denominator": len(ionic),
        "valid": valid,
        "rate": 0.0 if not ionic else valid / len(ionic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--v2-diagnostic", type=Path, required=True)
    parser.add_argument("--requested", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if int(args.requested) != 256:
        raise ValueError("C3FD-v2.1 pilot is frozen at requested256")
    known = train_formula_set(args.train_jsonl)
    cells = []
    records = {}
    for arm, run in (("p0", args.control_run), ("c3fd_v21", args.candidate_run)):
        for seed in SEEDS:
            prefix = "f0" if arm == "p0" else "c3"
            summary, rows = summarize_cell(
                arm,
                seed,
                run / f"{prefix}_seed{seed}/raw_generations.jsonl",
                requested=int(args.requested),
                known_train_formulas=known,
                allow_requested_prefix=arm == "p0",
            )
            cells.append(summary)
            records[(arm, seed)] = rows
    pooled = [pool(cells, arm) for arm in ("p0", "c3fd_v21")]
    by_arm = {row["arm"]: row for row in pooled}
    control = by_arm["p0"]
    candidate = by_arm["c3fd_v21"]
    pooled_records = {
        arm: [row for seed in SEEDS for row in records[(arm, seed)]]
        for arm in ("p0", "c3fd_v21")
    }
    distributions = {
        arm: {
            key: parsed_distribution(pooled_records[arm], key)
            for key in ("family", "arity", "n")
        }
        for arm in pooled_records
    }
    diagnostic = json.loads(args.v2_diagnostic.read_text(encoding="utf-8"))
    train_distributions = diagnostic["groups"]["train_full"]["distributions"]
    distance_to_train = {
        arm: {
            "family": tvd(distributions[arm]["family"], train_distributions["family"]),
            "arity": tvd(distributions[arm]["arity"], train_distributions["arity"]),
            "N": tvd(distributions[arm]["n"], train_distributions["N"]),
        }
        for arm in distributions
    }
    parsed_candidate = [row for row in pooled_records["c3fd_v21"] if row["formula_valid"]]
    candidate_all_metal_rate = (
        0.0
        if not parsed_candidate
        else sum(bool(row["all_metal"]) for row in parsed_candidate)
        / len(parsed_candidate)
    )
    train_all_metal_rate = float(diagnostic["groups"]["train_full"]["all_metal_rate"])
    ionic = {arm: ionic_rate(rows) for arm, rows in pooled_records.items()}
    left = [
        bool(row["legacy_comp_valid"])
        for seed in SEEDS
        for row in records[("p0", seed)]
    ]
    right = [
        bool(row["legacy_comp_valid"])
        for seed in SEEDS
        for row in records[("c3fd_v21", seed)]
    ]
    ci = paired_bootstrap_ci(left, right)
    seed_deltas = {
        str(seed): next(
            row for row in cells if row["arm"] == "c3fd_v21" and row["seed"] == seed
        )["legacy_comp_valid_rate"]
        - next(row for row in cells if row["arm"] == "p0" and row["seed"] == seed)[
            "legacy_comp_valid_rate"
        ]
        for seed in SEEDS
    }
    gates = {
        "requested256_frozen": int(args.requested) == 256,
        "pooled_comp_valid_positive": candidate["legacy_comp_valid_rate"]
        > control["legacy_comp_valid_rate"],
        "both_seed_comp_valid_positive": all(value > 0 for value in seed_deltas.values()),
        "paired_ci_lower_positive": ci["low"] > 0.0,
        "ionic_comp_valid_positive": float(ionic["c3fd_v21"]["rate"])
        > float(ionic["p0"]["rate"]),
        "parse_noninferior_1pp": candidate["plan_parsed_rate"]
        - control["plan_parsed_rate"]
        >= -0.01,
        "novel_unique_noninferior_1pp": candidate["novel_unique_rate"]
        - control["novel_unique_rate"]
        >= -0.01,
        "all_metal_within_3pp_of_full_train": abs(
            candidate_all_metal_rate - train_all_metal_rate
        )
        <= 0.03,
        "N_distance_not_worse_than_p0_plus_0p01": distance_to_train["c3fd_v21"]["N"]
        <= distance_to_train["p0"]["N"] + 0.01,
        "arity_distance_not_worse_than_p0_plus_0p01": distance_to_train["c3fd_v21"]["arity"]
        <= distance_to_train["p0"]["arity"] + 0.01,
        "family_distance_not_worse_than_p0_plus_0p01": distance_to_train["c3fd_v21"]["family"]
        <= distance_to_train["p0"]["family"] + 0.01,
    }
    gates["step5_pass"] = all(gates.values())
    report = {
        "schema": "h1a2_c3fd_v21_pilot_final_v1",
        "claim_boundary": "requested256 promotion screen only; not a headline result",
        "cells": cells,
        "pooled": pooled,
        "seed_comp_valid_deltas": seed_deltas,
        "paired_comp_valid_ci95": ci,
        "mcnemar_comp_valid": exact_mcnemar(left, right),
        "ionic": ionic,
        "candidate_all_metal_rate_parsed": candidate_all_metal_rate,
        "full_train_all_metal_rate": train_all_metal_rate,
        "distributions": distributions,
        "distance_to_full_train": distance_to_train,
        "gates": gates,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stem = "C3FD_V21_PILOT_FINAL"
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
        "# C³FD-v2.1 requested256 pilot",
        "",
        f"Step 5 pass: **{gates['step5_pass']}**. This is not a headline run.",
        "",
        "| Arm | Requested | Parsed | Comp-valid | Novel/Unique/NU | All-metal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['arm']} | {row['requested']} | {row['plan_parsed']} | "
            f"{row['legacy_comp_valid']} | {row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['all_metal']} |"
        )
    lines.extend(["", "## Distance to full train", "", f"`{distance_to_train}`", "", "## Gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
