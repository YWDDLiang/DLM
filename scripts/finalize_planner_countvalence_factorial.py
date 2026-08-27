#!/usr/bin/env python3
"""Finalize the count-fields versus count-valence Planner pre-downstream gate."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.r5_plan_state import validate_plan_state  # noqa: E402
from crystal_dlm.valence_assignment import annotate_plan_with_valence  # noqa: E402


ARMS = ("p0", "countfields", "countvalence")
SEEDS = (17, 18)
LATTICE_TO_SPACEGROUP = {
    "triclinic": "sg_001_002",
    "monoclinic": "sg_003_015",
    "orthorhombic": "sg_016_074",
    "tetragonal": "sg_075_142",
    "trigonal": "sg_143_167",
    "hexagonal": "sg_168_194",
    "cubic": "sg_195_230",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(value: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else value / denominator


def n_bin(value: int) -> str:
    if value <= 4:
        return "01_04"
    if value <= 8:
        return "05_08"
    if value <= 12:
        return "09_12"
    return "13_20"


def summarize_plans(
    *,
    arm: str,
    seed: int | str,
    requested: int,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    valid = 0
    assignable = 0
    emitted_known = 0
    emitted_neutral = 0
    lattice_sg = 0
    formulas: Counter[str] = Counter()
    plans: Counter[str] = Counter()
    distributions: dict[str, Counter[str]] = {
        key: Counter()
        for key in ("family", "charge", "lattice", "spacegroup", "volume", "arity", "n_bin")
    }
    for row in rows:
        plan = row.get("plan_state") or row.get("parsed_plan")
        if not isinstance(plan, dict):
            continue
        validation = validate_plan_state(plan)
        if not validation.valid:
            continue
        valid += 1
        assignment = annotate_plan_with_valence(plan).get("valence_assignment") or {}
        assignable += int(assignment.get("assigned") is True)
        if plan.get("generated_charge_sum_known") is True:
            emitted_known += 1
            emitted_neutral += int(int(plan.get("generated_charge_sum") or 0) == 0)
        lattice_sg += int(
            LATTICE_TO_SPACEGROUP.get(str(plan.get("lattice_system")))
            == str(plan.get("spacegroup_bucket"))
        )
        formula = str(plan.get("formula"))
        formulas[formula] += 1
        plans[str(row.get("plan_text") or formula)] += 1
        distributions["family"][str(plan.get("anion_framework"))] += 1
        distributions["charge"][str(plan.get("charge_bucket"))] += 1
        distributions["lattice"][str(plan.get("lattice_system"))] += 1
        distributions["spacegroup"][str(plan.get("spacegroup_bucket"))] += 1
        distributions["volume"][str(plan.get("volume_per_atom_bin"))] += 1
        distributions["arity"][str(len(plan.get("elements") or []))] += 1
        distributions["n_bin"][n_bin(int(plan.get("N") or 0))] += 1
    parsed = len(rows)
    all_metal = distributions["charge"].get("all_metal", 0)
    oxide = distributions["family"].get("oxide", 0)
    return {
        "arm": arm,
        "seed": seed,
        "requested": requested,
        "parsed": parsed,
        "composition_valid": valid,
        "physics_assignable": assignable,
        "emitted_charge_known": emitted_known,
        "emitted_charge_neutral": emitted_neutral,
        "lattice_spacegroup_match": lattice_sg,
        "unique_formula": len(formulas),
        "unique_plan": len(plans),
        "_formula_values": sorted(formulas),
        "_plan_values": sorted(plans),
        "parse_rate": rate(parsed, requested),
        "composition_valid_rate": rate(valid, requested),
        "physics_assignable_rate": rate(assignable, requested),
        "emitted_charge_neutral_rate": (
            rate(emitted_neutral, parsed) if arm == "countvalence" else None
        ),
        "lattice_spacegroup_match_rate": rate(lattice_sg, parsed),
        "unique_formula_rate": rate(len(formulas), parsed),
        "all_metal_rate": rate(all_metal, parsed),
        "oxide_rate": rate(oxide, parsed),
        "distributions": {
            key: dict(sorted(value.items())) for key, value in distributions.items()
        },
    }


def pooled(cells: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in cells if row["arm"] == arm]
    requested = sum(int(row["requested"]) for row in selected)
    count_keys = (
        "parsed",
        "composition_valid",
        "physics_assignable",
        "emitted_charge_known",
        "emitted_charge_neutral",
        "lattice_spacegroup_match",
    )
    result: dict[str, Any] = {
        "arm": arm,
        "seed": "pooled",
        "requested": requested,
    }
    for key in count_keys:
        result[key] = sum(int(row[key]) for row in selected)
    result["unique_formula"] = len(
        {value for row in selected for value in row["_formula_values"]}
    )
    result["unique_plan"] = len(
        {value for row in selected for value in row["_plan_values"]}
    )
    result["parse_rate"] = rate(result["parsed"], requested)
    result["composition_valid_rate"] = rate(result["composition_valid"], requested)
    result["physics_assignable_rate"] = rate(result["physics_assignable"], requested)
    result["emitted_charge_neutral_rate"] = (
        rate(result["emitted_charge_neutral"], result["parsed"])
        if arm == "countvalence"
        else None
    )
    result["lattice_spacegroup_match_rate"] = rate(
        result["lattice_spacegroup_match"], result["parsed"]
    )
    result["unique_formula_rate"] = rate(result["unique_formula"], result["parsed"])
    for key in ("all_metal_rate", "oxide_rate"):
        result[key] = sum(float(row[key]) * int(row["parsed"]) for row in selected) / max(
            1, result["parsed"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--p0-seed17", type=Path, required=True)
    parser.add_argument("--p0-seed18", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = read_json(args.audit / "PLANNER_COUNTVALENCE_AUDIT.json")
    if audit.get("gate", {}).get("candidate_training_authorized") is not True:
        raise RuntimeError("coverage audit did not authorize the candidate")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    cells: list[dict[str, Any]] = []
    p0_paths = {17: args.p0_seed17, 18: args.p0_seed18}
    for seed in SEEDS:
        p0_rows = read_jsonl(p0_paths[seed] / "plans_for_dlm.jsonl")
        p0_metrics = read_json(p0_paths[seed] / "sample_metrics.json")
        cells.append(
            summarize_plans(
                arm="p0",
                seed=seed,
                requested=int(p0_metrics.get("requested_samples") or 256),
                rows=p0_rows,
            )
        )
        for arm in ("countfields", "countvalence"):
            root = args.run / f"{arm}_seed{seed}"
            rows = read_jsonl(root / "plans" / "plans_for_dlm.jsonl")
            metrics = read_json(root / "plans" / "sample_metrics.json")
            row = summarize_plans(
                arm=arm,
                seed=seed,
                requested=int(metrics["requested"]),
                rows=rows,
            )
            row["final_val_nll"] = float(
                read_json(root / "train" / "train_metrics.json")["final_eval_loss"]
            )
            cells.append(row)

    pooled_rows = [pooled(cells, arm) for arm in ARMS]
    by_arm = {row["arm"]: row for row in pooled_rows}
    cv = by_arm["countvalence"]
    cf = by_arm["countfields"]
    p0 = by_arm["p0"]
    gates = {
        "coverage_audit": True,
        "both_seed_parse_at_least_250": all(
            row["parsed"] >= 250
            for row in cells
            if row["arm"] == "countvalence"
        ),
        "parse_noninferior_to_countfields_1pp": cv["parse_rate"] >= cf["parse_rate"] - 0.01,
        "parse_noninferior_to_p0_1pp": cv["parse_rate"] >= p0["parse_rate"] - 0.01,
        "assignability_noninferior_to_countfields": (
            cv["physics_assignable_rate"] >= cf["physics_assignable_rate"]
        ),
        "emitted_neutral_at_least_90pct": (
            cv["emitted_charge_neutral_rate"] is not None
            and cv["emitted_charge_neutral_rate"] >= 0.90
        ),
        "lattice_spacegroup_at_least_95pct": cv["lattice_spacegroup_match_rate"] >= 0.95,
        "lattice_spacegroup_noninferior_1pp": (
            cv["lattice_spacegroup_match_rate"]
            >= cf["lattice_spacegroup_match_rate"] - 0.01
        ),
        "unique_formula_noninferior_5pp": (
            cv["unique_formula_rate"] >= cf["unique_formula_rate"] - 0.05
        ),
        "no_all_metal_shortcut_10pp": abs(cv["all_metal_rate"] - p0["all_metal_rate"]) <= 0.10,
        "no_oxide_shortcut_10pp": abs(cv["oxide_rate"] - p0["oxide_rate"]) <= 0.10,
    }
    gates["downstream_authorized"] = all(gates.values())
    summary = {
        "schema": "h1a2_planner_countvalence_factorial_final_v1",
        "design": {
            "p0": "original rich-Plan checkpoint and frozen seed17/18 samples",
            "matched_control": "countfields without oxidation labels",
            "candidate": "countvalence with CrysVCD-first tiered catalog",
            "seeds": list(SEEDS),
            "requested_per_cell": 256,
            "rl": False,
            "rerank": False,
            "repair": False,
        },
        "audit_gate": audit["gate"],
        "cells": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in cells
        ],
        "pooled": pooled_rows,
        "gates": gates,
    }
    stem = "PLANNER_COUNTVALENCE_FACTORIAL_FINAL"
    (output / f"{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    flat_keys = [key for key in pooled_rows[0] if key != "distributions"]
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in flat_keys} for row in pooled_rows)
    lines = [
        "# Planner count-valence factorial pre-downstream screen",
        "",
        f"Downstream authorized: **{gates['downstream_authorized']}**",
        "",
        "| Arm | Requested | Parsed | Assignable | Emitted neutral | Lattice-SG | Unique formula | All-metal | Oxide |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled_rows:
        emitted = (
            "NA"
            if row["emitted_charge_neutral_rate"] is None
            else f"{row['emitted_charge_neutral_rate']:.2%}"
        )
        lines.append(
            f"| {row['arm']} | {row['requested']} | {row['parsed']} | "
            f"{row['physics_assignable_rate']:.2%} | {emitted} | "
            f"{row['lattice_spacegroup_match_rate']:.2%} | "
            f"{row['unique_formula_rate']:.2%} | {row['all_metal_rate']:.2%} | "
            f"{row['oxide_rate']:.2%} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
