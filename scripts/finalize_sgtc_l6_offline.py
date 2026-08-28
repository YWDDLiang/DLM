#!/usr/bin/env python3
"""Finalize matched SGTC L6 Direct/CHGNet evidence before official hull."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crystal_dlm.sgtc_eval import (  # noqa: E402
    paired_energy_stats,
    quantile,
    rate_delta_pp,
)


ARMS = ("base", "g0_all", "g1_strict")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cell_summary(eval_run: Path, seed: int, arm: str, *, raw: bool) -> dict[str, Any]:
    directory = f"raw_{arm}" if raw else arm
    root = eval_run / f"seed{seed}/{directory}/evaluation"
    direct = json.loads((root / "direct/report.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (root / "full_reconstructed/summary.json").read_text(encoding="utf-8")
    )
    labels = read_jsonl(root / "full_reconstructed/attempt_labels_preofficial.jsonl")
    by_ordinal = {int(row["ordinal"]): row for row in labels}
    if len(labels) != 256 or set(by_ordinal) != set(range(256)):
        raise ValueError(f"SGTC {seed}/{directory} labels changed denominator")
    energies = {
        (int(seed), ordinal): float(row["chgnet_energy_per_atom"])
        for ordinal, row in by_ordinal.items()
        if row.get("chgnet_relaxation_known") is True
        and row.get("chgnet_energy_per_atom") is not None
    }
    metrics = direct["metrics_unchanged_upstream"]
    return {
        "seed": int(seed),
        "arm": arm,
        "stage": "raw" if raw else "refined",
        "requested": 256,
        "generation_succeeded": int(summary["generation_succeeded"]),
        "direct_comp_valid": int(direct["comp_valid_count"]),
        "direct_struct_valid": int(direct["struct_valid_count"]),
        "direct_joint_valid": int(direct["valid_count"]),
        "reconstructed": int(summary["reconstructed"]),
        "novel": int(summary["novel"]),
        "unique": int(summary["unique_representatives"]),
        "novel_unique": int(summary["novel_unique"]),
        "chgnet_known": len(energies),
        "energy_q10": quantile(list(energies.values()), 0.10),
        "energy_q25": quantile(list(energies.values()), 0.25),
        "energy_q50": quantile(list(energies.values()), 0.50),
        "energy_q75": quantile(list(energies.values()), 0.75),
        "energy_q90": quantile(list(energies.values()), 0.90),
        "coverage_precision": metrics.get("cov_precision"),
        "coverage_recall": metrics.get("cov_recall"),
        "energies": energies,
    }


def pooled(cells: list[dict[str, Any]], arm: str, stage: str) -> dict[str, Any]:
    selected = [row for row in cells if row["arm"] == arm and row["stage"] == stage]
    if len(selected) != 2:
        raise ValueError(f"SGTC pooled {stage}/{arm} requires two seeds")
    energies = {}
    for row in selected:
        energies.update(row["energies"])
    output = {"arm": arm, "stage": stage}
    for key in (
        "requested",
        "generation_succeeded",
        "direct_comp_valid",
        "direct_struct_valid",
        "direct_joint_valid",
        "reconstructed",
        "novel",
        "unique",
        "novel_unique",
        "chgnet_known",
    ):
        output[key] = sum(int(row[key]) for row in selected)
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        output[f"energy_q{int(q * 100):02d}"] = quantile(list(energies.values()), q)
    output["energies"] = energies
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.eval_run / "_OFFLINE_SUCCESS").is_file():
        raise ValueError("SGTC offline evaluation is incomplete")

    cells = [
        cell_summary(args.eval_run, seed, arm, raw=raw)
        for raw in (False, True)
        for seed in (17, 18)
        for arm in ARMS
    ]
    pooled_rows = [pooled(cells, arm, stage) for stage in ("refined", "raw") for arm in ARMS]
    pooled_by = {(row["stage"], row["arm"]): row for row in pooled_rows}
    comparisons = {}
    for stage in ("refined", "raw"):
        base = pooled_by[(stage, "base")]
        for arm in ("g0_all", "g1_strict"):
            candidate = pooled_by[(stage, arm)]
            stats = paired_energy_stats(
                candidate["energies"],
                base["energies"],
                seed=82017 + (0 if arm == "g0_all" else 100) + (0 if stage == "refined" else 10),
            )
            stats.update(
                {
                    "body_delta_pp": rate_delta_pp(
                        candidate["generation_succeeded"],
                        base["generation_succeeded"],
                        512,
                    ),
                    "direct_delta_pp": rate_delta_pp(
                        candidate["direct_joint_valid"], base["direct_joint_valid"], 512
                    ),
                    "novel_delta_pp": rate_delta_pp(candidate["novel"], base["novel"], 512),
                    "unique_delta_pp": rate_delta_pp(candidate["unique"], base["unique"], 512),
                    "nu_delta_pp": rate_delta_pp(
                        candidate["novel_unique"], base["novel_unique"], 512
                    ),
                }
            )
            comparisons[f"{stage}:{arm}-base"] = stats
    comparisons["refined:g1_strict-g0_all"] = paired_energy_stats(
        pooled_by[("refined", "g1_strict")]["energies"],
        pooled_by[("refined", "g0_all")]["energies"],
        seed=82317,
    )

    candidate_gates = {}
    for arm in ("g0_all", "g1_strict"):
        stats = comparisons[f"refined:{arm}-base"]
        candidate_gates[arm] = {
            "paired_energy_direction": stats["mean_delta"] is not None
            and stats["mean_delta"] < 0.0
            and stats["fraction_candidate_lower"] > 0.50,
            "body_floor": stats["body_delta_pp"] >= -3.0,
            "direct_floor": stats["direct_delta_pp"] >= -3.0,
            "novel_floor": stats["novel_delta_pp"] >= -5.0,
            "unique_floor": stats["unique_delta_pp"] >= -5.0,
        }
        candidate_gates[arm]["preofficial_go"] = all(candidate_gates[arm].values())
    fresh_official_recommended = any(
        value["preofficial_go"] for value in candidate_gates.values()
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "SGTC_L6_OFFLINE_CELLS.csv"
    public_cells = [{key: value for key, value in row.items() if key != "energies"} for row in cells]
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_cells[0]))
        writer.writeheader()
        writer.writerows(public_cells)
    report = {
        "schema": "h1a2_sgtc_l6_offline_final_v1",
        "cells": public_cells,
        "pooled": [
            {key: value for key, value in row.items() if key != "energies"}
            for row in pooled_rows
        ],
        "comparisons": comparisons,
        "candidate_gates": candidate_gates,
        "fresh_official_recommended": fresh_official_recommended,
        "selection_policy": "both arms disclosed; no NLL selection",
    }
    (output / "SGTC_L6_OFFLINE_FINAL.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# SGTC-DLM-v1 L6 offline result",
        "",
        "All values use the frozen requested denominator; official hull is still pending.",
        "",
        "| Stage | Arm | Body | Direct | Reconstructed | Novel/Unique/NU | CHGNet known | median E |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["pooled"]:
        lines.append(
            f"| {row['stage']} | {row['arm']} | {row['generation_succeeded']}/512 | "
            f"{row['direct_joint_valid']}/512 | {row['reconstructed']}/512 | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['chgnet_known']} | {row['energy_q50']} |"
        )
    lines.extend(["", f"Fresh official recommended: `{fresh_official_recommended}`."])
    (output / "SGTC_L6_OFFLINE_FINAL.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
