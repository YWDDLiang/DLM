#!/usr/bin/env python3
"""Finalize the two-seed coarse model494 intermediate-timestep calibration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


SEEDS = (17, 18)
TAUS = (0, 200, 500, 800)
ATTEMPTS = 256
TARGETS = {"strict_attempt_rate": 0.10, "meta_attempt_rate": 0.50}


def rate(value: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else value / denominator


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def fmt_optional(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def summarize(
    rows: list[dict[str, Any]], direct_joint: int, denominator: int
) -> dict[str, Any]:
    energies = [
        float(row["official_e_above_hull"])
        for row in rows
        if row.get("novel_unique") is True
        and row.get("official_hull_status") == "known"
        and row.get("official_e_above_hull") is not None
    ]
    counts = {
        "reconstructed": sum(row.get("reconstructed") is True for row in rows),
        "direct_joint": direct_joint,
        "novel": sum(row.get("novel") is True for row in rows),
        "unique": sum(row.get("unique_representative") is True for row in rows),
        "novel_unique": sum(row.get("novel_unique") is True for row in rows),
        "hull_known": sum(row.get("official_hull_status") == "known" for row in rows),
        "hull_unknown": sum(
            row.get("reconstructed") is True
            and row.get("official_hull_status") != "known"
            for row in rows
        ),
        "strict_stable": sum(row.get("strict_stable") is True for row in rows),
        "strict_sun": sum(row.get("strict_sun") is True for row in rows),
        "meta_stable": sum(row.get("meta_stable") is True for row in rows),
        "meta_sun": sum(row.get("meta_sun") is True for row in rows),
    }
    return {
        **counts,
        "reconstructed_rate": rate(counts["reconstructed"], denominator),
        "direct_joint_rate": rate(counts["direct_joint"], denominator),
        "novel_rate": rate(counts["novel"], counts["reconstructed"]),
        "unique_rate": rate(counts["unique"], counts["reconstructed"]),
        "strict_attempt_rate": rate(counts["strict_sun"], denominator),
        "meta_attempt_rate": rate(counts["meta_sun"], denominator),
        "strict_retention": rate(counts["strict_sun"], counts["strict_stable"]),
        "meta_retention": rate(counts["meta_sun"], counts["meta_stable"]),
        "e_hull_quantiles": {
            key: quantile(energies, probability)
            for key, probability in (
                ("q10", 0.10),
                ("q25", 0.25),
                ("q50", 0.50),
                ("q75", 0.75),
                ("q90", 0.90),
            )
        },
        "e_hull_threshold_counts": {
            "eq_0": sum(value <= 0.0 for value in energies),
            "le_0p01": sum(value <= 0.01 for value in energies),
            "le_0p05": sum(value <= 0.05 for value in energies),
            "le_0p10": sum(value <= 0.10 for value in energies),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-eval-run", type=Path, required=True)
    parser.add_argument("--tau-eval-run", type=Path, required=True)
    parser.add_argument("--source-arm", choices=("full_axis", "hard_joint"), required=True)
    parser.add_argument("--official-cache-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.eval_runtime.resolve()))
    import protocol  # noqa: PLC0415
    from finalize_official import _evaluate_cell, _exact_mcnemar, _phase_diagrams  # noqa: PLC0415

    if protocol.DENOMINATOR != ATTEMPTS:
        raise RuntimeError("H1_ACTIVE_DENOMINATOR must be 256")
    cache = args.official_cache_run / "official_mp_cache"
    phase_diagrams = _phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    cells: list[dict[str, Any]] = []
    rows_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for seed in SEEDS:
        for tau in TAUS:
            if tau == 0:
                root = args.condition_eval_run / f"seed{seed}/raw_{args.source_arm}"
            elif tau == 800:
                root = args.condition_eval_run / f"seed{seed}/{args.source_arm}"
            else:
                root = args.tau_eval_run / f"seed{seed}/tau{tau}"
            rows, report = _evaluate_cell(
                cell_id=f"seed{seed}_tau{tau}",
                labels_path=root / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                generation_path=root / "generation/generation.jsonl",
                direct_path=root / "evaluation/direct/report.json",
                phase_diagrams=phase_diagrams,
                unresolved=unresolved,
                output_dir=output / f"cells/seed{seed}/tau{tau}",
            )
            rows_by_cell[(seed, tau)] = rows
            summary = summarize(rows, int(report["direct"]["joint_valid"]), ATTEMPTS)
            summary.update({"seed": seed, "tau": tau, "requested": ATTEMPTS})
            cells.append(summary)

    pooled: list[dict[str, Any]] = []
    for tau in TAUS:
        selected_rows = [row for seed in SEEDS for row in rows_by_cell[(seed, tau)]]
        direct_joint = sum(
            int(row["direct_joint"]) for row in cells if int(row["tau"]) == tau
        )
        item = summarize(selected_rows, direct_joint, ATTEMPTS * len(SEEDS))
        item.update({"seed": "pooled-repeat-sum", "tau": tau, "requested": 512})
        pooled.append(item)

    pooled_by_tau = {int(row["tau"]): row for row in pooled}
    baseline = pooled_by_tau[800]
    rate_keys = (
        "reconstructed_rate",
        "direct_joint_rate",
        "novel_rate",
        "unique_rate",
        "strict_attempt_rate",
        "meta_attempt_rate",
        "strict_retention",
        "meta_retention",
    )
    comparisons: dict[str, Any] = {}
    for tau in (0, 200, 500):
        candidate = pooled_by_tau[tau]
        delta = {key: candidate[key] - baseline[key] for key in rate_keys}
        seed_delta: dict[str, dict[str, float]] = {}
        for seed in SEEDS:
            left = next(row for row in cells if row["seed"] == seed and row["tau"] == 800)
            right = next(row for row in cells if row["seed"] == seed and row["tau"] == tau)
            seed_delta[str(seed)] = {key: right[key] - left[key] for key in rate_keys}
        gate = {
            "strict_positive": delta["strict_attempt_rate"] > 0.0,
            "meta_positive": delta["meta_attempt_rate"] > 0.0,
            "reconstructed_noninferior_1pp": delta["reconstructed_rate"] >= -0.01,
            "direct_noninferior_1pp": delta["direct_joint_rate"] >= -0.01,
            "novel_noninferior_1pp": delta["novel_rate"] >= -0.01,
            "unique_noninferior_1pp": delta["unique_rate"] >= -0.01,
            "strict_retention_noninferior_1pp": delta["strict_retention"] >= -0.01,
            "meta_retention_noninferior_1pp": delta["meta_retention"] >= -0.01,
            "both_seeds_strict_noninferior_1pp": all(
                seed_delta[str(seed)]["strict_attempt_rate"] >= -0.01 for seed in SEEDS
            ),
            "both_seeds_meta_noninferior_1pp": all(
                seed_delta[str(seed)]["meta_attempt_rate"] >= -0.01 for seed in SEEDS
            ),
        }
        gate["eligible"] = all(gate.values())
        comparisons[str(tau)] = {"delta_vs_tau800": delta, "seed_delta": seed_delta, "gate": gate}

    eligible = [tau for tau in (0, 200, 500) if comparisons[str(tau)]["gate"]["eligible"]]
    selected_tau = 800
    if eligible:
        selected_tau = max(
            eligible,
            key=lambda tau: (
                min(
                    pooled_by_tau[tau]["strict_attempt_rate"] / TARGETS["strict_attempt_rate"],
                    pooled_by_tau[tau]["meta_attempt_rate"] / TARGETS["meta_attempt_rate"],
                ),
                -tau,
            ),
        )

    mcnemar: dict[str, Any] = {}
    for tau in (0, 200, 500):
        mcnemar[str(tau)] = {}
        for seed in SEEDS:
            left = {int(row["ordinal"]): row for row in rows_by_cell[(seed, 800)]}
            right = {int(row["ordinal"]): row for row in rows_by_cell[(seed, tau)]}
            known = [
                idx
                for idx in range(ATTEMPTS)
                if left[idx]["official_hull_status"] == "known"
                and right[idx]["official_hull_status"] == "known"
            ]
            mcnemar[str(tau)][str(seed)] = {
                "known_both": len(known),
                "strict": _exact_mcnemar(
                    [bool(left[idx]["strict_sun"]) for idx in known],
                    [bool(right[idx]["strict_sun"]) for idx in known],
                ),
                "meta": _exact_mcnemar(
                    [bool(left[idx]["meta_sun"]) for idx in known],
                    [bool(right[idx]["meta_sun"]) for idx in known],
                ),
            }

    report = {
        "schema": "h1a2_refiner_tau_l6_final_v1",
        "source_arm": args.source_arm,
        "taus": list(TAUS),
        "tau800_control": True,
        "cells": cells,
        "pooled_repeat_sum": pooled,
        "comparisons": comparisons,
        "selected_tau": selected_tau,
        "selection_rule": "eligible candidate maximizing min(Strict/10%, Meta/50%) progress; ties prefer lower tau",
        "mcnemar": mcnemar,
        "unknown_policy": "excluded from hull-known denominators; never mapped to unstable",
    }
    stem = "DLM_REFINER_TAU_L6_FINAL"
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
        "# model494 intermediate-timestep L6 calibration",
        "",
        f"Source arm: `{args.source_arm}`. Selected tau: **{selected_tau}**.",
        "",
        "| tau | Requested | Reconstructed | Direct J | N/U/NU | Strict stable/SUN | Meta stable/SUN | E_hull q50 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['tau']} | {row['requested']} | {row['reconstructed']} | "
            f"{row['direct_joint']} | {row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['strict_stable']}/{row['strict_sun']} | {row['meta_stable']}/{row['meta_sun']} | "
            f"{fmt_optional(row['e_hull_quantiles']['q50'])} |"
        )
    for tau in (0, 200, 500):
        lines.extend(["", f"## tau{tau} versus tau800", ""])
        lines.extend(
            f"- {key}: `{value}`" for key, value in comparisons[str(tau)]["gate"].items()
        )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
