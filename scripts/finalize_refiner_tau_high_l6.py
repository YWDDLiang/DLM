#!/usr/bin/env python3
"""Finalize the explicitly authorized tau900/1000 extension with all prior taus."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from finalize_refiner_tau_l6 import SEEDS, ATTEMPTS, summarize


TAUS = (0, 200, 500, 800, 900, 1000)
TARGETS = {"strict_attempt_rate": 0.10, "meta_attempt_rate": 0.50}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-eval-run", type=Path, required=True)
    parser.add_argument("--low-tau-eval-run", type=Path, required=True)
    parser.add_argument("--high-tau-eval-run", type=Path, required=True)
    parser.add_argument("--source-arm", choices=("full_axis",), required=True)
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
            elif tau in (200, 500):
                root = args.low_tau_eval_run / f"seed{seed}/tau{tau}"
            else:
                root = args.high_tau_eval_run / f"seed{seed}/tau{tau}"
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
        direct_joint = sum(int(row["direct_joint"]) for row in cells if int(row["tau"]) == tau)
        item = summarize(selected_rows, direct_joint, ATTEMPTS * len(SEEDS))
        item.update({"seed": "pooled-repeat-sum", "tau": tau, "requested": 512})
        pooled.append(item)
    by_tau = {int(row["tau"]): row for row in pooled}
    baseline = by_tau[800]
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
    for tau in (900, 1000):
        candidate = by_tau[tau]
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

    eligible = [tau for tau in (900, 1000) if comparisons[str(tau)]["gate"]["eligible"]]
    selected_tau = 800
    if eligible:
        selected_tau = max(
            eligible,
            key=lambda tau: (
                min(
                    by_tau[tau]["strict_attempt_rate"] / TARGETS["strict_attempt_rate"],
                    by_tau[tau]["meta_attempt_rate"] / TARGETS["meta_attempt_rate"],
                ),
                -tau,
            ),
        )

    mcnemar: dict[str, Any] = {}
    for tau in (900, 1000):
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
        "schema": "h1a2_refiner_tau_high_l6_final_v1",
        "user_authorized_extension": [900, 1000],
        "source_arm": args.source_arm,
        "taus": list(TAUS),
        "cells": cells,
        "pooled_repeat_sum": pooled,
        "comparisons_high_vs_tau800": comparisons,
        "selected_tau": selected_tau,
        "selection_rule": "same frozen gate versus tau800; eligible candidate maximizing min(Strict/10%, Meta/50%), ties prefer lower tau",
        "mcnemar": mcnemar,
        "unknown_policy": "excluded from hull-known denominators; never unstable",
    }
    stem = "DLM_REFINER_TAU_HIGH_L6_FINAL"
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
        "# model494 high-timestep L6 extension",
        "",
        f"Selected tau: **{selected_tau}**.",
        "",
        "| tau | Requested | Reconstructed | Direct J | N/U/NU | Strict stable/SUN | Meta stable/SUN | E_hull q50 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        q50 = row["e_hull_quantiles"]["q50"]
        lines.append(
            f"| {row['tau']} | {row['requested']} | {row['reconstructed']} | {row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['strict_stable']}/{row['strict_sun']} | {row['meta_stable']}/{row['meta_sun']} | "
            f"{'NA' if q50 is None else f'{q50:.4f}'} |"
        )
    for tau in (900, 1000):
        lines.extend(["", f"## tau{tau} versus tau800", ""])
        lines.extend(f"- {key}: `{value}`" for key, value in comparisons[str(tau)]["gate"].items())
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
