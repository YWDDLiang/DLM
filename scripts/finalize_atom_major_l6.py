#!/usr/bin/env python3
"""Finalize full/hard conditioning x axis/atom-major L6 factorial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


SEEDS = (17, 18)
ARMS = ("full_axis", "hard_axis", "full_atom", "hard_atom")
ATTEMPTS = 256


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rate(value: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else value / denominator


def summarize(rows: list[dict[str, Any]], direct: dict[str, Any]) -> dict[str, Any]:
    reconstructed = sum(row.get("reconstructed") is True for row in rows)
    counts = {
        "reconstructed": reconstructed,
        "direct_comp": int(direct["composition_valid"]),
        "direct_struct": int(direct["structure_valid"]),
        "direct_joint": int(direct["joint_valid"]),
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
        "direct_joint_rate": rate(counts["direct_joint"], ATTEMPTS),
        "novel_rate": rate(counts["novel"], reconstructed),
        "unique_rate": rate(counts["unique"], reconstructed),
        "strict_attempt_rate": rate(counts["strict_sun"], ATTEMPTS),
        "meta_attempt_rate": rate(counts["meta_sun"], ATTEMPTS),
        "strict_retention": rate(counts["strict_sun"], counts["strict_stable"]),
        "meta_retention": rate(counts["meta_sun"], counts["meta_stable"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-eval-run", type=Path, required=True)
    parser.add_argument("--atom-eval-run", type=Path, required=True)
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
    rows_by_cell: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for seed in SEEDS:
        for arm in ARMS:
            parent = args.condition_eval_run if arm.endswith("axis") else args.atom_eval_run
            root = parent / f"seed{seed}/{arm}"
            rows, report = _evaluate_cell(
                cell_id=f"seed{seed}_{arm}",
                labels_path=root / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                generation_path=root / "generation/generation.jsonl",
                direct_path=root / "evaluation/direct/report.json",
                phase_diagrams=phase_diagrams,
                unresolved=unresolved,
                output_dir=output / f"cells/seed{seed}/{arm}",
            )
            rows_by_cell[(seed, arm)] = rows
            generation = read_json(root / "generation/generation_report.json")
            summary = summarize(rows, report["direct"])
            body = int(generation["body_success"])
            refined = int(generation["refined"])
            summary.update(
                {
                    "seed": seed,
                    "arm": arm,
                    "requested": ATTEMPTS,
                    "body": body,
                    "refined": refined,
                    "body_rate": rate(body, ATTEMPTS),
                }
            )
            cells.append(summary)

    count_keys = (
        "requested",
        "body",
        "refined",
        "reconstructed",
        "direct_comp",
        "direct_struct",
        "direct_joint",
        "novel",
        "unique",
        "novel_unique",
        "hull_known",
        "hull_unknown",
        "strict_stable",
        "strict_sun",
        "meta_stable",
        "meta_sun",
    )
    pooled: list[dict[str, Any]] = []
    for arm in ARMS:
        chosen = [row for row in cells if row["arm"] == arm]
        item: dict[str, Any] = {"seed": "pooled-repeat-sum", "arm": arm}
        for key in count_keys:
            item[key] = sum(int(row[key]) for row in chosen)
        item.update(
            {
                "body_rate": rate(item["body"], item["requested"]),
                "direct_joint_rate": rate(item["direct_joint"], item["requested"]),
                "novel_rate": rate(item["novel"], item["reconstructed"]),
                "unique_rate": rate(item["unique"], item["reconstructed"]),
                "strict_attempt_rate": rate(item["strict_sun"], item["requested"]),
                "meta_attempt_rate": rate(item["meta_sun"], item["requested"]),
                "strict_retention": rate(item["strict_sun"], item["strict_stable"]),
                "meta_retention": rate(item["meta_sun"], item["meta_stable"]),
            }
        )
        pooled.append(item)

    by_arm = {row["arm"]: row for row in pooled}
    control = by_arm["full_axis"]
    rate_keys = (
        "body_rate",
        "direct_joint_rate",
        "novel_rate",
        "unique_rate",
        "strict_attempt_rate",
        "meta_attempt_rate",
        "strict_retention",
        "meta_retention",
    )
    variant_deltas: dict[str, dict[str, float]] = {}
    seed_deltas: dict[str, dict[str, dict[str, float]]] = {}
    gates: dict[str, dict[str, bool]] = {}
    for arm in ("hard_axis", "full_atom", "hard_atom"):
        variant_deltas[arm] = {key: by_arm[arm][key] - control[key] for key in rate_keys}
        seed_deltas[arm] = {}
        for seed in SEEDS:
            left = next(row for row in cells if row["seed"] == seed and row["arm"] == "full_axis")
            right = next(row for row in cells if row["seed"] == seed and row["arm"] == arm)
            seed_deltas[arm][str(seed)] = {key: right[key] - left[key] for key in rate_keys}
        delta = variant_deltas[arm]
        gate = {
            "strict_positive": delta["strict_attempt_rate"] > 0.0,
            "meta_positive": delta["meta_attempt_rate"] > 0.0,
            "body_noninferior_1pp": delta["body_rate"] >= -0.01,
            "direct_noninferior_1pp": delta["direct_joint_rate"] >= -0.01,
            "novel_noninferior_1pp": delta["novel_rate"] >= -0.01,
            "unique_noninferior_1pp": delta["unique_rate"] >= -0.01,
            "strict_retention_noninferior_1pp": delta["strict_retention"] >= -0.01,
            "meta_retention_noninferior_1pp": delta["meta_retention"] >= -0.01,
            "both_seeds_strict_noninferior_1pp": all(
                seed_deltas[arm][str(seed)]["strict_attempt_rate"] >= -0.01
                for seed in SEEDS
            ),
            "both_seeds_meta_noninferior_1pp": all(
                seed_deltas[arm][str(seed)]["meta_attempt_rate"] >= -0.01
                for seed in SEEDS
            ),
        }
        gate["eligible"] = all(gate.values())
        gates[arm] = gate

    selected_arm = "full_axis"
    for arm in ("hard_atom", "hard_axis", "full_atom"):
        if gates[arm]["eligible"]:
            selected_arm = arm
            break

    factorial_effects = {
        "conditioning_hard_minus_full": {
            key: 0.5
            * (by_arm["hard_axis"][key] + by_arm["hard_atom"][key]
               - by_arm["full_axis"][key] - by_arm["full_atom"][key])
            for key in rate_keys
        },
        "schedule_atom_minus_axis": {
            key: 0.5
            * (by_arm["full_atom"][key] + by_arm["hard_atom"][key]
               - by_arm["full_axis"][key] - by_arm["hard_axis"][key])
            for key in rate_keys
        },
        "interaction_hard_x_atom": {
            key: by_arm["hard_atom"][key] - by_arm["hard_axis"][key]
            - by_arm["full_atom"][key] + by_arm["full_axis"][key]
            for key in rate_keys
        },
    }

    mcnemar: dict[str, Any] = {}
    for arm in ("hard_axis", "full_atom", "hard_atom"):
        mcnemar[arm] = {}
        for seed in SEEDS:
            left = {int(row["ordinal"]): row for row in rows_by_cell[(seed, "full_axis")]}
            right = {int(row["ordinal"]): row for row in rows_by_cell[(seed, arm)]}
            known = [
                idx for idx in range(ATTEMPTS)
                if left[idx]["official_hull_status"] == "known"
                and right[idx]["official_hull_status"] == "known"
            ]
            mcnemar[arm][str(seed)] = {
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
        "schema": "h1a2_atom_major_l6_final_v1",
        "trigger": "naive joint-coordinate confidence pool failed body validity",
        "cells": cells,
        "pooled_repeat_sum": pooled,
        "variant_deltas_vs_full_axis": variant_deltas,
        "seed_deltas": seed_deltas,
        "gates": gates,
        "selected_executor_arm": selected_arm,
        "selection_order": ["hard_atom", "hard_axis", "full_atom", "full_axis"],
        "factorial_effects": factorial_effects,
        "mcnemar": mcnemar,
        "unknown_policy": "excluded from hull-known denominators; never mapped to unstable",
    }
    stem = "DLM_ATOM_MAJOR_L6_FINAL"
    (output / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = list(pooled[0])
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in pooled)
    lines = [
        "# DLM atom-major L6 factorial",
        "",
        f"Selected executor arm: **{selected_arm}**.",
        "",
        "| Arm | Requested | Body | Direct J | N/U/NU | Strict stable/SUN | Meta stable/SUN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['arm']} | {row['requested']} | {row['body']} | {row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['strict_stable']}/{row['strict_sun']} | {row['meta_stable']}/{row['meta_sun']} |"
        )
    for arm in ("hard_axis", "full_atom", "hard_atom"):
        lines.extend(["", f"## {arm} gate", ""])
        lines.extend(f"- {key}: `{value}`" for key, value in gates[arm].items())
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
