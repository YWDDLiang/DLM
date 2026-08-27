#!/usr/bin/env python3
"""Finalize the two-seed conditioning x coordinate-schedule L6 factorial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


SEEDS = (17, 18)
ARMS = ("full_axis", "full_joint", "hard_axis", "hard_joint")
ATTEMPTS = 256


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rate(value: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else value / denominator


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reconstructed = sum(row.get("reconstructed") is True for row in rows)
    novel = sum(row.get("novel") is True for row in rows)
    unique = sum(row.get("unique_representative") is True for row in rows)
    novel_unique = sum(row.get("novel_unique") is True for row in rows)
    hull_known = sum(row.get("official_hull_status") == "known" for row in rows)
    hull_unknown = sum(
        row.get("reconstructed") is True and row.get("official_hull_status") != "known"
        for row in rows
    )
    strict_stable = sum(row.get("strict_stable") is True for row in rows)
    meta_stable = sum(row.get("meta_stable") is True for row in rows)
    strict_sun = sum(row.get("strict_sun") is True for row in rows)
    meta_sun = sum(row.get("meta_sun") is True for row in rows)
    return {
        "reconstructed": reconstructed,
        "novel": novel,
        "unique": unique,
        "novel_unique": novel_unique,
        "hull_known": hull_known,
        "hull_unknown": hull_unknown,
        "strict_stable": strict_stable,
        "strict_sun": strict_sun,
        "meta_stable": meta_stable,
        "meta_sun": meta_sun,
        "novel_rate": rate(novel, reconstructed),
        "unique_rate": rate(unique, reconstructed),
        "strict_attempt_rate": rate(strict_sun, len(rows)),
        "meta_attempt_rate": rate(meta_sun, len(rows)),
        "strict_retention": rate(strict_sun, strict_stable),
        "meta_retention": rate(meta_sun, meta_stable),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--generation-run", type=Path, required=True)
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
            cell = args.eval_run / f"seed{seed}/{arm}"
            rows, report = _evaluate_cell(
                cell_id=f"seed{seed}_{arm}",
                labels_path=cell / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                generation_path=cell / "generation/generation.jsonl",
                direct_path=cell / "evaluation/direct/report.json",
                phase_diagrams=phase_diagrams,
                unresolved=unresolved,
                output_dir=output / f"cells/seed{seed}/{arm}",
            )
            rows_by_cell[(seed, arm)] = rows
            body = read_json(args.generation_run / f"seed{seed}/{arm}/body/sample_metrics.json")
            refine = read_json(args.generation_run / f"seed{seed}/{arm}/refine/refinement_metrics.json")
            summary = summarize_rows(rows)
            summary.update(
                {
                    "seed": seed,
                    "arm": arm,
                    "requested": ATTEMPTS,
                    "parsed": int(body["parse_success"]),
                    "body": int(body["graph_success"]),
                    "refined": int(refine["num_proposals"]),
                    "direct_comp": int(report["direct"]["composition_valid"]),
                    "direct_struct": int(report["direct"]["structure_valid"]),
                    "direct_joint": int(report["direct"]["joint_valid"]),
                    "parse_rate": rate(int(body["parse_success"]), ATTEMPTS),
                    "body_rate": rate(int(body["graph_success"]), ATTEMPTS),
                    "direct_joint_rate": rate(int(report["direct"]["joint_valid"]), ATTEMPTS),
                }
            )
            cells.append(summary)

    pooled: list[dict[str, Any]] = []
    count_keys = (
        "requested",
        "parsed",
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
    for arm in ARMS:
        selected = [row for row in cells if row["arm"] == arm]
        item: dict[str, Any] = {"seed": "pooled-repeat-sum", "arm": arm}
        for key in count_keys:
            item[key] = sum(int(row[key]) for row in selected)
        item.update(
            {
                "parse_rate": rate(item["parsed"], item["requested"]),
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

    pooled_by_arm = {row["arm"]: row for row in pooled}
    control = pooled_by_arm["full_axis"]
    primary = pooled_by_arm["hard_joint"]
    rate_keys = (
        "parse_rate",
        "body_rate",
        "direct_joint_rate",
        "novel_rate",
        "unique_rate",
        "strict_attempt_rate",
        "meta_attempt_rate",
        "strict_retention",
        "meta_retention",
    )
    primary_delta = {key: primary[key] - control[key] for key in rate_keys}
    seed_deltas: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        baseline = next(row for row in cells if row["seed"] == seed and row["arm"] == "full_axis")
        candidate = next(row for row in cells if row["seed"] == seed and row["arm"] == "hard_joint")
        seed_deltas[str(seed)] = {key: candidate[key] - baseline[key] for key in rate_keys}

    gate = {
        "pooled_strict_positive": primary_delta["strict_attempt_rate"] > 0.0,
        "pooled_meta_positive": primary_delta["meta_attempt_rate"] > 0.0,
        "parse_noninferior_1pp": primary_delta["parse_rate"] >= -0.01,
        "body_noninferior_1pp": primary_delta["body_rate"] >= -0.01,
        "direct_noninferior_1pp": primary_delta["direct_joint_rate"] >= -0.01,
        "novel_noninferior_1pp": primary_delta["novel_rate"] >= -0.01,
        "unique_noninferior_1pp": primary_delta["unique_rate"] >= -0.01,
        "strict_retention_noninferior_1pp": primary_delta["strict_retention"] >= -0.01,
        "meta_retention_noninferior_1pp": primary_delta["meta_retention"] >= -0.01,
        "both_seeds_strict_noninferior_1pp": all(
            seed_deltas[str(seed)]["strict_attempt_rate"] >= -0.01 for seed in SEEDS
        ),
        "both_seeds_meta_noninferior_1pp": all(
            seed_deltas[str(seed)]["meta_attempt_rate"] >= -0.01 for seed in SEEDS
        ),
    }
    gate["promote_hard_joint"] = all(gate.values())

    mcnemar: dict[str, Any] = {}
    for seed in SEEDS:
        left = {int(row["ordinal"]): row for row in rows_by_cell[(seed, "full_axis")]}
        right = {int(row["ordinal"]): row for row in rows_by_cell[(seed, "hard_joint")]}
        known = [
            idx
            for idx in range(ATTEMPTS)
            if left[idx]["official_hull_status"] == "known"
            and right[idx]["official_hull_status"] == "known"
        ]
        mcnemar[str(seed)] = {
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
        "schema": "h1a2_condition_schedule_l6_final_v1",
        "design": {
            "factors": ["full_vs_hard_anchor_conditioning", "axis_vs_joint_coordinate_commitment"],
            "primary_candidate": "hard_joint",
            "control": "full_axis",
            "checkpoint": "public H1-A2 DLM checkpoint",
            "seeds": list(SEEDS),
            "rl": False,
            "rerank": False,
            "cfg_scale": 0,
        },
        "cells": cells,
        "pooled_repeat_sum": pooled,
        "primary_delta": primary_delta,
        "seed_deltas": seed_deltas,
        "gate": gate,
        "mcnemar": mcnemar,
        "unknown_policy": "excluded from hull-known denominators; never mapped to unstable",
        "uniqueness_note": "pooled rows sum independently recomputed seed-level uniqueness rather than recomputing one 512-sample cohort",
    }
    stem = "DLM_CONDITION_SCHEDULE_L6_FINAL"
    (output / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_keys = list(pooled[0])
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in csv_keys} for row in pooled)
    lines = [
        "# DLM conditioning × coordinate schedule L6",
        "",
        f"Promote hard_joint: **{gate['promote_hard_joint']}**",
        "",
        "| Arm | Requested | Body | Direct J | N/U/NU | Hull K/U | Strict stable/SUN | Meta stable/SUN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['arm']} | {row['requested']} | {row['body']} | {row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['hull_known']}/{row['hull_unknown']} | "
            f"{row['strict_stable']}/{row['strict_sun']} | {row['meta_stable']}/{row['meta_sun']} |"
        )
    lines.extend(["", "## hard_joint minus full_axis", ""])
    lines.extend(f"- {key}: `{value:+.4%}`" for key, value in primary_delta.items())
    lines.extend(["", "## Frozen gate", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gate.items())
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
