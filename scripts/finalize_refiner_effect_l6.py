#!/usr/bin/env python3
"""Compare raw DLM bodies with the same bodies after model494 refinement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


SEEDS = (17, 18)
SOURCE_ARMS = ("full_axis", "hard_joint")
ATTEMPTS = 256


def rate(value: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else value / denominator


def summarize(rows: list[dict[str, Any]], direct: dict[str, Any]) -> dict[str, Any]:
    counts = {
        "reconstructed": sum(row.get("reconstructed") is True for row in rows),
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
        "reconstructed_rate": rate(counts["reconstructed"], ATTEMPTS),
        "direct_joint_rate": rate(counts["direct_joint"], ATTEMPTS),
        "novel_rate": rate(counts["novel"], counts["reconstructed"]),
        "unique_rate": rate(counts["unique"], counts["reconstructed"]),
        "strict_attempt_rate": rate(counts["strict_sun"], ATTEMPTS),
        "meta_attempt_rate": rate(counts["meta_sun"], ATTEMPTS),
        "strict_retention": rate(counts["strict_sun"], counts["strict_stable"]),
        "meta_retention": rate(counts["meta_sun"], counts["meta_stable"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
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
    rows_by_cell: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for seed in SEEDS:
        for source_arm in SOURCE_ARMS:
            for stage, arm in (("raw", f"raw_{source_arm}"), ("model494", source_arm)):
                root = args.eval_run / f"seed{seed}/{arm}"
                rows, report = _evaluate_cell(
                    cell_id=f"seed{seed}_{source_arm}_{stage}",
                    labels_path=root / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                    generation_path=root / "generation/generation.jsonl",
                    direct_path=root / "evaluation/direct/report.json",
                    phase_diagrams=phase_diagrams,
                    unresolved=unresolved,
                    output_dir=output / f"cells/seed{seed}/{source_arm}/{stage}",
                )
                rows_by_cell[(seed, source_arm, stage)] = rows
                summary = summarize(rows, report["direct"])
                summary.update(
                    {
                        "seed": seed,
                        "source_arm": source_arm,
                        "stage": stage,
                        "requested": ATTEMPTS,
                    }
                )
                cells.append(summary)

    count_keys = (
        "requested",
        "reconstructed",
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
    for source_arm in SOURCE_ARMS:
        for stage in ("raw", "model494"):
            chosen = [
                row
                for row in cells
                if row["source_arm"] == source_arm and row["stage"] == stage
            ]
            item: dict[str, Any] = {
                "seed": "pooled-repeat-sum",
                "source_arm": source_arm,
                "stage": stage,
            }
            for key in count_keys:
                item[key] = sum(int(row[key]) for row in chosen)
            item.update(
                {
                    "reconstructed_rate": rate(item["reconstructed"], item["requested"]),
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

    delta: dict[str, dict[str, float]] = {}
    for source_arm in SOURCE_ARMS:
        raw = next(
            row for row in pooled if row["source_arm"] == source_arm and row["stage"] == "raw"
        )
        refined = next(
            row
            for row in pooled
            if row["source_arm"] == source_arm and row["stage"] == "model494"
        )
        rate_keys = [key for key in raw if key.endswith("_rate") or key.endswith("retention")]
        delta[source_arm] = {key: refined[key] - raw[key] for key in rate_keys}

    mcnemar: dict[str, Any] = {}
    for source_arm in SOURCE_ARMS:
        mcnemar[source_arm] = {}
        for seed in SEEDS:
            left = {
                int(row["ordinal"]): row
                for row in rows_by_cell[(seed, source_arm, "raw")]
            }
            right = {
                int(row["ordinal"]): row
                for row in rows_by_cell[(seed, source_arm, "model494")]
            }
            known = [
                idx
                for idx in range(ATTEMPTS)
                if left[idx]["official_hull_status"] == "known"
                and right[idx]["official_hull_status"] == "known"
            ]
            mcnemar[source_arm][str(seed)] = {
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
        "schema": "h1a2_refiner_effect_l6_diagnostic_v1",
        "question": "effect of 800-step model494 warm-start diffusion before common CHGNet relaxation",
        "diagnostic_only": True,
        "cells": cells,
        "pooled_repeat_sum": pooled,
        "model494_minus_raw_by_source_arm": delta,
        "mcnemar": mcnemar,
        "interpretation_rule": {
            "positive_stability": "distil or improve model494 low-energy outputs",
            "null_or_negative_stability": "shorten/calibrate warm-start timestep before executor retraining",
        },
        "unknown_policy": "excluded from hull-known denominators; never mapped to unstable",
    }
    stem = "DLM_REFINER_EFFECT_L6_DIAGNOSTIC"
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
        "# DLM raw-body versus model494 L6 diagnostic",
        "",
        "| Source arm | Stage | Requested | Reconstructed | Direct J | N/U/NU | Strict stable/SUN | Meta stable/SUN |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['source_arm']} | {row['stage']} | {row['requested']} | {row['reconstructed']} | "
            f"{row['direct_joint']} | {row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['strict_stable']}/{row['strict_sun']} | {row['meta_stable']}/{row['meta_sun']} |"
        )
    for source_arm in SOURCE_ARMS:
        lines.extend(["", f"## {source_arm}: model494 minus raw", ""])
        lines.extend(
            f"- {key}: `{value:+.4%}`" for key, value in delta[source_arm].items()
        )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
