#!/usr/bin/env python3
"""Finalize the eight fixed-256 grounding cells with the frozen official hull logic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def n_bin(value: int) -> str:
    if value <= 4:
        return "01-04"
    if value <= 8:
        return "05-08"
    if value <= 12:
        return "09-12"
    return "13-20"


def chemistry_signature(plan: dict) -> dict[str, str | bool]:
    from pymatgen.core import Element

    elements = [str(value) for value in plan.get("elements") or []]
    arity = int(len(set(elements)))
    return {
        "family": str(plan.get("family") or plan.get("anion_framework") or "unknown"),
        "arity": str(arity),
        "n_bin": n_bin(int(plan.get("N") or sum(plan.get("counts") or []))),
        "all_metal": bool(elements) and all(Element(symbol).is_metal for symbol in elements),
    }


def distribution(rows: list[dict], selected: set[int]) -> dict[str, dict[str, int] | int]:
    result: dict[str, Counter] = {
        "family": Counter(),
        "arity": Counter(),
        "n_bin": Counter(),
        "all_metal": Counter(),
    }
    for row in rows:
        ordinal = int(row["ordinal"])
        if ordinal not in selected:
            continue
        signature = chemistry_signature(row.get("plan_state") or {})
        for key, value in signature.items():
            result[key][str(value)] += 1
    return {"n": len(selected), **{key: dict(sorted(counts.items())) for key, counts in result.items()}}


def tvd(left: dict[str, int], right: dict[str, int]) -> float:
    left_n, right_n = sum(left.values()), sum(right.values())
    if left_n == 0 or right_n == 0:
        return 0.0
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0) / left_n - right.get(key, 0) / right_n) for key in keys)


def mean_sd_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 3.182 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--repeat-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-val-ce", type=float, required=True)
    parser.add_argument("--candidate-val-ce", type=float, required=True)
    parser.add_argument("--candidate-margin", type=float, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.eval_runtime.resolve()))
    import protocol  # noqa: PLC0415
    from finalize_official import _evaluate_cell, _exact_mcnemar, _phase_diagrams  # noqa: PLC0415

    if protocol.DENOMINATOR != 256:
        raise RuntimeError("H1_ACTIVE_DENOMINATOR must be 256")
    eval_run = args.eval_run.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cache = eval_run / "official_mp_cache"
    phase_diagrams = _phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {str(row["chemsys"]) for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")}

    rows_by_cell: dict[str, list[dict]] = {}
    reports: dict[str, dict] = {}
    cell_table: list[dict] = []
    chemistry: dict[str, dict] = {}
    for repeat in range(1, 5):
        for arm in ("control", "candidate"):
            cell_id = f"r{repeat}_{arm}"
            cell = eval_run / f"repeat{repeat:02d}" / arm
            rows, report = _evaluate_cell(
                cell_id=cell_id,
                labels_path=cell / "evaluation/full_reconstructed/attempt_labels_preofficial.jsonl",
                generation_path=cell / "generation/generation.jsonl",
                direct_path=cell / "evaluation/direct/report.json",
                phase_diagrams=phase_diagrams,
                unresolved=unresolved,
                output_dir=output / "cells" / cell_id,
            )
            rows_by_cell[cell_id] = rows
            reports[cell_id] = report
            body = read_json(args.repeat_run / f"repeat{repeat:02d}" / arm / "body/sample_metrics.json")
            refine = read_json(args.repeat_run / f"repeat{repeat:02d}" / arm / "refine/refinement_metrics.json")
            generation = read_jsonl(cell / "generation/generation.jsonl")
            successful = {int(row["ordinal"]) for row in generation if row.get("status") == "succeeded"}
            reconstructed = {int(row["ordinal"]) for row in rows if row.get("reconstructed") is True}
            novel_unique = {int(row["ordinal"]) for row in rows if row.get("novel_unique") is True}
            chemistry[cell_id] = {
                "input": distribution(generation, set(range(256))),
                "body_success": distribution(generation, successful),
                "reconstructed": distribution(generation, reconstructed),
                "novel_unique": distribution(generation, novel_unique),
            }
            counts = report["counts"]
            direct = report["direct"]
            cell_table.append(
                {
                    "repeat": repeat,
                    "arm": arm,
                    "requested": int(body["requested_samples"]),
                    "parsed": int(body["parse_success"]),
                    "body": int(body["graph_success"]),
                    "refined": int(refine["num_proposals"]),
                    "reconstructed": int(counts["reconstructed"]),
                    "direct_comp": int(direct["composition_valid"]),
                    "direct_struct": int(direct["structure_valid"]),
                    "direct_joint": int(direct["joint_valid"]),
                    "novel": int(counts["novel"]),
                    "unique": int(counts["unique_representatives"]),
                    "novel_unique": int(counts["novel_unique"]),
                    "hull_known": int(counts["hull_known_reconstructed"]),
                    "hull_unknown": int(counts["hull_unknown_reconstructed"]),
                    "strict": int(counts["strict_sun"]),
                    "meta": int(counts["meta_sun"]),
                    "strict_known_rate": float(report["rates"]["strict_sun_hull_known_reconstructed"]),
                    "meta_known_rate": float(report["rates"]["meta_sun_hull_known_reconstructed"]),
                    "strict_attempt_rate": float(report["rates"]["strict_sun_all_attempts"]),
                    "meta_attempt_rate": float(report["rates"]["meta_sun_all_attempts"]),
                }
            )

    repeat_results: list[dict] = []
    pooled_control_strict: list[bool] = []
    pooled_candidate_strict: list[bool] = []
    pooled_control_meta: list[bool] = []
    pooled_candidate_meta: list[bool] = []
    for repeat in range(1, 5):
        control = rows_by_cell[f"r{repeat}_control"]
        candidate = rows_by_cell[f"r{repeat}_candidate"]
        known_both = [
            index
            for index in range(256)
            if control[index]["official_hull_status"] == "known"
            and candidate[index]["official_hull_status"] == "known"
        ]
        control_strict = [bool(control[index]["strict_sun"]) for index in known_both]
        candidate_strict = [bool(candidate[index]["strict_sun"]) for index in known_both]
        control_meta = [bool(control[index]["meta_sun"]) for index in known_both]
        candidate_meta = [bool(candidate[index]["meta_sun"]) for index in known_both]
        pooled_control_strict.extend(control_strict)
        pooled_candidate_strict.extend(candidate_strict)
        pooled_control_meta.extend(control_meta)
        pooled_candidate_meta.extend(candidate_meta)
        control_report = reports[f"r{repeat}_control"]
        candidate_report = reports[f"r{repeat}_candidate"]
        deltas = {
            "strict_known_rate": candidate_report["rates"]["strict_sun_hull_known_reconstructed"]
            - control_report["rates"]["strict_sun_hull_known_reconstructed"],
            "meta_known_rate": candidate_report["rates"]["meta_sun_hull_known_reconstructed"]
            - control_report["rates"]["meta_sun_hull_known_reconstructed"],
            "strict_attempt_rate": candidate_report["rates"]["strict_sun_all_attempts"]
            - control_report["rates"]["strict_sun_all_attempts"],
            "meta_attempt_rate": candidate_report["rates"]["meta_sun_all_attempts"]
            - control_report["rates"]["meta_sun_all_attempts"],
        }
        repeat_results.append(
            {
                "repeat": repeat,
                "known_both": len(known_both),
                "deltas": deltas,
                "strict_mcnemar": _exact_mcnemar(control_strict, candidate_strict),
                "meta_mcnemar": _exact_mcnemar(control_meta, candidate_meta),
            }
        )

    pooled_cells: dict[str, dict] = {}
    for arm in ("control", "candidate"):
        cells = [row for row in cell_table if row["arm"] == arm]
        pooled_counts = {
            key: sum(int(row[key]) for row in cells)
            for key in (
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
                "strict",
                "meta",
            )
        }
        pooled_cells[arm] = {
            **pooled_counts,
            "body_rate": rate(pooled_counts["body"], pooled_counts["requested"]),
            "direct_joint_rate": rate(pooled_counts["direct_joint"], pooled_counts["requested"]),
            "novel_rate": rate(pooled_counts["novel"], pooled_counts["reconstructed"]),
            "unique_rate": rate(pooled_counts["unique"], pooled_counts["reconstructed"]),
            "strict_known_rate": rate(pooled_counts["strict"], pooled_counts["hull_known"]),
            "meta_known_rate": rate(pooled_counts["meta"], pooled_counts["hull_known"]),
            "strict_attempt_rate": rate(pooled_counts["strict"], pooled_counts["requested"]),
            "meta_attempt_rate": rate(pooled_counts["meta"], pooled_counts["requested"]),
        }

    deltas = {
        key: pooled_cells["candidate"][key] - pooled_cells["control"][key]
        for key in (
            "body_rate",
            "direct_joint_rate",
            "novel_rate",
            "unique_rate",
            "strict_known_rate",
            "meta_known_rate",
            "strict_attempt_rate",
            "meta_attempt_rate",
        )
    }
    chemistry_drift: dict[str, dict] = {}
    for repeat in range(1, 5):
        chemistry_drift[f"repeat{repeat}"] = {}
        for stage in ("input", "body_success", "reconstructed", "novel_unique"):
            left = chemistry[f"r{repeat}_control"][stage]
            right = chemistry[f"r{repeat}_candidate"][stage]
            chemistry_drift[f"repeat{repeat}"][stage] = {
                "family_tvd": tvd(left["family"], right["family"]),
                "arity_tvd": tvd(left["arity"], right["arity"]),
                "n_bin_tvd": tvd(left["n_bin"], right["n_bin"]),
                "all_metal_rate_delta": rate(right["all_metal"].get("True", 0), right["n"])
                - rate(left["all_metal"].get("True", 0), left["n"]),
            }

    strict_deltas = [item["deltas"]["strict_known_rate"] for item in repeat_results]
    meta_deltas = [item["deltas"]["meta_known_rate"] for item in repeat_results]
    max_reconstructed_family_tvd = max(
        item["reconstructed"]["family_tvd"] for item in chemistry_drift.values()
    )
    max_reconstructed_all_metal_delta = max(
        abs(item["reconstructed"]["all_metal_rate_delta"]) for item in chemistry_drift.values()
    )
    mechanism = {
        "control_factual_val_ce": args.control_val_ce,
        "candidate_factual_val_ce": args.candidate_val_ce,
        "candidate_minus_control": args.candidate_val_ce - args.control_val_ce,
        "candidate_true_vs_counterfactual_margin": args.candidate_margin,
    }
    criteria = {
        "mechanism": args.candidate_val_ce < args.control_val_ce and args.candidate_margin > 0.0,
        "strict_repeat_stability": sum(value >= 0.0 for value in strict_deltas) >= 3
        and deltas["strict_known_rate"] > 0.0,
        "meta_noninferiority": deltas["meta_known_rate"] >= -0.01
        and sum(value < 0.0 for value in meta_deltas) < 3,
        "body_noninferiority": deltas["body_rate"] >= -0.01,
        "direct_joint_noninferiority": deltas["direct_joint_rate"] >= -0.01,
        "novelty_no_collapse": deltas["novel_rate"] >= -0.01 and deltas["unique_rate"] >= -0.01,
        "composition_no_collapse": max_reconstructed_family_tvd <= 0.05
        and max_reconstructed_all_metal_delta <= 0.02,
    }
    criteria["contribution_pass"] = all(criteria.values())
    summary = {
        "schema": "h1a2_grounding_final_repeat4_v1",
        "unknown_policy": "excluded from hull-known denominators; never mapped to unstable",
        "mechanism": mechanism,
        "cells": cell_table,
        "cell_reports": reports,
        "per_repeat": repeat_results,
        "repeat_delta_statistics": {
            "strict_known_rate": mean_sd_ci(strict_deltas),
            "meta_known_rate": mean_sd_ci(meta_deltas),
        },
        "pooled": pooled_cells,
        "pooled_deltas": deltas,
        "pooled_known_both": len(pooled_control_strict),
        "pooled_strict_mcnemar": _exact_mcnemar(pooled_control_strict, pooled_candidate_strict),
        "pooled_meta_mcnemar": _exact_mcnemar(pooled_control_meta, pooled_candidate_meta),
        "chemistry": chemistry,
        "chemistry_drift": chemistry_drift,
        "criteria": criteria,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "cell_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_table[0]))
        writer.writeheader()
        writer.writerows(cell_table)

    lines = [
        "# Candidate A grounding — final repeat4",
        "",
        f"Contribution pass: **{criteria['contribution_pass']}**",
        "",
        "| Repeat | Arm | Body | Refined | Reconstructed | Direct joint | N∩U | Hull known/unknown | Strict | Meta |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cell_table:
        lines.append(
            f"| {row['repeat']} | {row['arm']} | {row['body']}/256 | {row['refined']} | "
            f"{row['reconstructed']} | {row['direct_joint']} | {row['novel_unique']} | "
            f"{row['hull_known']}/{row['hull_unknown']} | {row['strict']} | {row['meta']} |"
        )
    lines.extend(["", "## Criteria", ""])
    for key, value in criteria.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "The public 105/1000 Strict and 488/1000 Meta headline is not replaced by this repeat screen.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
