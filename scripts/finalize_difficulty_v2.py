#!/usr/bin/env python3
"""Finalize the two-seed normalized difficulty-Planner screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


SEEDS = (17, 18)
ARMS = ("control", "candidate")
ATTEMPTS = 256


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
    n_atoms = int(plan.get("N") or sum(plan.get("counts") or []))
    return {
        "family": str(plan.get("family") or plan.get("anion_framework") or "unknown"),
        "arity": str(len(set(elements))),
        "n_bin": n_bin(n_atoms),
        "all_metal": bool(elements) and all(Element(symbol).is_metal for symbol in elements),
    }


def distribution(rows: list[dict], selected: set[int]) -> dict:
    counters = {key: Counter() for key in ("family", "arity", "n_bin", "all_metal")}
    observed = 0
    missing_plan = 0
    for row in rows:
        ordinal = int(row["ordinal"])
        if ordinal not in selected:
            continue
        plan = row.get("plan_state")
        if not plan:
            missing_plan += 1
            continue
        observed += 1
        for key, value in chemistry_signature(plan).items():
            counters[key][str(value)] += 1
    return {
        "selected": len(selected),
        "observed_plans": observed,
        "missing_plan": missing_plan,
        **{key: dict(sorted(values.items())) for key, values in counters.items()},
    }


def tvd(left: dict[str, int], right: dict[str, int]) -> float:
    left_n, right_n = sum(left.values()), sum(right.values())
    if left_n == 0 or right_n == 0:
        return 0.0
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0) / left_n - right.get(key, 0) / right_n) for key in keys)


def mean_sd_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182}.get(len(values), 1.96)
    half = critical * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def two_proportion_z(success_a: int, n_a: int, success_b: int, n_b: int) -> dict:
    if n_a == 0 or n_b == 0:
        return {"z": None, "two_sided_p": None}
    pooled = (success_a + success_b) / (n_a + n_b)
    variance = pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b)
    if variance == 0.0:
        return {"z": 0.0, "two_sided_p": 1.0}
    z = (success_b / n_b - success_a / n_a) / math.sqrt(variance)
    return {"z": z, "two_sided_p": math.erfc(abs(z) / math.sqrt(2.0))}


def cell_rates(counts: dict) -> dict[str, float]:
    return {
        "planner_parse_rate": rate(counts["planner_parsed"], counts["requested"]),
        "body_rate": rate(counts["body"], counts["requested"]),
        "direct_joint_rate": rate(counts["direct_joint"], counts["requested"]),
        "novel_rate": rate(counts["novel"], counts["reconstructed"]),
        "unique_rate": rate(counts["unique"], counts["reconstructed"]),
        "novel_unique_rate": rate(counts["novel_unique"], counts["reconstructed"]),
        "strict_known_rate": rate(counts["strict"], counts["hull_known"]),
        "meta_known_rate": rate(counts["meta"], counts["hull_known"]),
        "strict_attempt_rate": rate(counts["strict"], counts["requested"]),
        "meta_attempt_rate": rate(counts["meta"], counts["requested"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--plan-run", type=Path, required=True)
    parser.add_argument("--downstream-run", type=Path, required=True)
    parser.add_argument("--eval-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-nll-seed17", type=float, required=True)
    parser.add_argument("--candidate-nll-seed17", type=float, required=True)
    parser.add_argument("--control-nll-seed18", type=float, required=True)
    parser.add_argument("--candidate-nll-seed18", type=float, required=True)
    parser.add_argument("--variant", choices=("v2", "strong20-v3"), default="v2")
    args = parser.parse_args()

    sys.path.insert(0, str(args.eval_runtime.resolve()))
    import protocol  # noqa: PLC0415
    from finalize_official import _evaluate_cell, _exact_mcnemar, _phase_diagrams  # noqa: PLC0415

    if protocol.DENOMINATOR != ATTEMPTS:
        raise RuntimeError("H1_ACTIVE_DENOMINATOR must be 256")
    eval_run = args.eval_run.resolve()
    plan_run = args.plan_run.resolve()
    downstream_run = args.downstream_run.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    cache = eval_run / "official_mp_cache"
    phase_diagrams = _phase_diagrams(cache / "official_slim_cache.jsonl")
    unresolved = {
        str(row["chemsys"])
        for row in protocol.read_jsonl(cache / "unresolved_chemsys.jsonl")
    }
    difficulty_manifest = read_json(plan_run / "weighted_data/difficulty_manifest.json")
    training_metrics = {}
    for seed in SEEDS:
        for arm in ARMS:
            metrics_path = plan_run / f"{arm}_seed{seed}" / "train_metrics.json"
            if not metrics_path.is_file() and arm == "control":
                parent_file = plan_run / "matched_control_parent.txt"
                if parent_file.is_file():
                    parent = Path(parent_file.read_text(encoding="utf-8").strip())
                    metrics_path = parent / f"control_seed{seed}" / "train_metrics.json"
            metrics = read_json(metrics_path)
            training_metrics[f"s{seed}_{arm}"] = {
                "global_step": int(metrics["global_step"]),
                "final_eval_loss": float(metrics["final_eval_loss"]),
                "sampled_rows": metrics.get("sampled_rows"),
                "sampled_self_improvement": metrics.get("sampled_self_improvement"),
                "sampled_self_improvement_fraction": metrics.get(
                    "sampled_self_improvement_fraction"
                ),
            }

    rows_by_cell: dict[str, list[dict]] = {}
    reports: dict[str, dict] = {}
    cells: list[dict] = []
    chemistry: dict[str, dict] = {}
    for seed in SEEDS:
        for arm in ARMS:
            cell_id = f"s{seed}_{arm}"
            cell = eval_run / f"seed{seed}" / arm
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

            generation = read_jsonl(cell / "generation/generation.jsonl")
            generation_report = read_json(cell / "generation/generation_report.json")
            body = read_json(downstream_run / f"seed{seed}" / arm / "body/sample_metrics.json")
            refine = read_json(downstream_run / f"seed{seed}" / arm / "refine/refinement_metrics.json")
            succeeded = {int(row["ordinal"]) for row in generation if row.get("status") == "succeeded"}
            reconstructed = {int(row["ordinal"]) for row in rows if row.get("reconstructed") is True}
            novel_unique = {int(row["ordinal"]) for row in rows if row.get("novel_unique") is True}
            parsed = {int(row["ordinal"]) for row in generation if row.get("plan_state")}
            chemistry[cell_id] = {
                "planner_parsed": distribution(generation, parsed),
                "body_success": distribution(generation, succeeded),
                "reconstructed": distribution(generation, reconstructed),
                "novel_unique": distribution(generation, novel_unique),
            }

            counts = report["counts"]
            direct = report["direct"]
            row = {
                "seed": seed,
                "arm": arm,
                "requested": ATTEMPTS,
                "planner_parsed": int(generation_report["planner_parsed"]),
                "body_attempted": int(body["requested_samples"]),
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
            }
            row.update(cell_rates(row))
            cells.append(row)

    per_seed: list[dict] = []
    pooled_pairing: dict[str, tuple[list[bool], list[bool]]] = {
        "strict": ([], []),
        "meta": ([], []),
    }
    for seed in SEEDS:
        control_map = {int(row["ordinal"]): row for row in rows_by_cell[f"s{seed}_control"]}
        candidate_map = {int(row["ordinal"]): row for row in rows_by_cell[f"s{seed}_candidate"]}
        known_both = [
            ordinal
            for ordinal in range(ATTEMPTS)
            if control_map[ordinal]["official_hull_status"] == "known"
            and candidate_map[ordinal]["official_hull_status"] == "known"
        ]
        endpoint_stats = {}
        for endpoint in ("strict", "meta"):
            key = f"{endpoint}_sun"
            control_values = [bool(control_map[ordinal][key]) for ordinal in known_both]
            candidate_values = [bool(candidate_map[ordinal][key]) for ordinal in known_both]
            pooled_pairing[endpoint][0].extend(control_values)
            pooled_pairing[endpoint][1].extend(candidate_values)
            endpoint_stats[f"{endpoint}_mcnemar"] = _exact_mcnemar(control_values, candidate_values)

        control = next(row for row in cells if row["seed"] == seed and row["arm"] == "control")
        candidate = next(row for row in cells if row["seed"] == seed and row["arm"] == "candidate")
        rate_keys = [key for key in control if key.endswith("_rate")]
        per_seed.append(
            {
                "seed": seed,
                "known_both": len(known_both),
                "deltas": {key: candidate[key] - control[key] for key in rate_keys},
                **endpoint_stats,
            }
        )

    pooled: dict[str, dict] = {}
    count_keys = (
        "requested",
        "planner_parsed",
        "body_attempted",
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
    for arm in ARMS:
        arm_cells = [row for row in cells if row["arm"] == arm]
        counts = {key: sum(int(row[key]) for row in arm_cells) for key in count_keys}
        pooled[arm] = {**counts, **cell_rates(counts)}

    rate_keys = [key for key in pooled["control"] if key.endswith("_rate")]
    pooled_deltas = {key: pooled["candidate"][key] - pooled["control"][key] for key in rate_keys}
    chemistry_drift: dict[str, dict] = {}
    for seed in SEEDS:
        chemistry_drift[f"seed{seed}"] = {}
        for stage in ("planner_parsed", "body_success", "reconstructed", "novel_unique"):
            control = chemistry[f"s{seed}_control"][stage]
            candidate = chemistry[f"s{seed}_candidate"][stage]
            chemistry_drift[f"seed{seed}"][stage] = {
                "family_tvd": tvd(control["family"], candidate["family"]),
                "arity_tvd": tvd(control["arity"], candidate["arity"]),
                "n_bin_tvd": tvd(control["n_bin"], candidate["n_bin"]),
                "all_metal_rate_delta": rate(
                    candidate["all_metal"].get("True", 0), candidate["observed_plans"]
                )
                - rate(control["all_metal"].get("True", 0), control["observed_plans"]),
            }

    strict_seed_deltas = [row["deltas"]["strict_attempt_rate"] for row in per_seed]
    meta_seed_deltas = [row["deltas"]["meta_attempt_rate"] for row in per_seed]
    criteria = {
        "strict_direction": pooled_deltas["strict_attempt_rate"] > 0.0,
        "meta_noninferiority_1pp": pooled_deltas["meta_attempt_rate"] >= -0.01
        and pooled_deltas["meta_known_rate"] >= -0.01,
        "planner_parse_noninferiority_1pp": pooled_deltas["planner_parse_rate"] >= -0.01,
        "body_noninferiority_1pp": pooled_deltas["body_rate"] >= -0.01,
        "direct_joint_noninferiority_1pp": pooled_deltas["direct_joint_rate"] >= -0.01,
        "novelty_noninferiority_1pp": pooled_deltas["novel_rate"] >= -0.01
        and pooled_deltas["unique_rate"] >= -0.01,
    }
    criteria["route_b_screen_useful"] = all(criteria.values())

    pooled_stats = {}
    for endpoint in ("strict", "meta"):
        control_values, candidate_values = pooled_pairing[endpoint]
        pooled_stats[f"{endpoint}_known_both_mcnemar"] = _exact_mcnemar(
            control_values, candidate_values
        )
        pooled_stats[f"{endpoint}_attempt_unpaired_z"] = two_proportion_z(
            pooled["control"][endpoint],
            pooled["control"]["requested"],
            pooled["candidate"][endpoint],
            pooled["candidate"]["requested"],
        )

    nll = {
        "seed17": {
            "control": args.control_nll_seed17,
            "candidate": args.candidate_nll_seed17,
            "candidate_minus_control": args.candidate_nll_seed17 - args.control_nll_seed17,
        },
        "seed18": {
            "control": args.control_nll_seed18,
            "candidate": args.candidate_nll_seed18,
            "candidate_minus_control": args.candidate_nll_seed18 - args.control_nll_seed18,
        },
    }
    is_v3 = args.variant == "strong20-v3"
    file_stem = (
        "PLANNER_DIFFICULTY_V3_STRONG20_FINAL"
        if is_v3
        else "PLANNER_DIFFICULTY_V2_FINAL"
    )
    summary = {
        "schema": (
            "h1a2_difficulty_decomposed_planner_strong20_v3_final_v1"
            if is_v3
            else "h1a2_difficulty_decomposed_planner_v2_final_v1"
        ),
        "variant": args.variant,
        "unknown_policy": "excluded from hull-known denominators; never mapped to unstable",
        "design": {
            "planner_seeds": list(SEEDS),
            "attempts_per_cell": ATTEMPTS,
            "downstream": "frozen CE-control DLM + model494; D1 exact-plan; no safe-axis",
            "pairing": "same Planner seed and ordinal; common DLM/refiner random numbers",
            "pairing_caveat": "compositions differ across arms; this is not a fixed-composition realization effect",
        },
        "difficulty_manifest": difficulty_manifest,
        "training_metrics": training_metrics,
        "validation_nll": nll,
        "cells": cells,
        "cell_reports": reports,
        "per_seed": per_seed,
        "seed_delta_statistics": {
            "strict_attempt_rate": mean_sd_ci(strict_seed_deltas),
            "meta_attempt_rate": mean_sd_ci(meta_seed_deltas),
        },
        "pooled": pooled,
        "pooled_deltas": pooled_deltas,
        "pooled_statistics": pooled_stats,
        "statistics_scope": {
            "mcnemar": "exact test on common-random-number ordinal pairs; end-to-end only, not fixed-composition",
            "unpaired_z": "descriptive attempt-level comparison of different proposal distributions",
            "planner_inference_unit": "Planner training seed; n=2, so this remains a screen rather than population-level confirmation",
        },
        "chemistry": chemistry,
        "chemistry_drift": chemistry_drift,
        "criteria": criteria,
    }

    json_path = output / f"{file_stem}.json"
    csv_path = output / f"{file_stem}.csv"
    md_path = output / f"{file_stem}.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    lines = [
        (
            "# Difficulty-Decomposed Self-Improving Planner strong20 V3 — final two-seed screen"
            if is_v3
            else "# Difficulty-Decomposed Self-Improving Planner V2 — final two-seed screen"
        ),
        "",
        f"Route-B screen useful: **{criteria['route_b_screen_useful']}**",
        "",
        "| Seed | Arm | Planner parsed | Body | Refined | Reconstructed | Direct C/S/J | N/U/N∩U | Hull K/U | Strict (attempt; known) | Meta (attempt; known) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['planner_parsed']}/{ATTEMPTS} | "
            f"{row['body']}/{ATTEMPTS} | {row['refined']} | {row['reconstructed']} | "
            f"{row['direct_comp']}/{row['direct_struct']}/{row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['hull_known']}/{row['hull_unknown']} | "
            f"{row['strict']}/{ATTEMPTS}={row['strict_attempt_rate']:.2%}; "
            f"{row['strict']}/{row['hull_known']}={row['strict_known_rate']:.2%} | "
            f"{row['meta']}/{ATTEMPTS}={row['meta_attempt_rate']:.2%}; "
            f"{row['meta']}/{row['hull_known']}={row['meta_known_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Pooled 512-attempt comparison",
            "",
            "| Arm | Planner parsed | Body/refined/reconstructed | Direct C/S/J | N/U/N∩U | Hull K/U | Strict (attempt; known) | Meta (attempt; known) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        row = pooled[arm]
        lines.append(
            f"| {arm} | {row['planner_parsed']}/{row['requested']} | "
            f"{row['body']}/{row['refined']}/{row['reconstructed']} | "
            f"{row['direct_comp']}/{row['direct_struct']}/{row['direct_joint']} | "
            f"{row['novel']}/{row['unique']}/{row['novel_unique']} | "
            f"{row['hull_known']}/{row['hull_unknown']} | "
            f"{row['strict']}/{row['requested']}={row['strict_attempt_rate']:.2%}; "
            f"{row['strict']}/{row['hull_known']}={row['strict_known_rate']:.2%} | "
            f"{row['meta']}/{row['requested']}={row['meta_attempt_rate']:.2%}; "
            f"{row['meta']}/{row['hull_known']}={row['meta_known_rate']:.2%} |"
        )
    lines.extend(["", "## Pooled deltas (candidate minus control)", ""])
    for key, value in pooled_deltas.items():
        lines.append(f"- {key}: `{value:+.4%}`")
    lines.extend(["", "## Decision criteria", ""])
    for key, value in criteria.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Seed stability and statistics", ""])
    for row in per_seed:
        lines.append(
            f"- seed {row['seed']}: Strict attempt delta "
            f"`{row['deltas']['strict_attempt_rate']:+.4%}`; Meta attempt delta "
            f"`{row['deltas']['meta_attempt_rate']:+.4%}`."
        )
    lines.append(
        "- Pooled known-both exact McNemar: Strict "
        f"candidate-only/control-only={pooled_stats['strict_known_both_mcnemar']['candidate_only']}/"
        f"{pooled_stats['strict_known_both_mcnemar']['control_only']}, "
        f"p={pooled_stats['strict_known_both_mcnemar']['two_sided_exact_p']:.4g}; Meta "
        f"candidate-only/control-only={pooled_stats['meta_known_both_mcnemar']['candidate_only']}/"
        f"{pooled_stats['meta_known_both_mcnemar']['control_only']}, "
        f"p={pooled_stats['meta_known_both_mcnemar']['two_sided_exact_p']:.4g}."
    )
    lines.extend(
        [
            "",
            "Proposal-mix changes and downstream conversion are reported separately. Because the Planner arms sample different compositions, ordinal pairing is only an end-to-end common-random-number comparison and is not evidence of a fixed-composition realization effect.",
            "",
        ]
    )
    if is_v3:
        lines.append(
            "This is the corrected strong20 treatment: dedicated replacement weighted sampling, 20% self-improvement probability, and 800 matched control/candidate updates."
        )
    else:
        lines.append(
            "The normalized V2 screen is not retained as a positive method result: seed 17 improved Strict/Meta, seed 18 reversed both, and pooled Strict plus novelty were negative despite improved Direct joint validity."
        )
    lines.extend(
        [
            "",
            "The public 105/1000 Strict and 488/1000 Meta headline remains unchanged pending user confirmation.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
