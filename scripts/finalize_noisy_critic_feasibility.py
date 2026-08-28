#!/usr/bin/env python3
"""Finalize independent MatterSim-vs-CHGNet same-Plan predictability gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        average = 0.5 * (cursor + stop - 1)
        for position in range(cursor, stop):
            output[order[position]] = average
        cursor = stop
    return output


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return 0.0 if left_scale == 0.0 or right_scale == 0.0 else numerator / (left_scale * right_scale)


def grouped(rows: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[int(row["sample_idx"])].append(row)
    return dict(result)


def summarize_groups(groups: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    rank_left: list[float] = []
    rank_right: list[float] = []
    concordant = discordant = extreme_agree = extreme_total = 0
    plan_spearman: list[float] = []
    rows = 0
    for values in groups.values():
        if len(values) < 2:
            continue
        values = sorted(values, key=lambda row: int(row["stream"]))
        chgnet = [float(row["chgnet_relaxed_energy_per_atom"]) for row in values]
        mattersim = [float(row["mattersim_energy_per_atom"]) for row in values]
        left_rank = rankdata(chgnet)
        right_rank = rankdata(mattersim)
        denominator = max(1, len(values) - 1)
        rank_left.extend(value / denominator for value in left_rank)
        rank_right.extend(value / denominator for value in right_rank)
        rows += len(values)
        if len(values) >= 3:
            plan_spearman.append(pearson(left_rank, right_rank))
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                chg_delta = chgnet[right_index] - chgnet[left_index]
                matter_delta = mattersim[right_index] - mattersim[left_index]
                if chg_delta == 0.0 or matter_delta == 0.0:
                    continue
                if chg_delta * matter_delta > 0.0:
                    concordant += 1
                else:
                    discordant += 1
        low = min(range(len(values)), key=lambda index: (chgnet[index], index))
        high = max(range(len(values)), key=lambda index: (chgnet[index], -index))
        if mattersim[low] != mattersim[high]:
            extreme_total += 1
            extreme_agree += int(mattersim[low] < mattersim[high])
    pair_total = concordant + discordant
    return {
        "plans": len(groups),
        "plans_with_at_least_two": sum(len(values) >= 2 for values in groups.values()),
        "plans_with_at_least_three": sum(len(values) >= 3 for values in groups.values()),
        "rows_in_rank_pool": rows,
        "pooled_within_plan_spearman": pearson(rank_left, rank_right),
        "pairwise_concordance_auc": None if pair_total == 0 else concordant / pair_total,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "extreme_pair_direction_agreement": None if extreme_total == 0 else extreme_agree / extreme_total,
        "extreme_pairs": extreme_total,
        "median_plan_spearman": None if not plan_spearman else sorted(plan_spearman)[len(plan_spearman) // 2],
        "positive_plan_spearman_fraction": None if not plan_spearman else sum(value > 0.0 for value in plan_spearman) / len(plan_spearman),
    }


def bootstrap_spearman(
    groups: dict[int, list[dict[str, Any]]], *, draws: int = 5000, seed: int = 20260828
) -> dict[str, Any]:
    eligible = {key: value for key, value in groups.items() if len(value) >= 2}
    plan_ids = sorted(eligible)
    if not plan_ids:
        return {"draws": 0, "low": None, "median": None, "high": None}
    rng = random.Random(int(seed))
    estimates: list[float] = []
    for _ in range(int(draws)):
        sampled: dict[int, list[dict[str, Any]]] = {}
        for position in range(len(plan_ids)):
            source = plan_ids[rng.randrange(len(plan_ids))]
            sampled[position] = eligible[source]
        estimates.append(float(summarize_groups(sampled)["pooled_within_plan_spearman"]))
    estimates.sort()
    return {
        "draws": len(estimates),
        "low": estimates[int(0.025 * (len(estimates) - 1))],
        "median": estimates[len(estimates) // 2],
        "high": estimates[int(0.975 * (len(estimates) - 1))],
        "resampling_unit": "exact Plan",
    }


def subset_groups(
    rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> dict[int, list[dict[str, Any]]]:
    return grouped(row for row in rows if predicate(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()

    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    prediction_manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    metadata = read_jsonl(args.metadata)
    predictions = read_jsonl(args.predictions)
    if len(metadata) != len(predictions) or [int(row["record_index"]) for row in predictions] != list(range(len(metadata))):
        raise RuntimeError("MatterSim predictions and metadata are not aligned")
    rows: list[dict[str, Any]] = []
    for meta, prediction in zip(metadata, predictions):
        row = dict(meta)
        row["mattersim_known"] = prediction.get("known") is True
        row["mattersim_energy_per_atom"] = prediction.get("energy_per_atom_eV")
        row["mattersim_error"] = prediction.get("error")
        rows.append(row)
    known = [row for row in rows if row["mattersim_known"] and row["mattersim_energy_per_atom"] is not None]
    split_rows = {
        split: [row for row in known if row["pair_split"] == split]
        for split in ("train", "validation")
    }
    summaries = {split: summarize_groups(grouped(values)) for split, values in split_rows.items()}
    validation_groups = grouped(split_rows["validation"])
    bootstrap = bootstrap_spearman(
        validation_groups, draws=int(args.bootstrap_draws)
    )
    strata_definitions: dict[str, Callable[[dict[str, Any]], bool]] = {
        "oxide": lambda row: row["family"] == "oxide",
        "sulfide": lambda row: row["family"] == "sulfide",
        "long_N_13_20": lambda row: 13 <= int(row["N"]) <= 20,
        "all_metal": lambda row: row["charge_bucket"] == "all_metal",
    }
    strata = {
        name: summarize_groups(subset_groups(split_rows["validation"], predicate))
        for name, predicate in strata_definitions.items()
    }
    def stratum_direction_ok(name: str) -> bool:
        value = strata[name]
        if value["plans_with_at_least_two"] < 5 or value["concordant_pairs"] + value["discordant_pairs"] < 20:
            return False
        return (
            value["pooled_within_plan_spearman"] > 0.0
            and value["pairwise_concordance_auc"] is not None
            and value["pairwise_concordance_auc"] > 0.5
        )

    validation = summaries["validation"]
    gates = {
        "all_mattersim_predictions_known": len(known) == len(rows),
        "validation_at_least_50_plans": validation["plans_with_at_least_two"] >= 50,
        "validation_spearman_lcb_positive": bootstrap["low"] is not None and bootstrap["low"] > 0.0,
        "validation_pairwise_auc_above_0p60": validation["pairwise_concordance_auc"] is not None and validation["pairwise_concordance_auc"] > 0.60,
        "validation_extreme_agreement_above_0p60": validation["extreme_pair_direction_agreement"] is not None and validation["extreme_pair_direction_agreement"] > 0.60,
        "oxide_direction_positive": stratum_direction_ok("oxide"),
        "sulfide_direction_positive": stratum_direction_ok("sulfide"),
        "long_N_direction_positive": stratum_direction_ok("long_N_13_20"),
        "formula_disjoint_from_l6_l7": data_manifest.get("formula_disjoint_from_l6_l7") is True,
        "model_is_exact_mattersim_5M": prediction_manifest.get("model_id") == "MatterSim-v1.0.0-5M.pth",
        "batch_size_one": int(prediction_manifest.get("batch_size", -1)) == 1,
    }
    gates["critic_training_authorized"] = all(gates.values())
    report = {
        "schema": "h1a2_noisy_critic_independent_audit_v1",
        "data_manifest": data_manifest,
        "prediction_manifest": prediction_manifest,
        "rows": len(rows),
        "known": len(known),
        "summaries": summaries,
        "validation_plan_bootstrap_ci95": bootstrap,
        "validation_strata": strata,
        "constant_feature_baseline": {
            "features": ["family", "arity", "N", "charge_bucket", "formula"],
            "within_exact_plan_pairwise_auc": 0.5,
            "reason": "all listed features are constant inside each exact-Plan comparison",
        },
        "official_hull_note": "within one exact formula the phase-diagram reference is constant, so official E_hull direction is identical to the chosen structure-energy direction and is not counted as an independent evaluator",
        "gates": gates,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stem = "NOISY_CRITIC_INDEPENDENT_AUDIT"
    (output / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = ("split", "plans_with_at_least_two", "rows_in_rank_pool", "pooled_within_plan_spearman", "pairwise_concordance_auc", "extreme_pair_direction_agreement")
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, summary in summaries.items():
            writer.writerow({"split": split, **{key: summary.get(key) for key in fields if key != "split"}})
    lines = [
        "# Noisy-state critic independent evaluator audit",
        "",
        f"Critic training authorized: **{gates['critic_training_authorized']}**",
        "",
        "| Split | Plans>=2 | Rows | Spearman | Pairwise AUC | Extreme agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, summary in summaries.items():
        lines.append(
            f"| {split} | {summary['plans_with_at_least_two']} | {summary['rows_in_rank_pool']} | "
            f"{summary['pooled_within_plan_spearman']:.4f} | {summary['pairwise_concordance_auc']:.4f} | "
            f"{summary['extreme_pair_direction_agreement']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Validation Plan-bootstrap Spearman 95% CI: `{bootstrap}`",
            "",
            "## Gates",
            "",
            *[f"- {key}: `{value}`" for key, value in gates.items()],
        ]
    )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
