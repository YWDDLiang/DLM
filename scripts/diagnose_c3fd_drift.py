#!/usr/bin/env python3
"""Diagnose proposal-mix drift in the C³FD-v2 matched experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import reduce
import json
from math import gcd, sqrt
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from crystal_dlm.ccfd_v2 import (  # noqa: E402
    BenchmarkReachability,
    CCFDv2State,
    SetAtomCount,
    compile_plan_actions,
    replay_actions,
)
from crystal_dlm.composition_pair_prior import (  # noqa: E402
    CompositionPairPrior,
    ValenceNode,
)
from crystal_dlm.composition_validity import classify_smact_validity  # noqa: E402
from crystal_dlm.fixed_slot import SYMBOL_TO_Z  # noqa: E402
from crystal_dlm.r5_plan_body import formula_from_symbol_counts  # noqa: E402
from crystal_dlm.r5_plan_state import anion_framework_from_symbols  # noqa: E402
from finalize_ccfd_phase1 import extract_formula, tvd  # noqa: E402


SEEDS = (17, 18)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def reduced(counts: Sequence[int]) -> list[int]:
    divisor = reduce(gcd, (int(value) for value in counts)) if counts else 1
    return [int(value) // max(1, int(divisor)) for value in counts]


def is_all_metal(symbols: Sequence[str]) -> bool:
    try:
        import smact

        return all(str(symbol) in smact.metals for symbol in symbols)
    except Exception:  # noqa: BLE001
        return False


def pair_score(nodes: Sequence[ValenceNode], prior: CompositionPairPrior) -> float | None:
    unique = tuple(sorted(set(nodes)))
    scores = [
        prior.pair_pmi(left, right)
        for idx, left in enumerate(unique)
        for right in unique[idx + 1 :]
    ]
    return statistics.fmean(scores) if scores else None


def record_from_plan(
    plan: Mapping[str, Any],
    *,
    prior: CompositionPairPrior,
    vocabulary_nodes: Sequence[ValenceNode] | None = None,
    species_ids: Sequence[int] | None = None,
) -> dict[str, Any] | None:
    symbols = [str(value) for value in (plan.get("elements") or ())]
    counts = [int(value) for value in (plan.get("counts") or ())]
    if not symbols or len(symbols) != len(counts):
        return None
    formula = formula_from_symbol_counts(symbols, counts)
    independent = classify_smact_validity(
        [int(SYMBOL_TO_Z[symbol]) for symbol in symbols], reduced(counts)
    )
    nodes: list[ValenceNode] = []
    if vocabulary_nodes is not None and species_ids is not None:
        nodes = [vocabulary_nodes[int(value)] for value in species_ids]
    elif independent.get("valid") is True:
        try:
            actions, _meta = compile_plan_actions(plan)
            state = replay_actions(actions)
            nodes = [ValenceNode.from_token(token) for token in state.tokens]
        except Exception:  # noqa: BLE001
            nodes = []
    return {
        "formula": formula,
        "N": int(sum(counts)),
        "arity": len(symbols),
        "family": anion_framework_from_symbols(symbols),
        "all_metal": is_all_metal(symbols),
        "unary": len(symbols) == 1,
        "benchmark_valid": independent.get("valid") is True,
        "pair_score": pair_score(nodes, prior),
    }


def load_training(
    path: Path, vocabulary: Mapping[str, Any], prior: CompositionPairPrior
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        ValenceNode(int(row["atomic_number"]), int(row["oxidation_state"]))
        for row in sorted(vocabulary["species"], key=lambda value: int(value["id"]))
    ]
    full: list[dict[str, Any]] = []
    benchmark: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        plan = row.get("plan_state") or {}
        record = record_from_plan(
            plan,
            prior=prior,
            vocabulary_nodes=nodes,
            species_ids=row.get("species_labels") or (),
        )
        if record is None:
            continue
        full.append(record)
        if row.get("composition_supervision") is True:
            benchmark.append(record)
    return full, benchmark


def load_arm(
    arm: str, run: Path, prior: CompositionPairPrior
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    prefix = "f0" if arm == "p0" else "c3"
    for seed in SEEDS:
        for row in iter_jsonl(run / f"{prefix}_seed{seed}/raw_generations.jsonl"):
            plan = row.get("plan_state") or row.get("parsed_plan")
            if not isinstance(plan, Mapping):
                extracted = extract_formula(row)
                if extracted is None:
                    continue
                _formula, symbols, counts = extracted
                plan = {"N": sum(counts), "elements": symbols, "counts": counts}
            record = record_from_plan(plan, prior=prior)
            if record is None:
                continue
            record["seed"] = seed
            record["sample_idx"] = int(row["sample_idx"])
            records.append(record)
    formula_counts = Counter(str(row["formula"]) for row in records)
    for row in records:
        row["arm_formula_count"] = formula_counts[str(row["formula"])]
        row["duplicated_formula"] = row["arm_formula_count"] > 1
    return records


def distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {key: None for key in ("mean", "q10", "q25", "q50", "q75", "q90")}
    def q(p: float) -> float:
        position = p * (len(ordered) - 1)
        lo = int(position)
        hi = min(len(ordered) - 1, lo + 1)
        frac = position - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac
    return {
        "mean": statistics.fmean(ordered),
        "q10": q(0.10),
        "q25": q(0.25),
        "q50": q(0.50),
        "q75": q(0.75),
        "q90": q(0.90),
    }


def summarize(rows: Sequence[Mapping[str, Any]], train_frequency: Counter[str]) -> dict[str, Any]:
    formulas = Counter(str(row["formula"]) for row in rows)
    pair_values = [float(row["pair_score"]) for row in rows if row.get("pair_score") is not None]
    duplicate_pair = [
        float(row["pair_score"])
        for row in rows
        if row.get("pair_score") is not None and row.get("duplicated_formula")
    ]
    unique_pair = [
        float(row["pair_score"])
        for row in rows
        if row.get("pair_score") is not None and not row.get("duplicated_formula")
    ]
    return {
        "rows": len(rows),
        "benchmark_valid_rate": sum(bool(row["benchmark_valid"]) for row in rows) / max(1, len(rows)),
        "all_metal_rate": sum(bool(row["all_metal"]) for row in rows) / max(1, len(rows)),
        "unary_rate": sum(bool(row["unary"]) for row in rows) / max(1, len(rows)),
        "unique_formulas": len(formulas),
        "duplicate_excess": sum(max(0, count - 1) for count in formulas.values()),
        "max_formula_multiplicity": max(formulas.values(), default=0),
        "top_formulas": formulas.most_common(20),
        "mean_train_formula_frequency": statistics.fmean(
            train_frequency.get(formula, 0) for formula in formulas.elements()
        ) if formulas else None,
        "distributions": {
            key: distribution(rows, key) for key in ("N", "arity", "family", "all_metal")
        },
        "pair_score": quantiles(pair_values),
        "pair_score_duplicated": quantiles(duplicate_pair),
        "pair_score_singleton": quantiles(unique_pair),
    }


def rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[cursor]]):
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right))
    denominator = sqrt(
        sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right)
    )
    return None if denominator == 0 else numerator / denominator


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return pearson(rank(left), rank(right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    vocabulary = json.loads((args.data_dir / "vocabulary.json").read_text(encoding="utf-8"))
    prior = CompositionPairPrior.from_dict(vocabulary["pair_prior"])
    train_full, train_benchmark = load_training(args.data_dir / "train.jsonl", vocabulary, prior)
    p0 = load_arm("p0", args.control_run, prior)
    c3 = load_arm("c3", args.candidate_run, prior)
    train_frequency = Counter(str(row["formula"]) for row in train_full)
    groups = {
        "train_full": summarize(train_full, train_frequency),
        "train_benchmark": summarize(train_benchmark, train_frequency),
        "p0": summarize(p0, train_frequency),
        "c3fd_v2": summarize(c3, train_frequency),
    }
    distances = {}
    for arm in ("p0", "c3fd_v2"):
        distances[arm] = {}
        for target in ("train_full", "train_benchmark"):
            distances[arm][target] = {
                key: tvd(
                    groups[arm]["distributions"][key],
                    groups[target]["distributions"][key],
                )
                for key in ("N", "arity", "family", "all_metal")
            }

    nodes = [
        (int(row["atomic_number"]), int(row["oxidation_state"]))
        for row in vocabulary["species"]
    ]
    oracle = BenchmarkReachability(nodes)
    legal_volume = {}
    for n_value in range(1, 21):
        tokens = oracle.legal_species_counts(
            CCFDv2State.start().apply(SetAtomCount(n_value))
        )
        legal_volume[str(n_value)] = {
            "total": len(tokens),
            "zero": sum(int(token.oxidation_state) == 0 for token in tokens),
            "ionic": sum(int(token.oxidation_state) != 0 for token in tokens),
        }
    c3_n = groups["c3fd_v2"]["distributions"]["N"]
    train_n = groups["train_benchmark"]["distributions"]["N"]
    n_values = [str(value) for value in range(1, 21)]
    correlations = {
        "c3_N_vs_total_legal": spearman(
            [float(c3_n.get(key, 0)) for key in n_values],
            [float(legal_volume[key]["total"]) for key in n_values],
        ),
        "c3_N_vs_zero_legal": spearman(
            [float(c3_n.get(key, 0)) for key in n_values],
            [float(legal_volume[key]["zero"]) for key in n_values],
        ),
        "c3_N_vs_benchmark_train_N": spearman(
            [float(c3_n.get(key, 0)) for key in n_values],
            [float(train_n.get(key, 0)) for key in n_values],
        ),
    }
    report = {
        "schema": "h1a2_c3fd_drift_diagnostic_v1",
        "groups": groups,
        "distance_to_training": distances,
        "legal_first_action_volume": legal_volume,
        "correlations": correlations,
        "hypotheses": {
            "benchmark_subset_selection": "compare C3/P0 distance to train_full versus train_benchmark",
            "small_head_mode_concentration": "compare duplicate excess, top formula multiplicity, and train formula frequency",
            "pair_prior_concentration": "compare pair-score distributions and duplicate versus singleton pair scores",
            "legal_volume_bias": "compare C3 N frequencies with legal first-action volume",
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stem = "C3FD_DRIFT_DIAGNOSTIC"
    (output / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# C³FD-v2 drift diagnostic",
        "",
        "| Group | Rows | Benchmark | All-metal | Unary | Unique formulas | Duplicate excess | Pair-score mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in groups.items():
        lines.append(
            f"| {name} | {row['rows']} | {row['benchmark_valid_rate']:.2%} | "
            f"{row['all_metal_rate']:.2%} | {row['unary_rate']:.2%} | "
            f"{row['unique_formulas']} | {row['duplicate_excess']} | "
            f"{row['pair_score']['mean']} |"
        )
    lines.extend(
        [
            "",
            f"Distances to training: `{distances}`",
            f"N correlations: `{correlations}`",
            f"Top C3 formulas: `{groups['c3fd_v2']['top_formulas']}`",
            f"Top P0 formulas: `{groups['p0']['top_formulas']}`",
        ]
    )
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
