#!/usr/bin/env python3
"""Audit a train-only semantic element/valence co-occurrence prior."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ccfd_v2 import (  # noqa: E402
    compile_plan_actions,
    replay_actions,
)
from crystal_dlm.composition_pair_prior import (  # noqa: E402
    CompositionPairPrior,
    ValenceNode,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"non-object row in {path}")
                yield value


def plan_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("plan_state") or row.get("r5_plan_state")
    return value if isinstance(value, Mapping) else None


def load_benchmark_compositions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    plans = compiled = benchmark = 0
    for row in iter_jsonl(path):
        plan = plan_from_row(row)
        if plan is None:
            failures["missing_plan_state"] += 1
            continue
        plans += 1
        try:
            actions, _metadata = compile_plan_actions(plan)
            compiled += 1
            state = replay_actions(actions)
            certificate = state.certificate()
        except Exception as exc:  # noqa: BLE001
            failures[f"{type(exc).__name__}:{str(exc)}"] += 1
            continue
        if not certificate.benchmark_compatible:
            continue
        benchmark += 1
        nodes = tuple(sorted({ValenceNode.from_token(token) for token in state.tokens}))
        records.append(
            {
                "formula": str(certificate.formula),
                "N": int(state.target_atoms or 0),
                "anion": str(plan.get("anion_framework") or "other"),
                "nodes": nodes,
            }
        )
    return records, {
        "path": str(path.resolve()),
        "plans": plans,
        "compiled": compiled,
        "benchmark_certified": benchmark,
        "failures": dict(failures.most_common()),
    }


def auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    values = sorted(
        [(float(value), 1) for value in positive]
        + [(float(value), 0) for value in negative],
        key=lambda item: item[0],
    )
    positive_rank_sum = 0.0
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[end][0] == values[cursor][0]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _score, label in values[cursor:end])
        cursor = end
    n_pos = len(positive)
    n_neg = len(negative)
    return (
        positive_rank_sum - n_pos * (n_pos + 1) / 2.0
    ) / float(n_pos * n_neg)


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    prior: CompositionPairPrior,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    vocabulary = list(prior.nodes)
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    known_nodes = total_nodes = seen_edges = total_edges = 0
    formulas_with_pairs = 0
    by_anion: dict[str, dict[str, list[float]]] = {}
    for record in records:
        nodes = tuple(record["nodes"])
        node_set = set(nodes)
        total_nodes += len(nodes)
        known_nodes += sum(node in prior.node_counts for node in nodes)
        if len(nodes) < 2:
            continue
        formulas_with_pairs += 1
        anion = str(record.get("anion") or "other")
        bucket = by_anion.setdefault(anion, {"positive": [], "negative": []})
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                score = prior.pair_pmi(left, right)
                positive_scores.append(score)
                bucket["positive"].append(score)
                total_edges += 1
                seen_edges += int(tuple(sorted((left, right))) in prior.pair_counts)
                candidates = [
                    candidate
                    for candidate in vocabulary
                    if candidate not in node_set
                    and candidate != left
                    and int(candidate.atomic_number) != int(left.atomic_number)
                ]
                if not candidates:
                    continue
                negative = rng.choice(candidates)
                negative_score = prior.pair_pmi(left, negative)
                negative_scores.append(negative_score)
                bucket["negative"].append(negative_score)

    strata = {}
    for name, values in sorted(by_anion.items()):
        strata[name] = {
            "positive_pairs": len(values["positive"]),
            "negative_pairs": len(values["negative"]),
            "auc": auc(values["positive"], values["negative"]),
            "positive_mean": (
                statistics.fmean(values["positive"]) if values["positive"] else None
            ),
            "negative_mean": (
                statistics.fmean(values["negative"]) if values["negative"] else None
            ),
        }
    return {
        "records": len(records),
        "formulas_with_pairs": formulas_with_pairs,
        "positive_pairs": len(positive_scores),
        "negative_pairs": len(negative_scores),
        "known_node_rate": 0.0 if total_nodes == 0 else known_nodes / total_nodes,
        "seen_edge_rate": 0.0 if total_edges == 0 else seen_edges / total_edges,
        "auc": auc(positive_scores, negative_scores),
        "positive_mean": statistics.fmean(positive_scores) if positive_scores else None,
        "negative_mean": statistics.fmean(negative_scores) if negative_scores else None,
        "positive_median": statistics.median(positive_scores) if positive_scores else None,
        "negative_median": statistics.median(negative_scores) if negative_scores else None,
        "strata": strata,
    }


def write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    stem = "CCFD_V2_PAIR_PRIOR_AUDIT"
    (output_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "split",
            "records",
            "positive_pairs",
            "known_node_rate",
            "seen_edge_rate",
            "auc",
            "positive_mean",
            "negative_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in ("validation_all", "validation_formula_disjoint"):
            writer.writerow({"split": split, **{key: payload[split].get(key) for key in fields if key != "split"}})
    gate = payload["gate"]
    lines = [
        "# CCFD-v2 semantic pair-prior audit",
        "",
        f"Soft prior authorized: **{gate['soft_pair_prior_authorized']}**",
        "",
        "The prior is train-only typed element/valence co-occurrence. It is not BPE and never changes the hard CCFD legality mask.",
        "",
        "| Split | Records | Positive pairs | Known nodes | Seen edges | AUC | Positive mean | Negative mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation_all", "validation_formula_disjoint"):
        row = payload[split]
        lines.append(
            f"| {split} | {row['records']} | {row['positive_pairs']} | "
            f"{row['known_node_rate']:.2%} | {row['seen_edge_rate']:.2%} | "
            f"{row['auc'] if row['auc'] is not None else 'NA'} | "
            f"{row['positive_mean'] if row['positive_mean'] is not None else 'NA'} | "
            f"{row['negative_mean'] if row['negative_mean'] is not None else 'NA'} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in gate.items())
    (output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "_SUCCESS").touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    train_records, train_manifest = load_benchmark_compositions(args.train_jsonl)
    val_records, val_manifest = load_benchmark_compositions(args.val_jsonl)
    prior = CompositionPairPrior.fit(
        [record["nodes"] for record in train_records], alpha=float(args.alpha)
    )
    train_formulas = {str(record["formula"]) for record in train_records}
    disjoint = [record for record in val_records if str(record["formula"]) not in train_formulas]
    val_all = evaluate_records(val_records, prior, seed=int(args.seed))
    val_disjoint = evaluate_records(disjoint, prior, seed=int(args.seed))
    disjoint_auc = val_disjoint["auc"]
    positive_mean = val_disjoint["positive_mean"]
    negative_mean = val_disjoint["negative_mean"]
    gate = {
        "train_only_fit": True,
        "validation_known_nodes_at_least_95pct": val_all["known_node_rate"] >= 0.95,
        "formula_disjoint_pairs_at_least_1000": val_disjoint["positive_pairs"] >= 1000,
        "formula_disjoint_auc_above_0_60": disjoint_auc is not None and disjoint_auc > 0.60,
        "formula_disjoint_positive_mean_above_negative": (
            positive_mean is not None
            and negative_mean is not None
            and positive_mean > negative_mean
        ),
        "hard_mask_unchanged": True,
    }
    gate["soft_pair_prior_authorized"] = all(gate.values())
    payload = {
        "schema": "h1a2_ccfd_v2_pair_prior_audit_v1",
        "claim_boundary": "soft composition prior only; no hard chemistry or stability claim",
        "train": train_manifest,
        "validation": val_manifest,
        "formula_overlap": {
            "train_unique": len(train_formulas),
            "validation_unique": len({str(record['formula']) for record in val_records}),
            "validation_disjoint_records": len(disjoint),
        },
        "prior": prior.to_dict(),
        "validation_all": val_all,
        "validation_formula_disjoint": val_disjoint,
        "gate": gate,
    }
    write_report(payload, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
