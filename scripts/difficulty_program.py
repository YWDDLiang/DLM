#!/usr/bin/env python3
"""Analyze historical S.U.N. ledgers or build a small Planner replay buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h1a2_repro.difficulty import (  # noqa: E402
    PRIMARY_FEATURES,
    cross_fitted_difficulty,
    deduplicate,
    difficulty_weights,
    kitagawa_decomposition,
    load_jsonl,
    summarize,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_cohort(value: str) -> tuple[str, str, Path]:
    try:
        method, cohort_id, path = value.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cohort must be METHOD:COHORT_ID:PATH") from exc
    return method, cohort_id, Path(path)


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Historical S.U.N. difficulty analysis", ""]
    lines.append(f"- unique attempts: `{report['unique_attempts']}`")
    lines.append(f"- removed replay rows: `{report['duplicates_removed']}`")
    lines.append("")
    for method, payload in report["methods"].items():
        lines.extend([f"## {method}", ""])
        for feature in PRIMARY_FEATURES:
            lines.extend([f"### {feature}", "", "| value | attempts | hull known | Strict | Meta |", "|---|---:|---:|---:|---:|"])
            for row in payload[feature]:
                strict = "NA" if row["strict_rate"] is None else f"{row['strict']}/{row['hull_known']} ({100*row['strict_rate']:.2f}%)"
                meta = "NA" if row["meta_rate"] is None else f"{row['meta']}/{row['hull_known']} ({100*row['meta_rate']:.2f}%)"
                lines.append(f"| {row['value']} | {row['attempts']} | {row['hull_known']} | {strict} | {meta} |")
            lines.append("")
    if report.get("decomposition"):
        lines.extend(["## Proposal/realization decomposition", "", "```json", json.dumps(report["decomposition"], indent=2), "```", ""])
    lines.append("Exact formulas and individual halogens are exploratory only and are not used as headline strata.")
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> None:
    attempts = []
    for method, cohort_id, path in args.cohort:
        attempts.extend(load_jsonl(path, cohort_id=cohort_id, method=method))
    unique, duplicates = deduplicate(attempts)
    methods: dict[str, Any] = {}
    for method in sorted({item.method for item in unique}):
        selected = [item for item in unique if item.method == method]
        methods[method] = {feature: summarize(selected, feature) for feature in PRIMARY_FEATURES}
    decomposition = None
    if args.baseline and args.candidate:
        baseline = [item for item in unique if item.method == args.baseline]
        candidate = [item for item in unique if item.method == args.candidate]
        decomposition = {
            endpoint: kitagawa_decomposition(baseline, candidate, feature=args.decomposition_feature, endpoint=endpoint)
            for endpoint in ("strict_sun", "meta_sun")
        }
    report = {
        "schema": "h1a2-sun-difficulty-analysis@1",
        "unique_attempts": len(unique),
        "duplicates_removed": duplicates,
        "methods": methods,
        "decomposition": decomposition,
        "primary_features": list(PRIMARY_FEATURES),
        "exploratory_only": ["individual_halogen", "oxyhalide", "exact_formula", "exact_element_set"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "analysis.md").write_text(_markdown(report), encoding="utf-8")


def _plan_answer(attempt) -> str:
    features = attempt.raw.get("planner_plan_features") or attempt.raw.get("plan_state") or {}
    required = {
        "formula": attempt.formula,
        "anion": features.get("anion_framework", attempt.family if attempt.family != "all_metal" else "other"),
        "charge": features.get("charge_bucket", "all_metal" if attempt.all_metal == "yes" else "validator_unavailable"),
        "lattice": features.get("lattice_system", "triclinic"),
        "spacegroup": features.get("spacegroup_bucket", "sg_001_002"),
        "volume": features.get("volume_per_atom_bin", "volpa_000_004"),
    }
    return "\n".join([*(f"{key}: {required[key]}" for key in required), "end: plan"])


def build_buffer(args: argparse.Namespace) -> None:
    attempts, duplicates = deduplicate(load_jsonl(args.attempts, cohort_id=args.cohort_id, method="self_improvement_source"))
    baselines = cross_fitted_difficulty(attempts, folds=args.folds, prior_strength=args.prior_strength)
    weights, weight_report = difficulty_weights(
        attempts,
        baselines,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        max_weight=args.max_weight,
        min_ess_ratio=args.min_ess_ratio,
    )
    anchor_train = _read_jsonl(args.anchor_dir / "train.jsonl")
    if not anchor_train:
        raise ValueError("anchor train.jsonl is empty")
    template = anchor_train[0]
    eligible = [item for item in attempts if item.key in weights and item.formula]
    if not eligible:
        raise ValueError("no eligible self-improvement Plan rows")
    target_total = (args.buffer_fraction / (1.0 - args.buffer_fraction)) * len(anchor_train)
    scale = target_total / sum(weights[item.key] for item in eligible)
    buffer_rows = []
    for item in eligible:
        row = {key: value for key, value in template.items() if key in {"prompt", "messages"}}
        row.update(
            {
                "answer": _plan_answer(item),
                "sample_weight": weights[item.key] * scale,
                "source_kind": "difficulty_decomposed_self_improvement",
                "source_attempt_key": item.key,
                "difficulty_baseline": baselines[item.key],
                "within_stratum_advantage": float(item.reward) - baselines[item.key],
                "difficulty_features": {feature: item.feature(feature) for feature in PRIMARY_FEATURES},
            }
        )
        buffer_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "train.jsonl", anchor_train + buffer_rows)
    for split in ("val", "test"):
        source = args.anchor_dir / f"{split}.jsonl"
        if source.exists():
            shutil.copyfile(source, args.output_dir / source.name)
    manifest = {
        "schema": "difficulty-decomposed-planner-buffer@1",
        "anchor_rows": len(anchor_train),
        "buffer_rows": len(buffer_rows),
        "buffer_fraction_by_total_weight": args.buffer_fraction,
        "duplicates_removed": duplicates,
        "reward": "I(meta_sun)+I(strict_sun); hull_unknown excluded",
        "features": list(PRIMARY_FEATURES),
        "cross_fitting": {"folds": args.folds, "prior_strength": args.prior_strength},
        "weighting": weight_report,
    }
    (args.output_dir / "difficulty_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--cohort", type=_parse_cohort, action="append", required=True)
    analysis.add_argument("--baseline")
    analysis.add_argument("--candidate")
    analysis.add_argument("--decomposition-feature", choices=PRIMARY_FEATURES, default="family")
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.set_defaults(func=analyze)

    buffer = subparsers.add_parser("build-buffer")
    buffer.add_argument("--attempts", type=Path, required=True)
    buffer.add_argument("--cohort-id", required=True)
    buffer.add_argument("--anchor-dir", type=Path, required=True)
    buffer.add_argument("--output-dir", type=Path, required=True)
    buffer.add_argument("--buffer-fraction", type=float, default=0.05)
    buffer.add_argument("--folds", type=int, default=5)
    buffer.add_argument("--prior-strength", type=float, default=20.0)
    buffer.add_argument("--alpha", type=float, default=1.0)
    buffer.add_argument("--beta", type=float, default=1.0)
    buffer.add_argument("--temperature", type=float, default=1.0)
    buffer.add_argument("--max-weight", type=float, default=5.0)
    buffer.add_argument("--min-ess-ratio", type=float, default=0.5)
    buffer.set_defaults(func=build_buffer)
    args = parser.parse_args()
    if getattr(args, "buffer_fraction", 0.05) <= 0 or getattr(args, "buffer_fraction", 0.05) >= 0.5:
        parser.error("buffer-fraction must be in (0, 0.5)")
    args.func(args)


if __name__ == "__main__":
    main()
