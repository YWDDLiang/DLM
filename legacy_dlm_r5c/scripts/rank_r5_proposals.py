#!/usr/bin/env python3
"""Rank R5 proposals with a lightweight verifier score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.dynamic_crystal import parse_dynamic_answer  # noqa: E402
from crystal_dlm.fixed_slot import parse_fixed_slot_answer, write_json  # noqa: E402
from crystal_dlm.r5_verifier import VerifierWeights, extract_verifier_features, rank_feature_rows  # noqa: E402


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def metric_row(metrics: Mapping[str, Any] | None, idx: int) -> Dict[str, Any]:
    if not metrics:
        return {}
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, list) and idx < len(value):
            out[key] = value[idx]
    return out


def parse_arrays(text: str, representation: str) -> Dict[str, Any]:
    if representation == "dynamic_v1":
        return parse_dynamic_answer(text)
    if representation == "fixed_slot":
        return parse_fixed_slot_answer(text)
    raise ValueError(f"Unsupported representation {representation!r}")


def weights_from_args(args) -> VerifierWeights:
    return VerifierWeights(
        comp_valid=args.comp_valid_weight,
        strict_valid=args.strict_valid_weight,
        graph_valid=args.graph_valid_weight,
        refine_success=args.refine_success_weight,
        meta_stable=args.meta_stable_weight,
        strict_stable=args.strict_stable_weight,
        novel=args.novel_weight,
        unique=args.unique_weight,
        single_penalty=args.single_penalty,
        all_metal_penalty=args.all_metal_penalty,
        high_sym_penalty=args.high_sym_penalty,
        duplicate_penalty=args.duplicate_penalty,
        plan_mismatch_penalty=args.plan_mismatch_penalty,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-generations-jsonl", type=Path, required=True)
    parser.add_argument("--representation", choices=["fixed_slot", "dynamic_v1"], default="dynamic_v1")
    parser.add_argument("--detailed-metrics-json", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--comp-valid-weight", type=float, default=1.0)
    parser.add_argument("--strict-valid-weight", type=float, default=1.0)
    parser.add_argument("--graph-valid-weight", type=float, default=0.5)
    parser.add_argument("--refine-success-weight", type=float, default=0.5)
    parser.add_argument("--meta-stable-weight", type=float, default=2.0)
    parser.add_argument("--strict-stable-weight", type=float, default=4.0)
    parser.add_argument("--novel-weight", type=float, default=1.0)
    parser.add_argument("--unique-weight", type=float, default=1.0)
    parser.add_argument("--single-penalty", type=float, default=2.0)
    parser.add_argument("--all-metal-penalty", type=float, default=0.2)
    parser.add_argument("--high-sym-penalty", type=float, default=1.0)
    parser.add_argument("--duplicate-penalty", type=float, default=2.0)
    parser.add_argument("--plan-mismatch-penalty", type=float, default=2.0)
    args = parser.parse_args()

    metrics = None
    if args.detailed_metrics_json is not None and args.detailed_metrics_json.exists():
        metrics = json.loads(args.detailed_metrics_json.read_text(encoding="utf-8"))
    rows = []
    parse_failures = 0
    for idx, row in enumerate(read_jsonl(args.raw_generations_jsonl)):
        text = str(row.get("text") or row.get("answer") or "")
        try:
            arrays = parse_arrays(text, args.representation)
            features = extract_verifier_features(
                arrays,
                plan_state=row.get("plan_state") or row.get("r5_plan_state"),
                sample_record=row,
                metric_record=metric_row(metrics, idx),
            )
            payload = dict(row)
            payload["r5_verifier_features"] = features
            payload.update(features)
            rows.append(payload)
        except Exception as exc:  # noqa: BLE001
            parse_failures += 1
            rows.append(
                {
                    "sample_idx": row.get("sample_idx", idx),
                    "text": text,
                    "parse_error": type(exc).__name__,
                    "parse_error_message": str(exc),
                    "graph_valid": False,
                    "comp_valid": False,
                    "strict_valid": False,
                    "high_sym_coord_fraction": 1.0,
                    "pbc_duplicate_count": 0,
                }
            )
    ranked = rank_feature_rows(rows, weights=weights_from_args(args))
    if args.top_k and args.top_k > 0:
        ranked = ranked[: int(args.top_k)]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in ranked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "input": str(args.raw_generations_jsonl),
        "representation": args.representation,
        "count": len(rows),
        "written": len(ranked),
        "parse_failures": parse_failures,
        "top_score": ranked[0]["r5_utility_score"] if ranked else None,
        "mean_score": sum(float(row["r5_utility_score"]) for row in ranked) / max(1, len(ranked)),
        "top_formulas": {},
    }
    formula_counts: Dict[str, int] = {}
    for row in ranked:
        formula = str(row.get("formula") or row.get("r5_verifier_features", {}).get("formula") or "unknown")
        formula_counts[formula] = formula_counts.get(formula, 0) + 1
    summary["top_formulas"] = dict(sorted(formula_counts.items(), key=lambda item: (-item[1], item[0]))[:20])
    write_json(str(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
