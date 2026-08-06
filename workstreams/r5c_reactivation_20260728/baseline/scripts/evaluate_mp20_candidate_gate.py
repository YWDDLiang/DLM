#!/usr/bin/env python3
"""Evaluate MP-20 candidate checkpoints against smoke/refined gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nested_summary(payload: dict[str, Any], preferred: str | None) -> dict[str, Any]:
    if preferred and isinstance(payload.get(preferred), dict):
        return payload[preferred]
    for key in ("refined_pt", "raw_pt", "raw_jsonl"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def metric(summary: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in summary and summary[name] is not None:
            return float(summary[name])
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke256", "refined1000", "final"], required=True)
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--composition-summary", type=Path, required=True)
    parser.add_argument("--composition-key", default=None)
    parser.add_argument("--crysllmgen-metrics", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-parse-rate", type=float, default=1.0)
    parser.add_argument("--min-graph-acceptance", type=float, default=0.95)
    parser.add_argument(
        "--min-comp-valid",
        type=float,
        default=None,
        help="Optional native/refined composition validity gate. By default comp_valid is reported and compared manually.",
    )
    parser.add_argument(
        "--baseline-comp-valid",
        type=float,
        default=None,
        help="Optional relative gate: require comp_valid to be at least baseline + min-comp-delta.",
    )
    parser.add_argument("--min-comp-delta", type=float, default=0.0)
    parser.add_argument("--min-strict-valid", type=float, default=None)
    parser.add_argument(
        "--baseline-strict-valid",
        type=float,
        default=None,
        help="Optional relative gate: require strict_valid to be at least baseline + min-strict-delta.",
    )
    parser.add_argument("--min-strict-delta", type=float, default=0.0)
    parser.add_argument(
        "--max-single-element",
        type=float,
        default=None,
        help="Optional diagnostic gate. By default shortcut distribution is reported but not blocking.",
    )
    parser.add_argument(
        "--max-all-metal",
        type=float,
        default=None,
        help="Optional shortcut diagnostic gate. By default all-metal rate is reported but not blocking.",
    )
    parser.add_argument("--max-pbc-duplicate", type=float, default=0.0)
    args = parser.parse_args()

    sample = load_json(args.sample_metrics)
    composition_payload = load_json(args.composition_summary)
    composition = nested_summary(composition_payload, args.composition_key)
    crys = load_json(args.crysllmgen_metrics)
    crys_metrics = crys.get("metrics", crys)

    parse_rate = metric(sample, "parse_rate")
    graph_rate = metric(sample, "graph_acceptance_rate", "graph_rate")
    comp_valid = metric(composition, "comp_valid_rate")
    shortcut = metric(composition, "shortcut_fraction")
    single_element = 0.0
    all_metal = 0.0
    reason_counts = composition.get("reason_counts") or {}
    count = max(1.0, float(composition.get("count") or sum(reason_counts.values()) or 1.0))
    if reason_counts:
        single_element = float(reason_counts.get("single_element_shortcut", 0)) / count
        all_metal = float(reason_counts.get("all_metal_shortcut", 0)) / count
    pbc_duplicate = metric(composition, "pbc_equivalent_duplicate_fraction")
    strict_valid = float(reason_counts.get("charge_neutral_pauling_valid", 0)) / count if reason_counts else 0.0
    if "strict_valid_rate" in composition:
        strict_valid = float(composition["strict_valid_rate"])

    target_comp = 0.90
    target_strict = 0.40 if args.mode == "smoke256" else 0.50

    failures: list[str] = []
    if parse_rate < args.min_parse_rate:
        failures.append(f"parse_rate {parse_rate:.4f} < {args.min_parse_rate:.4f}")
    if graph_rate < args.min_graph_acceptance:
        failures.append(f"graph_acceptance {graph_rate:.4f} < {args.min_graph_acceptance:.4f}")
    if args.min_comp_valid is not None and comp_valid < args.min_comp_valid:
        failures.append(f"comp_valid {comp_valid:.4f} < {args.min_comp_valid:.4f}")
    if args.baseline_comp_valid is not None:
        relative_floor = args.baseline_comp_valid + args.min_comp_delta
        if comp_valid < relative_floor:
            failures.append(f"comp_valid {comp_valid:.4f} < baseline+delta {relative_floor:.4f}")
    if args.min_strict_valid is not None and strict_valid < args.min_strict_valid:
        failures.append(f"strict_valid {strict_valid:.4f} < {args.min_strict_valid:.4f}")
    if args.baseline_strict_valid is not None:
        relative_floor = args.baseline_strict_valid + args.min_strict_delta
        if strict_valid < relative_floor:
            failures.append(
                f"strict_valid {strict_valid:.4f} < baseline+delta {relative_floor:.4f}"
            )
    if args.max_single_element is not None and single_element > args.max_single_element:
        failures.append(f"single_element {single_element:.4f} > {args.max_single_element:.4f}")
    if args.max_all_metal is not None and all_metal > args.max_all_metal:
        failures.append(f"all_metal {all_metal:.4f} > {args.max_all_metal:.4f}")
    if pbc_duplicate > args.max_pbc_duplicate:
        failures.append(f"pbc_duplicate {pbc_duplicate:.4f} > {args.max_pbc_duplicate:.4f}")

    if args.mode in {"refined1000", "final"} and crys_metrics:
        final_thresholds = {
            "comp_valid": 90.0,
            "struct_valid": 99.0,
            "wdist_density": 0.85,
            "cov_recall": 93.0,
            "cov_precision": 94.0,
        }
        if float(crys_metrics.get("comp_valid", 0.0)) < final_thresholds["comp_valid"]:
            failures.append("crysllmgen comp_valid below 90")
        if float(crys_metrics.get("struct_valid", 0.0)) < final_thresholds["struct_valid"]:
            failures.append("crysllmgen struct_valid below 99")
        if float(crys_metrics.get("wdist_density", 999.0)) > final_thresholds["wdist_density"]:
            failures.append("wdist_density above 0.85")
        if float(crys_metrics.get("cov_recall", 0.0)) < final_thresholds["cov_recall"]:
            failures.append("cov_recall below 93")
        if float(crys_metrics.get("cov_precision", 0.0)) < final_thresholds["cov_precision"]:
            failures.append("cov_precision below 94")

    result = {
        "mode": args.mode,
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "parse_rate": parse_rate,
            "graph_acceptance": graph_rate,
            "comp_valid": comp_valid,
            "strict_valid": strict_valid,
            "shortcut": shortcut,
            "single_element": single_element,
            "all_metal": all_metal,
            "pbc_duplicate": pbc_duplicate,
            "crysllmgen": crys_metrics,
        },
        "thresholds": {
            "min_parse_rate": args.min_parse_rate,
            "min_graph_acceptance": args.min_graph_acceptance,
            "target_comp_valid": target_comp,
            "min_comp_valid": args.min_comp_valid,
            "baseline_comp_valid": args.baseline_comp_valid,
            "min_comp_delta": args.min_comp_delta,
            "target_strict_valid": target_strict,
            "min_strict_valid": args.min_strict_valid,
            "baseline_strict_valid": args.baseline_strict_valid,
            "min_strict_delta": args.min_strict_delta,
            "max_single_element": args.max_single_element,
            "max_all_metal": args.max_all_metal,
            "max_pbc_duplicate": args.max_pbc_duplicate,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
