#!/usr/bin/env python3
"""Evaluate R5-B modular CrysLLMGen-answer DLM gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nested_summary(payload: Mapping[str, Any], preferred: str | None) -> Mapping[str, Any]:
    if preferred and isinstance(payload.get(preferred), Mapping):
        return payload[preferred]  # type: ignore[index]
    for key in ("refined_pt", "raw_pt", "raw_jsonl"):
        item = payload.get(key)
        if isinstance(item, Mapping):
            return item
    return payload


def rate(numerator: Any, denominator: Any) -> float:
    try:
        return float(numerator) / max(1.0, float(denominator))
    except Exception:
        return 0.0


def crys_metric(payload: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, Mapping):
        return default
    try:
        return float(metrics.get(name, default))
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke256", "refined1000"], required=True)
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--composition-summary", type=Path, required=True)
    parser.add_argument("--composition-key", default=None)
    parser.add_argument("--crysllmgen-metrics", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-composition-parse", type=float, default=0.99)
    parser.add_argument("--min-parse-rate", type=float, default=0.95)
    parser.add_argument("--min-graph-acceptance", type=float, default=0.85)
    parser.add_argument("--min-comp-valid", type=float, default=0.918)
    parser.add_argument("--max-single-element", type=float, default=0.01)
    parser.add_argument("--max-pbc-duplicate", type=float, default=0.0)
    parser.add_argument("--min-crys-comp-valid", type=float, default=90.0)
    parser.add_argument("--min-crys-struct-valid", type=float, default=99.0)
    parser.add_argument("--min-crys-cov-recall", type=float, default=90.0)
    args = parser.parse_args()

    sample = load_json(args.sample_metrics)
    composition_payload = load_json(args.composition_summary)
    composition = nested_summary(composition_payload, args.composition_key)
    crys_payload = load_json(args.crysllmgen_metrics)

    decoded = sample.get("decoded_samples") or sample.get("requested_samples") or 0
    composition_parse = rate(sample.get("composition_parse_success", 0), decoded)
    parse_rate = float(sample.get("parse_rate") or rate(sample.get("parse_success", 0), decoded))
    graph_acceptance = float(
        sample.get("graph_acceptance_rate")
        or sample.get("graph_rate")
        or rate(sample.get("graph_success", 0), decoded)
    )
    comp_valid = float(composition.get("comp_valid_rate") or 0.0)
    reason_counts = composition.get("reason_counts") or {}
    count = float(composition.get("count") or sum(reason_counts.values()) or 1.0)
    single_element = float(reason_counts.get("single_element_shortcut", 0)) / max(1.0, count)
    all_metal = float(reason_counts.get("all_metal_shortcut", 0)) / max(1.0, count)
    pbc_duplicate = float(composition.get("pbc_equivalent_duplicate_fraction") or 0.0)

    failures: list[str] = []
    if args.mode == "smoke256":
        if composition_parse < args.min_composition_parse:
            failures.append(
                f"composition_parse {composition_parse:.4f} < {args.min_composition_parse:.4f}"
            )
        if parse_rate < args.min_parse_rate:
            failures.append(f"parse_rate {parse_rate:.4f} < {args.min_parse_rate:.4f}")
        if graph_acceptance < args.min_graph_acceptance:
            failures.append(
                f"graph_acceptance {graph_acceptance:.4f} < {args.min_graph_acceptance:.4f}"
            )
        if comp_valid < args.min_comp_valid:
            failures.append(f"comp_valid {comp_valid:.4f} < {args.min_comp_valid:.4f}")
        if single_element > args.max_single_element:
            failures.append(f"single_element {single_element:.4f} > {args.max_single_element:.4f}")
        if pbc_duplicate > args.max_pbc_duplicate:
            failures.append(f"pbc_duplicate {pbc_duplicate:.4f} > {args.max_pbc_duplicate:.4f}")
    else:
        crys_comp_valid = crys_metric(crys_payload, "comp_valid")
        crys_struct_valid = crys_metric(crys_payload, "struct_valid")
        crys_cov_recall = crys_metric(crys_payload, "cov_recall")
        if graph_acceptance < args.min_graph_acceptance:
            failures.append(
                f"graph_acceptance {graph_acceptance:.4f} < {args.min_graph_acceptance:.4f}"
            )
        if comp_valid < args.min_comp_valid:
            failures.append(f"refined_comp_valid {comp_valid:.4f} < {args.min_comp_valid:.4f}")
        if crys_comp_valid < args.min_crys_comp_valid:
            failures.append(
                f"crysllmgen comp_valid {crys_comp_valid:.4f} < {args.min_crys_comp_valid:.4f}"
            )
        if crys_struct_valid < args.min_crys_struct_valid:
            failures.append(
                f"crysllmgen struct_valid {crys_struct_valid:.4f} < {args.min_crys_struct_valid:.4f}"
            )
        if crys_cov_recall < args.min_crys_cov_recall:
            failures.append(
                f"crysllmgen cov_recall {crys_cov_recall:.4f} < {args.min_crys_cov_recall:.4f}"
            )
        if pbc_duplicate > args.max_pbc_duplicate:
            failures.append(f"pbc_duplicate {pbc_duplicate:.4f} > {args.max_pbc_duplicate:.4f}")

    result = {
        "mode": args.mode,
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "composition_parse": composition_parse,
            "parse_rate": parse_rate,
            "graph_acceptance": graph_acceptance,
            "comp_valid": comp_valid,
            "single_element": single_element,
            "all_metal": all_metal,
            "pbc_duplicate": pbc_duplicate,
            "crysllmgen": crys_payload.get("metrics", crys_payload),
        },
        "thresholds": {
            "min_composition_parse": args.min_composition_parse,
            "min_parse_rate": args.min_parse_rate,
            "min_graph_acceptance": args.min_graph_acceptance,
            "min_comp_valid": args.min_comp_valid,
            "max_single_element": args.max_single_element,
            "max_pbc_duplicate": args.max_pbc_duplicate,
            "min_crys_comp_valid": args.min_crys_comp_valid,
            "min_crys_struct_valid": args.min_crys_struct_valid,
            "min_crys_cov_recall": args.min_crys_cov_recall,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
