#!/usr/bin/env python3
"""Evaluate R5-D plan-state 256 gate metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rate(numerator: Any, denominator: Any) -> float:
    try:
        return float(numerator) / max(1.0, float(denominator))
    except Exception:
        return 0.0


def top_fraction(histogram: Mapping[str, Any], denominator: int) -> float:
    if not histogram:
        return 0.0
    return rate(max(int(value) for value in histogram.values()), denominator)


def raw_examples(path: Path, *, limit: int = 8) -> dict[str, list[dict[str, Any]]]:
    invalid_formula: list[dict[str, Any]] = []
    invalid_n: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("parsed"):
                if len(parse_failures) < limit:
                    parse_failures.append(
                        {
                            "row_idx": row_idx,
                            "sample_idx": record.get("sample_idx", row_idx),
                            "reason": record.get("reason"),
                            "message": record.get("message"),
                            "text_prefix": str(record.get("text") or "")[:180],
                        }
                    )
                continue
            validation = record.get("plan_validation") or {}
            if not validation.get("valid_formula") and len(invalid_formula) < limit:
                plan = record.get("plan_state") or {}
                invalid_formula.append(
                    {
                        "row_idx": row_idx,
                        "sample_idx": record.get("sample_idx", row_idx),
                        "N": plan.get("N"),
                        "elements": plan.get("elements"),
                        "counts": plan.get("counts"),
                        "formula": plan.get("formula"),
                    }
                )
            if not validation.get("valid_N") and len(invalid_n) < limit:
                plan = record.get("plan_state") or {}
                invalid_n.append(
                    {
                        "row_idx": row_idx,
                        "sample_idx": record.get("sample_idx", row_idx),
                        "N": plan.get("N"),
                        "formula": plan.get("formula"),
                    }
                )
    return {
        "parse_failures": parse_failures,
        "invalid_formula": invalid_formula,
        "invalid_n": invalid_n,
    }


def normalized_metrics(sample: Mapping[str, Any]) -> dict[str, Any]:
    decoded = int(sample.get("decoded_samples") or 0)
    parsed = int(sample.get("parse_success") or 0)
    formula_hist = sample.get("formula_histogram") or {}
    prototype_hist = sample.get("prototype_histogram") or {}
    n_hist = sample.get("n_histogram") or {}
    return {
        "decoded_samples": decoded,
        "parse_rate": float(sample.get("parse_rate") or rate(parsed, decoded)),
        "valid_N_rate": float(sample.get("valid_N_rate") or rate(sample.get("valid_N", 0), parsed)),
        "valid_formula_rate": float(sample.get("valid_formula_rate") or rate(sample.get("valid_formula", 0), parsed)),
        "valid_counts_rate": float(sample.get("valid_counts_rate") or rate(sample.get("valid_counts", 0), parsed)),
        "valid_elements_rate": float(sample.get("valid_elements_rate") or rate(sample.get("valid_elements", 0), parsed)),
        "valid_plan_rate": float(sample.get("valid_plan_rate") or rate(sample.get("valid_plan", 0), parsed)),
        "smact_plausible_rate": float(
            sample.get("smact_plausible_rate") or rate(sample.get("smact_plausible", 0), sample.get("smact_checked", 0))
        ),
        "single_element_rate": float(sample.get("single_element_rate") or rate(sample.get("single_element", 0), parsed)),
        "all_metal_rate": float(sample.get("all_metal_rate") or rate(sample.get("all_metal", 0), parsed)),
        "unique_formula_count": int(sample.get("unique_formula_count") or len(formula_hist)),
        "unique_prototype_count": int(sample.get("unique_prototype_count") or len(prototype_hist)),
        "top_formula_fraction": float(sample.get("top_formula_fraction") or top_fraction(formula_hist, parsed)),
        "top_prototype_fraction": float(sample.get("top_prototype_fraction") or top_fraction(prototype_hist, parsed)),
        "n_max_fraction": float(sample.get("n_max_fraction") or top_fraction(n_hist, parsed)),
    }


def raw_plan_metrics(path: Path) -> dict[str, Any]:
    decoded = 0
    parsed = 0
    valid_n = 0
    valid_formula = 0
    valid_counts = 0
    valid_elements = 0
    valid_plan = 0
    smact_checked = 0
    smact_plausible = 0
    single_element = 0
    all_metal = 0
    formula_hist: Counter[str] = Counter()
    prototype_hist: Counter[str] = Counter()
    n_hist: Counter[str] = Counter()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            decoded += 1
            record = json.loads(line)
            if not record.get("parsed"):
                continue
            parsed += 1
            plan = record.get("plan_state") or {}
            validation = record.get("plan_validation") or {}
            validator = plan.get("validator") or record.get("smact") or {}
            formula_hist[str(plan.get("formula"))] += 1
            prototype_hist[str(plan.get("prototype_key"))] += 1
            n_hist[str(plan.get("N"))] += 1
            valid_n += int(bool(validation.get("valid_N")))
            valid_formula += int(bool(validation.get("valid_formula")))
            valid_counts += int(bool(validation.get("valid_counts")))
            valid_elements += int(bool(validation.get("valid_elements")))
            valid_plan += int(bool(validation.get("valid")))
            smact_checked += 1
            smact_plausible += int(validator.get("valid") is True)
            single_element += int(len(plan.get("elements") or []) <= 1)
            all_metal += int(str(validator.get("reason")) == "all_metal_shortcut")

    if parsed <= 0:
        return {"decoded_samples": decoded, "parse_rate": rate(parsed, decoded)}
    formula_values = list(formula_hist.values())
    prototype_values = list(prototype_hist.values())
    n_values = list(n_hist.values())
    return {
        "decoded_samples": decoded,
        "parse_rate": rate(parsed, decoded),
        "valid_N_rate": rate(valid_n, parsed),
        "valid_formula_rate": rate(valid_formula, parsed),
        "valid_counts_rate": rate(valid_counts, parsed),
        "valid_elements_rate": rate(valid_elements, parsed),
        "valid_plan_rate": rate(valid_plan, parsed),
        "smact_plausible_rate": rate(smact_plausible, smact_checked),
        "single_element_rate": rate(single_element, parsed),
        "all_metal_rate": rate(all_metal, parsed),
        "unique_formula_count": len(formula_hist),
        "unique_prototype_count": len(prototype_hist),
        "top_formula_fraction": rate(max(formula_values or [0]), parsed),
        "top_prototype_fraction": rate(max(prototype_values or [0]), parsed),
        "n_max_fraction": rate(max(n_values or [0]), parsed),
        "raw_histograms": {
            "n_histogram": dict(n_hist.most_common()),
            "formula_histogram": dict(formula_hist.most_common(30)),
            "prototype_histogram": dict(prototype_hist.most_common(30)),
        },
    }


def failure_list(metrics: Mapping[str, Any], args) -> list[str]:
    failures: list[str] = []
    if int(metrics["decoded_samples"]) < int(args.min_decoded_samples):
        failures.append(f"decoded_samples {metrics['decoded_samples']} < {int(args.min_decoded_samples)}")
    checks = [
        ("parse_rate", ">=", args.min_parse_rate),
        ("valid_N_rate", ">=", args.min_valid_n),
        ("valid_formula_rate", ">=", args.min_valid_formula),
        ("valid_plan_rate", ">=", args.min_valid_plan),
        ("smact_plausible_rate", ">=", args.min_smact_plausible),
        ("single_element_rate", "<=", args.max_single_element),
        ("all_metal_rate", "<=", args.max_all_metal),
        ("top_formula_fraction", "<=", args.max_top_formula_fraction),
        ("top_prototype_fraction", "<=", args.max_top_prototype_fraction),
        ("n_max_fraction", "<=", args.max_n_fraction),
    ]
    for key, op, threshold in checks:
        value = float(metrics[key])
        if op == ">=" and value < float(threshold):
            failures.append(f"{key} {value:.4f} < {float(threshold):.4f}")
        if op == "<=" and value > float(threshold):
            failures.append(f"{key} {value:.4f} > {float(threshold):.4f}")
    if int(metrics["unique_formula_count"]) < int(args.min_unique_formula):
        failures.append(f"unique_formula_count {metrics['unique_formula_count']} < {int(args.min_unique_formula)}")
    if int(metrics["unique_prototype_count"]) < int(args.min_unique_prototype):
        failures.append(f"unique_prototype_count {metrics['unique_prototype_count']} < {int(args.min_unique_prototype)}")
    return failures


def histogram_payload(sample: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "n_histogram",
        "formula_histogram",
        "prototype_histogram",
        "charge_bucket_histogram",
        "anion_framework_histogram",
        "lattice_system_histogram",
        "spacegroup_bucket_histogram",
        "volume_per_atom_bin_histogram",
        "failures",
    ]
    payload: dict[str, Any] = {}
    for key in keys:
        value = sample.get(key) or {}
        if key in {"formula_histogram", "prototype_histogram"}:
            payload[key] = dict(Counter({str(k): int(v) for k, v in value.items()}).most_common(30))
        else:
            payload[key] = value
    return payload


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    metrics = result["metrics"]
    lines = [
        "# R5-D Plan-State 256 Gate",
        "",
        f"- passed: `{result['passed']}`",
        f"- decoded_samples: `{metrics['decoded_samples']}`",
        f"- parse_rate: `{metrics['parse_rate']:.6f}`",
        f"- valid_N_rate: `{metrics['valid_N_rate']:.6f}`",
        f"- valid_formula_rate: `{metrics['valid_formula_rate']:.6f}`",
        f"- valid_plan_rate: `{metrics['valid_plan_rate']:.6f}`",
        f"- smact_plausible_rate: `{metrics['smact_plausible_rate']:.6f}`",
        f"- single_element_rate: `{metrics['single_element_rate']:.6f}`",
        f"- all_metal_rate: `{metrics['all_metal_rate']:.6f}`",
        f"- unique_formula_count: `{metrics['unique_formula_count']}`",
        f"- top_formula_fraction: `{metrics['top_formula_fraction']:.6f}`",
        f"- unique_prototype_count: `{metrics['unique_prototype_count']}`",
        f"- top_prototype_fraction: `{metrics['top_prototype_fraction']:.6f}`",
        f"- n_max_fraction: `{metrics['n_max_fraction']:.6f}`",
        "",
    ]
    if result["failures"]:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {item}" for item in result["failures"])
        lines.append("")
    lines.extend(
        [
            "## Histograms",
            "",
            "```json",
            json.dumps(result["histograms"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Examples",
            "",
            "```json",
            json.dumps(result["examples"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--raw-generations-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--min-decoded-samples", type=int, default=256)
    parser.add_argument("--min-parse-rate", type=float, default=0.95)
    parser.add_argument("--min-valid-n", type=float, default=0.99)
    parser.add_argument("--min-valid-formula", type=float, default=0.99)
    parser.add_argument("--min-valid-plan", type=float, default=0.95)
    parser.add_argument("--min-smact-plausible", type=float, default=0.918)
    parser.add_argument("--max-single-element", type=float, default=0.01)
    parser.add_argument("--max-all-metal", type=float, default=0.50)
    parser.add_argument("--min-unique-formula", type=int, default=128)
    parser.add_argument("--max-top-formula-fraction", type=float, default=0.08)
    parser.add_argument("--min-unique-prototype", type=int, default=96)
    parser.add_argument("--max-top-prototype-fraction", type=float, default=0.12)
    parser.add_argument("--max-n-fraction", type=float, default=0.35)
    args = parser.parse_args()

    sample = load_json(args.sample_metrics)
    metrics = normalized_metrics(sample)
    raw_metrics = raw_plan_metrics(args.raw_generations_jsonl)
    metrics.update({key: value for key, value in raw_metrics.items() if key != "raw_histograms"})
    result = {
        "passed": False,
        "failures": failure_list(metrics, args),
        "metrics": metrics,
        "thresholds": {
            "min_parse_rate": args.min_parse_rate,
            "min_decoded_samples": args.min_decoded_samples,
            "min_valid_n": args.min_valid_n,
            "min_valid_formula": args.min_valid_formula,
            "min_valid_plan": args.min_valid_plan,
            "min_smact_plausible": args.min_smact_plausible,
            "max_single_element": args.max_single_element,
            "max_all_metal": args.max_all_metal,
            "min_unique_formula": args.min_unique_formula,
            "max_top_formula_fraction": args.max_top_formula_fraction,
            "min_unique_prototype": args.min_unique_prototype,
            "max_top_prototype_fraction": args.max_top_prototype_fraction,
            "max_n_fraction": args.max_n_fraction,
        },
        "sample_metrics": sample,
        "histograms": histogram_payload(sample),
        "raw_histograms": raw_metrics.get("raw_histograms", {}),
        "examples": raw_examples(args.raw_generations_jsonl),
    }
    result["passed"] = not result["failures"]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
