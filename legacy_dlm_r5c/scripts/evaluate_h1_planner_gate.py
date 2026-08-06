#!/usr/bin/env python3
"""Evaluate H1 LLM formula-planner gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_r5c_plan_distribution import (  # noqa: E402
    compare_stats,
    load_generated,
    load_teacher,
    summarize,
)
from scripts.evaluate_r5c_de_novo_gate import raw_plan_stats, rate, threshold_failures  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    metrics = result["metrics"]
    lines = [
        "# H1 LLM Planner Gate",
        "",
        f"- passed_target: `{result['passed_target']}`",
        f"- passed_acceptable: `{result['passed_acceptable']}`",
        f"- decoded_samples: `{metrics['decoded_samples']}`",
        f"- formula_parse: `{metrics['formula_parse']:.6f}`",
        f"- end_marker: `{metrics['end_marker']:.6f}`",
        f"- valid_N: `{metrics['valid_N']:.6f}`",
        f"- single_element: `{metrics['single_element']:.6f}`",
        f"- all_metal: `{metrics['all_metal']:.6f}`",
        f"- mean_N: `{metrics['mean_N']:.6f}`",
        f"- N>=12: `{metrics['n_ge_12']:.6f}`",
        f"- ternary: `{metrics['ternary']:.6f}`",
        f"- 4+ elements: `{metrics['four_plus_elements']:.6f}`",
        f"- N TVD vs train: `{metrics.get('n_tvd', 0.0):.6f}`",
        f"- arity TVD vs train: `{metrics.get('arity_tvd', 0.0):.6f}`",
        f"- rich_field_valid: `{metrics.get('rich_field_valid', 0.0):.6f}`",
        "",
    ]
    if result["target_failures"]:
        lines.extend(["## Target Failures", ""])
        lines.extend(f"- {item}" for item in result["target_failures"])
        lines.append("")
    if result["acceptable_failures"]:
        lines.extend(["## Acceptable Failures", ""])
        lines.extend(f"- {item}" for item in result["acceptable_failures"])
        lines.append("")
    lines.extend(
        [
            "## Plan Distribution",
            "",
            "```json",
            json.dumps(result["plan_distribution"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    if result.get("distribution_comparison"):
        lines.extend(
            [
                "## Teacher Comparison",
                "",
                "```json",
                json.dumps(result["distribution_comparison"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--raw-generations-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-jsonl", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    sample = load_json(args.sample_metrics)
    raw_stats = raw_plan_stats(args.raw_generations_jsonl)
    decoded = int(sample.get("decoded_samples") or raw_stats.get("rows") or 0)
    metrics: dict[str, Any] = {
        "decoded_samples": decoded,
        "plan_parse": float(sample.get("plan_parse_rate") or rate(sample.get("plan_parse_success", 0), decoded)),
        "formula_parse": float(sample.get("formula_parse_rate") or raw_stats.get("formula_parse_rate") or 0.0),
        "end_marker": float(sample.get("plan_end_marker_rate") or raw_stats.get("end_marker_rate") or 0.0),
        "valid_N": float(sample.get("valid_n_rate") or rate(sample.get("valid_n_success", 0), decoded)),
        "valid_formula": float(sample.get("valid_formula_rate") or rate(sample.get("valid_formula_success", 0), decoded)),
        "single_element": float(raw_stats.get("single_element_rate") or sample.get("single_element_rate") or 0.0),
        "all_metal": float(raw_stats.get("all_metal_rate") or 0.0),
        "mean_N": float(raw_stats.get("mean_N") or 0.0),
        "n_ge_12": float(raw_stats.get("n_ge_12_rate") or 0.0),
        "ternary": float(raw_stats.get("ternary_rate") or 0.0),
        "four_plus_elements": float(raw_stats.get("four_plus_elements_rate") or 0.0),
        "rich_field_valid": float(
            sample.get("rich_field_valid_rate")
            or raw_stats.get("rich_field_valid_rate")
            or 0.0
        ),
    }
    rich_field_required = bool(sample.get("rich_field_required")) or int(raw_stats.get("rich_plan_rows") or 0) > 0
    distribution_result: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    if args.teacher_jsonl is not None and args.teacher_jsonl.exists():
        teacher_raw = load_teacher(args.teacher_jsonl, dedupe=True)
        generated_raw = load_generated(args.raw_generations_jsonl)
        comparison = compare_stats(teacher_raw, generated_raw)
        distribution_result = {
            "teacher": summarize(teacher_raw, top_k=30),
            "generated": summarize(generated_raw, top_k=30),
            "comparison": comparison,
        }
        metrics["n_tvd"] = float(comparison["n_tvd"])
        # H1 plans expose only formula text; arity is derived from the formula
        # rather than a semantic ``arity:`` field.
        metrics["arity_tvd"] = float(comparison["num_elements_tvd"])
        for key in ("anion_framework", "charge_bucket", "lattice_system", "spacegroup_bucket", "volume_per_atom_bin"):
            comparison_key = f"{key}_tvd"
            if comparison_key in comparison:
                metrics[comparison_key] = float(comparison[comparison_key])

    target_thresholds = {
        "min_formula_parse": 0.995,
        "min_end_marker": 0.995,
        "min_valid_N": 0.995,
        "max_single_element": 0.01,
        "max_all_metal": 0.35,
        "min_mean_N": 10.0,
        "max_mean_N": 10.9,
        "min_n_ge_12": 0.38,
        "min_ternary": 0.55,
        "max_four_plus_elements": 0.25,
    }
    acceptable_thresholds = {
        "min_formula_parse": 0.98,
        "min_end_marker": 0.98,
        "min_valid_N": 0.98,
        "max_single_element": 0.06,
        "max_all_metal": 0.40,
        "min_mean_N": 9.5,
        "max_mean_N": 11.3,
        "min_n_ge_12": 0.35,
        "min_ternary": 0.50,
        "max_four_plus_elements": 0.30,
    }
    if comparison is not None:
        target_thresholds.update({"max_n_tvd": 0.25, "max_arity_tvd": 0.25})
        acceptable_thresholds.update({"max_n_tvd": 0.35, "max_arity_tvd": 0.35})
        if rich_field_required:
            for key in ("anion_framework", "charge_bucket", "lattice_system", "spacegroup_bucket", "volume_per_atom_bin"):
                if f"{key}_tvd" in metrics:
                    target_thresholds[f"max_{key}_tvd"] = 0.25
                    acceptable_thresholds[f"max_{key}_tvd"] = 0.35
    if rich_field_required:
        target_thresholds["min_rich_field_valid"] = 0.995
        acceptable_thresholds["min_rich_field_valid"] = 0.98

    target_failures = threshold_failures(metrics, target_thresholds)
    acceptable_failures = threshold_failures(metrics, acceptable_thresholds)
    result = {
        "passed_target": not target_failures,
        "passed_acceptable": not acceptable_failures,
        "target_failures": target_failures,
        "acceptable_failures": acceptable_failures,
        "metrics": metrics,
        "target_thresholds": target_thresholds,
        "acceptable_thresholds": acceptable_thresholds,
        "sample_metrics": sample,
        "rich_field_required": rich_field_required,
        "plan_distribution": raw_stats,
        "distribution_comparison": distribution_result,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
