#!/usr/bin/env python3
"""Evaluate H2 LLM-plan + plain-text DLM proposal gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_r5c_de_novo_gate import rate, threshold_failures  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    metrics = result["metrics"]
    lines = [
        "# H2 Plain-Text DLM Proposal Gate",
        "",
        f"- passed_target: `{result['passed_target']}`",
        f"- passed_acceptable: `{result['passed_acceptable']}`",
        f"- planner_parse: `{metrics['planner_parse']:.6f}`",
        f"- dlm_text_parse: `{metrics['dlm_text_parse']:.6f}`",
        f"- composition_match: `{metrics['composition_match']:.6f}`",
        f"- graph_acceptance: `{metrics['graph_acceptance']:.6f}`",
        f"- pbc_duplicate: `{metrics['pbc_duplicate']:.6f}`",
        f"- all_metal: `{metrics['all_metal']:.6f}`",
        f"- n_tvd: `{metrics.get('n_tvd', 0.0):.6f}`",
        f"- arity_tvd: `{metrics.get('arity_tvd', 0.0):.6f}`",
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
            "## Planner Gate",
            "",
            "```json",
            json.dumps(result["planner_gate"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## H2 Raw Metrics",
            "",
            "```json",
            json.dumps(result["h2_sample_metrics"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-gate-json", type=Path, required=True)
    parser.add_argument("--h2-sample-metrics", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    planner_gate = load_json(args.planner_gate_json)
    sample = load_json(args.h2_sample_metrics)
    planner_metrics = planner_gate.get("metrics", {})
    decoded = int(sample.get("decoded_samples") or 0)
    metrics = {
        "planner_parse": float(planner_metrics.get("plan_parse") or planner_metrics.get("formula_parse") or 0.0),
        "dlm_text_parse": float(sample.get("parse_rate") or rate(sample.get("parse_success", 0), decoded)),
        "composition_match": float(sample.get("composition_match_rate") or rate(sample.get("composition_match_success", 0), decoded)),
        "graph_acceptance": float(sample.get("graph_acceptance_rate") or rate(sample.get("graph_success", 0), decoded)),
        "pbc_duplicate": float(sample.get("pbc_duplicate_rate") or rate(sample.get("pbc_duplicate_failures", 0), decoded)),
        "all_metal": float(planner_metrics.get("all_metal") or 0.0),
        "mean_N": float(planner_metrics.get("mean_N") or 0.0),
        "n_tvd": float(planner_metrics.get("n_tvd") or 0.0),
        "arity_tvd": float(planner_metrics.get("arity_tvd") or 0.0),
        "planner_distribution_acceptable": bool(planner_gate.get("passed_acceptable")),
        "decoded_samples": decoded,
    }
    target_thresholds = {
        "min_planner_parse": 0.995,
        "min_dlm_text_parse": 0.92,
        "min_composition_match": 0.92,
        "min_graph_acceptance": 0.85,
        "max_pbc_duplicate": 0.01,
        "max_all_metal": 0.35,
        "max_n_tvd": 0.25,
        "max_arity_tvd": 0.25,
    }
    acceptable_thresholds = {
        "min_planner_parse": 0.98,
        "min_dlm_text_parse": 0.85,
        "min_composition_match": 0.85,
        "min_graph_acceptance": 0.75,
        "max_pbc_duplicate": 0.05,
        "max_all_metal": 0.40,
        "max_n_tvd": 0.35,
        "max_arity_tvd": 0.35,
    }
    target_failures = threshold_failures(metrics, target_thresholds)
    acceptable_failures = threshold_failures(metrics, acceptable_thresholds)
    if not metrics["planner_distribution_acceptable"]:
        target_failures.append("planner distribution gate did not pass acceptable")
        acceptable_failures.append("planner distribution gate did not pass acceptable")
    result = {
        "passed_target": not target_failures,
        "passed_acceptable": not acceptable_failures,
        "target_failures": target_failures,
        "acceptable_failures": acceptable_failures,
        "metrics": metrics,
        "target_thresholds": target_thresholds,
        "acceptable_thresholds": acceptable_thresholds,
        "planner_gate": planner_gate,
        "h2_sample_metrics": sample,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
