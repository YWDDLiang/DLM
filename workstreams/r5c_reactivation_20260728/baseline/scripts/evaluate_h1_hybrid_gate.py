#!/usr/bin/env python3
"""Evaluate H1 hybrid planner + R5-C exact-body 256 gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_r5c_de_novo_gate import rate, threshold_failures


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    metrics = result["metrics"]
    lines = [
        "# H1 Hybrid Gate",
        "",
        f"- passed_target: `{result['passed_target']}`",
        f"- passed_acceptable: `{result['passed_acceptable']}`",
        f"- planner_decoded: `{metrics['planner_decoded']}`",
        f"- body_decoded: `{metrics['body_decoded']}`",
        f"- planner_parse: `{metrics['planner_parse']:.6f}`",
        f"- body_parse: `{metrics['body_parse']:.6f}`",
        f"- plan_body_match: `{metrics['plan_body_match']:.6f}`",
        f"- graph_acceptance: `{metrics['graph_acceptance']:.6f}`",
        f"- planner_distribution_acceptable: `{metrics['planner_distribution_acceptable']}`",
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
            "## Body Metrics",
            "",
            "```json",
            json.dumps(result["body_metrics"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-gate-json", type=Path, required=True)
    parser.add_argument("--body-sample-metrics", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    planner_gate = load_json(args.planner_gate_json)
    body_metrics = load_json(args.body_sample_metrics)
    planner_metrics = planner_gate.get("metrics", {})
    planner_decoded = int(planner_metrics.get("decoded_samples") or 0)
    body_decoded = int(body_metrics.get("decoded_samples") or 0)
    metrics = {
        "planner_decoded": planner_decoded,
        "body_decoded": body_decoded,
        "planner_parse": float(planner_metrics.get("plan_parse") or planner_metrics.get("formula_parse") or 0.0),
        "body_parse": float(body_metrics.get("parse_rate") or rate(body_metrics.get("parse_success", 0), body_decoded)),
        "plan_body_match": float(body_metrics.get("plan_match_rate") or rate(body_metrics.get("plan_match_success", 0), body_decoded)),
        "graph_acceptance": float(body_metrics.get("graph_acceptance_rate") or rate(body_metrics.get("graph_success", 0), body_decoded)),
        "planner_distribution_acceptable": bool(planner_gate.get("passed_acceptable")),
    }
    target_thresholds = {
        "min_planner_parse": 0.995,
        "min_body_parse": 0.95,
        "min_plan_body_match": 0.95,
        "min_graph_acceptance": 0.85,
    }
    acceptable_thresholds = {
        "min_planner_parse": 0.98,
        "min_body_parse": 0.90,
        "min_plan_body_match": 0.90,
        "min_graph_acceptance": 0.80,
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
        "body_metrics": body_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
