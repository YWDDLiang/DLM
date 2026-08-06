#!/usr/bin/env python3
"""Evaluate R5-C composition-plan de novo sampling gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping


METAL_SYMBOLS = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rate(numerator: Any, denominator: Any) -> float:
    try:
        return float(numerator) / max(1.0, float(denominator))
    except Exception:
        return 0.0


def raw_plan_stats(path: Path) -> dict[str, Any]:
    rows = 0
    parsed = 0
    formula_parsed = 0
    end_marker = 0
    tail_after_end_marker = 0
    single = 0
    all_metal = 0
    n_sum = 0
    n_ge_12 = 0
    ternary = 0
    four_plus = 0
    family_match = 0
    arity_match = 0
    size_match = 0
    rich_rows = 0
    rich_field_valid = 0
    n_hist: dict[str, int] = {}
    formula_hist: dict[str, int] = {}
    rich_histograms: dict[str, dict[str, int]] = {
        "anion_framework": {},
        "charge_bucket": {},
        "lattice_system": {},
        "spacegroup_bucket": {},
        "volume_per_atom_bin": {},
    }
    examples: list[dict[str, Any]] = []
    if not path.exists():
        return {
            "rows": 0,
            "parsed_plan_rows": 0,
            "formula_parse_rate": 0.0,
            "end_marker_rate": 0.0,
            "plan_tail_after_end_marker_rate": 0.0,
            "single_element_rate": 0.0,
            "all_metal_rate": 0.0,
            "n_histogram": {},
            "formula_top": {},
            "rich_plan_rows": 0,
            "rich_field_valid_rate": 0.0,
            "rich_field_histograms": {},
            "failure_examples": [],
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            raw_text = str(record.get("raw_plan_text") or record.get("plan_text") or "")
            marker_match = re.search(r"(?im)^\s*end\s*:\s*plan\s*$", raw_text)
            marker_present = bool(record.get("plan_end_marker_present")) or marker_match is not None
            if marker_present:
                end_marker += 1
            if bool(record.get("plan_tail_after_end_marker")):
                tail_after_end_marker += 1
            elif marker_match is not None:
                tail = raw_text[marker_match.end() :].strip()
                if re.search(r"(?im)^\s*body\s*:\s*$|<N_\d{3}>|<LA_\d{3}>|<E_[A-Z][a-z]?>", tail):
                    tail_after_end_marker += 1
            if record.get("formula_parse") is True:
                formula_parsed += 1
            plan = record.get("parsed_plan")
            if isinstance(plan, Mapping):
                parsed += 1
                if record.get("formula_parse") is not True and record.get("formula_parse") is not False:
                    formula_parsed += 1
                elements = [str(value) for value in (plan.get("elements") or [])]
                n_value = int(plan.get("N"))
                n_sum += n_value
                if n_value >= 12:
                    n_ge_12 += 1
                if len(elements) == 3:
                    ternary += 1
                if len(elements) >= 4:
                    four_plus += 1
                if len(elements) == 1:
                    single += 1
                if elements and all(symbol in METAL_SYMBOLS for symbol in elements):
                    all_metal += 1
                consistency = record.get("semantic_consistency") or plan.get("semantic_consistency") or {}
                if consistency.get("family_match_formula") is True or plan.get("family_match_formula") is True:
                    family_match += 1
                if consistency.get("arity_match_formula") is True or plan.get("arity_match_formula") is True:
                    arity_match += 1
                if consistency.get("size_match_formula") is True or plan.get("size_match_formula") is True:
                    size_match += 1
                if isinstance(plan.get("generated_rich_fields"), Mapping) or plan.get("plan_format") in (
                    "h1_rich_plan_v1",
                    "h1_rich_nocharge_plan_v1",
                ):
                    rich_rows += 1
                    if plan.get("rich_field_valid") is True:
                        rich_field_valid += 1
                    for key in rich_histograms:
                        value = str(plan.get(key, "unknown"))
                        rich_histograms[key][value] = rich_histograms[key].get(value, 0) + 1
                n_key = str(n_value)
                n_hist[n_key] = n_hist.get(n_key, 0) + 1
                formula = str(plan.get("formula"))
                formula_hist[formula] = formula_hist.get(formula, 0) + 1
            elif len(examples) < 10:
                examples.append(
                    {
                        "sample_idx": record.get("sample_idx"),
                        "stage": record.get("stage"),
                        "reason": record.get("reason"),
                        "raw_plan_text": str(record.get("raw_plan_text", ""))[:240],
                    }
                )
    formula_top = dict(sorted(formula_hist.items(), key=lambda item: item[1], reverse=True)[:25])
    return {
        "rows": rows,
        "parsed_plan_rows": parsed,
        "formula_parse_rate": formula_parsed / max(1, rows),
        "end_marker_rate": end_marker / max(1, rows),
        "plan_tail_after_end_marker_rate": tail_after_end_marker / max(1, rows),
        "single_element_rate": single / max(1, rows),
        "all_metal_rate": all_metal / max(1, rows),
        "mean_N": n_sum / max(1, parsed),
        "n_ge_12_rate": n_ge_12 / max(1, rows),
        "ternary_rate": ternary / max(1, rows),
        "four_plus_elements_rate": four_plus / max(1, rows),
        "family_match_formula_rate": family_match / max(1, rows),
        "arity_match_formula_rate": arity_match / max(1, rows),
        "size_match_formula_rate": size_match / max(1, rows),
        "rich_plan_rows": rich_rows,
        "rich_field_valid_rate": rich_field_valid / max(1, rows),
        "rich_field_valid_rate_parsed_rich": rich_field_valid / max(1, rich_rows),
        "rich_field_histograms": {
            key: dict(sorted(values.items(), key=lambda item: (-item[1], item[0])))
            for key, values in rich_histograms.items()
        },
        "n_histogram": dict(sorted(n_hist.items(), key=lambda item: (-item[1], item[0]))),
        "formula_top": formula_top,
        "failure_examples": examples,
    }


def threshold_failures(metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> list[str]:
    failures: list[str] = []
    for key, threshold in thresholds.items():
        value = float(metrics.get(key, 0.0))
        if key.startswith("max_"):
            metric_key = key.removeprefix("max_")
            value = float(metrics.get(metric_key, 0.0))
            if value > float(threshold):
                failures.append(f"{metric_key} {value:.4f} > {float(threshold):.4f}")
        elif key.startswith("min_"):
            metric_key = key.removeprefix("min_")
            value = float(metrics.get(metric_key, 0.0))
            if value < float(threshold):
                failures.append(f"{metric_key} {value:.4f} < {float(threshold):.4f}")
    return failures


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    metrics = result["metrics"]
    lines = [
        "# R5-C Composition-Plan De Novo Gate",
        "",
        f"- passed_target: `{result['passed_target']}`",
        f"- passed_acceptable: `{result['passed_acceptable']}`",
        f"- decoded_samples: `{metrics['decoded_samples']}`",
        f"- plan_parse: `{metrics['plan_parse']:.6f}`",
        f"- formula_parse: `{metrics['formula_parse']:.6f}`",
        f"- end_marker: `{metrics['end_marker']:.6f}`",
        f"- plan_tail_after_end_marker: `{metrics['plan_tail_after_end_marker']:.6f}`",
        f"- valid_N: `{metrics['valid_N']:.6f}`",
        f"- valid_formula: `{metrics['valid_formula']:.6f}`",
        f"- single_element: `{metrics['single_element']:.6f}`",
        f"- all_metal: `{metrics['all_metal']:.6f}`",
        f"- mean_N: `{metrics['mean_N']:.6f}`",
        f"- N>=12: `{metrics['n_ge_12']:.6f}`",
        f"- ternary: `{metrics['ternary']:.6f}`",
        f"- 4+ elements: `{metrics['four_plus_elements']:.6f}`",
        f"- family_match_formula: `{metrics['family_match_formula']:.6f}`",
        f"- arity_match_formula: `{metrics['arity_match_formula']:.6f}`",
        f"- size_match_formula: `{metrics['size_match_formula']:.6f}`",
        f"- body_parse: `{metrics['body_parse']:.6f}`",
        f"- generated_plan_body_match: `{metrics['generated_plan_body_match']:.6f}`",
        f"- graph_acceptance: `{metrics['graph_acceptance']:.6f}`",
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
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--raw-generations-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--enable-distribution-gates", action="store_true")
    parser.add_argument("--enable-semantic-gates", action="store_true")
    parser.add_argument("--enable-formula-end-gates", action="store_true")
    args = parser.parse_args()

    sample = load_json(args.sample_metrics)
    raw_stats = raw_plan_stats(args.raw_generations_jsonl)
    decoded = int(sample.get("decoded_samples") or raw_stats.get("rows") or 0)
    metrics = {
        "decoded_samples": decoded,
        "plan_parse": float(sample.get("plan_parse_rate") or rate(sample.get("plan_parse_success", 0), decoded)),
        "formula_parse": float(
            sample.get("formula_parse_rate")
            or raw_stats.get("formula_parse_rate")
            or sample.get("valid_formula_rate")
            or rate(sample.get("valid_formula_success", 0), decoded)
        ),
        "end_marker": float(sample.get("plan_end_marker_rate") or raw_stats.get("end_marker_rate") or 0.0),
        "plan_tail_after_end_marker": float(
            sample.get("plan_tail_after_end_marker_rate")
            or raw_stats.get("plan_tail_after_end_marker_rate")
            or 0.0
        ),
        "valid_N": float(sample.get("valid_n_rate") or rate(sample.get("valid_n_success", 0), decoded)),
        "valid_formula": float(sample.get("valid_formula_rate") or rate(sample.get("valid_formula_success", 0), decoded)),
        "single_element": float(raw_stats.get("single_element_rate") or sample.get("single_element_rate") or 0.0),
        "all_metal": float(raw_stats.get("all_metal_rate") or 0.0),
        "mean_N": float(raw_stats.get("mean_N") or 0.0),
        "n_ge_12": float(raw_stats.get("n_ge_12_rate") or 0.0),
        "ternary": float(raw_stats.get("ternary_rate") or 0.0),
        "four_plus_elements": float(raw_stats.get("four_plus_elements_rate") or 0.0),
        "family_match_formula": float(
            raw_stats.get("family_match_formula_rate")
            or sample.get("family_match_formula_rate")
            or 0.0
        ),
        "arity_match_formula": float(
            raw_stats.get("arity_match_formula_rate")
            or sample.get("arity_match_formula_rate")
            or 0.0
        ),
        "size_match_formula": float(
            raw_stats.get("size_match_formula_rate")
            or sample.get("size_match_formula_rate")
            or 0.0
        ),
        "body_parse": float(sample.get("body_parse_rate") or rate(sample.get("body_parse_success", 0), decoded)),
        "generated_plan_body_match": float(sample.get("plan_match_rate") or rate(sample.get("plan_match_success", 0), decoded)),
        "graph_acceptance": float(sample.get("graph_acceptance_rate") or rate(sample.get("graph_success", 0), decoded)),
    }
    target_thresholds = {
        "min_plan_parse": 0.99,
        "min_valid_N": 0.99,
        "min_valid_formula": 0.99,
        "max_single_element": 0.01,
        "min_body_parse": 0.95,
        "min_generated_plan_body_match": 0.95,
        "min_graph_acceptance": 0.85,
    }
    acceptable_thresholds = {
        "min_plan_parse": 0.94,
        "min_valid_N": 0.94,
        "min_valid_formula": 0.94,
        "max_single_element": 0.06,
        "min_body_parse": 0.90,
        "min_generated_plan_body_match": 0.90,
        "min_graph_acceptance": 0.80,
    }
    if args.enable_distribution_gates:
        target_thresholds.update(
            {
                "max_all_metal": 0.35,
                "min_mean_N": 10.0,
                "max_mean_N": 10.9,
                "min_n_ge_12": 0.38,
                "min_ternary": 0.55,
                "max_four_plus_elements": 0.25,
            }
        )
        acceptable_thresholds.update(
            {
                "max_all_metal": 0.40,
                "min_mean_N": 9.5,
                "max_mean_N": 11.3,
                "min_n_ge_12": 0.35,
                "min_ternary": 0.50,
                "max_four_plus_elements": 0.30,
            }
        )
    if args.enable_formula_end_gates:
        target_thresholds.update(
            {
                "min_formula_parse": 0.99,
                "min_end_marker": 0.99,
            }
        )
        acceptable_thresholds.update(
            {
                "min_formula_parse": 0.94,
                "min_end_marker": 0.94,
            }
        )
    if args.enable_semantic_gates:
        target_thresholds.update(
            {
                "min_family_match_formula": 0.95,
                "min_arity_match_formula": 0.98,
                "min_size_match_formula": 0.95,
            }
        )
        acceptable_thresholds.update(
            {
                "min_family_match_formula": 0.90,
                "min_arity_match_formula": 0.94,
                "min_size_match_formula": 0.90,
            }
        )
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
        "plan_distribution": raw_stats,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
