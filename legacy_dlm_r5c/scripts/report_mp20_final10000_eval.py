#!/usr/bin/env python3
"""Summarize the MP-20 10000-sample CrysLLMGen + MatterGen S.U.N evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_metric(payload: Mapping[str, Any], key: str) -> Any:
    if not payload:
        return None
    metrics = payload.get("metrics", payload)
    if isinstance(metrics, Mapping):
        value = metrics.get(key)
        if isinstance(value, Mapping) and "value" in value:
            return value["value"]
        return value
    return None


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return str(value)


def fmt_metric_pct_or_raw(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if 0.0 <= number <= 1.0:
        return f"{100.0 * number:.{digits}f}"
    return f"{number:.{digits}f}"


def fmt_named_metric(key: str, value: Any) -> str:
    raw_metric_keys = {
        "wdist_density",
        "wdist_num_elems",
        "avg_energy_above_hull_per_atom",
    }
    if key in raw_metric_keys:
        return fmt_float(value)
    return fmt_metric_pct_or_raw(value)


def sun_threshold_rates(sun: Mapping[str, Any]) -> Mapping[str, Any]:
    thresholds = sun.get("sun_thresholds") or {}
    if not isinstance(thresholds, Mapping):
        return {}
    rates = thresholds.get("rates") or thresholds.get("rates_submitted") or {}
    return rates if isinstance(rates, Mapping) else {}


def comp_summary(comp: Mapping[str, Any], key: str) -> dict[str, Any]:
    summary = comp.get(key, {})
    if not isinstance(summary, Mapping):
        return {}
    reasons = summary.get("reason_counts", {})
    total = float(summary.get("count") or 0)

    def rate(reason: str) -> float | None:
        if total <= 0:
            return None
        return float(reasons.get(reason, 0)) / total

    return {
        "count": summary.get("count"),
        "comp_valid": summary.get("comp_valid_rate"),
        "strict_valid": rate("charge_neutral_pauling_valid"),
        "single_element": rate("single_element_shortcut"),
        "all_metal": rate("all_metal_shortcut"),
        "pbc_duplicate": summary.get("pbc_equivalent_duplicate_fraction"),
        "top_reasons": dict(list(reasons.items())[:8]),
        "top_formulas": dict(list(summary.get("formula_top30", {}).items())[:10]),
    }


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--generation-schedule", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--composition-summary", type=Path, required=True)
    parser.add_argument("--failure-modes", type=Path, default=None)
    parser.add_argument("--crysllmgen-metrics", type=Path, required=True)
    parser.add_argument("--sun-summary", type=Path, required=True)
    parser.add_argument("--baseline-crysllmgen-metrics", type=Path, default=None)
    parser.add_argument("--baseline-composition-summary", type=Path, default=None)
    parser.add_argument("--baseline-sun-summary", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    sample = read_json(args.sample_metrics)
    comp = read_json(args.composition_summary)
    failure = read_json(args.failure_modes)
    crys = read_json(args.crysllmgen_metrics)
    sun = read_json(args.sun_summary)
    base_crys = read_json(args.baseline_crysllmgen_metrics)
    base_comp = read_json(args.baseline_composition_summary)
    base_sun = read_json(args.baseline_sun_summary)

    raw_comp = comp_summary(comp, "raw_jsonl")
    refined_comp = comp_summary(comp, "refined_pt")
    base_refined_comp = comp_summary(base_comp, "refined_pt")

    selected_note = (
        "The 1000-sample MatterGen S.U.N screen selected "
        "`20260522_142200-lowrl5-nelemseq-refined1000`, because its S.U.N was "
        f"{fmt_pct(base_sun.get('frac_novel_unique_stable_structures'))} versus "
        "`20260522_124500-sftbest-nelemseq-refined1000` at 6.90%."
        if base_sun
        else "The selected candidate is the low-lr 5epoch checkpoint under the n-elements sequential schedule."
    )

    crys_keys = [
        "comp_valid",
        "struct_valid",
        "valid",
        "wdist_density",
        "wdist_num_elems",
        "cov_recall",
        "cov_precision",
    ]
    sun_keys = [
        "frac_novel_unique_stable_structures",
        "frac_stable_structures",
        "frac_novel_structures",
        "frac_unique_structures",
        "frac_novel_unique_structures",
        "avg_comp_validity",
        "avg_structure_validity",
        "frac_successful_jobs",
        "avg_energy_above_hull_per_atom",
    ]
    sun_rate = sun_threshold_rates(sun)
    base_sun_rate = sun_threshold_rates(base_sun)

    lines: list[str] = [
        f"# MP-20 Final 10000 Evaluation: {args.run_id}",
        "",
        "## Selection",
        "",
        selected_note,
        "",
        "## Run Configuration",
        "",
        f"- checkpoint: `{args.checkpoint_path}`",
        f"- generation schedule: `{args.generation_schedule}`",
        f"- temperature: `{args.temperature}`",
        "- sampling constraints: `block_length=1`, schema mask, slot prefill, atom-count grammar, PBC-aware duplicate mask, lattice-volume mask",
        "- refinement: CrysLLMGen direct refinement, no SMACT reranking",
        "- S.U.N: MatterGen/MatterSim relaxation with disordered structure matcher; built-in stable threshold is treated as `meta_sun`",
        "",
        "## Sampling",
        "",
        *table(
            ["metric", "value"],
            [
                ["decoded_samples", str(sample.get("decoded_samples", "n/a"))],
                ["graph_success", str(sample.get("graph_success", "n/a"))],
                ["target_reached", str(sample.get("target_reached", "n/a"))],
                ["parse_rate", fmt_pct(sample.get("parse_rate"))],
                ["graph_acceptance_rate", fmt_pct(sample.get("graph_acceptance_rate"))],
                ["time_sec", fmt_float(sample.get("time_sec"), 1)],
            ],
        ),
        "",
        "## Composition Diagnosis",
        "",
        *table(
            ["scope", "count", "comp_valid", "strict", "single_element", "all_metal", "PBC duplicate"],
            [
                [
                    "raw",
                    str(raw_comp.get("count", "n/a")),
                    fmt_pct(raw_comp.get("comp_valid")),
                    fmt_pct(raw_comp.get("strict_valid")),
                    fmt_pct(raw_comp.get("single_element")),
                    fmt_pct(raw_comp.get("all_metal")),
                    fmt_pct(raw_comp.get("pbc_duplicate")),
                ],
                [
                    "refined",
                    str(refined_comp.get("count", "n/a")),
                    fmt_pct(refined_comp.get("comp_valid")),
                    fmt_pct(refined_comp.get("strict_valid")),
                    fmt_pct(refined_comp.get("single_element")),
                    fmt_pct(refined_comp.get("all_metal")),
                    fmt_pct(refined_comp.get("pbc_duplicate")),
                ],
                [
                    "1000 baseline refined",
                    str(base_refined_comp.get("count", "n/a")),
                    fmt_pct(base_refined_comp.get("comp_valid")),
                    fmt_pct(base_refined_comp.get("strict_valid")),
                    fmt_pct(base_refined_comp.get("single_element")),
                    fmt_pct(base_refined_comp.get("all_metal")),
                    fmt_pct(base_refined_comp.get("pbc_duplicate")),
                ],
            ],
        ),
        "",
        "Top refined composition reasons:",
        "",
        "```json",
        json.dumps(refined_comp.get("top_reasons", {}), ensure_ascii=False, indent=2),
        "```",
        "",
    ]

    if failure:
        lines.extend(
            [
                "Raw comp_valid bottleneck:",
                "",
                "```json",
                json.dumps(failure.get("headline", []), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## CrysLLMGen Metrics",
            "",
            *table(
                ["metric", "1000 baseline", "10000 final"],
                [
                    [
                        key,
                        fmt_named_metric(key, get_metric(base_crys, key)),
                        fmt_named_metric(key, get_metric(crys, key)),
                    ]
                    for key in crys_keys
                ],
            ),
            "",
        "## MatterGen S.U.N",
        "",
        *table(
                ["metric", "1000 baseline", "10000 final"],
                [
                    [
                        key,
                        fmt_named_metric(key, get_metric(base_sun, key)),
                        fmt_named_metric(key, get_metric(sun, key)),
                    ]
                    for key in sun_keys
                ],
        ),
        "",
        "Strict/meta S.U.N thresholds:",
        "",
        *table(
            ["metric", "1000 baseline", "10000 final"],
            [
                [
                    "strict_sun (Ehull < 0.0)",
                    fmt_metric_pct_or_raw(base_sun_rate.get("strict_sun")),
                    fmt_metric_pct_or_raw(sun_rate.get("strict_sun")),
                ],
                [
                    "meta_sun (Ehull < 0.1)",
                    fmt_metric_pct_or_raw(base_sun_rate.get("meta_sun")),
                    fmt_metric_pct_or_raw(sun_rate.get("meta_sun")),
                ],
                [
                    "strict_stable",
                    fmt_metric_pct_or_raw(base_sun_rate.get("strict_stable")),
                    fmt_metric_pct_or_raw(sun_rate.get("strict_stable")),
                ],
                [
                    "meta_stable",
                    fmt_metric_pct_or_raw(base_sun_rate.get("meta_stable")),
                    fmt_metric_pct_or_raw(sun_rate.get("meta_stable")),
                ],
            ],
        ),
        "",
            "S.U.N run health:",
            "",
            *table(
                ["metric", "value"],
                [
                    ["num_structures", str(sun.get("num_structures", "n/a"))],
                    ["num_supported_before_relax", str(sun.get("num_supported_structures_before_relax", "n/a"))],
                    ["n_unsupported_failed", str(sun.get("n_unsupported_failed", "n/a"))],
                    ["n_relax_failed", str(sun.get("n_relax_failed", "n/a"))],
                    ["metric_errors", ", ".join(sorted((sun.get("metric_errors") or {}).keys())) or "none"],
                ],
            ),
            "",
            "## Acceptance Check",
            "",
            *table(
                ["target", "10000 final"],
                [
                    ["comp_valid >= 90", fmt_metric_pct_or_raw(get_metric(crys, "comp_valid"))],
                    ["struct_valid >= 99", fmt_metric_pct_or_raw(get_metric(crys, "struct_valid"))],
                    ["wdist_density <= 0.85", fmt_float(get_metric(crys, "wdist_density"))],
                    ["cov_recall >= 93", fmt_metric_pct_or_raw(get_metric(crys, "cov_recall"))],
                    ["cov_precision >= 94", fmt_metric_pct_or_raw(get_metric(crys, "cov_precision"))],
                    ["strict_valid diagnostic", fmt_pct(refined_comp.get("strict_valid"))],
                    ["single_element <= 10 diagnostic", fmt_pct(refined_comp.get("single_element"))],
                    ["S.U.N diagnostic", fmt_pct(get_metric(sun, "frac_novel_unique_stable_structures"))],
                ],
            ),
            "",
            "## Files",
            "",
            f"- sample metrics: `{args.sample_metrics}`",
            f"- CrysLLMGen metrics: `{args.crysllmgen_metrics}`",
            f"- S.U.N summary: `{args.sun_summary}`",
            f"- composition diagnosis: `{args.composition_summary}`",
        ]
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {args.output_md}")


if __name__ == "__main__":
    main()
