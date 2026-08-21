"""CLI for stagewise chemistry attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .attribution import analyze_cohort, analyze_pair, normalize_attempt, paired_mcnemar


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name.strip(), Path(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no JSONL rows")
    return rows


def format_rate(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.2f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# H1-A2 stagewise attribution", "", "## Cohort funnels", ""]
    lines.append("| Cohort | Requested | Plan eligible | Body success | Refined | Reconstructed | N&U | Hull known/unknown | Strict all | Meta all | Strict known | Meta known |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, report in payload["cohorts"].items():
        counts = report["funnel"]
        known = report["hull_known_rates"]
        lines.append(
            f"| {name} | {counts['requested']} | {counts['plan_eligible']} | "
            f"{counts['body_success']} | {counts['refined']} | {counts['reconstructed']} | "
            f"{counts['novel_unique']} | "
            f"{counts['hull_known']}/{known['unknown']} | {format_rate(report['strict_rate'])} | "
            f"{format_rate(report['meta_rate'])} | {format_rate(known['strict_rate'])} | "
            f"{format_rate(known['meta_rate'])} |"
        )
    lines.extend(["", "## Pairwise chemistry attribution", ""])
    for pair_name, pair_report in payload["pairs"].items():
        lines.extend([f"### {pair_name}", ""])
        lines.append("| Outcome/scope | Common support A/B | Gap | Mix | Within-stratum | Residual |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for outcome, outcome_report in pair_report.items():
            for scope, key in (("all-attempt", "decomposition"), ("hull-known", "hull_known_decomposition")):
                decomposition = outcome_report[key]
                if not decomposition.get("estimable"):
                    lines.append(f"| {outcome}/{scope} | not estimable | NA | NA | NA | NA |")
                    continue
                lines.append(
                    f"| {outcome}/{scope} | {100 * decomposition['coverage_a']:.1f}% / "
                    f"{100 * decomposition['coverage_b']:.1f}% | "
                    f"{format_rate(decomposition['gap_common'])} | "
                    f"{format_rate(decomposition['mix_effect'])} | "
                    f"{format_rate(decomposition['conditional_effect'])} | "
                    f"{decomposition['identity_residual']:.3e} |"
                )
        lines.append("")
    if payload.get("paired_mcnemar"):
        lines.extend(["## Exact paired McNemar", ""])
        lines.append("| Pair | Outcome | Known both | A-only | B-only | Exact two-sided p |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for pair_name, outcomes in payload["paired_mcnemar"].items():
            for outcome, report in outcomes.items():
                lines.append(
                    f"| {pair_name} | {outcome} | {report['known_both_pairs']} | "
                    f"{report['discordant_a_only']} | {report['discordant_b_only']} | "
                    f"{report['exact_two_sided_p']:.6g} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "Mix and within-stratum terms are reported only on common support. "
            "Low support coverage or effective sample size invalidates broad causal language.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", action="append", required=True, type=parse_named_path)
    parser.add_argument("--pair", action="append", default=[], help="NAME_A:NAME_B")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Cohort used as a standardization mix; repeat for learned-P0 and MP-20 references",
    )
    parser.add_argument(
        "--paired-key",
        default=None,
        help="Optional shared row key for exact paired McNemar tests (for example ordinal)",
    )
    parser.add_argument(
        "--paired-known-stage",
        default="hull_known",
        help="Require this stage in both rows; use 'none' for all aligned pairs",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    cohort_paths = dict(args.cohort)
    if len(cohort_paths) != len(args.cohort):
        raise ValueError("cohort names must be unique")
    rows = {name: load_jsonl(path) for name, path in cohort_paths.items()}
    unknown_references = [name for name in args.reference if name not in rows]
    if unknown_references:
        raise ValueError(f"unknown reference cohorts {unknown_references!r}")

    payload: dict[str, Any] = {
        "schema": "h1a2_stagewise_attribution_v1",
        "cohorts": {name: analyze_cohort(items) for name, items in rows.items()},
        "pairs": {},
        "paired_mcnemar": {},
        "references": list(args.reference),
    }
    for pair in args.pair:
        if ":" not in pair:
            raise ValueError(f"expected pair NAME_A:NAME_B, got {pair!r}")
        name_a, name_b = pair.split(":", 1)
        if name_a not in rows or name_b not in rows:
            raise ValueError(f"unknown cohort in pair {pair!r}")
        references = args.reference or [name_a]
        for reference in references:
            label = pair if len(references) == 1 else f"{pair}@{reference}"
            payload["pairs"][label] = analyze_pair(
                rows[name_a], rows[name_b], reference_rows=rows[reference]
            )
        if args.paired_key is not None:
            normalized_a = [normalize_attempt(row) for row in rows[name_a]]
            normalized_b = [normalize_attempt(row) for row in rows[name_b]]
            known_stage = None if args.paired_known_stage.lower() == "none" else args.paired_known_stage
            payload["paired_mcnemar"][pair] = {
                outcome: paired_mcnemar(
                    normalized_a,
                    normalized_b,
                    outcome,
                    key=args.paired_key,
                    known_stage=known_stage,
                )
                for outcome in ("strict_sun", "meta_sun")
            }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
