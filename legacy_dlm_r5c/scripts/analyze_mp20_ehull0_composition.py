#!/usr/bin/env python3
"""Summarize composition-validity reasons for MP-20 entries with E_hull = 0."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REASONS_VALID = (
    "charge_neutral_pauling_valid",
    "all_metal_shortcut",
    "single_element_shortcut",
)


def rate(numerator: int, denominator: int) -> float:
    return float(numerator) / max(1, int(denominator))


def summarize_counts(counts: Counter[str], *, dataset_count: int) -> dict[str, Any]:
    ehull_eq0_count = int(sum(counts.values()))
    comp_valid_count = int(sum(counts[reason] for reason in REASONS_VALID))
    strict_count = int(counts["charge_neutral_pauling_valid"])
    all_metal_count = int(counts["all_metal_shortcut"])
    single_count = int(counts["single_element_shortcut"])
    invalid_count = int(ehull_eq0_count - comp_valid_count)
    return {
        "dataset_count": int(dataset_count),
        "ehull_eq0_count": ehull_eq0_count,
        "ehull_eq0_rate_in_dataset": rate(ehull_eq0_count, dataset_count),
        "reason_counts": dict(counts.most_common()),
        "comp_valid_count": comp_valid_count,
        "comp_valid_rate_within_eq0": rate(comp_valid_count, ehull_eq0_count),
        "strict_valid_count": strict_count,
        "strict_valid_rate_within_eq0": rate(strict_count, ehull_eq0_count),
        "all_metal_count": all_metal_count,
        "all_metal_rate_within_eq0": rate(all_metal_count, ehull_eq0_count),
        "single_element_count": single_count,
        "single_element_rate_within_eq0": rate(single_count, ehull_eq0_count),
        "invalid_count": invalid_count,
        "invalid_rate_within_eq0": rate(invalid_count, ehull_eq0_count),
    }


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    output: dict[str, Any] = {}
    aggregate_counts: Counter[str] = Counter()
    aggregate_dataset_count = 0

    for split in args.splits:
        path = args.run_dir / "notes" / f"mp20_{split}_distribution.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data["summary"]
        counts = Counter(summary.get("reason_by_ehull_bin", {}).get("=0", {}))
        aggregate_counts.update(counts)
        aggregate_dataset_count += int(summary["count"])
        output[split] = summarize_counts(counts, dataset_count=int(summary["count"]))

    output["all"] = summarize_counts(aggregate_counts, dataset_count=aggregate_dataset_count)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# MP-20 E_hull=0 Composition Summary",
        "",
        "| split | total | E_hull=0 | E_hull=0 rate | comp_valid in E0 | strict in E0 | all_metal in E0 | single in E0 | invalid in E0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in [*args.splits, "all"]:
        row = output[split]
        lines.append(
            f"| {split} | {row['dataset_count']} | {row['ehull_eq0_count']} | "
            f"{pct(row['ehull_eq0_rate_in_dataset'])} | "
            f"{pct(row['comp_valid_rate_within_eq0'])} | "
            f"{pct(row['strict_valid_rate_within_eq0'])} | "
            f"{pct(row['all_metal_rate_within_eq0'])} | "
            f"{pct(row['single_element_rate_within_eq0'])} | "
            f"{pct(row['invalid_rate_within_eq0'])} |"
        )
    lines.extend(
        [
            "",
            "## Reason Counts",
            "",
            "```json",
            json.dumps({key: value["reason_counts"] for key, value in output.items()}, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
