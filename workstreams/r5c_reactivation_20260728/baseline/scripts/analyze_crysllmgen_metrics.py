#!/usr/bin/env python3
"""Collect and summarize all saved CrysLLMGen metric JSON files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "comp_valid",
    "struct_valid",
    "valid",
    "wdist_density",
    "wdist_num_elems",
    "cov_recall",
    "cov_precision",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def corr(rows: list[dict[str, Any]], left: str, right: str) -> float | None:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if isinstance(row.get(left), (int, float)) and isinstance(row.get(right), (int, float))
    ]
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def candidate_metric_files(root: Path) -> list[Path]:
    """Find both current and historical CrysLLMGen metric filenames."""
    files: set[Path] = set()
    files.update(root.glob("runs/**/notes/crysllmgen_metrics*.json"))
    files.update(root.glob("runs/**/notes/eval_metrics.json"))
    return sorted(files)


def extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    # Some ad-hoc metric dumps put values at top level.
    top_level = {key: data.get(key) for key in METRIC_KEYS if key in data}
    return top_level


def load_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = candidate_metric_files(root)
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []

    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            bad.append({"path": str(path), "error": f"json_error:{exc}"})
            continue

        metrics = extract_metrics(data)
        run = path.parts[path.parts.index("runs") + 1] if "runs" in path.parts else ""
        if not metrics:
            bad.append(
                {
                    "path": str(path),
                    "run": run,
                    "file": path.name,
                    "error": "empty_metrics",
                    "returncode": data.get("returncode"),
                }
            )
            continue

        row: dict[str, Any] = {
            "path": str(path),
            "run": run,
            "file": path.name,
            "returncode": data.get("returncode"),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        rows.append(row)

    return rows, bad


def metric_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        output[key] = {
            "n": len(values),
            "mean": mean(values),
            "std": stdev(values),
            "min": min(values),
            "max": max(values),
        }
    return output


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def print_console(rows: list[dict[str, Any]], bad: list[dict[str, Any]]) -> None:
    print(f"FILES_WITH_METRICS {len(rows)}")
    print(f"EMPTY_OR_FAILED {len(bad)}")
    if bad:
        print("BAD_FILES")
        for row in bad:
            print(row.get("path"), row.get("error"), "returncode", row.get("returncode"))

    print("\nSUMMARY")
    for key, stats in metric_stats(rows).items():
        print(
            key,
            {
                "n": int(stats["n"]),
                "mean": round(stats["mean"], 4),
                "std": round(stats["std"], 4),
                "min": round(stats["min"], 4),
                "max": round(stats["max"], 4),
            },
        )

    print("\nTABLE_BY_RUN")
    header = ["run", "file", *METRIC_KEYS]
    print("\t".join(header))
    for row in rows:
        print("\t".join(fmt(row.get(column)) for column in header))

    print("\nRANK_COMP_VALID")
    for row in sorted(rows, key=lambda item: float(item.get("comp_valid") or -999), reverse=True):
        print(fmt(row.get("comp_valid")), fmt(row.get("cov_recall")), fmt(row.get("wdist_density")), row["run"], row["file"])

    print("\nRANK_COV_RECALL")
    for row in sorted(rows, key=lambda item: float(item.get("cov_recall") or -999), reverse=True):
        print(fmt(row.get("cov_recall")), fmt(row.get("comp_valid")), fmt(row.get("wdist_density")), row["run"], row["file"])

    print("\nCORRELATIONS")
    for left, right in [
        ("comp_valid", "cov_recall"),
        ("comp_valid", "wdist_density"),
        ("comp_valid", "wdist_num_elems"),
        ("cov_recall", "wdist_density"),
        ("cov_recall", "wdist_num_elems"),
        ("cov_precision", "cov_recall"),
    ]:
        value = corr(rows, left, right)
        print(f"{left} vs {right}", "NA" if value is None else round(value, 4))


def write_report(rows: list[dict[str, Any]], bad: list[dict[str, Any]], output: Path) -> None:
    lines: list[str] = []
    lines.append("# 2026-05-22 CrysLLMGen Metrics Analysis\n\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    lines.append(f"Valid metric files: `{len(rows)}`. Empty or failed metric files: `{len(bad)}`.\n")

    if bad:
        lines.append("\n## Empty Or Failed Metrics\n\n")
        for row in bad:
            lines.append(f"- `{row.get('path')}`: {row.get('error')}, returncode={row.get('returncode')}\n")

    lines.append("\n## Aggregate Statistics\n\n")
    lines.append("| metric | n | mean | std | min | max |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
    for key, stats in metric_stats(rows).items():
        lines.append(
            f"| `{key}` | {int(stats['n'])} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n"
        )

    lines.append("\n## Per-Run Metrics\n\n")
    lines.append("| run | file | comp_valid | struct_valid | valid | wdist_density | wdist_num_elems | cov_recall | cov_precision |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in rows:
        lines.append(
            f"| `{row['run']}` | `{row['file']}` | {fmt(row.get('comp_valid'))} | {fmt(row.get('struct_valid'))} | {fmt(row.get('valid'))} | {fmt(row.get('wdist_density'))} | {fmt(row.get('wdist_num_elems'))} | {fmt(row.get('cov_recall'))} | {fmt(row.get('cov_precision'))} |\n"
        )

    lines.append("\n## Ranking By Comp Valid\n\n")
    for row in sorted(rows, key=lambda item: float(item.get("comp_valid") or -999), reverse=True):
        lines.append(
            f"- `{row['run']}` `{row['file']}`: comp_valid={fmt(row.get('comp_valid'))}, cov_recall={fmt(row.get('cov_recall'))}, wdist_density={fmt(row.get('wdist_density'))}\n"
        )

    lines.append("\n## Ranking By Coverage Recall\n\n")
    for row in sorted(rows, key=lambda item: float(item.get("cov_recall") or -999), reverse=True):
        lines.append(
            f"- `{row['run']}` `{row['file']}`: cov_recall={fmt(row.get('cov_recall'))}, comp_valid={fmt(row.get('comp_valid'))}, wdist_density={fmt(row.get('wdist_density'))}\n"
        )

    lines.append("\n## Correlations\n\n")
    for left, right in [
        ("comp_valid", "cov_recall"),
        ("comp_valid", "wdist_density"),
        ("comp_valid", "wdist_num_elems"),
        ("cov_recall", "wdist_density"),
        ("cov_recall", "wdist_num_elems"),
        ("cov_precision", "cov_recall"),
    ]:
        value = corr(rows, left, right)
        lines.append(f"- `{left}` vs `{right}`: {'NA' if value is None else f'{value:.4f}'}\n")

    lines.append("\n## Interpretation\n\n")
    lines.append("- The best current composition-valid runs reach `comp_valid` around 88-89, with `cov_recall` around 88.7-88.8 and `cov_precision` near 100.\n")
    lines.append("- The older final07 retry has higher `cov_recall` at 92.98, but lower `comp_valid` at 83.6 and much worse `wdist_density` at 1.1642.\n")
    lines.append("- This is a validity/coverage tradeoff: newer sampling is precise and chemically better, but covers a narrower part of MP-20.\n")
    lines.append("- Because COV recall requires both structure and composition fingerprint proximity to ground truth, composition narrowing and formula/element-mode collapse can reduce recall even when density improves.\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/20260522_crysllmgen_metrics_analysis.md")
    args = parser.parse_args()

    rows, bad = load_rows(Path(args.root))
    print_console(rows, bad)
    write_report(rows, bad, Path(args.output))
    print(f"\nWROTE {args.output}")


if __name__ == "__main__":
    main()
