#!/usr/bin/env python3
"""Summarize MatterGen detailed metrics with strict/meta S.U.N thresholds."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


STRICT_STABLE_THRESHOLD = 0.0
META_STABLE_THRESHOLD = 0.1


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rate(count: int, total: int) -> float:
    return float(count) / max(1, int(total))


def quantiles(values: list[float]) -> dict[str, float | None]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return {"mean": None, "q10": None, "q50": None, "q90": None}

    def pick(frac: float) -> float:
        index = min(len(clean) - 1, max(0, int(frac * (len(clean) - 1))))
        return clean[index]

    return {
        "mean": float(sum(clean) / len(clean)),
        "q10": pick(0.10),
        "q50": pick(0.50),
        "q90": pick(0.90),
    }


def get_metric(summary: Mapping[str, Any], key: str) -> Any:
    metrics = summary.get("metrics", {})
    if isinstance(metrics, Mapping):
        return metrics.get(key)
    return None


def formula_from_entry(entry: Mapping[str, Any]) -> tuple[str, str]:
    composition = entry.get("composition") or {}
    if not isinstance(composition, Mapping):
        return "unknown", "unknown"
    items = sorted((str(symbol), float(count)) for symbol, count in composition.items())
    formula_parts: list[str] = []
    for symbol, count in items:
        rounded = int(round(count))
        formula_parts.append(symbol if rounded == 1 else f"{symbol}{rounded}")
    chemsys = "-".join(symbol for symbol, _ in items)
    return "".join(formula_parts), chemsys


def analyze(summary: Mapping[str, Any], detailed: Mapping[str, Any], label: str) -> dict[str, Any]:
    total_submitted = int(summary.get("num_structures") or 0)
    ehull = detailed.get("energy_above_hull") or detailed.get("energy_above_hull_per_atom")
    if not isinstance(ehull, list):
        return {
            "label": label,
            "error": "missing_energy_above_hull",
            "summary_metrics": summary.get("metrics", {}),
            "available_detailed_columns": sorted(str(key) for key in detailed.keys()),
        }

    n_success = len(ehull)
    if total_submitted <= 0:
        total_submitted = n_success
    novel = detailed.get("novel") if isinstance(detailed.get("novel"), list) else [False] * n_success
    unique = detailed.get("unique") if isinstance(detailed.get("unique"), list) else [False] * n_success
    novel_unique = (
        detailed.get("novel_unique")
        if isinstance(detailed.get("novel_unique"), list)
        else [False] * n_success
    )
    comp_valid = (
        detailed.get("comp_validity")
        if isinstance(detailed.get("comp_validity"), list)
        else [True] * n_success
    )
    entries = detailed.get("entry") if isinstance(detailed.get("entry"), list) else []
    n = min(n_success, len(novel), len(unique), len(novel_unique), len(comp_valid))
    ehull_values = [float(value) for value in ehull[:n]]
    novel_mask = [bool(value) for value in novel[:n]]
    unique_mask = [bool(value) for value in unique[:n]]
    novel_unique_mask = [bool(value) for value in novel_unique[:n]]
    comp_valid_mask = [bool(value) for value in comp_valid[:n]]
    strict_mask = [value < STRICT_STABLE_THRESHOLD for value in ehull_values]
    meta_mask = [value < META_STABLE_THRESHOLD for value in ehull_values]
    strict_sun_mask = [novel_unique_mask[idx] and strict_mask[idx] for idx in range(n)]
    meta_sun_mask = [novel_unique_mask[idx] and meta_mask[idx] for idx in range(n)]
    strict_sun_comp_mask = [
        strict_sun_mask[idx] and comp_valid_mask[idx] for idx in range(n)
    ]
    meta_sun_comp_mask = [
        meta_sun_mask[idx] and comp_valid_mask[idx] for idx in range(n)
    ]

    formula_counts: Counter[str] = Counter()
    chemsys_counts: Counter[str] = Counter()
    meta_sun_formulas: Counter[str] = Counter()
    strict_sun_formulas: Counter[str] = Counter()
    for idx, entry in enumerate(entries[:n]):
        if not isinstance(entry, Mapping):
            continue
        formula, chemsys = formula_from_entry(entry)
        formula_counts[formula] += 1
        chemsys_counts[chemsys] += 1
        if meta_sun_mask[idx]:
            meta_sun_formulas[formula] += 1
        if strict_sun_mask[idx]:
            strict_sun_formulas[formula] += 1

    counts = {
        "submitted": total_submitted,
        "successful": n,
        "unsupported_failed": int(summary.get("n_unsupported_failed") or 0),
        "relax_failed": int(summary.get("n_relax_failed") or 0),
        "strict_stable": sum(strict_mask),
        "meta_stable": sum(meta_mask),
        "novel": sum(novel_mask),
        "unique": sum(unique_mask),
        "novel_unique": sum(novel_unique_mask),
        "strict_sun": sum(strict_sun_mask),
        "meta_sun": sum(meta_sun_mask),
        "strict_sun_comp_valid": sum(strict_sun_comp_mask),
        "meta_sun_comp_valid": sum(meta_sun_comp_mask),
    }
    rates = {
        key: rate(value, total_submitted)
        for key, value in counts.items()
        if key not in {"submitted"}
    }
    rates_successful = {
        key: rate(value, n)
        for key, value in counts.items()
        if key not in {"submitted", "successful", "unsupported_failed", "relax_failed"}
    }
    novel_unique_ehull = [
        value for value, keep in zip(ehull_values, novel_unique_mask) if keep
    ]
    not_novel_unique_ehull = [
        value for value, keep in zip(ehull_values, novel_unique_mask) if not keep
    ]
    return {
        "label": label,
        "thresholds": {
            "strict_stable_ehull_lt": STRICT_STABLE_THRESHOLD,
            "meta_stable_ehull_lt": META_STABLE_THRESHOLD,
        },
        "counts": counts,
        "rates_submitted": rates,
        "rates_successful": rates_successful,
        "mattergen_builtin": {
            key: get_metric(summary, key)
            for key in [
                "frac_novel_unique_stable_structures",
                "frac_stable_structures",
                "frac_novel_structures",
                "frac_unique_structures",
                "frac_novel_unique_structures",
                "avg_energy_above_hull_per_atom",
                "frac_successful_jobs",
            ]
        },
        "ehull_quantiles": {
            "all_successful": quantiles(ehull_values),
            "novel_unique": quantiles(novel_unique_ehull),
            "not_novel_unique": quantiles(not_novel_unique_ehull),
        },
        "top_formula_counts": dict(formula_counts.most_common(30)),
        "top_chemsys_counts": dict(chemsys_counts.most_common(30)),
        "top_meta_sun_formula_counts": dict(meta_sun_formulas.most_common(30)),
        "top_strict_sun_formula_counts": dict(strict_sun_formulas.most_common(30)),
        "diagnosis": build_diagnosis(counts, total_submitted),
    }


def build_diagnosis(counts: Mapping[str, int], total: int) -> list[str]:
    notes = [
        f"strict_sun={100.0 * rate(int(counts['strict_sun']), total):.2f}%",
        f"meta_sun={100.0 * rate(int(counts['meta_sun']), total):.2f}%",
        f"strict_stable={100.0 * rate(int(counts['strict_stable']), total):.2f}%",
        f"meta_stable={100.0 * rate(int(counts['meta_stable']), total):.2f}%",
        f"novel_unique={100.0 * rate(int(counts['novel_unique']), total):.2f}%",
    ]
    if int(counts["meta_stable"]) > 0:
        notes.append(
            "meta SUN retention among meta-stable="
            f"{100.0 * int(counts['meta_sun']) / max(1, int(counts['meta_stable'])):.2f}%"
        )
    if int(counts["novel_unique"]) > 0:
        notes.append(
            "meta-stable retention among novel-unique="
            f"{100.0 * int(counts['meta_sun']) / max(1, int(counts['novel_unique'])):.2f}%"
        )
    return notes


def fmt_pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        f"# MatterGen SUN Detailed Analysis: {payload.get('label')}",
        "",
        "## Diagnosis",
    ]
    for note in payload.get("diagnosis", []):
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Rates",
            "| metric | submitted denominator | successful denominator |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in [
        "strict_stable",
        "meta_stable",
        "novel",
        "unique",
        "novel_unique",
        "strict_sun",
        "meta_sun",
        "strict_sun_comp_valid",
        "meta_sun_comp_valid",
    ]:
        lines.append(
            "| "
            + key
            + " | "
            + fmt_pct((payload.get("rates_submitted") or {}).get(key))
            + " | "
            + fmt_pct((payload.get("rates_successful") or {}).get(key))
            + " |"
        )
    lines.extend(
        [
            "",
            "## Ehull Quantiles",
            "```json",
            json.dumps(payload.get("ehull_quantiles", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Top Formula Counts",
            "```json",
            json.dumps(payload.get("top_formula_counts", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Top Chemical Systems",
            "```json",
            json.dumps(payload.get("top_chemsys_counts", {}), ensure_ascii=False, indent=2),
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--detailed-json", type=Path, required=True)
    parser.add_argument("--label", default="mattergen_sun")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payload = analyze(read_json(args.summary_json), read_json(args.detailed_json), args.label)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
