#!/usr/bin/env python3
"""Analyze MP-20 composition, structure validity, and e-hull distributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import composition_record, pbc_duplicate_record
from crystal_dlm.fixed_slot import write_json


def structure_validity(structure, cutoff: float = 0.5) -> bool:
    """Match reference/crysllmgen/eval_utils.py::structure_validity."""

    import numpy as np

    dist_mat = structure.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    return bool(dist_mat.min() >= cutoff and structure.volume >= 0.1)


def quantiles(values: list[float], qs: Iterable[float]) -> dict[str, float | None]:
    if not values:
        return {str(q): None for q in qs}
    ordered = sorted(values)
    result: dict[str, float] = {}
    for q in qs:
        idx = int(round(float(q) * (len(ordered) - 1)))
        idx = max(0, min(len(ordered) - 1, idx))
        result[str(q)] = float(ordered[idx])
    return result


def rate(count: int, total: int) -> float:
    return float(count) / max(1, int(total))


def ehull_bin(value: float) -> str:
    if value < 0:
        return "<0"
    if value == 0:
        return "=0"
    if value <= 0.001:
        return "(0,0.001]"
    if value <= 0.01:
        return "(0.001,0.01]"
    if value <= 0.025:
        return "(0.01,0.025]"
    if value <= 0.05:
        return "(0.025,0.05]"
    if value <= 0.08:
        return "(0.05,0.08]"
    if value <= 0.1:
        return "(0.08,0.1]"
    return ">0.1"


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    reason_counts = Counter(str(row["reason"]) for row in records)
    comp_valid = sum(1 for row in records if row["comp_valid"])
    strict = reason_counts["charge_neutral_pauling_valid"]
    single = reason_counts["single_element_shortcut"]
    all_metal = reason_counts["all_metal_shortcut"]
    struct_valid = sum(1 for row in records if row["struct_valid"])
    pbc_dup = sum(1 for row in records if row["pbc_duplicate"]["has_pbc_equivalent_duplicate"])
    ehulls = [float(row["e_above_hull"]) for row in records if math.isfinite(float(row["e_above_hull"]))]
    formations = [
        float(row["formation_energy_per_atom"])
        for row in records
        if row.get("formation_energy_per_atom") is not None
        and math.isfinite(float(row["formation_energy_per_atom"]))
    ]
    ehull_bins = Counter(ehull_bin(v) for v in ehulls)
    num_atoms = Counter(str(row["num_atoms"]) for row in records)
    num_elements = Counter(str(row["num_elements"]) for row in records)
    reason_by_ehull_bin: dict[str, Counter[str]] = defaultdict(Counter)
    ehull_by_reason: dict[str, list[float]] = defaultdict(list)
    for row in records:
        h = float(row["e_above_hull"])
        reason = str(row["reason"])
        reason_by_ehull_bin[ehull_bin(h)][reason] += 1
        ehull_by_reason[reason].append(h)

    thresholds = [-1e-12, 0.0, 0.001, 0.01, 0.025, 0.05, 0.08, 0.1]
    threshold_counts: dict[str, int] = {}
    for threshold in thresholds:
        if threshold < 0:
            threshold_counts["<0"] = sum(1 for value in ehulls if value < 0)
        else:
            threshold_counts[f"<={threshold:g}"] = sum(1 for value in ehulls if value <= threshold)

    return {
        "count": total,
        "comp_valid_count": comp_valid,
        "comp_valid_rate": rate(comp_valid, total),
        "strict_valid_count": strict,
        "strict_valid_rate": rate(strict, total),
        "single_element_count": single,
        "single_element_rate": rate(single, total),
        "all_metal_count": all_metal,
        "all_metal_rate": rate(all_metal, total),
        "shortcut_count": single + all_metal,
        "shortcut_rate": rate(single + all_metal, total),
        "struct_valid_count": struct_valid,
        "struct_valid_rate": rate(struct_valid, total),
        "pbc_equivalent_duplicate_count": pbc_dup,
        "pbc_equivalent_duplicate_rate": rate(pbc_dup, total),
        "reason_counts": dict(reason_counts.most_common()),
        "num_atoms_histogram": dict(num_atoms.most_common()),
        "num_elements_histogram": dict(num_elements.most_common()),
        "ehull": {
            "count": len(ehulls),
            "min": min(ehulls) if ehulls else None,
            "max": max(ehulls) if ehulls else None,
            "mean": sum(ehulls) / len(ehulls) if ehulls else None,
            "quantiles": quantiles(ehulls, [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
            "threshold_counts": threshold_counts,
            "threshold_rates": {key: rate(value, len(ehulls)) for key, value in threshold_counts.items()},
            "bin_counts": dict(ehull_bins),
            "bin_rates": {key: rate(value, len(ehulls)) for key, value in ehull_bins.items()},
        },
        "formation_energy_per_atom": {
            "count": len(formations),
            "min": min(formations) if formations else None,
            "max": max(formations) if formations else None,
            "mean": sum(formations) / len(formations) if formations else None,
            "quantiles": quantiles(formations, [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
        },
        "reason_by_ehull_bin": {key: dict(counter.most_common()) for key, counter in reason_by_ehull_bin.items()},
        "ehull_by_reason": {
            key: {
                "count": len(values),
                "mean": sum(values) / len(values),
                "quantiles": quantiles(values, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1]),
                "lt_0_rate": rate(sum(1 for value in values if value < 0), len(values)),
                "eq_0_rate": rate(sum(1 for value in values if value == 0), len(values)),
                "le_0_01_rate": rate(sum(1 for value in values if value <= 0.01), len(values)),
                "le_0_05_rate": rate(sum(1 for value in values if value <= 0.05), len(values)),
            }
            for key, values in sorted(ehull_by_reason.items())
        },
    }


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def write_markdown(path: Path, *, split: str, summary: dict[str, Any], examples: dict[str, list[dict[str, Any]]]) -> None:
    lines: list[str] = [
        f"# MP-20 {split} Distribution Analysis",
        "",
        "## Summary",
        "",
        f"- count: {summary['count']}",
        f"- comp_valid: {format_pct(summary['comp_valid_rate'])}",
        f"- strict_valid: {format_pct(summary['strict_valid_rate'])}",
        f"- shortcut: {format_pct(summary['shortcut_rate'])}",
        f"- single_element: {format_pct(summary['single_element_rate'])}",
        f"- all_metal: {format_pct(summary['all_metal_rate'])}",
        f"- struct_valid: {format_pct(summary['struct_valid_rate'])}",
        f"- PBC-equivalent duplicate: {format_pct(summary['pbc_equivalent_duplicate_rate'])}",
        "",
        "## E Above Hull",
        "",
        "| threshold | count | rate |",
        "| --- | ---: | ---: |",
    ]
    ehull = summary["ehull"]
    for key, count in ehull["threshold_counts"].items():
        lines.append(f"| `{key}` | {count} | {format_pct(ehull['threshold_rates'][key])} |")
    lines.extend(["", "### Quantiles", "", "```json", json.dumps(ehull["quantiles"], indent=2, sort_keys=True), "```", ""])
    lines.extend(["## Composition Reasons", "", "```json", json.dumps(summary["reason_counts"], indent=2, sort_keys=True), "```", ""])
    lines.extend(["## E-hull By Reason", "", "| reason | count | mean | q50 | <=0.01 | <=0.05 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for reason, payload in summary["ehull_by_reason"].items():
        lines.append(
            f"| `{reason}` | {payload['count']} | {payload['mean']:.5f} | "
            f"{payload['quantiles']['0.5']:.5f} | {format_pct(payload['le_0_01_rate'])} | {format_pct(payload['le_0_05_rate'])} |"
        )
    lines.extend(["", "## Histograms", "", "### num_elements", "", "```json", json.dumps(summary["num_elements_histogram"], indent=2, sort_keys=True), "```", ""])
    lines.extend(["### num_atoms", "", "```json", json.dumps(summary["num_atoms_histogram"], indent=2, sort_keys=True), "```", ""])
    if examples.get("struct_invalid"):
        lines.extend(["## Struct Invalid Examples", "", "```json", json.dumps(examples["struct_invalid"][:10], indent=2, ensure_ascii=False), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "reference/crysllmgen/data/mp_20/train.csv")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    from pymatgen.core import Structure

    records: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = {"struct_invalid": [], "parse_error": []}
    with args.csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            try:
                structure = Structure.from_str(row["cif"], fmt="cif")
                atom_types = [int(site.specie.Z) for site in structure.sites]
                comp = composition_record(atom_types)
                struct_ok = structure_validity(structure)
                pbc = pbc_duplicate_record([[float(value) for value in coord] for coord in structure.frac_coords])
                record = {
                    "row_idx": row_idx,
                    "material_id": row.get("material_id"),
                    "pretty_formula": row.get("pretty_formula"),
                    "formula": comp["formula"],
                    "num_atoms": len(structure),
                    "num_elements": comp["num_elements"],
                    "comp_valid": comp["comp_valid"],
                    "reason": comp["reason"],
                    "struct_valid": struct_ok,
                    "volume": float(structure.volume),
                    "min_distance": None,
                    "pbc_duplicate": pbc,
                    "e_above_hull": float(row["e_above_hull"]),
                    "formation_energy_per_atom": float(row["formation_energy_per_atom"]),
                    "band_gap": float(row["band_gap"]) if row.get("band_gap") not in (None, "") else None,
                    "spacegroup_number": row.get("spacegroup.number"),
                }
                if len(structure) > 0:
                    import numpy as np

                    dist_mat = structure.distance_matrix
                    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * 10.5)
                    record["min_distance"] = float(dist_mat.min())
                records.append(record)
                if not struct_ok and len(examples["struct_invalid"]) < 50:
                    examples["struct_invalid"].append(record)
            except Exception as exc:
                examples["parse_error"].append(
                    {
                        "row_idx": row_idx,
                        "material_id": row.get("material_id"),
                        "pretty_formula": row.get("pretty_formula"),
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    summary = summarize_records(records)
    payload = {
        "csv": str(args.csv),
        "split": args.split,
        "summary": summary,
        "examples": examples,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output_json), payload)
    write_markdown(args.output_md, split=args.split, summary=summary, examples=examples)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
