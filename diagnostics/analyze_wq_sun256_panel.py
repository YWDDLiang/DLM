#!/usr/bin/env python3
"""Diagnose composition validity and CHGNet/MP-hull outcomes for a WQ panel.

The script is intentionally read-only with respect to scientific inputs.  It
joins the immutable generation, CrysLLMGen attempt, and R5-C A100/CHGNet attempt
ledgers, reconstructs lightweight composition/structure features, and writes one
new exclusive JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymatgen.core import Element, Structure


STRICT_THRESHOLD = 0.0
META_THRESHOLD = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def index_unique(
    rows: Iterable[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value or value in result:
            raise ValueError(f"{label}: missing or duplicate {key}: {value!r}")
        result[value] = row
    return result


def finite_number(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def numeric_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    data = sorted(
        value
        for value in (finite_number(item) for item in values)
        if value is not None
    )
    if not data:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "max": None,
        }
    array = np.asarray(data, dtype=float)
    return {
        "count": len(data),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def formula_and_features(structure: Structure) -> dict[str, Any]:
    composition = structure.composition
    symbols = sorted(str(element) for element in composition.elements)
    atom_count = int(structure.num_sites)
    distances = np.asarray(structure.distance_matrix, dtype=float)
    if atom_count > 1:
        distances = distances + np.eye(atom_count) * 1.0e9
        min_distance = float(distances.min())
    else:
        min_distance = None
    metal_sites = sum(
        float(amount)
        for element, amount in composition.items()
        if Element(str(element)).is_metal
    )
    electronegativities = [
        float(Element(symbol).X)
        for symbol in symbols
        if Element(symbol).X is not None
    ]
    return {
        "formula": composition.reduced_formula,
        "anonymous_formula": composition.anonymized_formula,
        "chemsys": "-".join(symbols),
        "elements": symbols,
        "num_elements": len(symbols),
        "atom_count": atom_count,
        "volume": float(structure.volume),
        "volume_per_atom": float(structure.volume / atom_count),
        "density": float(structure.density),
        "min_distance": min_distance,
        "metal_fraction": float(metal_sites / atom_count),
        "all_metal": bool(metal_sites == atom_count),
        "contains_oxygen": "O" in symbols,
        "contains_halogen": any(
            symbol in {"F", "Cl", "Br", "I"} for symbol in symbols
        ),
        "electronegativity_range": (
            max(electronegativities) - min(electronegativities)
            if len(electronegativities) >= 2
            else 0.0
        ),
    }


def reduced_atomic_composition(structure: Structure) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts = Counter(int(value) for value in structure.atomic_numbers)
    elements = tuple(sorted(counts))
    values = np.asarray([counts[element] for element in elements], dtype=int)
    reduced = values // np.gcd.reduce(values)
    return elements, tuple(int(value) for value in reduced)


def top_counter(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in counter.most_common(limit)
    ]


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    element_material_counts: Counter[str] = Counter()
    for row in rows:
        element_material_counts.update(set(row["elements"]))
    return {
        "count": len(rows),
        "formulas": [row["formula"] for row in rows],
        "formula_counts": top_counter(Counter(row["formula"] for row in rows)),
        "chemsys_counts": top_counter(Counter(row["chemsys"] for row in rows)),
        "num_elements_histogram": dict(
            sorted(Counter(str(row["num_elements"]) for row in rows).items())
        ),
        "atom_count": numeric_summary(row["atom_count"] for row in rows),
        "density": numeric_summary(row["density"] for row in rows),
        "volume_per_atom": numeric_summary(row["volume_per_atom"] for row in rows),
        "min_distance": numeric_summary(row["min_distance"] for row in rows),
        "metal_fraction": numeric_summary(row["metal_fraction"] for row in rows),
        "electronegativity_range": numeric_summary(
            row["electronegativity_range"] for row in rows
        ),
        "e_above_hull": numeric_summary(row["e_above_hull"] for row in rows),
        "energy_per_atom": numeric_summary(row["energy_per_atom"] for row in rows),
        "comp_invalid_count": sum(not row["comp_valid"] for row in rows),
        "joint_invalid_count": sum(not row["valid"] for row in rows),
        "all_metal_count": sum(row["all_metal"] for row in rows),
        "contains_oxygen_count": sum(row["contains_oxygen"] for row in rows),
        "composition_changed_from_initial_count": sum(
            row["composition_changed_from_initial"] for row in rows
        ),
        "top_elements_by_material_prevalence": top_counter(
            element_material_counts
        ),
    }


def element_enrichment(
    positive: list[dict[str, Any]], negative: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    positive_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    for row in positive:
        positive_counts.update(set(row["elements"]))
    for row in negative:
        negative_counts.update(set(row["elements"]))
    records = []
    for element in sorted(set(positive_counts) | set(negative_counts)):
        a = positive_counts[element]
        b = len(positive) - a
        c = negative_counts[element]
        d = len(negative) - c
        if a + c < 3:
            continue
        positive_rate = a / len(positive) if positive else 0.0
        negative_rate = c / len(negative) if negative else 0.0
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        records.append(
            {
                "element": element,
                "positive_count": a,
                "positive_rate": positive_rate,
                "negative_count": c,
                "negative_rate": negative_rate,
                "rate_difference": positive_rate - negative_rate,
                "log2_odds_ratio": math.log2(odds_ratio),
            }
        )
    return {
        "enriched_in_meta_stable": sorted(
            records, key=lambda row: row["rate_difference"], reverse=True
        )[:15],
        "enriched_in_unstable": sorted(
            records, key=lambda row: row["rate_difference"]
        )[:15],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--attempt-metrics-jsonl", type=Path, required=True)
    parser.add_argument("--sun-attempt-results-jsonl", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    paths = [
        args.generation_jsonl.resolve(),
        args.attempt_metrics_jsonl.resolve(),
        args.sun_attempt_results_jsonl.resolve(),
    ]
    generation = read_jsonl(paths[0])
    metric_by_id = index_unique(
        read_jsonl(paths[1]), "attempt_id", "CrysLLMGen metrics"
    )
    sun_by_id = index_unique(
        read_jsonl(paths[2]), "attempt_id", "S.U.N. attempts"
    )
    generation_by_id = index_unique(generation, "attempt_id", "generation")
    if not (
        len(generation_by_id)
        == len(metric_by_id)
        == len(sun_by_id)
        == 256
        and set(generation_by_id) == set(metric_by_id) == set(sun_by_id)
    ):
        raise ValueError("the three attempt ledgers are not the same 256-attempt panel")

    snapshot = args.snapshot_root.resolve()
    sys.path.insert(0, str(snapshot))
    try:
        from compute_metrics import CrystalNNFP
        from eval_utils import smact_validity
    finally:
        sys.path.pop(0)

    attempts: list[dict[str, Any]] = []
    extra_joint_invalid: list[dict[str, Any]] = []
    for generation_row in sorted(generation, key=lambda row: int(row["ordinal"])):
        attempt_id = str(generation_row["attempt_id"])
        metric = metric_by_id[attempt_id]
        sun = sun_by_id[attempt_id]
        if generation_row.get("status") != "succeeded":
            raise ValueError(f"unexpected failed generation: {attempt_id}")
        structure = Structure.from_dict(dict(generation_row["structure"]))
        initial_structure = Structure.from_dict(dict(generation_row["initial_structure"]))
        features = formula_and_features(structure)
        initial_formula = initial_structure.composition.reduced_formula
        atomic_numbers, reduced_counts = reduced_atomic_composition(structure)

        comp_valid = bool(metric["comp_valid"])
        struct_valid = bool(metric["struct_valid"])
        valid = bool(metric["valid"])
        validity_mechanism = "valid"
        if not comp_valid:
            try:
                no_pauling = bool(
                    smact_validity(
                        atomic_numbers,
                        reduced_counts,
                        use_pauling_test=False,
                    )
                )
                with_pauling = bool(
                    smact_validity(
                        atomic_numbers,
                        reduced_counts,
                        use_pauling_test=True,
                    )
                )
                if with_pauling:
                    validity_mechanism = "recorded_vs_recomputed_mismatch"
                elif no_pauling:
                    validity_mechanism = "pauling_electronegativity_rejection"
                else:
                    validity_mechanism = (
                        "no_charge_neutral_oxidation_assignment_under_smact_states"
                    )
            except Exception as exc:
                validity_mechanism = (
                    f"smact_diagnostic_exception:{type(exc).__name__}:{exc}"
                )
        elif not struct_valid:
            validity_mechanism = "structure_validity_failure"
        elif not valid:
            validity_mechanism = "post_validity_fingerprint_failure"
            fingerprint_failure = None
            for site_index in range(len(structure)):
                try:
                    CrystalNNFP.featurize(structure, site_index)
                except Exception as exc:
                    fingerprint_failure = {
                        "site_index": site_index,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
            extra_joint_invalid.append(
                {
                    "attempt_id": attempt_id,
                    "ordinal": int(generation_row["ordinal"]),
                    "formula": features["formula"],
                    "fingerprint_failure": fingerprint_failure,
                }
            )

        sun_metrics = dict(sun["metrics"])
        e_hull = finite_number(sun_metrics.get("e_above_hull"))
        energy = finite_number(sun_metrics.get("energy_per_atom"))
        status = str(sun["evaluation_status"])
        if status == "not_novel_unique":
            stability_category = "not_novel_unique"
        elif status == "relaxation_or_hull_unknown":
            stability_category = (
                "relaxation_unknown" if energy is None else "hull_unknown"
            )
        elif status == "evaluated":
            if e_hull is None:
                raise ValueError(f"evaluated attempt lacks E_hull: {attempt_id}")
            if e_hull <= STRICT_THRESHOLD:
                stability_category = "strict_stable"
            elif e_hull <= META_THRESHOLD:
                stability_category = "meta_only_stable"
            else:
                stability_category = "unstable"
        else:
            raise ValueError(f"unexpected S.U.N. status: {status}")

        attempts.append(
            {
                "attempt_id": attempt_id,
                "source_ordinal": int(generation_row["ordinal"]),
                "evaluation_ordinal": int(sun["generation_ordinal"]),
                "structure_sha256": generation_row["structure_sha256"],
                **features,
                "initial_formula": initial_formula,
                "composition_changed_from_initial": (
                    initial_formula != features["formula"]
                ),
                "comp_valid": comp_valid,
                "struct_valid": struct_valid,
                "valid": valid,
                "validity_mechanism": validity_mechanism,
                "evaluation_status": status,
                "stability_category": stability_category,
                "novel": bool(sun_metrics["novel"]),
                "novel_unique": bool(sun_metrics["novel_unique"]),
                "strict_full_sun": bool(sun_metrics["strict_full_sun"]),
                "meta_full_sun": bool(sun_metrics["meta_full_sun"]),
                "energy_per_atom": energy,
                "e_above_hull": e_hull,
            }
        )

    groups = {
        category: [row for row in attempts if row["stability_category"] == category]
        for category in (
            "strict_stable",
            "meta_only_stable",
            "unstable",
            "hull_unknown",
            "relaxation_unknown",
            "not_novel_unique",
        )
    }
    comp_invalid = [row for row in attempts if not row["comp_valid"]]
    joint_invalid = [row for row in attempts if not row["valid"]]
    meta_stable = groups["strict_stable"] + groups["meta_only_stable"]
    unstable = groups["unstable"]
    numeric_comparison = {}
    for field in (
        "num_elements",
        "atom_count",
        "density",
        "volume_per_atom",
        "min_distance",
        "metal_fraction",
        "electronegativity_range",
    ):
        positive = numeric_summary(row[field] for row in meta_stable)
        negative = numeric_summary(row[field] for row in unstable)
        numeric_comparison[field] = {
            "meta_stable": positive,
            "unstable": negative,
            "median_difference": (
                None
                if positive["median"] is None or negative["median"] is None
                else float(positive["median"] - negative["median"])
            ),
        }

    report = {
        "schema": "wq_parent_csp_sun256_failure_taxonomy_v1",
        "created_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "inputs": {
            str(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        },
        "contracts": {
            "attempts": len(attempts),
            "same_attempt_ids": True,
            "retry_or_replacement_used": False,
            "scientific_inputs_modified": False,
            "strict_threshold_ev_per_atom": STRICT_THRESHOLD,
            "meta_threshold_ev_per_atom": META_THRESHOLD,
        },
        "counts": {
            "composition_invalid": len(comp_invalid),
            "joint_invalid": len(joint_invalid),
            "extra_joint_invalid_after_comp_and_structure_pass": len(
                extra_joint_invalid
            ),
            "stability_categories": {
                category: len(rows) for category, rows in groups.items()
            },
        },
        "composition_invalid": {
            "mechanism_counts": dict(
                Counter(row["validity_mechanism"] for row in comp_invalid)
            ),
            "summary": group_summary(comp_invalid),
            "attempts": comp_invalid,
        },
        "joint_invalid": {
            "summary": group_summary(joint_invalid),
            "extra_after_comp_and_structure_pass": extra_joint_invalid,
            "attempts": joint_invalid,
        },
        "stability_groups": {
            category: group_summary(rows) for category, rows in groups.items()
        },
        "meta_stable_vs_unstable": {
            "meta_stable_count": len(meta_stable),
            "unstable_count": len(unstable),
            "numeric_features": numeric_comparison,
            "element_enrichment": element_enrichment(meta_stable, unstable),
        },
        "unknown_followup_manifest": [
            {
                "attempt_id": row["attempt_id"],
                "source_ordinal": row["source_ordinal"],
                "formula": row["formula"],
                "chemsys": row["chemsys"],
                "elements": row["elements"],
                "energy_per_atom": row["energy_per_atom"],
                "structure_sha256": row["structure_sha256"],
                "unknown_type": row["stability_category"],
            }
            for row in attempts
            if row["stability_category"] in {"hull_unknown", "relaxation_unknown"}
        ],
        "all_attempts": attempts,
    }

    output = args.output_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": sha256_file(output),
                "counts": report["counts"],
                "composition_invalid_mechanisms": report[
                    "composition_invalid"
                ]["mechanism_counts"],
                "extra_joint_invalid": extra_joint_invalid,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
