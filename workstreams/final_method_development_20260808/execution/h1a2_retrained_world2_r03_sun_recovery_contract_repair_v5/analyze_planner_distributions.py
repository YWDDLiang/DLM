#!/usr/bin/env python3
"""Freeze a deep H1-A2 planner-distribution audit before R03 generation."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from protocol import (
    DENOMINATOR,
    canonical_sha256,
    read_jsonl,
    require_file,
    sha256_file,
    write_json_exclusive,
)


LANTHANIDES = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
}
ACTINIDES = {
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
}
HALOGENS = {"F", "Cl", "Br", "I"}
CHALCOGENS = {"O", "S", "Se", "Te"}
PNICTOGENS = {"N", "P", "As", "Sb", "Bi"}
DISTANCE_FIELDS = (
    "element_presence",
    "stoichiometric_atoms",
    "chemsys",
    "formula",
    "atom_count",
    "element_count",
    "charge_bucket",
    "anion_framework",
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
    "family",
)


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _normalize(values: Mapping[str, int]) -> dict[str, float]:
    total = float(sum(int(value) for value in values.values()))
    if total <= 0:
        return {}
    return {str(key): int(value) / total for key, value in values.items()}


def categorical_distance(
    reference: Mapping[str, int], observed: Mapping[str, int]
) -> dict[str, float | int]:
    left = _normalize(reference)
    right = _normalize(observed)
    keys = sorted(set(left) | set(right))
    if not keys:
        return {
            "total_variation": 0.0,
            "jensen_shannon_bits": 0.0,
            "hellinger": 0.0,
            "support_jaccard": 1.0,
            "reference_support": 0,
            "observed_support": 0,
        }
    total_variation = 0.5 * sum(
        abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys
    )
    jensen_shannon = 0.0
    hellinger_sum = 0.0
    for key in keys:
        p = left.get(key, 0.0)
        q = right.get(key, 0.0)
        middle = 0.5 * (p + q)
        if p > 0:
            jensen_shannon += 0.5 * p * math.log2(p / middle)
        if q > 0:
            jensen_shannon += 0.5 * q * math.log2(q / middle)
        hellinger_sum += (math.sqrt(p) - math.sqrt(q)) ** 2
    left_support = {key for key, value in left.items() if value > 0}
    right_support = {key for key, value in right.items() if value > 0}
    union = left_support | right_support
    return {
        "total_variation": total_variation,
        "jensen_shannon_bits": jensen_shannon,
        "hellinger": math.sqrt(0.5 * hellinger_sum),
        "support_jaccard": (
            len(left_support & right_support) / len(union) if union else 1.0
        ),
        "reference_support": len(left_support),
        "observed_support": len(right_support),
    }


def _top_rate_shifts(
    reference: Mapping[str, int],
    observed: Mapping[str, int],
    reference_denominator: int,
    observed_denominator: int,
    limit: int = 25,
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(reference) | set(observed)):
        reference_rate = int(reference.get(key, 0)) / reference_denominator
        observed_rate = int(observed.get(key, 0)) / observed_denominator
        rows.append(
            {
                "category": key,
                "reference_count": int(reference.get(key, 0)),
                "reference_rate_per_attempt": reference_rate,
                "observed_count": int(observed.get(key, 0)),
                "observed_rate_per_attempt": observed_rate,
                "rate_difference": observed_rate - reference_rate,
            }
        )
    return sorted(
        rows,
        key=lambda row: (-abs(float(row["rate_difference"])), row["category"]),
    )[:limit]


def summarize_rows(
    rows: Iterable[Mapping[str, Any]], *, cohort_id: str
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError(f"empty planner cohort: {cohort_id}")
    distributions: dict[str, Counter[str]] = {
        field: Counter() for field in DISTANCE_FIELDS
    }
    parsed = 0
    group_presence: Counter[str] = Counter()
    for row in records:
        if row.get("parsed") is not True or row.get("body_eligible") is not True:
            continue
        plan = row.get("plan_state")
        if not isinstance(plan, Mapping):
            raise ValueError(f"parsed planner row lacks plan_state: {cohort_id}")
        elements = [str(value) for value in plan.get("elements") or []]
        counts = [int(value) for value in plan.get("counts") or []]
        atom_count = int(plan.get("N", -1))
        if (
            not elements
            or len(elements) != len(counts)
            or len(set(elements)) != len(elements)
            or any(value <= 0 for value in counts)
            or sum(counts) != atom_count
        ):
            raise ValueError(f"invalid parsed composition: {cohort_id}")
        parsed += 1
        unique_elements = set(elements)
        for element in unique_elements:
            distributions["element_presence"][element] += 1
        for element, count in zip(elements, counts):
            distributions["stoichiometric_atoms"][element] += count
        distributions["chemsys"]["-".join(sorted(unique_elements))] += 1
        distributions["formula"][str(plan.get("formula") or "")] += 1
        distributions["atom_count"][str(atom_count)] += 1
        distributions["element_count"][str(len(elements))] += 1
        for field in (
            "charge_bucket",
            "anion_framework",
            "lattice_system",
            "spacegroup_bucket",
            "volume_per_atom_bin",
            "family",
        ):
            distributions[field][str(plan.get(field) or "<missing>")] += 1
        if unique_elements & LANTHANIDES:
            group_presence["lanthanide"] += 1
        if unique_elements & ACTINIDES:
            group_presence["actinide"] += 1
        if unique_elements & HALOGENS:
            group_presence["halogen"] += 1
        if unique_elements & CHALCOGENS:
            group_presence["chalcogen"] += 1
        if unique_elements & PNICTOGENS:
            group_presence["pnictogen"] += 1
        if "O" in unique_elements:
            group_presence["oxygen"] += 1
        if "H" in unique_elements:
            group_presence["hydrogen"] += 1

    fixed_denominator = len(records)
    formula_counts = distributions["formula"]
    chemsys_counts = distributions["chemsys"]
    result = {
        "cohort_id": cohort_id,
        "attempts": fixed_denominator,
        "parsed": parsed,
        "planner_failed": fixed_denominator - parsed,
        "parse_rate": parsed / fixed_denominator,
        "unique_formula_count": len(formula_counts),
        "duplicate_formula_rows": parsed - len(formula_counts),
        "unique_chemsys_count": len(chemsys_counts),
        "duplicate_chemsys_rows": parsed - len(chemsys_counts),
        "group_presence": _sorted_counts(group_presence),
        "group_presence_rate_per_attempt": {
            key: int(value) / fixed_denominator
            for key, value in _sorted_counts(group_presence).items()
        },
        "distributions": {
            field: _sorted_counts(counter)
            for field, counter in distributions.items()
        },
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def compare_summaries(
    reference: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    reference_distributions = reference["distributions"]
    observed_distributions = observed["distributions"]
    return {
        "reference_cohort_id": reference["cohort_id"],
        "observed_cohort_id": observed["cohort_id"],
        "parse_rate_difference": float(observed["parse_rate"])
        - float(reference["parse_rate"]),
        "unique_formula_rate_difference": (
            int(observed["unique_formula_count"]) / int(observed["attempts"])
            - int(reference["unique_formula_count"])
            / int(reference["attempts"])
        ),
        "distances": {
            field: categorical_distance(
                reference_distributions[field], observed_distributions[field]
            )
            for field in DISTANCE_FIELDS
        },
        "largest_element_presence_rate_shifts": _top_rate_shifts(
            reference_distributions["element_presence"],
            observed_distributions["element_presence"],
            int(reference["attempts"]),
            int(observed["attempts"]),
        ),
        "largest_chemsys_rate_shifts": _top_rate_shifts(
            reference_distributions["chemsys"],
            observed_distributions["chemsys"],
            int(reference["attempts"]),
            int(observed["attempts"]),
        ),
    }


def build_audit(
    *, run_root: Path, config: Mapping[str, Any], output_path: Path
) -> dict[str, Any]:
    reference_spec = config["historical_planner_reference"]
    reference_path = require_file(
        reference_spec["cohort256"],
        reference_spec["cohort256_sha256"],
        "historical H1-A2 world2 planner cohort",
    )
    reference_rows = read_jsonl(reference_path)
    if len(reference_rows) != DENOMINATOR:
        raise ValueError("historical planner denominator changed")
    reference = summarize_rows(
        reference_rows, cohort_id=str(reference_spec["cohort_id"])
    )

    observed: list[dict[str, Any]] = []
    observed_rows: list[list[dict[str, Any]]] = []
    identities: list[dict[str, Any]] = []
    for cohort_id in config["downstream_cohorts"]:
        cohort_path = (
            run_root / "planner" / str(cohort_id) / "frozen" / "cohort256.jsonl"
        )
        rows = read_jsonl(cohort_path)
        if len(rows) != DENOMINATOR:
            raise ValueError(f"fresh planner denominator changed: {cohort_id}")
        observed_rows.append(rows)
        observed.append(summarize_rows(rows, cohort_id=str(cohort_id)))
        identities.append(
            {
                "cohort_id": str(cohort_id),
                "path": str(cohort_path.resolve()),
                "bytes": cohort_path.stat().st_size,
                "sha256": sha256_file(cohort_path),
            }
        )

    pooled = summarize_rows(
        [row for rows in observed_rows for row in rows],
        cohort_id="retrained_world2_four_cohorts_pooled1024_descriptive_only",
    )
    pairwise = []
    for left_index in range(len(observed)):
        for right_index in range(left_index + 1, len(observed)):
            pairwise.append(
                compare_summaries(observed[left_index], observed[right_index])
            )
    audit = {
        "schema": "h1a2_retrained_world2_deep_plan_distribution_audit_v1",
        "status": "complete",
        "reference": {
            "identity": {
                "cohort_id": reference_spec["cohort_id"],
                "path": str(reference_path),
                "bytes": reference_path.stat().st_size,
                "sha256": sha256_file(reference_path),
            },
            "summary": reference,
        },
        "fresh_cohort_identities": identities,
        "fresh_cohort_summaries": observed,
        "fresh_pooled1024_descriptive_only": pooled,
        "comparisons_to_historical_world2": [
            compare_summaries(reference, row) for row in observed
        ],
        "fresh_pairwise_comparisons": pairwise,
        "comparison_fields": list(DISTANCE_FIELDS),
        "no_selection_filter_retry_replacement_repair_or_rerank": True,
    }
    audit["audit_payload_sha256"] = canonical_sha256(audit)
    write_json_exclusive(output_path, audit)
    return audit


if __name__ == "__main__":
    raise SystemExit(
        "This module is invoked by assemble_planners.py after all four cohorts pass."
    )
