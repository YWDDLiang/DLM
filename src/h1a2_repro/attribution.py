"""Stagewise chemistry attribution for H1-A2 attempt ledgers.

The implementation is deliberately dependency-free.  It treats every JSONL
row as one requested attempt so failed stages remain in the denominator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import comb, gcd, log
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


STAGES = (
    "requested",
    "decoded",
    "plan_eligible",
    "body_attempted",
    "body_success",
    "refined",
    "reconstructed",
    "hull_known",
)

PRIMARY_STRATUM_FIELDS = ("family", "arity", "n_bin", "shortcut")
DISTRIBUTION_FIELDS = (
    "family",
    "arity",
    "n_bin",
    "shortcut",
    "generated_anion",
    "lattice",
    "spacegroup",
    "volume",
)

NONMETALS = {
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Se",
    "Br",
    "Ge",
    "As",
    "Sb",
    "Te",
    "I",
    "At",
}

HALOGENS = {"F", "Cl", "Br", "I", "At"}
CHALCOGENS = {"S", "Se", "Te", "Po"}
PNICTOGENS = {"N", "P", "As", "Sb", "Bi"}


def parse_formula(formula: str) -> list[tuple[str, int]]:
    """Parse a flat integer-count formula such as ``Li2O``.

    Parenthesized, fractional, signed, and occupancy formulas are rejected so
    analysis cannot silently reinterpret unsupported chemistry.
    """

    text = "".join(str(formula).split())
    if not text:
        raise ValueError("formula is empty")
    parts: list[tuple[str, int]] = []
    index = 0
    while index < len(text):
        if not text[index].isupper():
            raise ValueError(f"unsupported formula syntax at offset {index}: {text!r}")
        end_symbol = index + 1
        while end_symbol < len(text) and text[end_symbol].islower():
            end_symbol += 1
        symbol = text[index:end_symbol]
        end_count = end_symbol
        while end_count < len(text) and text[end_count].isdigit():
            end_count += 1
        count = int(text[end_symbol:end_count] or "1")
        if count <= 0:
            raise ValueError(f"non-positive count for {symbol}: {count}")
        parts.append((symbol, count))
        index = end_count
    merged: Counter[str] = Counter()
    for symbol, count in parts:
        merged[symbol] += count
    return sorted(merged.items())


def reduced_counts(parts: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    divisor = 0
    for _, count in parts:
        divisor = gcd(divisor, int(count))
    divisor = max(1, divisor)
    return [(symbol, int(count) // divisor) for symbol, count in parts]


def composition_family(elements: Iterable[str]) -> str:
    elems = set(elements)
    if not elems:
        return "unknown"
    if not (elems & NONMETALS):
        return "all-metal/intermetallic"
    has_oxygen = "O" in elems
    has_halogen = bool(elems & HALOGENS)
    has_chalcogen = bool(elems & CHALCOGENS)
    has_pnictogen = bool(elems & PNICTOGENS)
    if has_oxygen and has_halogen:
        return "mixed oxide+halide"
    if has_oxygen and has_chalcogen:
        return "mixed oxide+chalcogen"
    if has_oxygen and has_pnictogen:
        return "mixed oxide+pnictide"
    if has_oxygen:
        return "oxide"
    if has_halogen:
        return "halide"
    if has_chalcogen:
        return "nonoxide-chalcogen"
    if has_pnictogen:
        return "pnictide"
    return "other/mixed"


def atom_count_bin(num_atoms: int) -> str:
    value = int(num_atoms)
    if value <= 4:
        return "01-04"
    if value <= 8:
        return "05-08"
    if value <= 12:
        return "09-12"
    if value <= 16:
        return "13-16"
    return "17-20+"


def _nested(row: Mapping[str, Any], section: str, key: str, default: Any = None) -> Any:
    payload = row.get(section)
    if isinstance(payload, Mapping) and key in payload:
        return payload[key]
    return row.get(key, default)


def normalize_attempt(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    parts = parse_formula(str(row.get("formula", "")))
    elements = [symbol for symbol, _ in parts]
    num_atoms = int(row.get("N") or row.get("num_atoms") or sum(count for _, count in parts))
    family = str(row.get("family") or composition_family(elements))
    arity = int(row.get("arity") or len(elements))
    shortcut = str(
        row.get("shortcut")
        or ("unary" if arity == 1 else "all-metal" if family == "all-metal/intermetallic" else "none")
    )
    result.update(
        {
            "elements": elements,
            "elements_key": "-".join(sorted(elements)),
            "counts": [count for _, count in parts],
            "reduced_counts": [count for _, count in reduced_counts(parts)],
            "N": num_atoms,
            "family": family,
            "arity": str(arity),
            "n_bin": str(row.get("n_bin") or atom_count_bin(num_atoms)),
            "shortcut": shortcut,
            "generated_anion": str(row.get("generated_anion") or row.get("anion") or "unknown"),
            "lattice": str(row.get("lattice") or "unknown"),
            "spacegroup": str(row.get("spacegroup") or "unknown"),
            "volume": str(row.get("volume") or "unknown"),
        }
    )

    stages: dict[str, bool] = {}
    previous = True
    for stage in STAGES:
        value = bool(_nested(row, "stages", stage, stage == "requested"))
        if value and not previous:
            raise ValueError(f"non-monotone stages for {row.get('attempt_id')}: {stage}=true after failure")
        stages[stage] = value
        previous = value
    result["stages"] = stages
    result["outcomes"] = {
        key: bool(_nested(row, "outcomes", key, False))
        for key in ("novel", "unique", "strict_sun", "meta_sun")
    }
    return result


def funnel(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {stage: sum(bool(row["stages"][stage]) for row in rows) for stage in STAGES}
    counts.update(
        {
            "novel": sum(bool(row["outcomes"]["novel"]) for row in rows),
            "unique": sum(bool(row["outcomes"]["unique"]) for row in rows),
            "novel_unique": sum(
                bool(row["outcomes"]["novel"] and row["outcomes"]["unique"])
                for row in rows
            ),
            "strict_sun": sum(bool(row["outcomes"]["strict_sun"]) for row in rows),
            "meta_sun": sum(bool(row["outcomes"]["meta_sun"]) for row in rows),
        }
    )
    return counts


def distribution(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    if not rows:
        return {}
    counts = Counter(str(row.get(field, "unknown")) for row in rows)
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())}


def element_presence_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(set(str(element) for element in row.get("elements", [])))
    return {key: value / len(rows) for key, value in sorted(counts.items())}


def atom_weighted_element_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in rows:
        for element, count in zip(row.get("elements", []), row.get("counts", [])):
            counts[str(element)] += int(count)
    total = sum(counts.values())
    return {} if not total else {key: value / total for key, value in sorted(counts.items())}


def total_variation(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(float(a.get(key, 0.0)) - float(b.get(key, 0.0))) for key in keys)


def jensen_shannon(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    midpoint = {key: 0.5 * (float(a.get(key, 0.0)) + float(b.get(key, 0.0))) for key in keys}

    def kl_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
        return sum(float(p.get(key, 0.0)) * log(float(p[key]) / float(q[key])) for key in p if p[key] > 0)

    return 0.5 * kl_divergence(a, midpoint) + 0.5 * kl_divergence(b, midpoint)


def stage_rows(rows: Sequence[Mapping[str, Any]], stage: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if bool(row["stages"][stage])]


def stage_distribution_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    previous_stage: str | None = None
    for stage in STAGES:
        current = stage_rows(rows, stage)
        distributions = {field: distribution(current, field) for field in DISTRIBUTION_FIELDS}
        distributions.update(
            {
                "elements_key": distribution(current, "elements_key"),
                "element_presence": element_presence_distribution(current),
                "element_atom_weighted": atom_weighted_element_distribution(current),
            }
        )
        payload: dict[str, Any] = {"n": len(current), "distributions": distributions}
        if previous_stage is not None:
            previous = result[previous_stage]["distributions"]
            probability_fields = list(DISTRIBUTION_FIELDS) + [
                "elements_key",
                "element_atom_weighted",
            ]
            payload["drift_from_previous"] = {
                field: {
                    "tvd": total_variation(previous[field], distributions[field]),
                    "jsd_nats": jensen_shannon(previous[field], distributions[field]),
                }
                for field in probability_fields
            }
            presence_keys = set(previous["element_presence"]) | set(distributions["element_presence"])
            presence_deltas = [
                abs(
                    float(previous["element_presence"].get(key, 0.0))
                    - float(distributions["element_presence"].get(key, 0.0))
                )
                for key in presence_keys
            ]
            payload["element_presence_drift"] = {
                "max_abs_delta": max(presence_deltas, default=0.0),
                "mean_abs_delta": sum(presence_deltas) / max(1, len(presence_deltas)),
            }
        result[stage] = payload
        previous_stage = stage
    return result


def stratum_survival_report(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str] = PRIMARY_STRATUM_FIELDS
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for previous_stage, next_stage in zip(STAGES, STAGES[1:]):
        groups: dict[tuple[str, ...], list[int]] = defaultdict(lambda: [0, 0])
        for row in rows:
            if not row["stages"][previous_stage]:
                continue
            key = stratum_key(row, fields)
            groups[key][0] += 1
            groups[key][1] += int(bool(row["stages"][next_stage]))
        result[f"{previous_stage}->{next_stage}"] = {
            "fields": list(fields),
            "strata": {
                "|".join(key): {
                    "from_n": values[0],
                    "to_n": values[1],
                    "survival_rate": values[1] / values[0],
                }
                for key, values in sorted(groups.items())
            },
        }
    return result


def discovery_pareto(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requested = len(rows)
    novel_unique = sum(
        bool(row["outcomes"]["novel"] and row["outcomes"]["unique"])
        for row in rows
    )
    strict = sum(bool(row["outcomes"]["strict_sun"]) for row in rows)
    meta = sum(bool(row["outcomes"]["meta_sun"]) for row in rows)
    return {
        "requested": requested,
        "novel_unique": novel_unique,
        "novel_unique_supply": novel_unique / max(1, requested),
        "strict_sun": strict,
        "meta_sun": meta,
        "strict_conversion_within_novel_unique": strict / novel_unique if novel_unique else None,
        "meta_conversion_within_novel_unique": meta / novel_unique if novel_unique else None,
        "strict_implies_novel_unique": all(
            not row["outcomes"]["strict_sun"]
            or (row["outcomes"]["novel"] and row["outcomes"]["unique"])
            for row in rows
        ),
        "meta_implies_novel_unique": all(
            not row["outcomes"]["meta_sun"]
            or (row["outcomes"]["novel"] and row["outcomes"]["unique"])
            for row in rows
        ),
    }


def exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    discordant = int(a_only) + int(b_only)
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, value) for value in range(min(int(a_only), int(b_only)) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_mcnemar(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    outcome: str,
    *,
    key: str = "attempt_id",
    known_stage: str | None = "hull_known",
) -> dict[str, Any]:
    if known_stage is not None and known_stage not in STAGES:
        raise ValueError(f"unknown paired-known stage {known_stage!r}")

    def index(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if key not in row:
                raise ValueError(f"{label} row has no paired key {key!r}")
            value = str(row[key])
            if value in result:
                raise ValueError(f"duplicate paired key {value!r} in {label}")
            result[value] = row
        return result

    indexed_a = index(rows_a, "A")
    indexed_b = index(rows_b, "B")
    common = sorted(set(indexed_a) & set(indexed_b))
    cells = {"a0_b0": 0, "a0_b1": 0, "a1_b0": 0, "a1_b1": 0}
    known_both = 0
    for pair_key in common:
        row_a = indexed_a[pair_key]
        row_b = indexed_b[pair_key]
        if known_stage is not None and not (
            bool(row_a["stages"][known_stage]) and bool(row_b["stages"][known_stage])
        ):
            continue
        known_both += 1
        value_a = int(bool(row_a["outcomes"][outcome]))
        value_b = int(bool(row_b["outcomes"][outcome]))
        cells[f"a{value_a}_b{value_b}"] += 1
    return {
        "outcome": outcome,
        "paired_key": key,
        "known_stage": known_stage,
        "common_pairs": len(common),
        "known_both_pairs": known_both,
        "cells": cells,
        "discordant_a_only": cells["a1_b0"],
        "discordant_b_only": cells["a0_b1"],
        "exact_two_sided_p": exact_mcnemar_pvalue(cells["a1_b0"], cells["a0_b1"]),
    }


def stratum_key(row: Mapping[str, Any], fields: Sequence[str] = PRIMARY_STRATUM_FIELDS) -> tuple[str, ...]:
    return tuple(str(row.get(field, "unknown")) for field in fields)


def _group_outcomes(
    rows: Sequence[Mapping[str, Any]], outcome: str, fields: Sequence[str]
) -> dict[tuple[str, ...], tuple[int, int]]:
    counts: dict[tuple[str, ...], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = stratum_key(row, fields)
        counts[key][0] += 1
        counts[key][1] += int(bool(row["outcomes"][outcome]))
    return {key: (values[0], values[1]) for key, values in counts.items()}


def symmetric_decomposition(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    outcome: str,
    fields: Sequence[str] = PRIMARY_STRATUM_FIELDS,
) -> dict[str, Any]:
    grouped_a = _group_outcomes(rows_a, outcome, fields)
    grouped_b = _group_outcomes(rows_b, outcome, fields)
    common = sorted(set(grouped_a) & set(grouped_b))
    common_a = sum(grouped_a[key][0] for key in common)
    common_b = sum(grouped_b[key][0] for key in common)
    if not common or not common_a or not common_b:
        return {
            "fields": list(fields),
            "outcome": outcome,
            "common_strata": 0,
            "coverage_a": 0.0,
            "coverage_b": 0.0,
            "estimable": False,
        }

    mix = 0.0
    conditional = 0.0
    for key in common:
        n_a, y_a = grouped_a[key]
        n_b, y_b = grouped_b[key]
        p_a = n_a / common_a
        p_b = n_b / common_b
        mu_a = y_a / n_a
        mu_b = y_b / n_b
        mix += 0.5 * (p_a - p_b) * (mu_a + mu_b)
        conditional += 0.5 * (p_a + p_b) * (mu_a - mu_b)
    rate_a = sum(grouped_a[key][1] for key in common) / common_a
    rate_b = sum(grouped_b[key][1] for key in common) / common_b
    return {
        "fields": list(fields),
        "outcome": outcome,
        "common_strata": len(common),
        "coverage_a": common_a / max(1, len(rows_a)),
        "coverage_b": common_b / max(1, len(rows_b)),
        "rate_a_common": rate_a,
        "rate_b_common": rate_b,
        "gap_common": rate_a - rate_b,
        "mix_effect": mix,
        "conditional_effect": conditional,
        "identity_residual": (rate_a - rate_b) - mix - conditional,
        "estimable": True,
    }


def standardized_rate(
    rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    outcome: str,
    fields: Sequence[str] = PRIMARY_STRATUM_FIELDS,
) -> dict[str, Any]:
    grouped = _group_outcomes(rows, outcome, fields)
    reference_counts = Counter(stratum_key(row, fields) for row in reference_rows)
    row_counts = Counter(stratum_key(row, fields) for row in rows)
    reference_total = sum(reference_counts.values())
    common = sorted(set(grouped) & set(reference_counts))
    reference_coverage = sum(reference_counts[key] for key in common) / max(1, reference_total)
    covered_reference_total = sum(reference_counts[key] for key in common)
    if not common or not covered_reference_total:
        return {"estimable": False, "reference_coverage": 0.0}

    value = 0.0
    weights: list[float] = []
    for key in common:
        n, successes = grouped[key]
        normalized_reference_weight = reference_counts[key] / covered_reference_total
        value += normalized_reference_weight * (successes / n)
        p_reference = reference_counts[key] / reference_total
        p_rows = row_counts[key] / len(rows)
        weights.extend([p_reference / p_rows] * n)
    weight_sum = sum(weights)
    ess = weight_sum * weight_sum / sum(weight * weight for weight in weights)
    return {
        "estimable": True,
        "outcome": outcome,
        "fields": list(fields),
        "standardized_rate_on_overlap": value,
        "reference_coverage": reference_coverage,
        "common_strata": len(common),
        "effective_sample_size": ess,
        "max_weight": max(weights),
        "median_weight": median(weights),
        "max_to_median_weight": max(weights) / max(median(weights), 1e-12),
        "trimming": "none",
        "estimand": "reference-standardized on observed common support",
    }


def analyze_cohort(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_attempt(row) for row in rows]
    hull_known = [row for row in normalized if row["stages"]["hull_known"]]
    reconstructed = [row for row in normalized if row["stages"]["reconstructed"]]
    return {
        "n": len(normalized),
        "funnel": funnel(normalized),
        "stages": stage_distribution_report(normalized),
        "stratum_survival": stratum_survival_report(normalized),
        "discovery_pareto": discovery_pareto(normalized),
        "strict_rate": sum(row["outcomes"]["strict_sun"] for row in normalized) / max(1, len(normalized)),
        "meta_rate": sum(row["outcomes"]["meta_sun"] for row in normalized) / max(1, len(normalized)),
        "all_attempt_rates": {
            "denominator": len(normalized),
            "strict_numerator": sum(row["outcomes"]["strict_sun"] for row in normalized),
            "meta_numerator": sum(row["outcomes"]["meta_sun"] for row in normalized),
        },
        "hull_known_rates": {
            "denominator": len(hull_known),
            "unknown": len(reconstructed) - len(hull_known),
            "not_reconstructed": len(normalized) - len(reconstructed),
            "strict_numerator": sum(row["outcomes"]["strict_sun"] for row in hull_known),
            "meta_numerator": sum(row["outcomes"]["meta_sun"] for row in hull_known),
            "strict_rate": (
                None
                if not hull_known
                else sum(row["outcomes"]["strict_sun"] for row in hull_known) / len(hull_known)
            ),
            "meta_rate": (
                None
                if not hull_known
                else sum(row["outcomes"]["meta_sun"] for row in hull_known) / len(hull_known)
            ),
        },
    }


def analyze_pair(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    reference_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_a = [normalize_attempt(row) for row in rows_a]
    normalized_b = [normalize_attempt(row) for row in rows_b]
    reference = normalized_a if reference_rows is None else [normalize_attempt(row) for row in reference_rows]
    hull_a = [row for row in normalized_a if row["stages"]["hull_known"]]
    hull_b = [row for row in normalized_b if row["stages"]["hull_known"]]
    hull_reference = [row for row in reference if row["stages"]["hull_known"]]
    return {
        outcome: {
            "decomposition": symmetric_decomposition(normalized_a, normalized_b, outcome),
            "standardized_a": standardized_rate(normalized_a, reference, outcome),
            "standardized_b": standardized_rate(normalized_b, reference, outcome),
            "hull_known_decomposition": symmetric_decomposition(hull_a, hull_b, outcome),
            "hull_known_standardized_a": standardized_rate(hull_a, hull_reference, outcome),
            "hull_known_standardized_b": standardized_rate(hull_b, hull_reference, outcome),
        }
        for outcome in ("strict_sun", "meta_sun")
    }


__all__ = [
    "DISTRIBUTION_FIELDS",
    "PRIMARY_STRATUM_FIELDS",
    "STAGES",
    "analyze_cohort",
    "analyze_pair",
    "atom_count_bin",
    "composition_family",
    "discovery_pareto",
    "distribution",
    "element_presence_distribution",
    "exact_mcnemar_pvalue",
    "funnel",
    "jensen_shannon",
    "normalize_attempt",
    "parse_formula",
    "paired_mcnemar",
    "reduced_counts",
    "standardized_rate",
    "stratum_survival_report",
    "symmetric_decomposition",
    "total_variation",
    "atom_weighted_element_distribution",
]
