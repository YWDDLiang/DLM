"""Plan collision, entropy, and downstream-conversion audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from typing import Any, Mapping, Sequence

from .attribution import atom_count_bin, composition_family, parse_formula, reduced_counts


PLAN_DISTANCE_FIELDS = (
    "formula",
    "composition_key",
    "family",
    "anion",
    "charge",
    "N",
    "lattice",
    "spacegroup",
    "volume",
)


def extract_plan(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("plan_state", "r5_plan_state", "parsed_plan"):
        payload = row.get(key)
        if isinstance(payload, Mapping):
            return payload
    return row


def _outcome(row: Mapping[str, Any], key: str) -> bool | None:
    outcomes = row.get("outcomes")
    if isinstance(outcomes, Mapping) and key in outcomes:
        return bool(outcomes[key])
    if key in row:
        return bool(row[key])
    return None


def normalize_plan(row: Mapping[str, Any], index: int = 0) -> dict[str, Any]:
    plan = extract_plan(row)
    formula = str(plan.get("formula") or row.get("formula") or "")
    elements = [str(value) for value in (plan.get("elements") or [])]
    counts = [int(value) for value in (plan.get("counts") or [])]
    if not elements or not counts:
        parsed = parse_formula(formula)
        elements = [symbol for symbol, _ in parsed]
        counts = [count for _, count in parsed]
    if len(elements) != len(counts) or not elements:
        raise ValueError(f"Plan row {index}: invalid elements/counts")
    merged: Counter[str] = Counter()
    for element, count in zip(elements, counts):
        if count <= 0:
            raise ValueError(f"Plan row {index}: non-positive count for {element}")
        merged[element] += count
    reduced = reduced_counts(sorted(merged.items()))
    composition_key = "|".join(f"{element}:{count}" for element, count in reduced)
    num_atoms = int(plan.get("N") or sum(counts))
    if num_atoms != sum(counts):
        raise ValueError(f"Plan row {index}: N {num_atoms} != sum(counts) {sum(counts)}")
    result = {
        "plan_id": str(
            row.get("plan_id")
            or row.get("sample_idx")
            or row.get("raw_ordinal")
            or f"row-{index:06d}"
        ),
        "formula": formula,
        "composition_key": composition_key,
        "elements": sorted(merged),
        "counts": [merged[element] for element in sorted(merged)],
        "family": str(plan.get("family") or composition_family(merged)),
        "anion": str(plan.get("anion_framework") or "unknown"),
        "charge": str(plan.get("charge_bucket") or "unknown"),
        "arity": str(len(merged)),
        "N": str(num_atoms),
        "n_bin": atom_count_bin(num_atoms),
        "lattice": str(plan.get("lattice_system") or "unknown"),
        "spacegroup": str(plan.get("spacegroup_bucket") or "unknown"),
        "volume": str(plan.get("volume_per_atom_bin") or "unknown"),
    }
    result["full_tuple"] = tuple(str(result[field]) for field in PLAN_DISTANCE_FIELDS)
    result["outcomes"] = {
        key: _outcome(row, key) for key in ("body_success", "strict_sun", "meta_sun")
    }
    return result


def entropy(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * log(count / total) for count in counts.values())


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def nearest_hamming_distances(
    generated: Sequence[Mapping[str, Any]], train: Sequence[Mapping[str, Any]]
) -> list[int]:
    if not train:
        raise ValueError("training Plan reference is empty")
    by_field: dict[str, dict[str, set[int]]] = {
        field: defaultdict(set) for field in PLAN_DISTANCE_FIELDS
    }
    train_tuples = [tuple(str(row[field]) for field in PLAN_DISTANCE_FIELDS) for row in train]
    for index, row in enumerate(train):
        for field in PLAN_DISTANCE_FIELDS:
            by_field[field][str(row[field])].add(index)

    distances: list[int] = []
    maximum = len(PLAN_DISTANCE_FIELDS)
    for row in generated:
        values = tuple(str(row[field]) for field in PLAN_DISTANCE_FIELDS)
        candidates: set[int] = set()
        for field, value in zip(PLAN_DISTANCE_FIELDS, values):
            candidates.update(by_field[field].get(value, set()))
        if not candidates:
            distances.append(maximum)
            continue
        distances.append(
            min(sum(left != right for left, right in zip(values, train_tuples[index])) for index in candidates)
        )
    return distances


def grouped_conversion(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    result: dict[str, Any] = {}
    for value, group in sorted(groups.items()):
        payload: dict[str, Any] = {"n": len(group)}
        for outcome in ("body_success", "strict_sun", "meta_sun"):
            available = [row["outcomes"][outcome] for row in group if row["outcomes"][outcome] is not None]
            payload[outcome] = {
                "known": len(available),
                "numerator": sum(bool(item) for item in available),
                "rate": None if not available else sum(bool(item) for item in available) / len(available),
            }
        result[value] = payload
    return result


def audit_plans(
    generated_rows: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    generated = [normalize_plan(row, index) for index, row in enumerate(generated_rows)]
    train = [normalize_plan(row, index) for index, row in enumerate(train_rows)]
    train_formula = {row["formula"] for row in train}
    train_composition = {row["composition_key"] for row in train}
    train_full = {row["full_tuple"] for row in train}
    distances = nearest_hamming_distances(generated, train)
    marginals = {
        field: {
            "unique": len({row[field] for row in generated}),
            "entropy_nats": entropy([row[field] for row in generated]),
            "distribution": dict(Counter(str(row[field]) for row in generated)),
        }
        for field in PLAN_DISTANCE_FIELDS
    }
    full_tuples = [row["full_tuple"] for row in generated]
    return {
        "schema": "h1a2_plan_audit_v1",
        "generated_n": len(generated),
        "train_n": len(train),
        "collisions": {
            "formula_exact": sum(row["formula"] in train_formula for row in generated),
            "composition_exact": sum(row["composition_key"] in train_composition for row in generated),
            "full_tuple_exact": sum(row["full_tuple"] in train_full for row in generated),
        },
        "generated_diversity": {
            "unique_formula": len({row["formula"] for row in generated}),
            "unique_composition": len({row["composition_key"] for row in generated}),
            "unique_full_tuple": len(set(full_tuples)),
            "full_tuple_entropy_nats": entropy(full_tuples),
            "unique_full_tuple_rate": len(set(full_tuples)) / max(1, len(generated)),
        },
        "nearest_train_tuple_hamming": {
            "fields": list(PLAN_DISTANCE_FIELDS),
            "counts": dict(sorted(Counter(distances).items())),
            "min": min(distances, default=None),
            "median": _quantile(distances, 0.5),
            "p90": _quantile(distances, 0.9),
            "max": max(distances, default=None),
        },
        "marginals": marginals,
        "downstream_conversion": {
            field: grouped_conversion(generated, field)
            for field in ("family", "arity", "n_bin", "anion", "charge", "lattice", "spacegroup", "volume")
        },
    }


__all__ = [
    "PLAN_DISTANCE_FIELDS",
    "audit_plans",
    "entropy",
    "extract_plan",
    "grouped_conversion",
    "nearest_hamming_distances",
    "normalize_plan",
]
