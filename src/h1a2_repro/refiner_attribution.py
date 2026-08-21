"""Summaries for aligned discrete-proposal and continuous-refiner rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any, Mapping, Sequence


IDENTITY_FIELDS = ("n_invariant", "composition_invariant")
BOOLEAN_FIELDS = (
    "structure_match",
    "spacegroup_same",
    "pre_p1",
    "post_p1",
    "pre_struct_valid",
    "post_struct_valid",
    "pre_comp_valid",
    "post_comp_valid",
    "pre_joint_valid",
    "post_joint_valid",
    "plan_lattice_match_pre",
    "plan_lattice_match_post",
    "plan_spacegroup_match_pre",
    "plan_spacegroup_match_post",
    "plan_volume_match_pre",
    "plan_volume_match_post",
)
NUMERIC_FIELDS = (
    "coordinate_periodic_rms",
    "lattice_length_mae",
    "lattice_angle_mae",
    "min_distance_pre",
    "min_distance_post",
    "energy_pre",
    "energy_post",
    "e_hull_pre",
    "e_hull_post",
)


def _known_boolean(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    known = [bool(row[field]) for row in rows if row.get(field) is not None]
    return {
        "known": len(known),
        "true": sum(known),
        "rate": None if not known else sum(known) / len(known),
    }


def _numeric(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return {
        "known": len(values),
        "mean": None if not values else sum(values) / len(values),
        "median": None if not values else median(values),
        "min": None if not values else min(values),
        "max": None if not values else max(values),
    }


def _delta(
    rows: Sequence[Mapping[str, Any]],
    before: str,
    after: str,
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    values = [
        float(row[after]) - float(row[before])
        for row in rows
        if row.get(before) is not None and row.get(after) is not None
    ]
    return {
        "known_pairs": len(values),
        "mean_delta": None if not values else sum(values) / len(values),
        "median_delta": None if not values else median(values),
        "improved": sum((value < 0) if lower_is_better else (value > 0) for value in values),
        "worsened": sum((value > 0) if lower_is_better else (value < 0) for value in values),
        "unchanged": sum(value == 0 for value in values),
    }


def _conversion_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    known = 0
    for row in rows:
        if row.get("body_good") is None or row.get("final_good") is None:
            continue
        known += 1
        before = "good" if bool(row["body_good"]) else "bad"
        after = "good" if bool(row["final_good"]) else "bad"
        counts[f"{before}_body->{after}_final"] += 1
    return {"known_pairs": known, "counts": dict(sorted(counts.items()))}


def _grouped_final_rate(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row.get(field) is None or row.get("final_good") is None:
            continue
        groups[str(row[field])].append(bool(row["final_good"]))
    return {
        value: {"n": len(items), "final_good": sum(items), "rate": sum(items) / len(items)}
        for value, items in sorted(groups.items())
    }


def summarize_refiner_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requested = len(rows)
    body_success_rows = [row for row in rows if bool(row.get("body_success"))]
    refined_rows = [row for row in rows if bool(row.get("refined"))]
    return {
        "schema": "h1a2_refiner_attribution_v1",
        "requested_attempts": requested,
        "body_successes": len(body_success_rows),
        "body_failures": requested - len(body_success_rows),
        "refined": len(refined_rows),
        "identity": {field: _known_boolean(refined_rows, field) for field in IDENTITY_FIELDS},
        "booleans": {field: _known_boolean(refined_rows, field) for field in BOOLEAN_FIELDS},
        "numeric": {field: _numeric(refined_rows, field) for field in NUMERIC_FIELDS},
        "paired_deltas": {
            "minimum_distance": _delta(
                refined_rows, "min_distance_pre", "min_distance_post", lower_is_better=False
            ),
            "energy": _delta(refined_rows, "energy_pre", "energy_post", lower_is_better=True),
            "e_hull": _delta(refined_rows, "e_hull_pre", "e_hull_post", lower_is_better=True),
        },
        "body_to_final": _conversion_matrix(refined_rows),
        "final_rate_by_body_feature": {
            field: _grouped_final_rate(refined_rows, field)
            for field in (
                "pre_struct_valid",
                "pre_joint_valid",
                "plan_lattice_match_pre",
                "plan_spacegroup_match_pre",
                "plan_volume_match_pre",
                "arm",
                "plan_source",
            )
        },
    }


__all__ = ["BOOLEAN_FIELDS", "IDENTITY_FIELDS", "NUMERIC_FIELDS", "summarize_refiner_rows"]
