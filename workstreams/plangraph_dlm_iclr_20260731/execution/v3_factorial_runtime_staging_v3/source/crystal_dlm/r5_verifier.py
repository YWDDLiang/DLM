"""Lightweight R5 proposal verifier and ranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from crystal_dlm.composition_validity import (
    composition_record,
    element_symbols,
    formula_from_composition,
    pbc_duplicate_record,
    reduced_composition,
)
from crystal_dlm.diagnostic_remask import geometry_degeneracy_record
from crystal_dlm.r5_plan_state import validate_plan_state


@dataclass(frozen=True)
class VerifierWeights:
    comp_valid: float = 1.0
    strict_valid: float = 1.0
    graph_valid: float = 0.5
    refine_success: float = 0.5
    meta_stable: float = 2.0
    strict_stable: float = 4.0
    novel: float = 1.0
    unique: float = 1.0
    single_penalty: float = 2.0
    all_metal_penalty: float = 0.2
    high_sym_penalty: float = 1.0
    duplicate_penalty: float = 2.0
    plan_mismatch_penalty: float = 2.0


def plan_composition_matches_arrays(plan_state: Mapping[str, Any] | None, arrays: Mapping[str, Any]) -> bool | None:
    if not plan_state:
        return None
    validation = validate_plan_state(plan_state)
    if not validation.valid_N:
        return False
    expected = dict(zip(plan_state.get("elements", []), plan_state.get("counts", [])))
    if not expected:
        return None
    actual: Dict[str, int] = {}
    for symbol in arrays.get("species", []):
        actual[str(symbol)] = actual.get(str(symbol), 0) + 1
    return {str(k): int(v) for k, v in expected.items()} == actual


def bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def extract_verifier_features(
    arrays: Mapping[str, Any],
    *,
    plan_state: Mapping[str, Any] | None = None,
    sample_record: Mapping[str, Any] | None = None,
    metric_record: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    sample_record = dict(sample_record or {})
    metric_record = dict(metric_record or {})
    atom_types = arrays.get("atom_types", [])
    try:
        comp = composition_record(atom_types)
    except Exception as exc:  # noqa: BLE001 - SMACT/numpy may be absent in local smoke envs.
        elems, counts = reduced_composition(atom_types)
        comp = {
            "elems": list(elems),
            "symbols": list(element_symbols(elems)),
            "counts": list(counts),
            "formula": formula_from_composition(elems, counts),
            "num_atoms": len(atom_types),
            "num_elements": len(elems),
            "comp_valid": False,
            "reason": "validator_unavailable",
            "validator_error": type(exc).__name__,
        }
    geom = geometry_degeneracy_record(arrays)
    dup = pbc_duplicate_record(arrays.get("frac_coords", []))
    ehull_raw = metric_record.get("energy_above_hull_per_atom", metric_record.get("e_hull"))
    try:
        ehull = None if ehull_raw is None else float(ehull_raw)
    except Exception:
        ehull = None
    novel = bool_or_none(metric_record.get("novel"))
    unique = bool_or_none(metric_record.get("unique"))
    novel_unique = bool_or_none(metric_record.get("novel_unique"))
    plan_match = plan_composition_matches_arrays(plan_state, arrays)
    reason = str(comp.get("reason", ""))
    return {
        "num_atoms": int(arrays.get("num_atoms", comp.get("num_atoms", 0))),
        "num_elements": int(comp.get("num_elements", 0)),
        "formula": comp.get("formula"),
        "comp_valid": bool(comp.get("comp_valid")),
        "strict_valid": reason == "charge_neutral_pauling_valid",
        "composition_reason": reason,
        "single_element": reason == "single_element_shortcut",
        "all_metal": reason == "all_metal_shortcut",
        "graph_valid": bool(sample_record.get("parsed", True)),
        "plan_match": plan_match,
        "a_eq_b_eq_c": bool(geom["all_lengths_equal"]),
        "all_90_angles": bool(geom["all_angles_90"]),
        "high_sym_coord_fraction": float(geom["high_symmetry_coord_fraction"]),
        "pbc_duplicate_count": int(dup["pbc_equivalent_duplicate_site_count"]),
        "ehull": ehull,
        "meta_stable": None if ehull is None else ehull < 0.1,
        "strict_stable": None if ehull is None else ehull < 0.0,
        "novel": novel,
        "unique": unique,
        "novel_unique": novel_unique,
        "refine_success": bool_or_none(metric_record.get("refine_success", metric_record.get("relax_success"))),
    }


def utility_score(features: Mapping[str, Any], weights: VerifierWeights = VerifierWeights()) -> float:
    score = 0.0
    score += weights.comp_valid * float(bool(features.get("comp_valid")))
    score += weights.strict_valid * float(bool(features.get("strict_valid")))
    score += weights.graph_valid * float(bool(features.get("graph_valid")))
    if features.get("refine_success") is not None:
        score += weights.refine_success * float(bool(features.get("refine_success")))
    if features.get("meta_stable") is not None:
        score += weights.meta_stable * float(bool(features.get("meta_stable")))
    if features.get("strict_stable") is not None:
        score += weights.strict_stable * float(bool(features.get("strict_stable")))
    if features.get("novel") is not None:
        score += weights.novel * float(bool(features.get("novel")))
    if features.get("unique") is not None:
        score += weights.unique * float(bool(features.get("unique")))
    score -= weights.single_penalty * float(bool(features.get("single_element")))
    score -= weights.all_metal_penalty * float(bool(features.get("all_metal")))
    score -= weights.high_sym_penalty * float(features.get("high_sym_coord_fraction") or 0.0)
    score -= weights.duplicate_penalty * float(features.get("pbc_duplicate_count") or 0)
    if features.get("plan_match") is False:
        score -= weights.plan_mismatch_penalty
    return float(score)


def rank_feature_rows(rows: Iterable[Mapping[str, Any]], weights: VerifierWeights = VerifierWeights()) -> List[Dict[str, Any]]:
    ranked = []
    for idx, row in enumerate(rows):
        payload = dict(row)
        payload["r5_utility_score"] = utility_score(payload, weights=weights)
        payload["rank_input_idx"] = idx
        ranked.append(payload)
    ranked.sort(key=lambda item: (-float(item["r5_utility_score"]), int(item["rank_input_idx"])))
    for rank, row in enumerate(ranked):
        row["r5_rank"] = rank
    return ranked


__all__ = [
    "VerifierWeights",
    "extract_verifier_features",
    "rank_feature_rows",
    "utility_score",
]
