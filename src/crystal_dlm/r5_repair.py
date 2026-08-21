"""R5 failure-conditioned repair helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from crystal_dlm.diagnostic_remask import geometry_degeneracy_record
from crystal_dlm.r5_plan_state import plan_state_to_json


KNOWN_VIOLATION_LABELS = {
    "high_sym",
    "a_eq_b_eq_c",
    "all_90_angles",
    "pbc_duplicate",
    "single_element",
    "all_metal",
    "charge_fail",
    "pauling_fail",
    "refinement_fail",
    "sun_negative",
    "parse_fail",
    "graph_fail",
}

GEOMETRY_LABELS = {"high_sym", "a_eq_b_eq_c", "all_90_angles", "pbc_duplicate", "refinement_fail"}
COMPOSITION_LABELS = {"single_element", "all_metal", "charge_fail", "pauling_fail"}


def normalize_violation_labels(labels: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    for label in labels:
        text = str(label).strip().lower().replace("-", "_")
        aliases = {
            "high_symmetry": "high_sym",
            "all_lengths_equal": "a_eq_b_eq_c",
            "all_angles_90": "all_90_angles",
            "duplicate": "pbc_duplicate",
            "pbc_equivalent_duplicate": "pbc_duplicate",
            "not_sun": "sun_negative",
        }
        text = aliases.get(text, text)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def labels_from_arrays(
    arrays: Mapping[str, Any],
    *,
    max_high_symmetry_coord_fraction: float = 0.55,
) -> List[str]:
    record = geometry_degeneracy_record(arrays)
    labels: List[str] = []
    if float(record["high_symmetry_coord_fraction"]) > float(max_high_symmetry_coord_fraction):
        labels.append("high_sym")
    if bool(record["all_lengths_equal"]):
        labels.append("a_eq_b_eq_c")
    if bool(record["all_angles_90"]):
        labels.append("all_90_angles")
    if int(record["pbc_equivalent_duplicate_site_count"]) > 0:
        labels.append("pbc_duplicate")
    species = [str(symbol) for symbol in arrays.get("species", [])]
    if len(set(species)) == 1 and species:
        labels.append("single_element")
    return labels


def choose_masked_block(labels: Sequence[str]) -> str:
    label_set = set(normalize_violation_labels(labels))
    if label_set & COMPOSITION_LABELS:
        return "composition"
    if label_set & GEOMETRY_LABELS:
        if "a_eq_b_eq_c" in label_set or "all_90_angles" in label_set:
            return "lattice+coords"
        return "coords"
    if "parse_fail" in label_set or "graph_fail" in label_set:
        return "full_body"
    return "lattice+coords"


def build_repair_prompt(
    *,
    plan_state: Mapping[str, Any],
    visible_proposal: str,
    violation_labels: Sequence[str],
    masked_block: str | None = None,
) -> str:
    labels = normalize_violation_labels(violation_labels)
    block = masked_block or choose_masked_block(labels)
    return (
        "Repair the visible crystal proposal under the fixed plan_state. "
        "Keep valid unaffected blocks unchanged and rewrite only the requested block. "
        "Return only the corrected block text.\n"
        f"plan_state: {plan_state_to_json(plan_state)}\n"
        f"violation_labels: {json.dumps(labels, sort_keys=True)}\n"
        f"masked_block: {block}\n"
        f"visible_proposal: {visible_proposal}\n"
        "corrected_block:"
    )


def make_repair_record(
    *,
    plan_state: Mapping[str, Any],
    visible_proposal: str,
    target: str,
    violation_labels: Sequence[str],
    masked_block: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    sample_weight: float = 1.0,
) -> Dict[str, Any]:
    labels = normalize_violation_labels(violation_labels)
    block = masked_block or choose_masked_block(labels)
    prompt = build_repair_prompt(
        plan_state=plan_state,
        visible_proposal=visible_proposal,
        violation_labels=labels,
        masked_block=block,
    )
    return {
        "task": "r5_corrective_repair",
        "representation": "r5_repair_text",
        "prompt": prompt,
        "answer": str(target),
        "text": prompt.rstrip() + "\n" + str(target),
        "plan_state": dict(plan_state),
        "violation_labels": labels,
        "masked_block": block,
        "visible_proposal": str(visible_proposal),
        "metadata": dict(metadata or {}),
        "loss_profile": "text",
        "sample_weight": float(sample_weight),
    }


__all__ = [
    "KNOWN_VIOLATION_LABELS",
    "build_repair_prompt",
    "choose_masked_block",
    "labels_from_arrays",
    "make_repair_record",
    "normalize_violation_labels",
]
