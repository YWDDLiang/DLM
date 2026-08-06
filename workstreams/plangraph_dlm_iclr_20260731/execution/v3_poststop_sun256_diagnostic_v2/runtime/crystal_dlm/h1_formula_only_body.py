"""H1-B formula-only DLM body executor helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer
from crystal_dlm.r5_dynamic_length import exact_body_token_count, validate_answer_matches_plan
from crystal_dlm.r5_plan_body import H1_RICH_PLAN_FORMAT, composition_plan_from_state, format_composition_plan
from crystal_dlm.r5_plan_state import PLAN_STATE_VERSION


H1_FORMULA_ONLY_BODY_REPRESENTATION = "h1_formula_only_body_v1"
H1_FORMULA_ONLY_BODY_PROMPT_VERSION = "h1_formula_only_body_prompt_v1"
H1G1_ROBUST_BODY_REPRESENTATION = "h1g1_robust_exact_body_v1"
H1G1_ROBUST_BODY_PROMPT_VERSION = "h1g1_condition_view_body_prompt_v1"
H1G1_CONDITION_VIEWS = (
    "full-rich",
    "condition-dropout",
    "formula-volume-sg",
    "formula-volume-only",
)


def build_formula_only_body_prompt(plan_state: Mapping[str, Any]) -> str:
    plan = composition_plan_from_state(plan_state)
    elements = ", ".join(str(symbol) for symbol in plan["elements"])
    counts = ", ".join(str(int(count)) for count in plan["counts"])
    return (
        "Generate only the exact-length dynamic crystal body for this generated formula plan. "
        "The first token must match N and the element multiset must match elements/counts.\n"
        f"formula: {plan['formula']}\n"
        f"elements: {elements}\n"
        f"counts: {counts}\n"
        f"N: {int(plan['N'])}\n"
        "dynamic_crystal_body:"
    )


def _plan_execution_lines(plan_state: Mapping[str, Any]) -> list[str]:
    plan = composition_plan_from_state(plan_state)
    elements = ", ".join(str(symbol) for symbol in plan["elements"])
    counts = ", ".join(str(int(count)) for count in plan["counts"])
    return [
        f"formula: {plan['formula']}",
        f"elements: {elements}",
        f"counts: {counts}",
        f"N: {int(plan['N'])}",
    ]


def build_condition_view_body_prompt(plan_state: Mapping[str, Any], *, condition_view: str = "full-rich") -> str:
    """Build a robust exact-body prompt for one H1-G1 condition view.

    All views execute only the formula-derived N/elements/counts. Rich fields are
    ordinary conditioning text; they never override the composition.
    """

    view = str(condition_view).strip()
    if view not in H1G1_CONDITION_VIEWS:
        raise ValueError(f"unknown H1-G1 condition view {view!r}; expected {H1G1_CONDITION_VIEWS}")
    plan = composition_plan_from_state(plan_state)
    volume = str(plan_state.get("volume_per_atom_bin", "volpa_unknown"))
    spacegroup = str(plan_state.get("spacegroup_bucket", "sg_unknown"))
    anion = str(plan_state.get("anion_framework", "unknown"))
    charge = str(plan_state.get("charge_bucket", "unknown"))
    rich_text = format_composition_plan(plan_state, plan_style=H1_RICH_PLAN_FORMAT)

    lines = [
        "Generate only the exact-length dynamic crystal body for this generated plan view.",
        "The first token must match N and the element multiset must match elements/counts.",
        "Use the conditioning fields as geometry-basin hints, not as composition overrides.",
        f"condition_view: {view}",
    ]
    if view == "full-rich":
        lines.extend(["plan:", rich_text])
    elif view == "condition-dropout":
        lines.extend(
            [
                f"formula: {plan['formula']}",
                f"anion: {anion}",
                f"charge: {charge}",
                "lattice: unknown",
                "spacegroup: unknown",
                f"volume: {volume}",
                "end: plan",
            ]
        )
    elif view == "formula-volume-sg":
        lines.extend(
            [
                f"formula: {plan['formula']}",
                f"volume: {volume}",
                f"spacegroup: {spacegroup}",
            ]
        )
    elif view == "formula-volume-only":
        lines.extend([f"formula: {plan['formula']}", f"volume: {volume}"])
    lines.extend(_plan_execution_lines(plan_state))
    lines.append("dynamic_crystal_body:")
    return "\n".join(lines)


def build_formula_only_body_record(
    *,
    plan_state: Mapping[str, Any],
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    answer_separator: str = "",
    sample_weight: float = 1.0,
    context_prefix: str | None = None,
    task: str = "h1_formula_only_body_replay",
) -> Dict[str, Any]:
    answer, diagnostics = arrays_to_dynamic_answer(
        lengths=arrays["lengths"],
        angles=arrays["angles"],
        species=arrays["species"],
        frac_coords=arrays["frac_coords"],
        separator=answer_separator,
    )
    validate_answer_matches_plan(plan_state, answer)
    body_prompt = build_formula_only_body_prompt(plan_state)
    prompt = body_prompt if context_prefix is None else str(context_prefix).rstrip() + "\n" + body_prompt
    return {
        "task": task,
        "representation": "dynamic_v1",
        "r5_representation": H1_FORMULA_ONLY_BODY_REPRESENTATION,
        "prompt_version": H1_FORMULA_ONLY_BODY_PROMPT_VERSION,
        "plan_state_version": PLAN_STATE_VERSION,
        "prompt": prompt,
        "answer": answer,
        "text": prompt.rstrip() + "\n" + answer,
        "answer_semantic_length": exact_body_token_count(plan_state),
        "answer_token_count": exact_body_token_count(plan_state),
        "num_atoms": int(plan_state["N"]),
        "plan_state": dict(plan_state),
        "r5_plan_state": dict(plan_state),
        "metadata": dict(metadata or {}),
        "sample_weight": float(sample_weight),
        "loss_profile": "fixed_slot",
        "mask_policy": "normal",
        "encode_diagnostics": diagnostics.to_dict(),
    }


def build_condition_view_body_record(
    *,
    plan_state: Mapping[str, Any],
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    answer_separator: str = "",
    sample_weight: float = 1.0,
    condition_view: str = "full-rich",
    context_prefix: str | None = None,
    task: str | None = None,
) -> Dict[str, Any]:
    answer, diagnostics = arrays_to_dynamic_answer(
        lengths=arrays["lengths"],
        angles=arrays["angles"],
        species=arrays["species"],
        frac_coords=arrays["frac_coords"],
        separator=answer_separator,
    )
    validate_answer_matches_plan(plan_state, answer)
    body_prompt = build_condition_view_body_prompt(plan_state, condition_view=condition_view)
    prompt = body_prompt if context_prefix is None else str(context_prefix).rstrip() + "\n" + body_prompt
    return {
        "task": task or f"h1g1_{condition_view.replace('-', '_')}_body_replay",
        "condition_view": str(condition_view),
        "representation": "dynamic_v1",
        "r5_representation": H1G1_ROBUST_BODY_REPRESENTATION,
        "prompt_version": H1G1_ROBUST_BODY_PROMPT_VERSION,
        "plan_state_version": PLAN_STATE_VERSION,
        "prompt": prompt,
        "answer": answer,
        "text": prompt.rstrip() + "\n" + answer,
        "answer_semantic_length": exact_body_token_count(plan_state),
        "answer_token_count": exact_body_token_count(plan_state),
        "num_atoms": int(plan_state["N"]),
        "plan_state": dict(plan_state),
        "r5_plan_state": dict(plan_state),
        "metadata": dict(metadata or {}),
        "sample_weight": float(sample_weight),
        "loss_profile": "fixed_slot",
        "mask_policy": "normal",
        "encode_diagnostics": diagnostics.to_dict(),
    }


__all__ = [
    "H1G1_CONDITION_VIEWS",
    "H1G1_ROBUST_BODY_PROMPT_VERSION",
    "H1G1_ROBUST_BODY_REPRESENTATION",
    "H1_FORMULA_ONLY_BODY_PROMPT_VERSION",
    "H1_FORMULA_ONLY_BODY_REPRESENTATION",
    "build_condition_view_body_prompt",
    "build_condition_view_body_record",
    "build_formula_only_body_prompt",
    "build_formula_only_body_record",
]
