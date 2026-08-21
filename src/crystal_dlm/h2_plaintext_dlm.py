"""H2 LLM-plan conditioned plain-text DLM proposal helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping

from crystal_dlm.crysllmgen_text import CRYSLLMGEN_TEXT_PROMPT
from crystal_dlm.r5_plan_body import H1_RICH_PLAN_FORMAT, composition_plan_from_state, format_composition_plan


H2_PLAINTEXT_DLM_REPRESENTATION = "h2_plaintext_dlm_proposal_v1"
H2_PLAINTEXT_DLM_PROMPT_VERSION = "crysllmgen_text_with_h1_rich_plan_format_locked_v3"


def build_h2_plaintext_prompt(plan_state: Mapping[str, Any]) -> str:
    """Build the H2 DLM prompt from a generated or teacher rich plan.

    The prompt includes only the ordinary-text plan and asks the DLM to produce
    the raw CrysLLMGen text structure.  Coordinates/lattice values are never
    copied into the prompt.
    """

    plan_text = format_composition_plan(plan_state, plan_style=H1_RICH_PLAN_FORMAT)
    return (
        f"{CRYSLLMGEN_TEXT_PROMPT.rstrip()}\n\n"
        "plan:\n"
        f"{plan_text}\n"
        "structure:\n"
        "Return only compact CrysLLMGen plain text.\n"
        "The first generated line must be three lattice lengths as numbers separated by spaces.\n"
        "The second generated line must be three lattice angles as numbers separated by spaces.\n"
        "Use exactly the formula counts in the plan.\n"
        "For each atom, write two lines: first one element symbol only, then three fractional-coordinate numbers only.\n"
        "Never put an element symbol and coordinates on the same line.\n"
        "Do not write labels, prose, markdown, units, YAML, bullet lists, or end markers.\n"
    )


def composition_signature_from_plan(plan_state: Mapping[str, Any]) -> Dict[str, int]:
    plan = composition_plan_from_state(plan_state)
    return {str(symbol): int(count) for symbol, count in zip(plan["elements"], plan["counts"])}


def composition_signature_from_arrays(arrays: Mapping[str, Any]) -> Dict[str, int]:
    counts = Counter(str(symbol) for symbol in (arrays.get("species") or []))
    return dict(sorted(counts.items()))


def proposal_matches_plan_composition(arrays: Mapping[str, Any], plan_state: Mapping[str, Any]) -> bool:
    return composition_signature_from_arrays(arrays) == composition_signature_from_plan(plan_state)


__all__ = [
    "H2_PLAINTEXT_DLM_PROMPT_VERSION",
    "H2_PLAINTEXT_DLM_REPRESENTATION",
    "build_h2_plaintext_prompt",
    "composition_signature_from_arrays",
    "composition_signature_from_plan",
    "proposal_matches_plan_composition",
]
