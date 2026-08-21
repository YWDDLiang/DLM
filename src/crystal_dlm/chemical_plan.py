"""Chemical-plan helpers for plan-conditioned fixed-slot generation."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

from crystal_dlm.composition_validity import formula_from_composition
from crystal_dlm.fixed_slot import Z_TO_SYMBOL


COMMON_PLAN_STATES = {
    "Li": [1],
    "Na": [1],
    "K": [1],
    "Rb": [1],
    "Cs": [1],
    "Mg": [2],
    "Ca": [2],
    "Sr": [2],
    "Ba": [2],
    "Al": [3],
    "Ga": [3],
    "In": [3],
    "Ti": [4, 3],
    "V": [5, 4, 3],
    "Mn": [2, 3, 4],
    "Fe": [2, 3],
    "Co": [2, 3],
    "Ni": [2, 3],
    "Cu": [1, 2],
    "Zn": [2],
    "W": [6, 4],
    "O": [-2],
    "F": [-1],
    "Cl": [-1],
    "Br": [-1],
    "S": [-2, 6],
    "Se": [-2],
    "N": [-3, 5],
    "P": [5, -3],
}


def classify_composition_for_plan(elems: Sequence[int], counts: Sequence[int]) -> dict[str, Any]:
    """Classify composition for a plan, with a no-SMACT local fallback."""

    try:
        from crystal_dlm.composition_validity import classify_smact_validity

        return classify_smact_validity(elems, counts)
    except Exception:  # noqa: BLE001 - local planning mirrors may lack numpy/smact.
        symbols = [Z_TO_SYMBOL[int(elem)] for elem in elems]
        if len(set(symbols)) == 1:
            return {"valid": True, "reason": "single_element_shortcut"}
        all_metalish = all(symbol not in {"O", "F", "Cl", "Br", "S", "Se", "N", "P"} for symbol in symbols)
        if all_metalish:
            return {"valid": True, "reason": "all_metal_shortcut"}
        state_options = [COMMON_PLAN_STATES.get(symbol, []) for symbol in symbols]
        if any(not options for options in state_options):
            return {"valid": False, "reason": "oxidation_state_missing"}
        import itertools

        for states in itertools.product(*state_options):
            if sum(int(state) * int(count) for state, count in zip(states, counts)) == 0:
                return {
                    "valid": True,
                    "reason": "charge_neutral_pauling_valid",
                    "oxidation_states": tuple(int(state) for state in states),
                }
        return {"valid": False, "reason": "charge_neutrality_fail"}


PLAN_PROMPT = (
    "Generate a concise chemistry plan for one new MP-20-like inorganic crystal. "
    "Use ordinary chemical language so element valence and charge balance are explicit. "
    "Prefer charge-balanced low-energy oxides, halides, chalcogenides, pnictides, or intermetallics. "
    "Avoid single-element shortcuts and avoid unsupported actinide-heavy formulas unless necessary. "
    "End the plan with the exact line: crystal tokens:"
)


STRUCTURE_PROMPT_PREFIX = (
    "Below is a chemistry plan for one bulk material. Generate only the fixed-slot "
    "crystal tokens that implement this plan. Preserve charge-balanced element "
    "counts when possible, use non-degenerate lattice parameters, avoid duplicate "
    "or PBC-equivalent coordinates, and do not force cubic/high-symmetry geometry "
    "unless the chemistry requires it."
)


def _format_signed_state(state: int) -> str:
    if int(state) > 0:
        return f"+{int(state)}"
    return str(int(state))


def _counts_from_symbols(symbols: Sequence[str]) -> tuple[list[int], list[int], list[str]]:
    symbol_to_z = {symbol: z for z, symbol in Z_TO_SYMBOL.items()}
    counter = Counter(str(symbol) for symbol in symbols)
    elems = sorted(symbol_to_z[symbol] for symbol in counter if symbol in symbol_to_z)
    counts = [int(counter[Z_TO_SYMBOL[elem]]) for elem in elems]
    symbols_sorted = [Z_TO_SYMBOL[elem] for elem in elems]
    return elems, counts, symbols_sorted


def _reduced_counts(counts: Sequence[int]) -> list[int]:
    gcd_value = 0
    for count in counts:
        gcd_value = math.gcd(gcd_value, int(count))
    gcd_value = max(1, gcd_value)
    return [int(count) // gcd_value for count in counts]


def ehull_tier(metadata: Mapping[str, Any] | None) -> str:
    metadata = metadata or {}
    raw_value = (
        metadata.get("e_above_hull")
        if metadata.get("e_above_hull") not in (None, "")
        else metadata.get("ehull")
    )
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 0.0:
        return "stable_ehull_0"
    if value <= 0.01:
        return "near_stable_ehull_le_0.01"
    if value <= 0.1:
        return "metastable_ehull_le_0.1"
    return "higher_ehull"


def chemical_plan_from_symbols(
    symbols: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_reason: str | None = None,
) -> dict[str, Any]:
    elems, counts, symbols_sorted = _counts_from_symbols(symbols)
    if not elems:
        return {
            "formula": "unknown",
            "reason": "empty",
            "plan": (
                "formula: unknown\n"
                "composition: unknown\n"
                "chemistry: choose a charge-balanced inorganic composition.\n"
                "stability_hint: unknown\n"
                "geometry_hint: use a valid non-degenerate lattice and unique fractional coordinates.\n"
                "crystal tokens:"
            ),
        }
    reduced = _reduced_counts(counts)
    formula = formula_from_composition(elems, reduced)
    classification = classify_composition_for_plan(elems, reduced)
    reason = str(composition_reason or classification.get("reason") or "unknown")
    oxidation_states = classification.get("oxidation_states")
    parts = [
        f"{symbol}{'' if count == 1 else count}"
        for symbol, count in zip(symbols_sorted, reduced)
    ]
    if oxidation_states:
        valence_text = ", ".join(
            f"{symbol}{_format_signed_state(int(state))}"
            for symbol, state in zip(symbols_sorted, oxidation_states)
        )
        chemistry = f"charge-balanced composition with likely oxidation states {valence_text}."
    elif reason == "all_metal_shortcut":
        chemistry = "all-metal intermetallic/alloy composition; prioritize plausible metallic bonding and avoid collapsing to one element."
    elif reason == "single_element_shortcut":
        chemistry = "single-element composition; this is a weak shortcut and should be avoided for discovery unless deliberately requested."
    elif reason == "charge_neutrality_fail":
        chemistry = "SMACT charge-neutrality fails; prefer nearby charge-balanced ratios before using this pattern."
    elif reason == "pauling_fail_or_ratio_rejected":
        chemistry = "charge ratios are difficult or Pauling electronegativity rejects the combination; prefer compatible cation-anion ordering."
    elif reason == "oxidation_state_missing":
        chemistry = "one or more elements lack reliable oxidation states; avoid unsupported chemistry when possible."
    else:
        chemistry = "inorganic composition; prefer charge balance and chemically compatible element roles."
    plan = (
        f"formula: {formula}\n"
        f"composition: {' '.join(parts)}\n"
        f"composition_reason: {reason}\n"
        f"chemistry: {chemistry}\n"
        f"stability_hint: {ehull_tier(metadata)}\n"
        "geometry_hint: use a non-degenerate lattice, avoid exact/PBC duplicate coordinates, "
        "and avoid overusing a=b=c or all-90-degree cells unless appropriate.\n"
        "crystal tokens:"
    )
    return {
        "formula": formula,
        "reason": reason,
        "plan": plan,
        "symbols": symbols_sorted,
        "counts": reduced,
    }


def chemical_plan_from_fixed_arrays(
    arrays: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    composition_reason: str | None = None,
) -> dict[str, Any]:
    return chemical_plan_from_symbols(
        list(arrays.get("species") or []),
        metadata=metadata,
        composition_reason=composition_reason,
    )


def build_plan_conditioned_prompt(plan: str) -> str:
    plan = str(plan).strip()
    if "crystal tokens:" not in plan:
        plan = plan.rstrip() + "\ncrystal tokens:"
    return f"{STRUCTURE_PROMPT_PREFIX}\n\n{plan}"


def truncate_generated_plan(text: str) -> str:
    text = str(text)
    marker = "crystal tokens:"
    lower = text.lower()
    idx = lower.find(marker)
    if idx >= 0:
        return text[: idx + len(marker)].strip()
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "formula: unknown\nchemistry: choose a charge-balanced inorganic composition.\ncrystal tokens:"
    return "\n".join(lines[:8]).strip() + "\ncrystal tokens:"
