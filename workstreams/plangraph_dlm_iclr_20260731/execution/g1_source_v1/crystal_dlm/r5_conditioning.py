"""R5 prompt-side physical-plan conditioning helpers.

R5 keeps the proven 107-token fixed-slot answer unchanged.  The physical state
``z`` is represented in the prompt, not prepended to the answer.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, Iterable, Mapping, Sequence

from crystal_dlm.composition_validity import (
    element_symbols,
    formula_from_composition,
    reduced_composition,
)
from crystal_dlm.fixed_slot import ANSWER_TOKEN_COUNT, parse_fixed_slot_answer
from crystal_dlm.physical_header import build_physical_header_tokens_from_arrays


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _str_or_unknown(value: Any) -> str:
    if value in (None, ""):
        return "unknown"
    return str(value)


def ehull_tier(value: Any) -> str:
    """Return the R5 stability tier for an MP-20/generation Ehull value."""

    ehull = _float_or_none(value)
    if ehull is None:
        return "unknown_ehull"
    if ehull <= 0.0:
        return "strict_anchor"
    if ehull < 0.1:
        return "meta_anchor"
    return "higher_ehull"


def tier_sample_weight(tier: str) -> float:
    if tier == "strict_anchor":
        return 1.4
    if tier == "meta_anchor":
        return 1.15
    if tier == "higher_ehull":
        return 0.85
    return 1.0


def formula_payload_from_arrays(arrays: Mapping[str, Any]) -> Dict[str, Any]:
    atom_counter: Counter[int] = Counter(int(value) for value in arrays["atom_types"])
    elems = sorted(atom_counter)
    full_counts = [int(atom_counter[elem]) for elem in elems]
    full_formula = formula_from_composition(elems, full_counts)
    reduced_elems, reduced_counts = reduced_composition(arrays["atom_types"])
    reduced_formula = formula_from_composition(reduced_elems, reduced_counts)
    symbols = list(element_symbols(elems))
    special_formula = " + ".join(
        f"<E_{symbol}> x{count}" for symbol, count in zip(symbols, full_counts)
    )
    chemsys = "-".join(sorted(symbols))
    return {
        "full_formula": full_formula,
        "reduced_formula": reduced_formula,
        "special_formula": special_formula,
        "element_equivalence": "; ".join(f"{symbol} == <E_{symbol}>" for symbol in symbols),
        "symbols": symbols,
        "full_counts": full_counts,
        "chemsys": chemsys,
        "num_elements": len(symbols),
        "num_atoms": int(arrays["num_atoms"]),
        "atom_count_token": f"<N_{int(arrays['num_atoms']):03d}>",
    }


def _coarse_volpa_label(volpa_token: str) -> str:
    try:
        value = int(str(volpa_token).split("_", 1)[1].rstrip(">"))
    except Exception:
        return "volpa_unknown"
    low = int(math.floor(value / 5.0) * 5)
    high = low + 4
    return f"volpa_{low:03d}_{high:03d}"


def prototype_key_from_labels(labels: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    """Deterministic v0 prototype key; no clustering or Wyckoff dependence."""

    sg = _str_or_unknown(
        metadata.get("spacegroup.number.conv", metadata.get("spacegroup.number"))
    )
    return "|".join(
        [
            f"anion={labels.get('anion_framework', 'unknown')}",
            f"count={labels.get('count_pattern', 'unknown')}",
            f"lat={labels.get('lattice_system', 'unknown')}",
            f"vol={labels.get('volume_per_atom_coarse', 'volpa_unknown')}",
            f"sg={sg}",
        ]
    )


def build_z_payload_from_arrays(
    arrays: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    formula = formula_payload_from_arrays(arrays)
    _, header_labels = build_physical_header_tokens_from_arrays(arrays)
    comp_record = header_labels.get("composition_record") or {}
    reason = str(comp_record.get("reason") or "unknown")
    volpa_token = str(header_labels.get("volume_per_atom_token") or "<VOLPA_000>")
    tier = ehull_tier(metadata.get("e_above_hull"))
    labels: Dict[str, Any] = {
        **formula,
        "composition_reason": reason,
        "composition_token": header_labels.get("composition_token"),
        "oxidation_token": header_labels.get("oxidation_token"),
        "anion_framework": str(header_labels.get("anion_token") or "<ANION_OTHER>").removeprefix("<").removesuffix(">"),
        "count_pattern": str(header_labels.get("count_pattern_token") or "<COUNT_PATTERN_OTHER>").removeprefix("<").removesuffix(">"),
        "lattice_system": str(header_labels.get("lattice_system_token") or "<LAT_SYS_TRIC>").removeprefix("<").removesuffix(">"),
        "volume_per_atom_token": volpa_token,
        "volume_per_atom_coarse": _coarse_volpa_label(volpa_token),
        "volume_per_atom": header_labels.get("volume_per_atom"),
        "high_symmetry_token": header_labels.get("high_symmetry_token"),
        "high_symmetry_coord_fraction": header_labels.get("high_symmetry_coord_fraction"),
        "spacegroup_number": metadata.get("spacegroup.number"),
        "spacegroup_number_conv": metadata.get("spacegroup.number.conv"),
        "material_id": metadata.get("material_id"),
        "pretty_formula": metadata.get("pretty_formula"),
        "e_above_hull": metadata.get("e_above_hull"),
        "ehull_tier": tier,
    }
    labels["prototype_key"] = prototype_key_from_labels(labels, metadata)
    return labels


def build_z_payload_from_answer(
    answer: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> Dict[str, Any]:
    arrays = parse_fixed_slot_answer(answer, strict=strict)
    payload = build_z_payload_from_arrays(arrays, metadata=metadata)
    if int(payload["num_atoms"]) != int(arrays["num_atoms"]):
        raise ValueError("z atom count does not match fixed-slot body")
    return payload


def build_r5_prompt(z: Mapping[str, Any]) -> str:
    """Build the prompt-side v0 physical-plan condition."""

    return (
        "Generate only the 107 fixed-slot crystal body tokens. Do not output a physical header.\n"
        "The physical plan z below is persistent conditioning context; the body must match it.\n"
        f"formula: {z['full_formula']}\n"
        f"reduced_formula: {z['reduced_formula']}\n"
        f"special_formula: {z['special_formula']}\n"
        f"element_equivalence: {z['element_equivalence']}\n"
        f"atom_count_token: {z['atom_count_token']}\n"
        f"prototype_key: {z['prototype_key']}\n"
        f"anion_framework: {z['anion_framework']}\n"
        f"composition_reason: {z['composition_reason']}\n"
        f"oxidation_token: {z['oxidation_token']}\n"
        f"count_pattern: {z['count_pattern']}\n"
        f"lattice_system_target: {z['lattice_system']}\n"
        f"volume_per_atom_bin: {z['volume_per_atom_token']}\n"
        f"high_symmetry_source_bucket: {z['high_symmetry_token']}\n"
        f"spacegroup_number: {_str_or_unknown(z.get('spacegroup_number_conv') or z.get('spacegroup_number'))}\n"
        f"ehull_tier: {z['ehull_tier']}\n"
        "Keep the atom multiset exactly equal to formula and special_formula. "
        "Keep lattice volume non-zero and avoid exact or PBC-equivalent duplicate coordinates.\n"
        "fixed_slot_crystal_body:"
    )


def answer_has_only_fixed_slot_body(answer: str) -> bool:
    arrays = parse_fixed_slot_answer(answer, strict=True)
    return len(arrays["tokens"]) == ANSWER_TOKEN_COUNT


def formula_signature(z: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            (str(symbol), int(count))
            for symbol, count in zip(z.get("symbols", []), z.get("full_counts", []))
        )
    )


def arrays_formula_signature(arrays: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(str(symbol) for symbol in arrays["species"]).items()))


def validate_z_matches_answer(z: Mapping[str, Any], answer: str) -> None:
    arrays = parse_fixed_slot_answer(answer, strict=True)
    if int(z["num_atoms"]) != int(arrays["num_atoms"]):
        raise ValueError("z atom count and answer atom count differ")
    if formula_signature(z) != arrays_formula_signature(arrays):
        raise ValueError("z formula and answer atom multiset differ")


def formula_cap_weight(base_weight: float, formula_count: int, cap: int) -> float:
    if cap <= 0 or formula_count <= cap:
        return float(base_weight)
    return float(base_weight) / math.sqrt(float(formula_count) / float(cap))


def bucket_counts(rows: Iterable[Mapping[str, Any]], key: str) -> Dict[str, int]:
    counter = Counter(str(row.get(key, "unknown")) for row in rows)
    return dict(counter.most_common())

