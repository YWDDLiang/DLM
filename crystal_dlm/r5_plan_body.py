"""R5-C de novo composition-plan-to-body records.

The de novo variant keeps the user-facing prompt in the CrysLLMGen style, asks
the model to generate an ordinary text composition plan, then uses that generated
plan as the only source of atom count/composition for the exact-length body.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from crystal_dlm.composition_validity import classify_smact_validity
from crystal_dlm.crysllmgen_text import CRYSLLMGEN_TEXT_PROMPT, CRYSLLMGEN_TEXT_PROMPT_VERSION
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.r5_dynamic_length import exact_body_token_count, validate_answer_matches_plan
from crystal_dlm.r5_plan_state import (
    ALLOWED_LATTICE_SYSTEMS,
    ALLOWED_SPACEGROUP_BUCKETS,
    CHARGE_BUCKET_TO_CODE,
    PLAN_STATE_VERSION,
    anion_framework_from_symbols,
    charge_bucket_from_classification,
    prototype_key,
    validate_plan_state,
)


R5C_FORMULA_TEXT_PLAN_BODY_REPRESENTATION = "r5c_formula_text_plan_body_v2"
R5C_FORMULA_END_PLAN_BODY_REPRESENTATION = "r5c_formula_end_plan_body_v1"
R5C_SEMANTIC_FORMULA_PLAN_BODY_REPRESENTATION = "r5c_semantic_formula_plan_body_v1"
H1_RICH_PLAN_BODY_REPRESENTATION = "h1_rich_plan_body_v1"
H1_RICH_NOCHARGE_PLAN_BODY_REPRESENTATION = "h1_rich_nocharge_plan_body_v1"
R5C_PLAN_BODY_REPRESENTATION = R5C_FORMULA_TEXT_PLAN_BODY_REPRESENTATION
R5C_PLAN_BODY_PROMPT_VERSION = f"{CRYSLLMGEN_TEXT_PROMPT_VERSION}_r5c_plan_body"
R5C_PLAN_FORMAT = "formula_text"
R5C_FORMULA_END_PLAN_FORMAT = "formula_end_v1"
R5C_SEMANTIC_PLAN_FORMAT = "semantic_formula_v1"
H1_RICH_PLAN_FORMAT = "h1_rich_plan_v1"
H1_RICH_NOCHARGE_PLAN_FORMAT = "h1_rich_nocharge_plan_v1"
R5C_PLAN_STYLES = (
    R5C_PLAN_FORMAT,
    R5C_FORMULA_END_PLAN_FORMAT,
    R5C_SEMANTIC_PLAN_FORMAT,
    H1_RICH_PLAN_FORMAT,
    H1_RICH_NOCHARGE_PLAN_FORMAT,
)
R5C_PLAN_BODY_PLAN_LABEL = "plan:"
R5C_PLAN_BODY_BODY_LABEL = "body:"
R5C_PLAN_FIELDS = ("formula",)
H1_RICH_PLAN_FIELDS = ("formula", "anion", "charge", "lattice", "spacegroup", "volume")
H1_RICH_NOCHARGE_PLAN_FIELDS = ("formula", "anion", "lattice", "spacegroup", "volume")
R5C_PLAN_END_FIELD = "end"
R5C_PLAN_END_VALUE = "plan"
H1_RICH_ANION_FRAMEWORKS = {
    "oxide",
    "sulfide",
    "chalcogenide",
    "halide",
    "nitride",
    "phosphide_or_phosphate",
    "other",
}

METAL_SYMBOLS = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
}
HALOGEN_SYMBOLS = {"F", "Cl", "Br", "I"}
CHALCOGEN_SYMBOLS = {"S", "Se", "Te"}
PNICTIDE_SYMBOLS = {"N", "P", "As", "Sb", "Bi"}
CARBIDE_BORIDE_SYMBOLS = {"B", "C", "Si", "Ge"}


def normalize_plan_style(plan_style: str | None = None) -> str:
    style = R5C_PLAN_FORMAT if plan_style is None else str(plan_style).strip()
    if style not in R5C_PLAN_STYLES:
        raise ValueError(f"unknown R5-C plan style {style!r}; expected one of {R5C_PLAN_STYLES}")
    return style


def representation_for_plan_style(plan_style: str | None = None) -> str:
    style = normalize_plan_style(plan_style)
    if style == R5C_FORMULA_END_PLAN_FORMAT:
        return R5C_FORMULA_END_PLAN_BODY_REPRESENTATION
    if style == R5C_SEMANTIC_PLAN_FORMAT:
        return R5C_SEMANTIC_FORMULA_PLAN_BODY_REPRESENTATION
    if style == H1_RICH_PLAN_FORMAT:
        return H1_RICH_PLAN_BODY_REPRESENTATION
    if style == H1_RICH_NOCHARGE_PLAN_FORMAT:
        return H1_RICH_NOCHARGE_PLAN_BODY_REPRESENTATION
    return R5C_FORMULA_TEXT_PLAN_BODY_REPRESENTATION


def _canonical_symbol_counts(symbols: Sequence[str], counts: Sequence[int]) -> tuple[list[str], list[int]]:
    counter: Counter[str] = Counter()
    for symbol, count in zip(symbols, counts):
        symbol = str(symbol).strip()
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported element symbol {symbol!r}")
        count_value = int(count)
        if count_value <= 0:
            raise ValueError(f"element count for {symbol} must be positive, got {count_value}")
        counter[symbol] += count_value
    ordered = sorted(counter, key=lambda item: SYMBOL_TO_Z[item])
    return ordered, [int(counter[symbol]) for symbol in ordered]


def formula_from_symbol_counts(symbols: Sequence[str], counts: Sequence[int]) -> str:
    parts: list[str] = []
    for symbol, count in zip(symbols, counts):
        count = int(count)
        parts.append(str(symbol) if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def symbol_counts_from_formula(formula: str) -> tuple[list[str], list[int]]:
    """Parse a flat integer-count formula and canonicalize it by atomic number."""

    compact = re.sub(r"\s+", "", str(formula))
    if not compact:
        raise ValueError("composition plan formula is empty")
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", compact)
    if not tokens:
        raise ValueError(f"composition plan formula {compact!r} contains no element symbols")
    reconstructed = "".join(symbol + count for symbol, count in tokens)
    if reconstructed != compact:
        raise ValueError(f"composition plan formula {compact!r} is not a flat integer-count formula")
    symbols: list[str] = []
    counts: list[int] = []
    for symbol, count_text in tokens:
        count = int(count_text) if count_text else 1
        symbols.append(symbol)
        counts.append(count)
    return _canonical_symbol_counts(symbols, counts)


def composition_plan_from_state(plan_state: Mapping[str, Any]) -> Dict[str, Any]:
    symbols = [str(symbol) for symbol in (plan_state.get("elements") or [])]
    counts = [int(value) for value in (plan_state.get("counts") or [])]
    symbols, counts = _canonical_symbol_counts(symbols, counts)
    num_atoms = int(sum(counts))
    formula = formula_from_symbol_counts(symbols, counts)
    if int(plan_state.get("N", num_atoms)) != num_atoms:
        raise ValueError(f"plan_state N {plan_state.get('N')} does not match counts sum {num_atoms}")
    return {
        "formula": formula,
        "elements": symbols,
        "counts": counts,
        "N": num_atoms,
    }


def arity_label(num_elements: int) -> str:
    if num_elements <= 1:
        return "unary"
    if num_elements == 2:
        return "binary"
    if num_elements == 3:
        return "ternary"
    if num_elements == 4:
        return "quaternary"
    return "multi"


def size_label(num_atoms: int) -> str:
    atoms = int(num_atoms)
    if atoms <= 3:
        return "tiny"
    if atoms <= 6:
        return "small"
    if atoms <= 10:
        return "medium"
    if atoms <= 16:
        return "large"
    return "xlarge"


def family_label(elements: Sequence[str]) -> str:
    symbols = {str(symbol) for symbol in elements}
    if not symbols:
        return "other"
    if all(symbol in METAL_SYMBOLS for symbol in symbols):
        return "intermetallic"

    has_oxygen = "O" in symbols
    has_halogen = bool(symbols & HALOGEN_SYMBOLS)
    has_chalcogen = bool(symbols & CHALCOGEN_SYMBOLS)
    has_pnictide = bool(symbols & PNICTIDE_SYMBOLS)
    has_carbide_boride = bool(symbols & CARBIDE_BORIDE_SYMBOLS)
    group_count = sum(
        bool(value)
        for value in (
            has_oxygen,
            has_halogen,
            has_chalcogen,
            has_pnictide,
            has_carbide_boride,
        )
    )
    if has_oxygen and group_count == 1:
        return "oxide"
    if has_oxygen and has_halogen and group_count == 2:
        return "oxyhalide"
    if has_oxygen and has_chalcogen and group_count == 2:
        return "oxychalcogenide"
    if has_oxygen and group_count > 1:
        return "mixed_anion"
    if has_halogen and group_count == 1:
        return "halide"
    if has_chalcogen and group_count == 1:
        return "chalcogenide"
    if has_pnictide and group_count == 1:
        return "pnictide"
    if has_carbide_boride and group_count == 1:
        return "carbide_boride"
    if group_count > 1:
        return "mixed_anion"
    return "other"


def semantic_fields_from_plan(plan_state: Mapping[str, Any]) -> Dict[str, str]:
    plan = composition_plan_from_state(plan_state)
    return {
        "family": family_label(plan["elements"]),
        "arity": arity_label(len(plan["elements"])),
        "size": size_label(int(plan["N"])),
    }


def semantic_consistency_from_plan(plan_state: Mapping[str, Any]) -> Dict[str, bool | None]:
    expected = semantic_fields_from_plan(plan_state)
    result: Dict[str, bool | None] = {}
    for key in ("family", "arity", "size"):
        generated = plan_state.get(f"generated_{key}")
        result[f"{key}_match_formula"] = None if generated is None else str(generated) == expected[key]
    return result


def _normalize_rich_anion(value: Any) -> str:
    normalized = _normalize_generated_label(str(value))
    if normalized not in H1_RICH_ANION_FRAMEWORKS:
        raise ValueError(f"invalid rich-plan anion field {value!r}")
    return normalized


def _normalize_rich_charge(value: Any) -> str:
    normalized = _normalize_generated_label(str(value))
    if normalized not in CHARGE_BUCKET_TO_CODE:
        raise ValueError(f"invalid rich-plan charge field {value!r}")
    return normalized


def _normalize_rich_lattice(value: Any) -> str:
    normalized = _normalize_generated_label(str(value))
    if normalized not in ALLOWED_LATTICE_SYSTEMS:
        raise ValueError(f"invalid rich-plan lattice field {value!r}")
    return normalized


def _normalize_rich_spacegroup(value: Any) -> str:
    normalized = _normalize_generated_label(str(value))
    if normalized not in ALLOWED_SPACEGROUP_BUCKETS:
        raise ValueError(f"invalid rich-plan spacegroup field {value!r}")
    return normalized


def _normalize_rich_volume(value: Any) -> str:
    normalized = _normalize_generated_label(str(value))
    if not re.fullmatch(r"volpa_\d{3}_\d{3}", normalized):
        raise ValueError(f"invalid rich-plan volume field {value!r}")
    return normalized


def rich_fields_from_plan_state(plan_state: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "anion": _normalize_rich_anion(plan_state.get("anion_framework", "other")),
        "charge": _normalize_rich_charge(plan_state.get("charge_bucket", "validator_unavailable")),
        "lattice": _normalize_rich_lattice(plan_state.get("lattice_system", "triclinic")),
        "spacegroup": _normalize_rich_spacegroup(plan_state.get("spacegroup_bucket", "sg_001_002")),
        "volume": _normalize_rich_volume(plan_state.get("volume_per_atom_bin", "volpa_000_004")),
    }


def nocharge_rich_fields_from_plan_state(plan_state: Mapping[str, Any]) -> Dict[str, str]:
    """Return only fields emitted by the no-charge Planner schema."""

    return {
        "anion": _normalize_rich_anion(plan_state.get("anion_framework", "other")),
        "lattice": _normalize_rich_lattice(plan_state.get("lattice_system", "triclinic")),
        "spacegroup": _normalize_rich_spacegroup(plan_state.get("spacegroup_bucket", "sg_001_002")),
        "volume": _normalize_rich_volume(plan_state.get("volume_per_atom_bin", "volpa_000_004")),
    }


def _derived_formula_classification(symbols: Sequence[str], counts: Sequence[int]) -> Dict[str, Any]:
    """Classify a generated formula without changing it or its raw denominator."""

    elems = tuple(SYMBOL_TO_Z[symbol] for symbol in symbols if symbol in SYMBOL_TO_Z)
    if len(elems) != len(symbols):
        return {"valid": False, "reason": "unsupported_element"}
    try:
        return classify_smact_validity(elems, tuple(int(value) for value in counts))
    except Exception as exc:  # noqa: BLE001 - evaluator availability is reported in the plan_state.
        return {
            "valid": None,
            "reason": "validator_unavailable",
            "validator_error": type(exc).__name__,
        }


def format_composition_plan(plan_state: Mapping[str, Any], *, plan_style: str | None = None) -> str:
    plan = composition_plan_from_state(plan_state)
    style = normalize_plan_style(plan_style)
    if style == R5C_FORMULA_END_PLAN_FORMAT:
        return f"formula: {plan['formula']}\n{R5C_PLAN_END_FIELD}: {R5C_PLAN_END_VALUE}"
    if style == R5C_SEMANTIC_PLAN_FORMAT:
        semantic = semantic_fields_from_plan(plan)
        return "\n".join(
            [
                f"family: {semantic['family']}",
                f"arity: {semantic['arity']}",
                f"size: {semantic['size']}",
                f"formula: {plan['formula']}",
            ]
        )
    if style == H1_RICH_PLAN_FORMAT:
        rich = rich_fields_from_plan_state(plan_state)
        return "\n".join(
            [
                f"formula: {plan['formula']}",
                f"anion: {rich['anion']}",
                f"charge: {rich['charge']}",
                f"lattice: {rich['lattice']}",
                f"spacegroup: {rich['spacegroup']}",
                f"volume: {rich['volume']}",
                f"{R5C_PLAN_END_FIELD}: {R5C_PLAN_END_VALUE}",
            ]
        )
    if style == H1_RICH_NOCHARGE_PLAN_FORMAT:
        rich = nocharge_rich_fields_from_plan_state(plan_state)
        return "\n".join(
            [
                f"formula: {plan['formula']}",
                f"anion: {rich['anion']}",
                f"lattice: {rich['lattice']}",
                f"spacegroup: {rich['spacegroup']}",
                f"volume: {rich['volume']}",
                f"{R5C_PLAN_END_FIELD}: {R5C_PLAN_END_VALUE}",
            ]
        )
    return f"formula: {plan['formula']}"


def _strip_special_tail(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
    for marker in ("<|endoftext|>", "</s>", "<s>"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned


def _normalize_generated_label(value: str) -> str:
    cleaned = str(value).split("<", 1)[0].strip().lower()
    match = re.match(r"([a-z][a-z0-9_+-]*)", cleaned)
    return match.group(1).replace("-", "_") if match else cleaned


def has_plan_end_marker(text: str) -> bool:
    cleaned = _strip_special_tail(text)
    return re.search(rf"(?im)^\s*{R5C_PLAN_END_FIELD}\s*:\s*{R5C_PLAN_END_VALUE}\s*$", cleaned) is not None


def has_plan_tail_after_end_marker(text: str) -> bool:
    cleaned = _strip_special_tail(text)
    match = re.search(rf"(?im)^\s*{R5C_PLAN_END_FIELD}\s*:\s*{R5C_PLAN_END_VALUE}\s*$", cleaned)
    if match is None:
        return False
    tail = cleaned[match.end() :].strip()
    if not tail:
        return False
    return bool(re.search(r"(?im)^\s*body\s*:\s*$|<N_\d{3}>|<LA_\d{3}>|<E_[A-Z][a-z]?>", tail))


def parse_composition_plan(
    text: str,
    *,
    max_atoms: int = 20,
    plan_style: str | None = None,
) -> Dict[str, Any]:
    """Parse a text plan into a minimal plan_state dict.

    ``counts`` and ``N`` are intentionally derived by Python from ``formula``.
    This avoids making the model emit redundant arithmetic fields that can
    contradict each other during de novo sampling. DN4 semantic fields are
    diagnostics only; they never override the formula-derived composition. H1
    rich fields are generated conditioning fields; they condition the body
    executor, but never override formula-derived composition.
    """

    cleaned = _strip_special_tail(text)
    requested_style = normalize_plan_style(plan_style) if plan_style is not None else None
    fields: Dict[str, str] = {}
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        if re.fullmatch(r"(?i)plan\s*:", line):
            continue
        if re.fullmatch(r"(?i)body\s*:", line):
            break
        match = re.match(
            r"(?i)^(family|arity|size|formula|elements|counts|n|anion|charge|lattice|spacegroup|volume|end)\s*:\s*(.+?)\s*$",
            line,
        )
        if match:
            key = match.group(1).lower()
            fields[key] = match.group(2).strip()
            if key == R5C_PLAN_END_FIELD and _normalize_generated_label(fields[key]) == R5C_PLAN_END_VALUE:
                break
            continue
        if fields and not set(R5C_PLAN_FIELDS).issubset(fields):
            # Ignore natural-language crumbs before/after the fixed fields.
            continue
    missing = [field for field in R5C_PLAN_FIELDS if field.lower() not in fields]
    if missing:
        raise ValueError(f"composition plan missing fields: {','.join(missing)}")
    if requested_style == R5C_FORMULA_END_PLAN_FORMAT:
        marker_value = _normalize_generated_label(fields.get(R5C_PLAN_END_FIELD, ""))
        if marker_value != R5C_PLAN_END_VALUE:
            raise ValueError("composition plan missing required end: plan marker")
    if requested_style == H1_RICH_PLAN_FORMAT:
        missing_rich = [field for field in H1_RICH_PLAN_FIELDS if field not in fields]
        if missing_rich:
            raise ValueError(f"rich composition plan missing fields: {','.join(missing_rich)}")
        marker_value = _normalize_generated_label(fields.get(R5C_PLAN_END_FIELD, ""))
        if marker_value != R5C_PLAN_END_VALUE:
            raise ValueError("rich composition plan missing required end: plan marker")
    if requested_style == H1_RICH_NOCHARGE_PLAN_FORMAT:
        missing_rich = [field for field in H1_RICH_NOCHARGE_PLAN_FIELDS if field not in fields]
        if missing_rich:
            raise ValueError(f"no-charge rich composition plan missing fields: {','.join(missing_rich)}")
        if "charge" in fields:
            raise ValueError("no-charge rich composition plan must not emit a charge field")
        marker_value = _normalize_generated_label(fields.get(R5C_PLAN_END_FIELD, ""))
        if marker_value != R5C_PLAN_END_VALUE:
            raise ValueError("no-charge rich composition plan missing required end: plan marker")

    formula_value = fields["formula"].split("<", 1)[0].strip()
    if not formula_value:
        raise ValueError("composition plan formula is empty")
    generated_formula = formula_value.split()[0]
    symbols, counts = symbol_counts_from_formula(generated_formula)
    num_atoms = int(sum(counts))
    if not 1 <= num_atoms <= int(max_atoms):
        raise ValueError(f"composition plan N {num_atoms} outside 1..{max_atoms}")
    formula = formula_from_symbol_counts(symbols, counts)
    has_rich_fields = any(key in fields for key in ("anion", "charge", "lattice", "spacegroup", "volume"))
    has_nocharge_rich_fields = all(key in fields for key in H1_RICH_NOCHARGE_PLAN_FIELDS[1:]) and "charge" not in fields
    inferred_style = (
        R5C_FORMULA_END_PLAN_FORMAT
        if _normalize_generated_label(fields.get(R5C_PLAN_END_FIELD, "")) == R5C_PLAN_END_VALUE
        and not has_rich_fields
        else H1_RICH_NOCHARGE_PLAN_FORMAT
        if has_nocharge_rich_fields
        else H1_RICH_PLAN_FORMAT
        if has_rich_fields
        else
        R5C_SEMANTIC_PLAN_FORMAT
        if any(key in fields for key in ("family", "arity", "size"))
        else R5C_PLAN_FORMAT
    )
    output_style = requested_style or inferred_style
    plan: Dict[str, Any] = {
        "plan_state_version": PLAN_STATE_VERSION,
        "N": num_atoms,
        "elements": symbols,
        "counts": counts,
        "formula": formula,
        "reduced_formula": formula,
        "charge_bucket": "unknown",
        "oxidation_candidates": "unknown",
        "anion_framework": "unknown",
        "lattice_system": "unknown",
        "spacegroup_bucket": "sg_unknown",
        "volume_per_atom_bin": "volpa_unknown",
        "prototype_key": f"formula={formula}|N={num_atoms}",
        "plan_format": output_style,
        "plan_end_marker_present": _normalize_generated_label(fields.get(R5C_PLAN_END_FIELD, "")) == R5C_PLAN_END_VALUE,
        "derived_counts_from_formula": True,
        "derived_n_from_formula": True,
    }
    expected = semantic_fields_from_plan(plan)
    plan.update(expected)
    if output_style in (H1_RICH_PLAN_FORMAT, H1_RICH_NOCHARGE_PLAN_FORMAT) or has_rich_fields:
        is_nocharge = output_style == H1_RICH_NOCHARGE_PLAN_FORMAT
        classification = _derived_formula_classification(symbols, counts) if is_nocharge else None
        generated_rich = {
            "anion": _normalize_rich_anion(fields.get("anion", "")),
            "lattice": _normalize_rich_lattice(fields.get("lattice", "")),
            "spacegroup": _normalize_rich_spacegroup(fields.get("spacegroup", "")),
            "volume": _normalize_rich_volume(fields.get("volume", "")),
        }
        if not is_nocharge:
            generated_rich["charge"] = _normalize_rich_charge(fields.get("charge", ""))
        derived_charge = (
            charge_bucket_from_classification(classification or {})
            if is_nocharge
            else generated_rich["charge"]
        )
        expected_anion = anion_framework_from_symbols(symbols)
        plan.update(
            {
                "anion_framework": generated_rich["anion"],
                "charge_bucket": derived_charge,
                "lattice_system": generated_rich["lattice"],
                "spacegroup_bucket": generated_rich["spacegroup"],
                "volume_per_atom_bin": generated_rich["volume"],
                "generated_rich_fields": dict(generated_rich),
                "expected_anion_framework": expected_anion,
                "anion_match_formula": generated_rich["anion"] == expected_anion,
                "rich_field_valid": True,
                "derived_charge_bucket_from_formula": is_nocharge,
            }
        )
        if classification is not None:
            plan["validator"] = dict(classification)
        plan["prototype_key"] = prototype_key(plan)
    generated_semantic: Dict[str, str] = {}
    for key in ("family", "arity", "size"):
        if key in fields:
            generated = _normalize_generated_label(fields[key])
            generated_semantic[key] = generated
            plan[f"generated_{key}"] = generated
            plan[f"expected_{key}"] = expected[key]
            plan[f"{key}_match_formula"] = generated == expected[key]
    plan["generated_semantic_fields"] = generated_semantic
    plan["semantic_consistency"] = semantic_consistency_from_plan(plan)
    return plan


def build_plan_body_answer(plan_text: str, body_answer: str) -> str:
    return (
        f"{R5C_PLAN_BODY_PLAN_LABEL}\n"
        f"{str(plan_text).strip()}\n"
        f"{R5C_PLAN_BODY_BODY_LABEL}\n"
        f"{str(body_answer).strip()}"
    )


def split_plan_body_answer(text: str) -> Tuple[str, str]:
    """Extract compact plan text and dynamic body text from a generated answer."""

    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    plan_match = re.search(r"(?im)^\s*plan\s*:\s*$", normalized)
    body_match = re.search(r"(?im)^\s*body\s*:\s*$", normalized)
    if plan_match is None:
        raise ValueError("plan-body answer is missing a plan: block")
    if body_match is None:
        raise ValueError("plan-body answer is missing a body: block")
    if body_match.start() <= plan_match.end():
        raise ValueError("plan-body answer has body: before plan content")
    plan_text = normalized[plan_match.end() : body_match.start()].strip()
    body_text = normalized[body_match.end() :].strip()
    if not plan_text:
        raise ValueError("plan-body answer has an empty plan block")
    if not body_text:
        raise ValueError("plan-body answer has an empty body block")
    return plan_text, body_text


def token_len(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def _assert_plan_round_trip(
    plan_state: Mapping[str, Any],
    plan_text: str,
    *,
    plan_style: str | None = None,
) -> Dict[str, Any]:
    parsed_plan = parse_composition_plan(plan_text, plan_style=plan_style)
    validation = validate_plan_state(parsed_plan)
    if not validation.valid:
        raise ValueError(f"composition-text plan failed validation: {validation.to_dict()}")
    for key in ("N", "formula", "elements", "counts"):
        if parsed_plan.get(key) != plan_state.get(key):
            expected = composition_plan_from_state(plan_state).get(key)
            if parsed_plan.get(key) != expected:
                raise ValueError(f"composition-text plan {key} mismatch: {parsed_plan.get(key)!r} != {expected!r}")
    return parsed_plan


def build_plan_body_record(
    *,
    plan_state: Mapping[str, Any],
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    tokenizer: Any = None,
    prompt: str | None = None,
    answer_separator: str = "",
    sample_weight: float = 1.0,
    plan_style: str | None = None,
) -> Dict[str, Any]:
    """Build one joint SFT row for de novo semi-autoregressive R5-C.

    The row uses ``representation=dynamic_v1`` so training still registers the
    crystal special-token vocabulary, but ``loss_profile=text`` keeps the mixed
    plan/body answer from inheriting body-only positional loss weights.
    """

    normalized_plan_style = normalize_plan_style(plan_style)
    r5_representation = representation_for_plan_style(normalized_plan_style)
    plan_text = format_composition_plan(plan_state, plan_style=normalized_plan_style)
    parsed_plan = _assert_plan_round_trip(plan_state, plan_text, plan_style=normalized_plan_style)
    body_answer, diagnostics = arrays_to_dynamic_answer(
        lengths=arrays["lengths"],
        angles=arrays["angles"],
        species=arrays["species"],
        frac_coords=arrays["frac_coords"],
        separator=answer_separator,
    )
    validate_answer_matches_plan(parsed_plan, body_answer)

    prompt_text = (CRYSLLMGEN_TEXT_PROMPT if prompt is None else str(prompt)).rstrip()
    answer = build_plan_body_answer(plan_text, body_answer)
    plan_block_text, body_block_text = split_plan_body_answer(answer)
    body_prefix = f"{R5C_PLAN_BODY_PLAN_LABEL}\n{plan_block_text}\n{R5C_PLAN_BODY_BODY_LABEL}\n"
    body_char_start = len(body_prefix)
    body_semantic_length = exact_body_token_count(parsed_plan)
    prompt_with_newline = prompt_text + "\n"
    plan_model_length = token_len(tokenizer, f"{R5C_PLAN_BODY_PLAN_LABEL}\n{plan_block_text}\n")
    body_prefix_model_length = token_len(tokenizer, body_prefix)
    body_model_length = token_len(tokenizer, body_block_text)
    answer_model_length = token_len(tokenizer, answer)

    return {
        "task": "r5c_plan_body_generation",
        "module": "full",
        "module_id": 0,
        "representation": "dynamic_v1",
        "r5_representation": r5_representation,
        "prompt_version": R5C_PLAN_BODY_PROMPT_VERSION,
        "plan_state_version": PLAN_STATE_VERSION,
        "plan_format": normalized_plan_style,
        "prompt": prompt_text,
        "answer": answer,
        "text": prompt_with_newline + answer,
        "plan_text": plan_text,
        "body_answer": body_answer,
        "num_atoms": int(parsed_plan["N"]),
        "body_semantic_length": int(body_semantic_length),
        "prompt_length": token_len(tokenizer, prompt_with_newline),
        "answer_model_length": answer_model_length,
        "plan_model_length": plan_model_length,
        "body_prefix_model_length": body_prefix_model_length,
        "body_model_length": body_model_length,
        "block_spans": {
            "plan": {
                "answer_char_start": len(f"{R5C_PLAN_BODY_PLAN_LABEL}\n"),
                "answer_char_end": len(f"{R5C_PLAN_BODY_PLAN_LABEL}\n") + len(plan_block_text),
                "model_length": plan_model_length,
            },
            "body": {
                "answer_char_start": body_char_start,
                "answer_char_end": body_char_start + len(body_block_text),
                "semantic_length": int(body_semantic_length),
                "prefix_model_length": body_prefix_model_length,
                "model_length": body_model_length,
            },
        },
        "plan_state": dict(parsed_plan),
        "r5_plan_state": dict(parsed_plan),
        "source_plan_state": dict(plan_state),
        "metadata": dict(metadata or {}),
        "sample_weight": float(sample_weight),
        "loss_profile": "text",
        "mask_policy": "normal",
        "encode_diagnostics": diagnostics.to_dict(),
    }


def build_plan_only_record(
    *,
    plan_state: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    tokenizer: Any = None,
    prompt: str | None = None,
    sample_weight: float = 1.5,
    plan_style: str | None = None,
) -> Dict[str, Any]:
    normalized_plan_style = normalize_plan_style(plan_style)
    r5_representation = representation_for_plan_style(normalized_plan_style)
    plan_text = format_composition_plan(plan_state, plan_style=normalized_plan_style)
    parsed_plan = _assert_plan_round_trip(plan_state, plan_text, plan_style=normalized_plan_style)
    prompt_text = (CRYSLLMGEN_TEXT_PROMPT if prompt is None else str(prompt)).rstrip()
    answer = f"{R5C_PLAN_BODY_PLAN_LABEL}\n{plan_text}"
    prompt_with_newline = prompt_text + "\n"
    return {
        "task": "r5c_composition_plan_only",
        "module": "plan_only",
        "module_id": 0,
        "representation": "dynamic_v1",
        "r5_representation": r5_representation,
        "prompt_version": R5C_PLAN_BODY_PROMPT_VERSION,
        "plan_state_version": PLAN_STATE_VERSION,
        "plan_format": normalized_plan_style,
        "prompt": prompt_text,
        "answer": answer,
        "text": prompt_with_newline + answer,
        "plan_text": plan_text,
        "body_answer": "",
        "num_atoms": int(parsed_plan["N"]),
        "prompt_length": token_len(tokenizer, prompt_with_newline),
        "answer_model_length": token_len(tokenizer, answer),
        "plan_model_length": token_len(tokenizer, answer),
        "body_model_length": 0,
        "body_semantic_length": 0,
        "block_spans": {
            "plan": {
                "answer_char_start": len(f"{R5C_PLAN_BODY_PLAN_LABEL}\n"),
                "answer_char_end": len(answer),
                "model_length": token_len(tokenizer, answer),
            }
        },
        "plan_state": dict(parsed_plan),
        "r5_plan_state": dict(parsed_plan),
        "source_plan_state": dict(plan_state),
        "metadata": dict(metadata or {}),
        "sample_weight": float(sample_weight),
        "loss_profile": "text",
        "mask_policy": "normal",
    }


def build_body_replay_record(
    *,
    plan_state: Mapping[str, Any],
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    tokenizer: Any = None,
    prompt: str | None = None,
    answer_separator: str = "",
    sample_weight: float = 0.25,
    plan_style: str | None = None,
) -> Dict[str, Any]:
    normalized_plan_style = normalize_plan_style(plan_style)
    r5_representation = representation_for_plan_style(normalized_plan_style)
    plan_text = format_composition_plan(plan_state, plan_style=normalized_plan_style)
    parsed_plan = _assert_plan_round_trip(plan_state, plan_text, plan_style=normalized_plan_style)
    body_answer, diagnostics = arrays_to_dynamic_answer(
        lengths=arrays["lengths"],
        angles=arrays["angles"],
        species=arrays["species"],
        frac_coords=arrays["frac_coords"],
        separator=answer_separator,
    )
    validate_answer_matches_plan(parsed_plan, body_answer)
    base_prompt = (CRYSLLMGEN_TEXT_PROMPT if prompt is None else str(prompt)).rstrip()
    prompt_text = f"{base_prompt}\n{R5C_PLAN_BODY_PLAN_LABEL}\n{plan_text}\n{R5C_PLAN_BODY_BODY_LABEL}"
    prompt_with_newline = prompt_text + "\n"
    return {
        "task": "r5c_composition_body_replay",
        "module": "body_replay",
        "module_id": 0,
        "representation": "dynamic_v1",
        "r5_representation": r5_representation,
        "prompt_version": R5C_PLAN_BODY_PROMPT_VERSION,
        "plan_state_version": PLAN_STATE_VERSION,
        "plan_format": normalized_plan_style,
        "prompt": prompt_text,
        "answer": body_answer,
        "text": prompt_with_newline + body_answer,
        "plan_text": plan_text,
        "body_answer": body_answer,
        "num_atoms": int(parsed_plan["N"]),
        "body_semantic_length": int(exact_body_token_count(parsed_plan)),
        "prompt_length": token_len(tokenizer, prompt_with_newline),
        "answer_model_length": token_len(tokenizer, body_answer),
        "plan_model_length": token_len(tokenizer, f"{R5C_PLAN_BODY_PLAN_LABEL}\n{plan_text}\n"),
        "body_model_length": token_len(tokenizer, body_answer),
        "block_spans": {
            "body": {
                "answer_char_start": 0,
                "answer_char_end": len(body_answer),
                "semantic_length": int(exact_body_token_count(parsed_plan)),
                "model_length": token_len(tokenizer, body_answer),
            }
        },
        "plan_state": dict(parsed_plan),
        "r5_plan_state": dict(parsed_plan),
        "source_plan_state": dict(plan_state),
        "metadata": dict(metadata or {}),
        "sample_weight": float(sample_weight),
        "loss_profile": "fixed_slot",
        "mask_policy": "normal",
        "encode_diagnostics": diagnostics.to_dict(),
    }


__all__ = [
    "H1_RICH_NOCHARGE_PLAN_BODY_REPRESENTATION",
    "H1_RICH_NOCHARGE_PLAN_FIELDS",
    "H1_RICH_NOCHARGE_PLAN_FORMAT",
    "H1_RICH_PLAN_BODY_REPRESENTATION",
    "H1_RICH_PLAN_FIELDS",
    "H1_RICH_PLAN_FORMAT",
    "R5C_FORMULA_TEXT_PLAN_BODY_REPRESENTATION",
    "R5C_FORMULA_END_PLAN_BODY_REPRESENTATION",
    "R5C_FORMULA_END_PLAN_FORMAT",
    "R5C_PLAN_BODY_BODY_LABEL",
    "R5C_PLAN_BODY_PLAN_LABEL",
    "R5C_PLAN_FORMAT",
    "R5C_PLAN_BODY_PROMPT_VERSION",
    "R5C_PLAN_BODY_REPRESENTATION",
    "R5C_PLAN_STYLES",
    "R5C_SEMANTIC_FORMULA_PLAN_BODY_REPRESENTATION",
    "R5C_SEMANTIC_PLAN_FORMAT",
    "arity_label",
    "build_body_replay_record",
    "build_plan_only_record",
    "build_plan_body_answer",
    "build_plan_body_record",
    "composition_plan_from_state",
    "family_label",
    "format_composition_plan",
    "formula_from_symbol_counts",
    "has_plan_end_marker",
    "has_plan_tail_after_end_marker",
    "normalize_plan_style",
    "nocharge_rich_fields_from_plan_state",
    "parse_composition_plan",
    "representation_for_plan_style",
    "rich_fields_from_plan_state",
    "semantic_consistency_from_plan",
    "semantic_fields_from_plan",
    "size_label",
    "split_plan_body_answer",
]
