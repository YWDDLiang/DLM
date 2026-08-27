"""R5 composition-first crystal plan-state helpers.

The plan state is a small, JSON-serializable physical state used as persistent
conditioning context before dynamic/exact-length body generation.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Dict, Mapping, Sequence

from crystal_dlm.composition_validity import (
    classify_smact_validity,
    element_symbols,
    formula_from_composition,
    reduced_composition,
)
from crystal_dlm.fixed_slot import SYMBOL_TO_Z


PLAN_STATE_VERSION = "r5_plan_state_v1"

PLAN_STATE_FIELDS = [
    "N",
    "elements",
    "counts",
    "formula",
    "reduced_formula",
    "charge_bucket",
    "oxidation_candidates",
    "anion_framework",
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
    "prototype_key",
]

PLAN_STATE_PROMPT = (
    "Generate one crystal plan_state JSON object for an MP-20 bulk material. "
    "Use keys N, elements, counts, formula, reduced_formula, charge_bucket, "
    "oxidation_candidates, anion_framework, lattice_system, spacegroup_bucket, "
    "volume_per_atom_bin, prototype_key. Return only JSON:"
)

PLAN_STATE_COMPACT_PROMPT = (
    "Generate one compact MP-20 crystal plan. Format exactly: "
    "N=<atoms>;E=<Element>:<count>,<Element>:<count>;LS=<lattice_system>;"
    "SG=<spacegroup_bucket>;VP=<volume_per_atom_bin>. "
    "Hard rules: N must be 1..20 and must equal the sum of all E counts; "
    "do not add elements after the count sum reaches N. Return only the compact line:"
)

PLAN_STATE_ATOMSEQ_PROMPT = (
    "Generate one atom-sequence MP-20 crystal plan. Format exactly: "
    "A=<Element>,<Element>,...;LS=<lattice_system>;SG=<spacegroup_bucket>;"
    "VP=<volume_per_atom_bin>. Hard rules: A must contain 1..20 atom symbols, "
    "with one symbol per atom. Return only the atom-sequence line:"
)

PLAN_STATE_ATOMSLOTS_PROMPT = (
    "Generate one atom-slots MP-20 crystal plan. Format exactly: "
    "S=<slot1>,<slot2>,...,<slot20>;LS=<lattice_system>;"
    "SG=<spacegroup_bucket>;VP=<volume_per_atom_bin>. Hard rules: each slot is "
    "one element symbol or _ for empty, S must have at most 20 slots, and at "
    "least one slot must be an element. Return only the atom-slots line:"
)

PLAN_STATE_ATOMFIELDS_PROMPT = (
    "Generate one atom-fields MP-20 crystal plan. Format exactly: "
    "S01=<slot>;S02=<slot>;S03=<slot>;S04=<slot>;S05=<slot>;S06=<slot>;"
    "S07=<slot>;S08=<slot>;S09=<slot>;S10=<slot>;S11=<slot>;S12=<slot>;"
    "S13=<slot>;S14=<slot>;S15=<slot>;S16=<slot>;S17=<slot>;S18=<slot>;"
    "S19=<slot>;S20=<slot>;LS=<lattice_system>;SG=<spacegroup_bucket>;"
    "VP=<volume_per_atom_bin>. Hard rules: every S01..S20 value is one element "
    "symbol or _ for empty, all 20 slot fields must appear exactly once, and at "
    "least one slot must be an element. Return only the atom-fields line:"
)

PLAN_STATE_COUNTFIELDS_PROMPT = (
    "Generate one count-fields MP-20 crystal plan. Format exactly: "
    "P01=Z###:C###;P02=Z###:C###;P03=Z###:C###;P04=Z###:C###;"
    "P05=Z###:C###;P06=Z###:C###;P07=Z###:C###;LS=L_<code>;"
    "SG=G######;VP=V######. Hard rules: each P field is an element atomic "
    "number code and count code, Z000:C000 means empty, non-empty counts must "
    "sum to 1..20, all P01..P07 fields must appear exactly once, and no extra "
    "P fields are allowed. Return only the count-fields line:"
)

PLAN_STATE_COUNTVALENCE_PROMPT = (
    "Generate one chemistry-labeled MP-20 crystal plan. Format exactly: "
    "P01=Z###:C###:QX##;P02=Z###:C###:QX##;P03=Z###:C###:QX##;"
    "P04=Z###:C###:QX##;P05=Z###:C###:QX##;P06=Z###:C###:QX##;"
    "P07=Z###:C###:QX##;CB=B_<code>;LS=L_<code>;SG=G######;VP=V######. "
    "Hard rules: QP## is positive oxidation, QM## is negative oxidation, "
    "QZ00 is zero or empty, QU00 is unknown oxidation, Z000:C000:QZ00 means "
    "empty, non-empty counts must sum to 1..20, all P01..P07 fields must "
    "appear exactly once, and no extra P fields are allowed. Return only the "
    "chemistry-labeled line:"
)

PLAN_STATE_COMPACT_REPAIR_PROMPT_TEMPLATE = (
    "Repair one compact MP-20 crystal plan. The original line may violate syntax, "
    "composition, or count rules. Return only one corrected compact line in the exact "
    "format N=<atoms>;E=<Element>:<count>,<Element>:<count>;LS=<lattice_system>;"
    "SG=<spacegroup_bucket>;VP=<volume_per_atom_bin>. Hard rules: N must be 1..20 "
    "and must equal the sum of all E counts.\n"
    "violation_labels: {violation_labels}\n"
    "original_compact_plan: {visible_plan}\n"
    "corrected_compact_plan:"
)

ALLOWED_LATTICE_SYSTEMS = {
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
}

ALLOWED_SPACEGROUP_BUCKETS = {
    "sg_001_002",
    "sg_003_015",
    "sg_016_074",
    "sg_075_142",
    "sg_143_167",
    "sg_168_194",
    "sg_195_230",
}

LATTICE_SYSTEM_TO_CODE = {
    "triclinic": "L_TRI",
    "monoclinic": "L_MON",
    "orthorhombic": "L_ORH",
    "tetragonal": "L_TET",
    "trigonal": "L_TRG",
    "hexagonal": "L_HEX",
    "cubic": "L_CUB",
}
CODE_TO_LATTICE_SYSTEM = {value: key for key, value in LATTICE_SYSTEM_TO_CODE.items()}

SPACEGROUP_BUCKET_TO_CODE = {
    "sg_001_002": "G001002",
    "sg_003_015": "G003015",
    "sg_016_074": "G016074",
    "sg_075_142": "G075142",
    "sg_143_167": "G143167",
    "sg_168_194": "G168194",
    "sg_195_230": "G195230",
}
CODE_TO_SPACEGROUP_BUCKET = {value: key for key, value in SPACEGROUP_BUCKET_TO_CODE.items()}
Z_TO_SYMBOL = {z: symbol for symbol, z in SYMBOL_TO_Z.items()}

CHARGE_BUCKET_TO_CODE = {
    "neutral_plausible": "B_NEU",
    "single_element": "B_ONE",
    "all_metal": "B_MET",
    "charge_fail": "B_CHF",
    "pauling_fail": "B_PAU",
    "oxidation_missing": "B_OXM",
    "validator_unavailable": "B_UNK",
}
CODE_TO_CHARGE_BUCKET = {value: key for key, value in CHARGE_BUCKET_TO_CODE.items()}


@dataclass(frozen=True)
class PlanValidation:
    valid_N: bool
    valid_formula: bool
    valid_counts: bool
    valid_elements: bool
    valid_generated_N: bool = True

    @property
    def valid(self) -> bool:
        return self.valid_N and self.valid_generated_N and self.valid_formula and self.valid_counts and self.valid_elements

    def to_dict(self) -> Dict[str, bool]:
        return {
            "valid": self.valid,
            "valid_N": self.valid_N,
            "valid_generated_N": self.valid_generated_N,
            "valid_formula": self.valid_formula,
            "valid_counts": self.valid_counts,
            "valid_elements": self.valid_elements,
        }


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _formula_from_symbols(symbols: Sequence[str], counts: Sequence[int]) -> str:
    parts: list[str] = []
    for symbol, count in zip(symbols, counts):
        count = int(count)
        parts.append(str(symbol) if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def _canonical_symbol_counts(symbols: Sequence[str], counts: Sequence[int]) -> tuple[list[str], list[int]]:
    counter: Counter[str] = Counter()
    for symbol, count in zip(symbols, counts):
        if symbol not in SYMBOL_TO_Z:
            continue
        count_value = int(count)
        if count_value > 0:
            counter[symbol] += count_value
    ordered = sorted(counter, key=lambda symbol: SYMBOL_TO_Z.get(symbol, 10_000))
    return ordered, [int(counter[symbol]) for symbol in ordered]


def _full_composition_from_species(species: Sequence[str]) -> tuple[list[str], list[int]]:
    counter = Counter(str(symbol) for symbol in species)
    symbols = sorted(counter, key=lambda symbol: SYMBOL_TO_Z.get(symbol, 10_000))
    return symbols, [int(counter[symbol]) for symbol in symbols]


def _safe_smact_classification(symbols: Sequence[str], counts: Sequence[int]) -> Dict[str, Any]:
    elems = tuple(SYMBOL_TO_Z[symbol] for symbol in symbols if symbol in SYMBOL_TO_Z)
    if len(elems) != len(symbols):
        return {"valid": False, "reason": "unsupported_element"}
    try:
        return classify_smact_validity(elems, tuple(int(value) for value in counts))
    except Exception as exc:  # noqa: BLE001 - validators are optional in local envs.
        return {
            "valid": None,
            "reason": "validator_unavailable",
            "validator_error": type(exc).__name__,
        }


def charge_bucket_from_classification(classification: Mapping[str, Any]) -> str:
    reason = str(classification.get("reason", "unknown"))
    valid = classification.get("valid")
    if valid is True and reason == "charge_neutral_pauling_valid":
        return "neutral_plausible"
    if valid is True and reason == "single_element_shortcut":
        return "single_element"
    if valid is True and reason == "all_metal_shortcut":
        return "all_metal"
    if reason == "charge_neutrality_fail":
        return "charge_fail"
    if reason == "pauling_fail_or_ratio_rejected":
        return "pauling_fail"
    if reason == "oxidation_state_missing":
        return "oxidation_missing"
    if reason == "validator_unavailable":
        return "validator_unavailable"
    return reason


def anion_framework_from_symbols(symbols: Sequence[str]) -> str:
    symbol_set = set(symbols)
    if "O" in symbol_set:
        return "oxide"
    if "S" in symbol_set:
        return "sulfide"
    if "Se" in symbol_set or "Te" in symbol_set:
        return "chalcogenide"
    if symbol_set.intersection({"F", "Cl", "Br", "I"}):
        return "halide"
    if "N" in symbol_set:
        return "nitride"
    if "P" in symbol_set:
        return "phosphide_or_phosphate"
    return "other"


def lattice_system_from_lattice(lengths: Sequence[float], angles: Sequence[float], tol: float = 1e-2) -> str:
    a, b, c = [float(value) for value in lengths]
    alpha, beta, gamma = [float(value) for value in angles]
    eq_ab = abs(a - b) <= tol
    eq_bc = abs(b - c) <= tol
    right = all(abs(value - 90.0) <= tol for value in (alpha, beta, gamma))
    if eq_ab and eq_bc and right:
        return "cubic"
    if eq_ab and not eq_bc and right:
        return "tetragonal"
    if right:
        return "orthorhombic"
    if eq_ab and abs(alpha - 90.0) <= tol and abs(beta - 90.0) <= tol and abs(gamma - 120.0) <= tol:
        return "hexagonal"
    if eq_ab and eq_bc and abs(alpha - beta) <= tol and abs(beta - gamma) <= tol:
        return "trigonal"
    if sum(abs(value - 90.0) <= tol for value in (alpha, beta, gamma)) == 2:
        return "monoclinic"
    return "triclinic"


def lattice_volume(lengths: Sequence[float], angles: Sequence[float]) -> float:
    a, b, c = [float(value) for value in lengths]
    alpha, beta, gamma = [math.radians(float(value)) for value in angles]
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    radicand = 1.0 + 2.0 * cos_a * cos_b * cos_g - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g
    return a * b * c * math.sqrt(max(radicand, 0.0))


def volume_per_atom_bin(lengths: Sequence[float], angles: Sequence[float], num_atoms: int) -> str:
    if int(num_atoms) <= 0:
        return "volpa_unknown"
    value = lattice_volume(lengths, angles) / float(num_atoms)
    if not math.isfinite(value) or value <= 0:
        return "volpa_unknown"
    low = int(math.floor(value / 5.0) * 5)
    high = low + 4
    return f"volpa_{low:03d}_{high:03d}"


def spacegroup_bucket(metadata: Mapping[str, Any]) -> str:
    raw = metadata.get("spacegroup.number.conv", metadata.get("spacegroup.number"))
    value = _safe_float(raw)
    if value is None:
        return "sg_unknown"
    number = int(value)
    if number <= 2:
        return "sg_001_002"
    if number <= 15:
        return "sg_003_015"
    if number <= 74:
        return "sg_016_074"
    if number <= 142:
        return "sg_075_142"
    if number <= 167:
        return "sg_143_167"
    if number <= 194:
        return "sg_168_194"
    if number <= 230:
        return "sg_195_230"
    return "sg_unknown"


def prototype_key(plan: Mapping[str, Any]) -> str:
    reduced_formula = str(plan.get("reduced_formula", "unknown"))
    return "|".join(
        [
            f"formula={reduced_formula}",
            f"anion={plan.get('anion_framework', 'unknown')}",
            f"charge={plan.get('charge_bucket', 'unknown')}",
            f"lat={plan.get('lattice_system', 'unknown')}",
            f"sg={plan.get('spacegroup_bucket', 'sg_unknown')}",
            f"vol={plan.get('volume_per_atom_bin', 'volpa_unknown')}",
        ]
    )


def _enum_or_default(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def _volume_bin_or_default(value: Any) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"volpa_\d{3}_\d{3}", text):
        return text
    return "volpa_unknown"


def _volume_bin_search_or_default(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"volpa_\d{3}_\d{3}", text)
    return match.group(0) if match else "volpa_unknown"


def _spacegroup_bucket_or_default(value: Any) -> str:
    text = str(value or "").strip()
    if text in ALLOWED_SPACEGROUP_BUCKETS:
        return text
    number_match = re.search(r"sg[_-]?0*(\d{1,3})(?:[_-]0*(\d{1,3}))?", text)
    if number_match:
        number = int(number_match.group(1))
        if number <= 2:
            return "sg_001_002"
        if number <= 15:
            return "sg_003_015"
        if number <= 74:
            return "sg_016_074"
        if number <= 142:
            return "sg_075_142"
        if number <= 167:
            return "sg_143_167"
        if number <= 194:
            return "sg_168_194"
        if number <= 230:
            return "sg_195_230"
    return "sg_001_002"


def _volume_bin_to_code(value: Any) -> str:
    text = _volume_bin_or_default(value)
    match = re.fullmatch(r"volpa_(\d{3})_(\d{3})", text)
    if not match:
        return "V999999"
    return f"V{match.group(1)}{match.group(2)}"


def _volume_code_to_bin(value: Any) -> str:
    text = str(value or "").strip().upper()
    match = re.fullmatch(r"V(\d{3})(\d{3})", text)
    if not match:
        return "volpa_unknown"
    if match.group(1) == "999" and match.group(2) == "999":
        return "volpa_unknown"
    return f"volpa_{match.group(1)}_{match.group(2)}"


def _oxidation_to_code(value: Any) -> str:
    if value in (None, "", "unknown"):
        return "QU00"
    try:
        oxidation = int(value)
    except Exception:
        return "QU00"
    magnitude = abs(oxidation)
    if magnitude > 99:
        raise ValueError(f"oxidation state {oxidation} outside encodable range")
    if oxidation < 0:
        return f"QM{magnitude:02d}"
    if oxidation > 0:
        return f"QP{magnitude:02d}"
    return "QZ00"


def _oxidation_code_to_int(value: Any) -> int | None:
    text = str(value or "").strip().upper()
    match = re.fullmatch(r"Q([PMZU])(\d{2})", text)
    if not match:
        raise ValueError(f"invalid oxidation code {value!r}")
    sign, magnitude_text = match.groups()
    magnitude = int(magnitude_text)
    if sign == "U":
        if magnitude != 0:
            raise ValueError(f"unknown oxidation code must use QU00, got {value!r}")
        return None
    if sign == "Z":
        if magnitude != 0:
            raise ValueError(f"zero oxidation code must use QZ00, got {value!r}")
        return 0
    if sign == "M":
        return -magnitude
    return magnitude


def _parse_compact_fields(text: str) -> Dict[str, str]:
    first_line = str(text).strip().splitlines()[0] if str(text).strip() else ""
    fields: Dict[str, str] = {}
    for chunk in first_line.split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        normalized_key = key.strip().upper()
        if normalized_key:
            fields[normalized_key] = value.strip()
    return fields


def _ordered_symbol_counts_from_compact_field(element_field: str) -> tuple[list[str], list[int]]:
    ordered: OrderedDict[str, int] = OrderedDict()
    for match in re.finditer(r"([A-Z][a-z]?)[\s:=xX*_-]*(\d{1,3})", str(element_field or "")):
        symbol = match.group(1)
        if symbol not in SYMBOL_TO_Z:
            continue
        count = int(match.group(2))
        if count <= 0:
            continue
        ordered[symbol] = int(ordered.get(symbol, 0)) + count
    return list(ordered.keys()), list(ordered.values())


def _trim_ordered_counts_to_max_atoms(
    symbols: Sequence[str],
    counts: Sequence[int],
    *,
    max_atoms: int,
) -> tuple[list[str], list[int]]:
    trimmed_symbols = [str(symbol) for symbol in symbols]
    trimmed_counts = [int(count) for count in counts]
    while sum(trimmed_counts) > int(max_atoms) and trimmed_counts:
        tail_idx = len(trimmed_counts) - 1
        excess = sum(trimmed_counts) - int(max_atoms)
        if trimmed_counts[tail_idx] > 1:
            trimmed_counts[tail_idx] -= min(excess, trimmed_counts[tail_idx] - 1)
        else:
            trimmed_symbols.pop()
            trimmed_counts.pop()
    if not trimmed_counts:
        return [], []
    return trimmed_symbols, trimmed_counts


def normalize_compact_plan_for_repair_target(text: str, *, max_atoms: int = 20) -> str:
    """Create a deterministic training target for compact-plan repair examples."""
    fields = _parse_compact_fields(text)
    symbols, counts = _ordered_symbol_counts_from_compact_field(fields.get("E", ""))
    if not symbols:
        raise ValueError("cannot normalize compact plan without supported element-count pairs")
    symbols, counts = _trim_ordered_counts_to_max_atoms(symbols, counts, max_atoms=int(max_atoms))
    symbols, counts = _canonical_symbol_counts(symbols, counts)
    if not symbols:
        raise ValueError("cannot normalize compact plan to an empty composition")
    n_value = sum(int(count) for count in counts)
    pairs = ",".join(f"{symbol}:{int(count)}" for symbol, count in zip(symbols, counts))
    return (
        f"N={n_value};"
        f"E={pairs};"
        f"LS={_enum_or_default(fields.get('LS'), ALLOWED_LATTICE_SYSTEMS, 'triclinic')};"
        f"SG={_spacegroup_bucket_or_default(fields.get('SG'))};"
        f"VP={_volume_bin_search_or_default(fields.get('VP', fields.get('VOLPA')))}"
    )


def plan_state_to_compact(plan: Mapping[str, Any]) -> str:
    elements = list(plan.get("elements") or [])
    counts = [int(value) for value in (plan.get("counts") or [])]
    pairs = ",".join(f"{symbol}:{count}" for symbol, count in zip(elements, counts))
    return (
        f"N={int(plan['N'])};"
        f"E={pairs};"
        f"LS={plan.get('lattice_system', 'triclinic')};"
        f"SG={plan.get('spacegroup_bucket', 'sg_001_002')};"
        f"VP={plan.get('volume_per_atom_bin', 'volpa_unknown')}"
    )


def plan_state_to_atomseq(plan: Mapping[str, Any]) -> str:
    symbols = [str(symbol) for symbol in (plan.get("elements") or [])]
    counts = [int(value) for value in (plan.get("counts") or [])]
    atom_symbols = [symbol for symbol, count in zip(symbols, counts) for _ in range(int(count))]
    return (
        f"A={','.join(atom_symbols)};"
        f"LS={plan.get('lattice_system', 'triclinic')};"
        f"SG={plan.get('spacegroup_bucket', 'sg_001_002')};"
        f"VP={plan.get('volume_per_atom_bin', 'volpa_unknown')}"
    )


def plan_state_to_atomslots(plan: Mapping[str, Any], *, max_slots: int = 20) -> str:
    symbols = [str(symbol) for symbol in (plan.get("elements") or [])]
    counts = [int(value) for value in (plan.get("counts") or [])]
    atom_symbols = [symbol for symbol, count in zip(symbols, counts) for _ in range(int(count))]
    if len(atom_symbols) > int(max_slots):
        raise ValueError(f"atom-slots plan has {len(atom_symbols)} atoms for {max_slots} slots")
    slots = atom_symbols + ["_"] * (int(max_slots) - len(atom_symbols))
    return (
        f"S={','.join(slots)};"
        f"LS={plan.get('lattice_system', 'triclinic')};"
        f"SG={plan.get('spacegroup_bucket', 'sg_001_002')};"
        f"VP={plan.get('volume_per_atom_bin', 'volpa_unknown')}"
    )


def plan_state_to_atomfields(plan: Mapping[str, Any], *, max_slots: int = 20) -> str:
    symbols = [str(symbol) for symbol in (plan.get("elements") or [])]
    counts = [int(value) for value in (plan.get("counts") or [])]
    atom_symbols = [symbol for symbol, count in zip(symbols, counts) for _ in range(int(count))]
    if len(atom_symbols) > int(max_slots):
        raise ValueError(f"atom-fields plan has {len(atom_symbols)} atoms for {max_slots} slots")
    slots = atom_symbols + ["_"] * (int(max_slots) - len(atom_symbols))
    slot_fields = ";".join(f"S{idx:02d}={token}" for idx, token in enumerate(slots, start=1))
    return (
        f"{slot_fields};"
        f"LS={plan.get('lattice_system', 'triclinic')};"
        f"SG={plan.get('spacegroup_bucket', 'sg_001_002')};"
        f"VP={plan.get('volume_per_atom_bin', 'volpa_unknown')}"
    )


def plan_state_to_countfields(plan: Mapping[str, Any], *, max_pairs: int = 7) -> str:
    symbols = [str(symbol) for symbol in (plan.get("elements") or [])]
    counts = [int(value) for value in (plan.get("counts") or [])]
    symbols, counts = _canonical_symbol_counts(symbols, counts)
    if len(symbols) > int(max_pairs):
        raise ValueError(f"count-fields plan has {len(symbols)} element-count pairs for {max_pairs} fields")
    fields: list[str] = []
    for idx in range(int(max_pairs)):
        if idx < len(symbols):
            z_value = int(SYMBOL_TO_Z[symbols[idx]])
            count_value = int(counts[idx])
        else:
            z_value = 0
            count_value = 0
        fields.append(f"P{idx + 1:02d}=Z{z_value:03d}:C{count_value:03d}")
    lattice_code = LATTICE_SYSTEM_TO_CODE.get(str(plan.get("lattice_system", "triclinic")), "L_TRI")
    sg_code = SPACEGROUP_BUCKET_TO_CODE.get(str(plan.get("spacegroup_bucket", "sg_001_002")), "G001002")
    volume_code = _volume_bin_to_code(plan.get("volume_per_atom_bin"))
    return ";".join(fields + [f"LS={lattice_code}", f"SG={sg_code}", f"VP={volume_code}"])


def plan_state_to_countvalencefields(plan: Mapping[str, Any], *, max_pairs: int = 7) -> str:
    symbols_in = [str(symbol) for symbol in (plan.get("elements") or [])]
    counts_in = [int(value) for value in (plan.get("counts") or [])]
    if len(symbols_in) != len(counts_in):
        raise ValueError("count-valence plan requires aligned elements/counts")

    raw_species = plan.get("valence_species")
    rows: list[tuple[str, int, int | None]] = []
    if isinstance(raw_species, Sequence) and not isinstance(raw_species, (str, bytes)):
        merged_species: Counter[tuple[str, int | None]] = Counter()
        for value in raw_species:
            if not isinstance(value, Mapping):
                raise ValueError("valence_species entries must be mappings")
            symbol = str(value.get("element") or value.get("symbol") or "")
            count_value = int(value.get("count") or 0)
            oxidation_value = value.get("oxidation_state")
            if symbol not in SYMBOL_TO_Z or count_value <= 0:
                raise ValueError(f"invalid valence species {value!r}")
            oxidation_value = (
                None
                if oxidation_value in (None, "unknown")
                else int(oxidation_value)
            )
            merged_species[(symbol, oxidation_value)] += count_value
        rows = [
            (symbol, int(count), oxidation)
            for (symbol, oxidation), count in merged_species.items()
        ]
        source_symbols, source_counts = _canonical_symbol_counts(symbols_in, counts_in)
        species_symbols, species_counts = _canonical_symbol_counts(
            [symbol for symbol, _, _ in rows],
            [count for _, count, _ in rows],
        )
        if (source_symbols, source_counts) != (species_symbols, species_counts):
            raise ValueError("valence_species composition disagrees with Plan elements/counts")
    else:
        oxidation = (plan.get("validator") or {}).get("oxidation_states")
        if oxidation is None:
            oxidation = plan.get("oxidation_candidates")
        if isinstance(oxidation, Sequence) and not isinstance(oxidation, (str, bytes)):
            oxidation_in = [
                None if value in (None, "unknown") else int(value)
                for value in oxidation
            ]
        else:
            oxidation_in = [None for _ in symbols_in]
        if len(oxidation_in) != len(symbols_in):
            oxidation_in = [None for _ in symbols_in]
        for symbol, count, ox in zip(symbols_in, counts_in, oxidation_in):
            if symbol not in SYMBOL_TO_Z:
                continue
            count_value = int(count)
            if count_value > 0:
                rows.append((symbol, count_value, None if ox is None else int(ox)))
    rows.sort(
        key=lambda item: (
            SYMBOL_TO_Z.get(item[0], 10_000),
            10_000 if item[2] is None else int(item[2]),
        )
    )
    if len(rows) > int(max_pairs):
        raise ValueError(f"count-valence plan has {len(rows)} element-count pairs for {max_pairs} fields")

    fields: list[str] = []
    for idx in range(int(max_pairs)):
        if idx < len(rows):
            symbol, count_value, oxidation_value = rows[idx]
            z_value = int(SYMBOL_TO_Z[symbol])
            oxidation_code = _oxidation_to_code(oxidation_value)
        else:
            z_value = 0
            count_value = 0
            oxidation_code = "QZ00"
        fields.append(f"P{idx + 1:02d}=Z{z_value:03d}:C{count_value:03d}:{oxidation_code}")
    charge_code = CHARGE_BUCKET_TO_CODE.get(str(plan.get("charge_bucket", "validator_unavailable")), "B_UNK")
    lattice_code = LATTICE_SYSTEM_TO_CODE.get(str(plan.get("lattice_system", "triclinic")), "L_TRI")
    sg_code = SPACEGROUP_BUCKET_TO_CODE.get(str(plan.get("spacegroup_bucket", "sg_001_002")), "G001002")
    volume_code = _volume_bin_to_code(plan.get("volume_per_atom_bin"))
    return ";".join(fields + [f"CB={charge_code}", f"LS={lattice_code}", f"SG={sg_code}", f"VP={volume_code}"])


def parse_compact_plan_state(text: str, *, max_atoms: int = 20) -> Dict[str, Any]:
    fields = _parse_compact_fields(text)
    element_field = fields.get("E", "")
    symbols: list[str] = []
    counts: list[int] = []
    for match in re.finditer(r"([A-Z][a-z]?)[\s:=xX*_-]*(\d{1,2})", element_field):
        symbol = match.group(1)
        if symbol not in SYMBOL_TO_Z:
            continue
        count = int(match.group(2))
        if count > 0:
            symbols.append(symbol)
            counts.append(count)
    symbols, counts = _canonical_symbol_counts(symbols, counts)
    if not symbols:
        raise ValueError("compact plan contains no supported element-count pairs")
    num_atoms = sum(counts)
    if num_atoms < 1 or num_atoms > int(max_atoms):
        raise ValueError(f"compact plan atom count {num_atoms} outside 1..{max_atoms}")

    atom_types = [SYMBOL_TO_Z[symbol] for symbol, count in zip(symbols, counts) for _ in range(int(count))]
    reduced_elems, reduced_counts = reduced_composition(atom_types)
    reduced_symbols = list(element_symbols(reduced_elems))
    classification = _safe_smact_classification(symbols, counts)
    oxidation = classification.get("oxidation_states")
    plan: Dict[str, Any] = {
        "plan_state_version": PLAN_STATE_VERSION,
        "N": int(num_atoms),
        "generated_N": int(fields["N"]) if fields.get("N", "").isdigit() else None,
        "elements": symbols,
        "counts": counts,
        "formula": _formula_from_symbols(symbols, counts),
        "reduced_formula": _formula_from_symbols(reduced_symbols, reduced_counts),
        "charge_bucket": charge_bucket_from_classification(classification),
        "oxidation_candidates": "unknown" if oxidation is None else list(oxidation),
        "anion_framework": anion_framework_from_symbols(symbols),
        "lattice_system": _enum_or_default(fields.get("LS"), ALLOWED_LATTICE_SYSTEMS, "triclinic"),
        "spacegroup_bucket": _enum_or_default(fields.get("SG"), ALLOWED_SPACEGROUP_BUCKETS, "sg_001_002"),
        "volume_per_atom_bin": _volume_bin_or_default(fields.get("VP")),
        "validator": classification,
        "compact_fields": fields,
    }
    plan["prototype_key"] = prototype_key(plan)
    return plan


def parse_atomseq_plan_state(text: str, *, max_atoms: int = 20) -> Dict[str, Any]:
    fields = _parse_compact_fields(text)
    atom_field = fields.get("A", "")
    atom_symbols = [match.group(0) for match in re.finditer(r"[A-Z][a-z]?", atom_field)]
    atom_symbols = [symbol for symbol in atom_symbols if symbol in SYMBOL_TO_Z]
    if not atom_symbols:
        raise ValueError("atom-sequence plan contains no supported atom symbols")
    num_atoms = len(atom_symbols)
    if num_atoms < 1 or num_atoms > int(max_atoms):
        raise ValueError(f"atom-sequence plan atom count {num_atoms} outside 1..{max_atoms}")

    symbols, counts = _full_composition_from_species(atom_symbols)
    atom_types = [SYMBOL_TO_Z[symbol] for symbol in atom_symbols]
    reduced_elems, reduced_counts = reduced_composition(atom_types)
    reduced_symbols = list(element_symbols(reduced_elems))
    classification = _safe_smact_classification(symbols, counts)
    oxidation = classification.get("oxidation_states")
    plan: Dict[str, Any] = {
        "plan_state_version": PLAN_STATE_VERSION,
        "N": int(num_atoms),
        "elements": symbols,
        "counts": counts,
        "formula": _formula_from_symbols(symbols, counts),
        "reduced_formula": _formula_from_symbols(reduced_symbols, reduced_counts),
        "charge_bucket": charge_bucket_from_classification(classification),
        "oxidation_candidates": "unknown" if oxidation is None else list(oxidation),
        "anion_framework": anion_framework_from_symbols(symbols),
        "lattice_system": _enum_or_default(fields.get("LS"), ALLOWED_LATTICE_SYSTEMS, "triclinic"),
        "spacegroup_bucket": _enum_or_default(fields.get("SG"), ALLOWED_SPACEGROUP_BUCKETS, "sg_001_002"),
        "volume_per_atom_bin": _volume_bin_or_default(fields.get("VP")),
        "validator": classification,
        "atom_sequence": atom_symbols,
        "compact_fields": fields,
    }
    if fields.get("N", "").isdigit():
        plan["generated_N"] = int(fields["N"])
    plan["prototype_key"] = prototype_key(plan)
    return plan


def parse_atomslots_plan_state(text: str, *, max_atoms: int = 20) -> Dict[str, Any]:
    fields = _parse_compact_fields(text)
    slot_field = fields.get("S", "")
    raw_slots = [chunk.strip() for chunk in str(slot_field).split(",") if chunk.strip()]
    if not raw_slots:
        raise ValueError("atom-slots plan contains no slots")
    if len(raw_slots) > int(max_atoms):
        raise ValueError(f"atom-slots plan slot count {len(raw_slots)} outside 1..{max_atoms}")

    empty_tokens = {"_", "-", "PAD", "EMPTY", "empty", "none", "None", "NULL", "null"}
    atom_symbols: list[str] = []
    for token in raw_slots:
        if token in empty_tokens:
            continue
        if not re.fullmatch(r"[A-Z][a-z]?", token):
            raise ValueError(f"atom-slots plan contains invalid slot token {token!r}")
        if token not in SYMBOL_TO_Z:
            raise ValueError(f"atom-slots plan contains unsupported element {token!r}")
        atom_symbols.append(token)
    if not atom_symbols:
        raise ValueError("atom-slots plan contains no supported atom symbols")
    num_atoms = len(atom_symbols)
    if num_atoms < 1 or num_atoms > int(max_atoms):
        raise ValueError(f"atom-slots plan atom count {num_atoms} outside 1..{max_atoms}")

    symbols, counts = _full_composition_from_species(atom_symbols)
    atom_types = [SYMBOL_TO_Z[symbol] for symbol in atom_symbols]
    reduced_elems, reduced_counts = reduced_composition(atom_types)
    reduced_symbols = list(element_symbols(reduced_elems))
    classification = _safe_smact_classification(symbols, counts)
    oxidation = classification.get("oxidation_states")
    plan: Dict[str, Any] = {
        "plan_state_version": PLAN_STATE_VERSION,
        "N": int(num_atoms),
        "elements": symbols,
        "counts": counts,
        "formula": _formula_from_symbols(symbols, counts),
        "reduced_formula": _formula_from_symbols(reduced_symbols, reduced_counts),
        "charge_bucket": charge_bucket_from_classification(classification),
        "oxidation_candidates": "unknown" if oxidation is None else list(oxidation),
        "anion_framework": anion_framework_from_symbols(symbols),
        "lattice_system": _enum_or_default(fields.get("LS"), ALLOWED_LATTICE_SYSTEMS, "triclinic"),
        "spacegroup_bucket": _enum_or_default(fields.get("SG"), ALLOWED_SPACEGROUP_BUCKETS, "sg_001_002"),
        "volume_per_atom_bin": _volume_bin_or_default(fields.get("VP")),
        "validator": classification,
        "atom_slots": raw_slots,
        "atom_sequence": atom_symbols,
        "compact_fields": fields,
    }
    if fields.get("N", "").isdigit():
        plan["generated_N"] = int(fields["N"])
    plan["prototype_key"] = prototype_key(plan)
    return plan


def parse_atomfields_plan_state(text: str, *, max_atoms: int = 20) -> Dict[str, Any]:
    first_line = str(text).strip().splitlines()[0] if str(text).strip() else ""
    fields = _parse_compact_fields(text)
    seen_slot_keys: set[str] = set()
    for chunk in first_line.split(";"):
        if "=" not in chunk:
            continue
        key = chunk.split("=", 1)[0].strip().upper()
        match = re.fullmatch(r"S(\d{2})", key)
        if not match:
            continue
        index = int(match.group(1))
        if index < 1 or index > int(max_atoms):
            raise ValueError(f"atom-fields plan contains out-of-range slot field {key}")
        if key in seen_slot_keys:
            raise ValueError(f"atom-fields plan duplicates slot field {key}")
        seen_slot_keys.add(key)

    expected_keys = [f"S{idx:02d}" for idx in range(1, int(max_atoms) + 1)]
    missing = [key for key in expected_keys if key not in fields]
    if missing:
        raise ValueError(f"atom-fields plan missing slot fields {','.join(missing[:4])}")

    empty_tokens = {"_", "-", "PAD", "EMPTY", "empty", "none", "None", "NULL", "null"}
    atom_symbols: list[str] = []
    raw_slots: list[str] = []
    for key in expected_keys:
        token = str(fields.get(key, "")).strip()
        raw_slots.append(token)
        if token in empty_tokens:
            continue
        if not re.fullmatch(r"[A-Z][a-z]?", token):
            raise ValueError(f"atom-fields plan contains invalid slot token {token!r} at {key}")
        if token not in SYMBOL_TO_Z:
            raise ValueError(f"atom-fields plan contains unsupported element {token!r} at {key}")
        atom_symbols.append(token)
    if not atom_symbols:
        raise ValueError("atom-fields plan contains no supported atom symbols")
    num_atoms = len(atom_symbols)
    if num_atoms < 1 or num_atoms > int(max_atoms):
        raise ValueError(f"atom-fields plan atom count {num_atoms} outside 1..{max_atoms}")

    symbols, counts = _full_composition_from_species(atom_symbols)
    atom_types = [SYMBOL_TO_Z[symbol] for symbol in atom_symbols]
    reduced_elems, reduced_counts = reduced_composition(atom_types)
    reduced_symbols = list(element_symbols(reduced_elems))
    classification = _safe_smact_classification(symbols, counts)
    oxidation = classification.get("oxidation_states")
    plan: Dict[str, Any] = {
        "plan_state_version": PLAN_STATE_VERSION,
        "N": int(num_atoms),
        "elements": symbols,
        "counts": counts,
        "formula": _formula_from_symbols(symbols, counts),
        "reduced_formula": _formula_from_symbols(reduced_symbols, reduced_counts),
        "charge_bucket": charge_bucket_from_classification(classification),
        "oxidation_candidates": "unknown" if oxidation is None else list(oxidation),
        "anion_framework": anion_framework_from_symbols(symbols),
        "lattice_system": _enum_or_default(fields.get("LS"), ALLOWED_LATTICE_SYSTEMS, "triclinic"),
        "spacegroup_bucket": _enum_or_default(fields.get("SG"), ALLOWED_SPACEGROUP_BUCKETS, "sg_001_002"),
        "volume_per_atom_bin": _volume_bin_or_default(fields.get("VP")),
        "validator": classification,
        "atom_slots": raw_slots,
        "atom_sequence": atom_symbols,
        "compact_fields": fields,
    }
    plan["prototype_key"] = prototype_key(plan)
    return plan


def parse_countfields_plan_state(text: str, *, max_atoms: int = 20, max_pairs: int = 7) -> Dict[str, Any]:
    first_line = str(text).strip().splitlines()[0] if str(text).strip() else ""
    fields = _parse_compact_fields(text)
    seen_pair_keys: set[str] = set()
    for chunk in first_line.split(";"):
        if "=" not in chunk:
            continue
        key = chunk.split("=", 1)[0].strip().upper()
        match = re.fullmatch(r"P(\d{2})", key)
        if not match:
            continue
        index = int(match.group(1))
        if index < 1 or index > int(max_pairs):
            raise ValueError(f"count-fields plan contains out-of-range pair field {key}")
        if key in seen_pair_keys:
            raise ValueError(f"count-fields plan duplicates pair field {key}")
        seen_pair_keys.add(key)

    expected_keys = [f"P{idx:02d}" for idx in range(1, int(max_pairs) + 1)]
    missing = [key for key in expected_keys if key not in fields]
    if missing:
        raise ValueError(f"count-fields plan missing pair fields {','.join(missing[:4])}")

    raw_pairs: list[str] = []
    symbols_raw: list[str] = []
    counts_raw: list[int] = []
    for key in expected_keys:
        token = str(fields.get(key, "")).strip().upper()
        raw_pairs.append(token)
        match = re.fullmatch(r"Z(\d{3}):C(\d{3})", token)
        if not match:
            raise ValueError(f"count-fields plan contains invalid pair token {token!r} at {key}")
        z_value = int(match.group(1))
        count_value = int(match.group(2))
        if z_value == 0 and count_value == 0:
            continue
        if z_value == 0 or count_value <= 0:
            raise ValueError(f"count-fields plan contains inconsistent empty/nonempty pair {token!r} at {key}")
        if z_value not in Z_TO_SYMBOL:
            raise ValueError(f"count-fields plan contains unsupported atomic number {z_value} at {key}")
        symbols_raw.append(Z_TO_SYMBOL[z_value])
        counts_raw.append(count_value)

    symbols, counts = _canonical_symbol_counts(symbols_raw, counts_raw)
    if not symbols:
        raise ValueError("count-fields plan contains no supported element-count pairs")
    num_atoms = sum(counts)
    if num_atoms < 1 or num_atoms > int(max_atoms):
        raise ValueError(f"count-fields plan atom count {num_atoms} outside 1..{max_atoms}")

    atom_types = [SYMBOL_TO_Z[symbol] for symbol, count in zip(symbols, counts) for _ in range(int(count))]
    reduced_elems, reduced_counts = reduced_composition(atom_types)
    reduced_symbols = list(element_symbols(reduced_elems))
    classification = _safe_smact_classification(symbols, counts)
    oxidation = classification.get("oxidation_states")
    lattice_code = str(fields.get("LS", "")).strip().upper()
    sg_code = str(fields.get("SG", "")).strip().upper()
    plan: Dict[str, Any] = {
        "plan_state_version": PLAN_STATE_VERSION,
        "N": int(num_atoms),
        "elements": symbols,
        "counts": counts,
        "formula": _formula_from_symbols(symbols, counts),
        "reduced_formula": _formula_from_symbols(reduced_symbols, reduced_counts),
        "charge_bucket": charge_bucket_from_classification(classification),
        "oxidation_candidates": "unknown" if oxidation is None else list(oxidation),
        "anion_framework": anion_framework_from_symbols(symbols),
        "lattice_system": CODE_TO_LATTICE_SYSTEM.get(lattice_code, "triclinic"),
        "spacegroup_bucket": CODE_TO_SPACEGROUP_BUCKET.get(sg_code, "sg_001_002"),
        "volume_per_atom_bin": _volume_code_to_bin(fields.get("VP")),
        "validator": classification,
        "count_fields": raw_pairs,
        "compact_fields": fields,
    }
    plan["prototype_key"] = prototype_key(plan)
    return plan


def parse_countvalence_plan_state(text: str, *, max_atoms: int = 20, max_pairs: int = 7) -> Dict[str, Any]:
    first_line = str(text).strip().splitlines()[0] if str(text).strip() else ""
    fields = _parse_compact_fields(text)
    seen_pair_keys: set[str] = set()
    for chunk in first_line.split(";"):
        if "=" not in chunk:
            continue
        key = chunk.split("=", 1)[0].strip().upper()
        match = re.fullmatch(r"P(\d{2})", key)
        if not match:
            continue
        index = int(match.group(1))
        if index < 1 or index > int(max_pairs):
            raise ValueError(f"count-valence plan contains out-of-range pair field {key}")
        if key in seen_pair_keys:
            raise ValueError(f"count-valence plan duplicates pair field {key}")
        seen_pair_keys.add(key)

    expected_keys = [f"P{idx:02d}" for idx in range(1, int(max_pairs) + 1)]
    missing = [key for key in expected_keys if key not in fields]
    if missing:
        raise ValueError(f"count-valence plan missing pair fields {','.join(missing[:4])}")

    raw_pairs: list[str] = []
    species_by_key: OrderedDict[tuple[str, int | None], int] = OrderedDict()
    for key in expected_keys:
        token = str(fields.get(key, "")).strip().upper()
        raw_pairs.append(token)
        match = re.fullmatch(r"Z(\d{3}):C(\d{3}):(Q[PMZU]\d{2})", token)
        if not match:
            raise ValueError(f"count-valence plan contains invalid pair token {token!r} at {key}")
        z_value = int(match.group(1))
        count_value = int(match.group(2))
        oxidation_value = _oxidation_code_to_int(match.group(3))
        if z_value == 0 and count_value == 0 and oxidation_value in (0, None):
            continue
        if z_value == 0 or count_value <= 0:
            raise ValueError(f"count-valence plan contains inconsistent empty/nonempty pair {token!r} at {key}")
        if z_value not in Z_TO_SYMBOL:
            raise ValueError(f"count-valence plan contains unsupported atomic number {z_value} at {key}")
        symbol = Z_TO_SYMBOL[z_value]
        species_key = (symbol, oxidation_value)
        species_by_key[species_key] = species_by_key.get(species_key, 0) + count_value

    species_rows = sorted(
        (
            (symbol, int(count), oxidation)
            for (symbol, oxidation), count in species_by_key.items()
        ),
        key=lambda item: (
            SYMBOL_TO_Z.get(item[0], 10_000),
            10_000 if item[2] is None else int(item[2]),
        ),
    )
    states_by_symbol: OrderedDict[str, set[int | None]] = OrderedDict()
    counts_by_symbol: Counter[str] = Counter()
    for symbol, count, oxidation in species_rows:
        states_by_symbol.setdefault(symbol, set()).add(oxidation)
        counts_by_symbol[symbol] += int(count)
    for symbol, states in states_by_symbol.items():
        if None in states and len(states) > 1:
            raise ValueError(f"count-valence plan mixes known and unknown oxidation for {symbol}")
    symbols = sorted(counts_by_symbol, key=lambda value: SYMBOL_TO_Z.get(value, 10_000))
    counts = [int(counts_by_symbol[symbol]) for symbol in symbols]
    generated_oxidation: list[int | str | None] = []
    for symbol in symbols:
        states = states_by_symbol[symbol]
        if len(states) == 1:
            value = next(iter(states))
            generated_oxidation.append(None if value is None else int(value))
        else:
            generated_oxidation.append("mixed")
    if not symbols:
        raise ValueError("count-valence plan contains no supported element-count pairs")
    num_atoms = sum(counts)
    if num_atoms < 1 or num_atoms > int(max_atoms):
        raise ValueError(f"count-valence plan atom count {num_atoms} outside 1..{max_atoms}")
    charge_sum = None
    if all(oxidation is not None for _, _, oxidation in species_rows):
        charge_sum = int(
            sum(
                int(count) * int(oxidation)
                for _, count, oxidation in species_rows
                if oxidation is not None
            )
        )

    atom_types = [SYMBOL_TO_Z[symbol] for symbol, count in zip(symbols, counts) for _ in range(int(count))]
    reduced_elems, reduced_counts = reduced_composition(atom_types)
    reduced_symbols = list(element_symbols(reduced_elems))
    classification = _safe_smact_classification(symbols, counts)
    generated_bucket = CODE_TO_CHARGE_BUCKET.get(str(fields.get("CB", "")).strip().upper(), "validator_unavailable")
    if charge_sum is None:
        expected_bucket = charge_bucket_from_classification(classification)
    elif charge_sum != 0:
        expected_bucket = "charge_fail"
    elif all(int(oxidation or 0) == 0 for _, _, oxidation in species_rows):
        expected_bucket = "single_element" if len(symbols) == 1 else "all_metal"
    else:
        expected_bucket = "neutral_plausible"
    lattice_code = str(fields.get("LS", "")).strip().upper()
    sg_code = str(fields.get("SG", "")).strip().upper()
    plan: Dict[str, Any] = {
        "plan_state_version": PLAN_STATE_VERSION,
        "N": int(num_atoms),
        "elements": symbols,
        "counts": counts,
        "formula": _formula_from_symbols(symbols, counts),
        "reduced_formula": _formula_from_symbols(reduced_symbols, reduced_counts),
        "charge_bucket": expected_bucket,
        "generated_charge_bucket": generated_bucket,
        "charge_bucket_match": generated_bucket == expected_bucket,
        "oxidation_candidates": generated_oxidation,
        "anion_framework": anion_framework_from_symbols(symbols),
        "lattice_system": CODE_TO_LATTICE_SYSTEM.get(lattice_code, "triclinic"),
        "spacegroup_bucket": CODE_TO_SPACEGROUP_BUCKET.get(sg_code, "sg_001_002"),
        "volume_per_atom_bin": _volume_code_to_bin(fields.get("VP")),
        "validator": classification,
        "generated_oxidation_states": generated_oxidation,
        "generated_oxidation_states_by_species": [
            None if oxidation is None else int(oxidation)
            for _, _, oxidation in species_rows
        ],
        "valence_species": [
            {
                "element": symbol,
                "count": int(count),
                "oxidation_state": None if oxidation is None else int(oxidation),
            }
            for symbol, count, oxidation in species_rows
        ],
        "generated_charge_sum": charge_sum,
        "generated_charge_sum_known": charge_sum is not None,
        "count_valence_validator": {
            "known": charge_sum is not None,
            "charge_sum": charge_sum,
            "neutral": charge_sum == 0 if charge_sum is not None else None,
            "mixed_valence": any(value == "mixed" for value in generated_oxidation),
        },
        "count_valence_fields": raw_pairs,
        "compact_fields": fields,
    }
    plan["prototype_key"] = prototype_key(plan)
    return plan


def plan_state_from_arrays(
    arrays: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    species = [str(symbol) for symbol in arrays["species"]]
    symbols, full_counts = _full_composition_from_species(species)
    atom_types = [SYMBOL_TO_Z[symbol] for symbol in species if symbol in SYMBOL_TO_Z]
    reduced_elems, reduced_counts = reduced_composition(atom_types)
    reduced_symbols = list(element_symbols(reduced_elems))
    classification = _safe_smact_classification(symbols, full_counts)
    oxidation = classification.get("oxidation_states")
    plan: Dict[str, Any] = {
        "plan_state_version": PLAN_STATE_VERSION,
        "N": int(arrays["num_atoms"]),
        "elements": symbols,
        "counts": full_counts,
        "formula": _formula_from_symbols(symbols, full_counts),
        "reduced_formula": _formula_from_symbols(reduced_symbols, reduced_counts),
        "charge_bucket": charge_bucket_from_classification(classification),
        "oxidation_candidates": "unknown" if oxidation is None else list(oxidation),
        "anion_framework": anion_framework_from_symbols(symbols),
        "lattice_system": lattice_system_from_lattice(arrays["lengths"], arrays["angles"]),
        "spacegroup_bucket": spacegroup_bucket(metadata),
        "volume_per_atom_bin": volume_per_atom_bin(arrays["lengths"], arrays["angles"], int(arrays["num_atoms"])),
        "validator": classification,
    }
    plan["prototype_key"] = prototype_key(plan)
    if metadata:
        plan["metadata"] = {
            key: metadata[key]
            for key in (
                "material_id",
                "pretty_formula",
                "e_above_hull",
                "spacegroup.number",
                "spacegroup.number.conv",
            )
            if key in metadata
        }
    return plan


def canonical_plan_state(plan: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: plan.get(key) for key in PLAN_STATE_FIELDS}


def plan_state_to_json(plan: Mapping[str, Any], *, canonical_only: bool = True) -> str:
    payload = canonical_plan_state(plan) if canonical_only else dict(plan)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_plan_state_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("plan_state JSON object not found")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("plan_state JSON must decode to an object")
    return payload


def validate_plan_state(plan: Mapping[str, Any], *, max_atoms: int = 20) -> PlanValidation:
    try:
        num_atoms = int(plan.get("N"))
    except Exception:
        num_atoms = -1
    generated_raw = plan.get("generated_N", num_atoms)
    try:
        generated_num_atoms = int(generated_raw)
    except Exception:
        generated_num_atoms = -1
    elements = plan.get("elements")
    counts = plan.get("counts")
    valid_elements = (
        isinstance(elements, list)
        and len(elements) > 0
        and all(isinstance(symbol, str) and symbol in SYMBOL_TO_Z for symbol in elements)
    )
    valid_counts = (
        isinstance(counts, list)
        and isinstance(elements, list)
        and len(counts) == len(elements)
        and all(isinstance(count, int) and count > 0 for count in counts)
        and sum(int(count) for count in counts) == num_atoms
    )
    formula = plan.get("formula")
    expected_formula = _formula_from_symbols(elements, counts) if valid_elements and valid_counts else None
    valid_n_range = 1 <= num_atoms <= int(max_atoms)
    valid_generated_n = generated_num_atoms == num_atoms
    return PlanValidation(
        valid_N=valid_n_range and valid_generated_n,
        valid_generated_N=valid_generated_n,
        valid_formula=isinstance(formula, str) and formula == expected_formula,
        valid_counts=valid_counts,
        valid_elements=valid_elements,
    )


def build_plan_prompt() -> str:
    return PLAN_STATE_PROMPT


def build_compact_plan_prompt() -> str:
    return PLAN_STATE_COMPACT_PROMPT


def build_atomseq_plan_prompt() -> str:
    return PLAN_STATE_ATOMSEQ_PROMPT


def build_atomslots_plan_prompt() -> str:
    return PLAN_STATE_ATOMSLOTS_PROMPT


def build_atomfields_plan_prompt() -> str:
    return PLAN_STATE_ATOMFIELDS_PROMPT


def build_countfields_plan_prompt() -> str:
    return PLAN_STATE_COUNTFIELDS_PROMPT


def build_countvalence_plan_prompt() -> str:
    return PLAN_STATE_COUNTVALENCE_PROMPT


def build_compact_plan_repair_prompt(*, visible_plan: str, violation_labels: Sequence[str]) -> str:
    labels = [str(label).strip() for label in violation_labels if str(label).strip()]
    return PLAN_STATE_COMPACT_REPAIR_PROMPT_TEMPLATE.format(
        violation_labels=json.dumps(labels, sort_keys=True),
        visible_plan=str(visible_plan).strip().splitlines()[0] if str(visible_plan).strip() else "",
    )


def build_body_prompt(plan: Mapping[str, Any]) -> str:
    return (
        "Generate only the exact-length dynamic crystal body for this fixed plan_state. "
        "The first token must match N and the element multiset must match elements/counts.\n"
        f"plan_state: {plan_state_to_json(plan)}\n"
        "dynamic_crystal_body:"
    )


def hard_anchor_plan_state(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep composition/cardinality anchors while removing uncertain soft fields."""

    payload = canonical_plan_state(plan)
    payload.update(
        {
            "charge_bucket": "unknown",
            "oxidation_candidates": "unknown",
            "anion_framework": "unknown",
            "lattice_system": "unknown",
            "spacegroup_bucket": "sg_unknown",
            "volume_per_atom_bin": "volpa_unknown",
            "prototype_key": "unknown",
        }
    )
    return payload


def build_hard_anchor_body_prompt(plan: Mapping[str, Any]) -> str:
    return build_body_prompt(hard_anchor_plan_state(plan))


__all__ = [
    "PLAN_STATE_FIELDS",
    "PLAN_STATE_PROMPT",
    "PLAN_STATE_VERSION",
    "PlanValidation",
    "build_body_prompt",
    "build_hard_anchor_body_prompt",
    "build_atomfields_plan_prompt",
    "build_atomseq_plan_prompt",
    "build_atomslots_plan_prompt",
    "build_compact_plan_prompt",
    "build_compact_plan_repair_prompt",
    "build_countfields_plan_prompt",
    "build_countvalence_plan_prompt",
    "build_plan_prompt",
    "canonical_plan_state",
    "hard_anchor_plan_state",
    "parse_compact_plan_state",
    "parse_atomfields_plan_state",
    "parse_atomseq_plan_state",
    "parse_atomslots_plan_state",
    "parse_countfields_plan_state",
    "parse_countvalence_plan_state",
    "parse_plan_state_json",
    "normalize_compact_plan_for_repair_target",
    "plan_state_from_arrays",
    "plan_state_to_atomfields",
    "plan_state_to_atomseq",
    "plan_state_to_atomslots",
    "plan_state_to_compact",
    "plan_state_to_countfields",
    "plan_state_to_countvalencefields",
    "plan_state_to_json",
    "validate_plan_state",
]
