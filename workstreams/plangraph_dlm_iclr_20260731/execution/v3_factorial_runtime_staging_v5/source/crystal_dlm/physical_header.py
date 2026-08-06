"""Compact physical-header extension for fixed-slot crystal DLM answers.

The representation keeps the proven 107-token fixed-slot body unchanged and
prepends ten global state tokens:

    <H_START> <NE_*> <ANION_*> <COMP_*> <COUNT_PATTERN_*>
    <OX_*> <LAT_SYS_*> <VOLPA_*> <HS_*> <H_END>

The header is intentionally derived from the same structure encoded by the body
so it can be used as a denoising state without leaking external labels.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Dict, List, Mapping, Sequence

from crystal_dlm.composition_validity import composition_record
from crystal_dlm.fixed_slot import (
    ANSWER_TOKEN_COUNT,
    FixedSlotConfig,
    arrays_to_tokens,
    parse_fixed_slot_answer,
    tokens_to_arrays,
)

PHYSICAL_HEADER_TOKEN_COUNT = 10
PHYSICAL_HEADER_ANSWER_TOKEN_COUNT = PHYSICAL_HEADER_TOKEN_COUNT + ANSWER_TOKEN_COUNT
PHYSICAL_HEADER_BODY_OFFSET = PHYSICAL_HEADER_TOKEN_COUNT

PHYSICAL_HEADER_CANONICAL_PROMPT = (
    "Generate compact physical header tokens followed by the fixed-slot crystal "
    "structure tokens:"
)

PHYSICAL_HEADER_PROMPT_POOL = [
    PHYSICAL_HEADER_CANONICAL_PROMPT,
    (
        "Generate a physical header for the crystal, then generate the lattice, "
        "atom species, and fractional coordinates in fixed-slot format:"
    ),
    (
        "Provide global physical state tokens first, followed by a fixed-slot "
        "crystal structure:"
    ),
]

NE_TOKENS = [f"<NE_{i:03d}>" for i in range(1, 9)] + ["<NE_009P>"]
ANION_TOKENS = [
    "<ANION_NONE>",
    "<ANION_O>",
    "<ANION_F>",
    "<ANION_CL>",
    "<ANION_BR>",
    "<ANION_I>",
    "<ANION_S>",
    "<ANION_SE>",
    "<ANION_TE>",
    "<ANION_N>",
    "<ANION_P>",
    "<ANION_AS>",
    "<ANION_MIXED>",
    "<ANION_OTHER>",
]
COMP_TOKENS = [
    "<COMP_STRICT>",
    "<COMP_ALL_METAL>",
    "<COMP_SINGLE>",
    "<COMP_INVALID>",
]
COUNT_PATTERN_TOKENS = [
    "<COUNT_PATTERN_SINGLE>",
    "<COUNT_PATTERN_1_1>",
    "<COUNT_PATTERN_1_2>",
    "<COUNT_PATTERN_1_3>",
    "<COUNT_PATTERN_1_4>",
    "<COUNT_PATTERN_2_3>",
    "<COUNT_PATTERN_1_1_1>",
    "<COUNT_PATTERN_1_1_2>",
    "<COUNT_PATTERN_1_1_3>",
    "<COUNT_PATTERN_1_1_4>",
    "<COUNT_PATTERN_1_1_6>",
    "<COUNT_PATTERN_1_2_2>",
    "<COUNT_PATTERN_1_2_3>",
    "<COUNT_PATTERN_1_2_4>",
    "<COUNT_PATTERN_1_3_3>",
    "<COUNT_PATTERN_2_3_4>",
    "<COUNT_PATTERN_GE4>",
    "<COUNT_PATTERN_OTHER>",
]
OX_TOKENS = [
    "<OX_BALANCED>",
    "<OX_ALLOY>",
    "<OX_SINGLE>",
    "<OX_CHARGE_FAIL>",
    "<OX_PAULING_FAIL>",
    "<OX_MISSING>",
    "<OX_INVALID>",
]
LAT_SYS_TOKENS = [
    "<LAT_SYS_CUBIC>",
    "<LAT_SYS_TETR>",
    "<LAT_SYS_ORTH>",
    "<LAT_SYS_HEX>",
    "<LAT_SYS_RHOMBO>",
    "<LAT_SYS_MONO>",
    "<LAT_SYS_TRIC>",
]
VOLPA_TOKENS = [f"<VOLPA_{i:03d}>" for i in range(0, 101)]
HS_TOKENS = ["<HS_LOW>", "<HS_MED>", "<HS_HIGH>"]

HEADER_ALLOWED_TOKEN_STRINGS = [
    ["<H_START>"],
    NE_TOKENS,
    ANION_TOKENS,
    COMP_TOKENS,
    COUNT_PATTERN_TOKENS,
    OX_TOKENS,
    LAT_SYS_TOKENS,
    VOLPA_TOKENS,
    HS_TOKENS,
    ["<H_END>"],
]

HEADER_TOKEN_RE = re.compile(
    r"<(?:H_START|H_END|NE_(?:\d{3}|009P)|ANION_[A-Z]+|COMP_[A-Z_]+|"
    r"COUNT_PATTERN_[A-Z0-9_]+|OX_[A-Z_]+|LAT_SYS_[A-Z]+|VOLPA_\d{3}|HS_[A-Z]+)>"
)

ANION_PRIORITY = ["O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "P", "As"]
ANION_TOKEN_BY_SYMBOL = {
    "O": "<ANION_O>",
    "F": "<ANION_F>",
    "Cl": "<ANION_CL>",
    "Br": "<ANION_BR>",
    "I": "<ANION_I>",
    "S": "<ANION_S>",
    "Se": "<ANION_SE>",
    "Te": "<ANION_TE>",
    "N": "<ANION_N>",
    "P": "<ANION_P>",
    "As": "<ANION_AS>",
}

HIGH_SYMMETRY_COORD_VALUES = (0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0)


def build_physical_header_special_tokens() -> List[str]:
    """Return all additional tokens needed by the physical-header variant."""

    tokens: List[str] = []
    for bucket in HEADER_ALLOWED_TOKEN_STRINGS:
        tokens.extend(bucket)
    return list(dict.fromkeys(tokens))


def physical_header_allowed_token_strings() -> List[List[str]]:
    """Return per-header-position allowed token strings."""

    return [list(bucket) for bucket in HEADER_ALLOWED_TOKEN_STRINGS]


def tokenize_physical_header_text(text: str) -> List[str]:
    """Extract physical-header tokens from decoded text."""

    return HEADER_TOKEN_RE.findall(text)


def _close(value: float, target: float, tol: float) -> bool:
    return abs(float(value) - float(target)) <= float(tol)


def _all_close(values: Sequence[float], target: float, tol: float) -> bool:
    return all(_close(value, target, tol) for value in values)


def _lengths_equal(a: float, b: float, tol: float = 0.11) -> bool:
    return abs(float(a) - float(b)) <= tol


def _volume(lengths: Sequence[float], angles: Sequence[float]) -> float:
    a, b, c = [float(value) for value in lengths]
    alpha, beta, gamma = [math.radians(float(value)) for value in angles]
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    radicand = 1.0 + 2.0 * cos_a * cos_b * cos_g - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g
    return a * b * c * math.sqrt(max(radicand, 0.0))


def _num_elements_token(symbols: Sequence[str]) -> str:
    count = len(set(str(symbol) for symbol in symbols))
    if count >= 9:
        return "<NE_009P>"
    return f"<NE_{max(1, count):03d}>"


def _anion_token(symbols: Sequence[str]) -> str:
    present = [symbol for symbol in ANION_PRIORITY if symbol in set(symbols)]
    if not present:
        return "<ANION_NONE>"
    if len(present) > 1:
        return "<ANION_MIXED>"
    return ANION_TOKEN_BY_SYMBOL.get(present[0], "<ANION_OTHER>")


def _count_pattern_token(symbols: Sequence[str]) -> str:
    counts = sorted(Counter(symbols).values())
    if not counts:
        return "<COUNT_PATTERN_OTHER>"
    if len(counts) == 1:
        return "<COUNT_PATTERN_SINGLE>"
    gcd_value = 0
    for count in counts:
        gcd_value = math.gcd(gcd_value, int(count))
    gcd_value = max(1, gcd_value)
    reduced = tuple(int(count // gcd_value) for count in counts)
    if len(reduced) >= 4:
        return "<COUNT_PATTERN_GE4>"
    token = "<COUNT_PATTERN_" + "_".join(str(value) for value in reduced) + ">"
    return token if token in COUNT_PATTERN_TOKENS else "<COUNT_PATTERN_OTHER>"


def _composition_and_ox_tokens(atom_types: Sequence[int]) -> tuple[str, str, Dict[str, Any]]:
    try:
        record = composition_record(atom_types)
    except Exception as exc:  # Keep data building robust when SMACT is unavailable/misconfigured.
        record = {"comp_valid": False, "reason": "classification_error", "error": str(exc)}
    reason = str(record.get("reason", "invalid"))
    if reason == "charge_neutral_pauling_valid":
        return "<COMP_STRICT>", "<OX_BALANCED>", record
    if reason == "all_metal_shortcut":
        return "<COMP_ALL_METAL>", "<OX_ALLOY>", record
    if reason == "single_element_shortcut":
        return "<COMP_SINGLE>", "<OX_SINGLE>", record
    if reason == "charge_neutrality_fail":
        return "<COMP_INVALID>", "<OX_CHARGE_FAIL>", record
    if reason == "oxidation_state_missing":
        return "<COMP_INVALID>", "<OX_MISSING>", record
    if reason == "pauling_fail_or_ratio_rejected":
        return "<COMP_INVALID>", "<OX_PAULING_FAIL>", record
    return "<COMP_INVALID>", "<OX_INVALID>", record


def _lattice_system_token(lengths: Sequence[float], angles: Sequence[float]) -> str:
    a, b, c = [float(value) for value in lengths]
    alpha, beta, gamma = [float(value) for value in angles]
    all_90 = _all_close((alpha, beta, gamma), 90.0, 1.0)
    all_equal = _lengths_equal(a, b) and _lengths_equal(b, c)
    if all_90 and all_equal:
        return "<LAT_SYS_CUBIC>"
    if all_90 and _lengths_equal(a, b):
        return "<LAT_SYS_TETR>"
    if all_90:
        return "<LAT_SYS_ORTH>"
    if _lengths_equal(a, b) and _close(alpha, 90.0, 1.0) and _close(beta, 90.0, 1.0) and _close(gamma, 120.0, 2.0):
        return "<LAT_SYS_HEX>"
    if all_equal and _close(alpha, beta, 1.0) and _close(beta, gamma, 1.0):
        return "<LAT_SYS_RHOMBO>"
    ninety_count = sum(1 for angle in (alpha, beta, gamma) if _close(angle, 90.0, 1.0))
    if ninety_count >= 2:
        return "<LAT_SYS_MONO>"
    return "<LAT_SYS_TRIC>"


def _volpa_token(lengths: Sequence[float], angles: Sequence[float], num_atoms: int) -> str:
    if int(num_atoms) <= 0:
        return "<VOLPA_000>"
    value = _volume(lengths, angles) / float(num_atoms)
    bin_value = int(round(max(0.0, min(100.0, value))))
    return f"<VOLPA_{bin_value:03d}>"


def high_symmetry_coord_fraction(frac_coords: Sequence[Sequence[float]]) -> float:
    if not frac_coords:
        return 0.0
    high = 0
    total = 0
    for coord in frac_coords:
        for value in coord:
            wrapped = float(value) % 1.0
            total += 1
            if any(_close(wrapped, target % 1.0, 0.011) for target in HIGH_SYMMETRY_COORD_VALUES):
                high += 1
    return high / max(1, total)


def _high_symmetry_token(arrays: Mapping[str, Any], lattice_token: str) -> str:
    coord_fraction = high_symmetry_coord_fraction(arrays["frac_coords"])
    lengths = [float(value) for value in arrays["lengths"]]
    angles = [float(value) for value in arrays["angles"]]
    all_equal = max(lengths) - min(lengths) <= 0.11
    all_90 = _all_close(angles, 90.0, 1.0)
    if coord_fraction >= 0.60 or (all_equal and all_90) or lattice_token == "<LAT_SYS_CUBIC>":
        return "<HS_HIGH>"
    if coord_fraction >= 0.35 or all_90:
        return "<HS_MED>"
    return "<HS_LOW>"


def build_physical_header_tokens_from_arrays(arrays: Mapping[str, Any]) -> tuple[List[str], Dict[str, Any]]:
    """Derive physical-header tokens from parsed fixed-slot arrays."""

    symbols = [str(symbol) for symbol in arrays["species"]]
    atom_types = [int(value) for value in arrays["atom_types"]]
    lengths = [float(value) for value in arrays["lengths"]]
    angles = [float(value) for value in arrays["angles"]]
    num_atoms = int(arrays["num_atoms"])
    comp_token, ox_token, comp_record = _composition_and_ox_tokens(atom_types)
    lat_token = _lattice_system_token(lengths, angles)
    hs_token = _high_symmetry_token(arrays, lat_token)
    tokens = [
        "<H_START>",
        _num_elements_token(symbols),
        _anion_token(symbols),
        comp_token,
        _count_pattern_token(symbols),
        ox_token,
        lat_token,
        _volpa_token(lengths, angles, num_atoms),
        hs_token,
        "<H_END>",
    ]
    labels = {
        "tokens": tokens,
        "num_elements_token": tokens[1],
        "anion_token": tokens[2],
        "composition_token": tokens[3],
        "count_pattern_token": tokens[4],
        "oxidation_token": tokens[5],
        "lattice_system_token": tokens[6],
        "volume_per_atom_token": tokens[7],
        "high_symmetry_token": tokens[8],
        "composition_record": comp_record,
        "high_symmetry_coord_fraction": high_symmetry_coord_fraction(arrays["frac_coords"]),
        "volume_per_atom": None if num_atoms <= 0 else _volume(lengths, angles) / float(num_atoms),
    }
    return tokens, labels


def build_physical_header_tokens(
    lengths: Sequence[float],
    angles: Sequence[float],
    species: Sequence[str],
    frac_coords: Sequence[Sequence[float]],
    config: FixedSlotConfig = FixedSlotConfig(),
) -> tuple[List[str], Dict[str, Any]]:
    body_tokens, _ = arrays_to_tokens(lengths, angles, species, frac_coords, config=config)
    arrays = tokens_to_arrays(body_tokens, config=config, strict=True)
    return build_physical_header_tokens_from_arrays(arrays)


def prepend_physical_header_to_answer(
    fixed_slot_answer: str,
    *,
    separator: str = "",
    config: FixedSlotConfig = FixedSlotConfig(),
) -> tuple[str, Dict[str, Any]]:
    """Return ``header + fixed body`` answer text and derived header labels."""

    arrays = parse_fixed_slot_answer(fixed_slot_answer, config=config, strict=True)
    header_tokens, labels = build_physical_header_tokens_from_arrays(arrays)
    header_text = separator.join(header_tokens)
    return header_text + separator + fixed_slot_answer, labels


def parse_physical_header_answer(
    text: str,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> Dict[str, Any]:
    """Parse a generated physical-header answer.

    The fixed-slot body is parsed with the existing robust fixed-slot parser;
    header tokens are returned for diagnostics and may be incomplete for failed
    samples.
    """

    header_tokens = tokenize_physical_header_text(text)
    arrays = parse_fixed_slot_answer(text, config=config)
    arrays["physical_header_tokens"] = header_tokens[:PHYSICAL_HEADER_TOKEN_COUNT]
    return arrays


__all__ = [
    "PHYSICAL_HEADER_TOKEN_COUNT",
    "PHYSICAL_HEADER_ANSWER_TOKEN_COUNT",
    "PHYSICAL_HEADER_BODY_OFFSET",
    "PHYSICAL_HEADER_CANONICAL_PROMPT",
    "PHYSICAL_HEADER_PROMPT_POOL",
    "build_physical_header_special_tokens",
    "physical_header_allowed_token_strings",
    "tokenize_physical_header_text",
    "build_physical_header_tokens",
    "build_physical_header_tokens_from_arrays",
    "prepend_physical_header_to_answer",
    "parse_physical_header_answer",
    "high_symmetry_coord_fraction",
]
