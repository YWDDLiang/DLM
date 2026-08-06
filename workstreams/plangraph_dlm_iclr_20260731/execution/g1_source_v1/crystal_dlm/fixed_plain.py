"""Fixed-schema crystal text without tokenizer special tokens.

This representation keeps the successful fixed-slot semantics but writes each
semantic token as ordinary text, for example ``N_012`` instead of ``<N_012>``.
Sampling is intended to be staged: atom count, lattice, elements, coordinates.
The inactive slots are filled deterministically by the sampler.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from crystal_dlm.cif_lite import CifLiteError, pbc_coordinate_key
from crystal_dlm.fixed_slot import (
    FixedSlotConfig,
    FixedSlotError,
    SYMBOL_TO_Z,
    Z_TO_SYMBOL,
    arrays_to_structure,
    arrays_to_tokens,
    arrays_to_torch_payload,
    metadata_from_csv_row,
    tokens_to_arrays,
    write_json,
)
from crystal_dlm.lattice_geometry import lattice_angle_rad


FIXED_PLAIN_PROMPT_VERSION = "fixed_plain_v1_staged"
FIXED_PLAIN_MODULE_TO_ID = {
    "count": 10,
    "lattice": 11,
    "elements": 12,
    "coords": 13,
    "full": 14,
}
FIXED_PLAIN_COUNT_PROMPT = (
    "Generate exactly one fixed crystal atom-count token. "
    "Use the form N_###, with ### from 001 to 020:\n"
)
FIXED_PLAIN_LATTICE_PROMPT_TEMPLATE = (
    "Atom count token:\n"
    "{count_text}\n"
    "Generate exactly six fixed lattice tokens in this order: "
    "LA_### LB_### LC_### AA_### AB_### AG_###.\n"
)
FIXED_PLAIN_ELEMENTS_PROMPT_TEMPLATE = (
    "Atom count token:\n"
    "{count_text}\n"
    "Lattice tokens:\n"
    "{lattice_text}\n"
    "Generate exactly {num_atoms} element tokens in order. "
    "Use tokens like E_Li, E_V, E_O and no coordinates:\n"
)
FIXED_PLAIN_COORDS_PROMPT_TEMPLATE = (
    "Atom count token:\n"
    "{count_text}\n"
    "Lattice tokens:\n"
    "{lattice_text}\n"
    "Element tokens:\n"
    "{elements_text}\n"
    "Generate exactly {num_atoms} coordinate triples in the same order. "
    "Each triple must be X_### Y_### Z_### with ### from 000 to 100:\n"
)

PLAIN_SCHEMA_TOKEN_RE = re.compile(
    r"\b(?:N_\d{3}|L[ABC]_\d{3}|A[ABG]_\d{3}|S\d{2}|"
    r"E_[A-Z][a-z]?|[XYZ]_\d{1,4}|EMPTY|[XYZ]_PAD)\b"
)


def strip_fixed_token(token: str) -> str:
    token = str(token).strip()
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    return token


def wrap_plain_token(token: str) -> str:
    token = strip_fixed_token(token)
    if token.startswith("<") and token.endswith(">"):
        return token
    return f"<{token}>"


def tokenize_plain_answer(text: str) -> List[str]:
    return PLAIN_SCHEMA_TOKEN_RE.findall(str(text))


def plain_tokens_to_arrays(tokens: Sequence[str], config: FixedSlotConfig = FixedSlotConfig()) -> Dict[str, Any]:
    return tokens_to_arrays([wrap_plain_token(token) for token in tokens], config=config, strict=True)


def arrays_to_plain_tokens(
    lengths: Sequence[float],
    angles: Sequence[float],
    species: Sequence[str],
    frac_coords: Sequence[Sequence[float]],
    config: FixedSlotConfig = FixedSlotConfig(),
) -> Tuple[List[str], Any]:
    fixed_tokens, diagnostics = arrays_to_tokens(lengths, angles, species, frac_coords, config=config)
    return [strip_fixed_token(token) for token in fixed_tokens], diagnostics


def structure_to_plain_answer(
    structure: Any,
    config: FixedSlotConfig = FixedSlotConfig(),
    separator: str = " ",
) -> Tuple[str, Any]:
    species = [site.specie.symbol for site in structure.sites]
    tokens, diagnostics = arrays_to_plain_tokens(
        structure.lattice.abc,
        structure.lattice.angles,
        species,
        structure.frac_coords.tolist(),
        config=config,
    )
    return separator.join(tokens), diagnostics


def split_plain_tokens(tokens: Sequence[str], config: FixedSlotConfig = FixedSlotConfig()) -> Dict[str, Any]:
    token_list = [strip_fixed_token(token) for token in tokens]
    arrays = plain_tokens_to_arrays(token_list, config=config)
    num_atoms = int(arrays["num_atoms"])
    elements: List[str] = []
    coords: List[str] = []
    for slot_index in range(num_atoms):
        base = 7 + slot_index * 5
        elements.append(token_list[base + 1])
        coords.extend(token_list[base + 2 : base + 5])
    return {
        "count_text": token_list[0],
        "lattice_text": " ".join(token_list[1:7]),
        "elements_text": " ".join(elements),
        "coords_text": " ".join(coords),
        "answer": " ".join(token_list),
        "arrays": arrays,
    }


def parse_plain_answer(text: str, config: FixedSlotConfig = FixedSlotConfig()) -> Dict[str, Any]:
    tokens = tokenize_plain_answer(text)
    if len(tokens) < 1 + 6 + config.max_atoms * 5:
        raise FixedSlotError(f"Expected at least {1 + 6 + config.max_atoms * 5} fixed-plain tokens, got {len(tokens)}")
    return plain_tokens_to_arrays(tokens[: 1 + 6 + config.max_atoms * 5], config=config)


def parse_count_text(text: str, config: FixedSlotConfig = FixedSlotConfig()) -> Dict[str, Any]:
    for token in tokenize_plain_answer(text):
        if not token.startswith("N_"):
            continue
        count = int(token.split("_", 1)[1])
        if 1 <= count <= config.max_atoms:
            return {"num_atoms": count, "count_text": f"N_{count:03d}"}
    raise FixedSlotError("Could not parse fixed-plain atom count")


def _lattice_from_tokens(tokens: Sequence[str], config: FixedSlotConfig = FixedSlotConfig()) -> Dict[str, Any]:
    if len(tokens) != 6:
        raise FixedSlotError("Expected six lattice tokens")
    expected = ("LA", "LB", "LC", "AA", "AB", "AG")
    values: List[int] = []
    for token, prefix in zip(tokens, expected):
        if not token.startswith(prefix + "_"):
            raise FixedSlotError(f"Expected {prefix}_###, got {token!r}")
        value = int(token.split("_", 1)[1])
        values.append(value)
    if any(value <= 0 for value in values[:3]):
        raise FixedSlotError(f"Lattice length bins must be positive: {tokens}")
    if any(not config.angle_min_bin <= value <= config.angle_max_bin for value in values[3:]):
        raise FixedSlotError(f"Lattice angle bins outside range: {tokens}")
    if lattice_angle_rad(values[3], values[4], values[5]) <= 1e-4:
        raise FixedSlotError(f"Illegal lattice angle triple: {tokens[3:]}")
    lengths = [value * config.length_step for value in values[:3]]
    angles = [float(value) for value in values[3:]]
    return {"lengths": lengths, "angles": angles, "lattice_tokens": list(tokens), "lattice_text": " ".join(tokens)}


def parse_lattice_text(text: str, config: FixedSlotConfig = FixedSlotConfig()) -> Dict[str, Any]:
    tokens = [token for token in tokenize_plain_answer(text) if token[:2] in {"LA", "LB", "LC", "AA", "AB", "AG"}]
    first_error: str | None = None
    for idx in range(0, max(0, len(tokens) - 5)):
        window = tokens[idx : idx + 6]
        try:
            return _lattice_from_tokens(window, config=config)
        except Exception as exc:
            if first_error is None:
                first_error = str(exc)
            continue
    if first_error:
        raise FixedSlotError(f"Could not parse valid fixed-plain lattice: {first_error}")
    raise FixedSlotError("Could not parse fixed-plain lattice")


def parse_elements_text(text: str, num_atoms: int, config: FixedSlotConfig = FixedSlotConfig()) -> Dict[str, Any]:
    symbols: List[str] = []
    tokens: List[str] = []
    for token in tokenize_plain_answer(text):
        if not token.startswith("E_"):
            continue
        symbol = token.split("_", 1)[1]
        if symbol not in SYMBOL_TO_Z or SYMBOL_TO_Z[symbol] > config.max_atomic_number:
            continue
        symbols.append(symbol)
        tokens.append(token)
        if len(tokens) >= int(num_atoms):
            break
    if len(tokens) != int(num_atoms):
        raise FixedSlotError(f"Expected {num_atoms} element tokens, got {len(tokens)}")
    return {
        "species": symbols,
        "atom_types": [int(SYMBOL_TO_Z[symbol]) for symbol in symbols],
        "elements_tokens": tokens,
        "elements_text": " ".join(tokens),
        "composition": dict(sorted(Counter(symbols).items(), key=lambda item: SYMBOL_TO_Z[item[0]])),
    }


def _coord_value(token: str, axis: str, config: FixedSlotConfig = FixedSlotConfig()) -> int:
    if not token.startswith(axis + "_"):
        raise FixedSlotError(f"Expected {axis}_###, got {token!r}")
    value = int(token.split("_", 1)[1])
    if not config.coord_min_bin <= value <= config.coord_max_bin:
        raise FixedSlotError(f"Coordinate bin {value} outside range")
    return value


def _coord_token(axis: str, value: int) -> str:
    return f"{axis}_{int(value):03d}"


def parse_coords_text(
    text: str,
    num_atoms: int,
    *,
    deduplicate_pbc: bool = True,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> Dict[str, Any]:
    coord_tokens = [token for token in tokenize_plain_answer(text) if token[:2] in {"X_", "Y_", "Z_"}]
    triples: List[List[str]] = []
    coords: List[List[float]] = []
    seen_pbc: set[Tuple[int, int, int]] = set()
    idx = 0
    while idx + 2 < len(coord_tokens) and len(triples) < int(num_atoms):
        window = coord_tokens[idx : idx + 3]
        try:
            bins = [
                _coord_value(window[0], "X", config=config),
                _coord_value(window[1], "Y", config=config),
                _coord_value(window[2], "Z", config=config),
            ]
        except Exception:
            idx += 1
            continue
        coord = [value / float(config.coord_max_bin) for value in bins]
        if deduplicate_pbc:
            key = pbc_coordinate_key(coord, bins=config.coord_max_bin)
            if key in seen_pbc:
                idx += 3
                continue
            seen_pbc.add(key)
        triples.append([_coord_token(axis, value) for axis, value in zip(("X", "Y", "Z"), bins)])
        coords.append(coord)
        idx += 3
    if len(triples) != int(num_atoms):
        raise FixedSlotError(f"Expected {num_atoms} coordinate triples, got {len(triples)}")
    flat_tokens = [token for triple in triples for token in triple]
    return {"frac_coords": coords, "coords_tokens": flat_tokens, "coords_text": " ".join(flat_tokens)}


def assemble_plain_answer(
    count_text: str,
    lattice_text: str,
    elements_text: str,
    coords_text: str,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> Dict[str, Any]:
    count = parse_count_text(count_text, config=config)
    lattice = parse_lattice_text(lattice_text, config=config)
    elements = parse_elements_text(elements_text, int(count["num_atoms"]), config=config)
    coords = parse_coords_text(coords_text, int(count["num_atoms"]), config=config)
    tokens: List[str] = [count["count_text"], *lattice["lattice_tokens"]]
    coord_tokens = list(coords["coords_tokens"])
    for slot_index in range(config.max_atoms):
        tokens.append(f"S{slot_index:02d}")
        if slot_index < int(count["num_atoms"]):
            tokens.append(elements["elements_tokens"][slot_index])
            tokens.extend(coord_tokens[slot_index * 3 : slot_index * 3 + 3])
        else:
            tokens.extend(["EMPTY", "X_PAD", "Y_PAD", "Z_PAD"])
    arrays = plain_tokens_to_arrays(tokens, config=config)
    answer = " ".join(tokens)
    arrays["answer"] = answer
    arrays["tokens"] = tokens
    return arrays


def count_prompt() -> str:
    return FIXED_PLAIN_COUNT_PROMPT


def lattice_prompt(count_text: str) -> str:
    return FIXED_PLAIN_LATTICE_PROMPT_TEMPLATE.format(count_text=parse_count_text(count_text)["count_text"])


def elements_prompt(count_text: str, lattice_text: str) -> str:
    count = parse_count_text(count_text)
    lattice = parse_lattice_text(lattice_text)
    return FIXED_PLAIN_ELEMENTS_PROMPT_TEMPLATE.format(
        count_text=count["count_text"],
        lattice_text=lattice["lattice_text"],
        num_atoms=int(count["num_atoms"]),
    )


def coords_prompt(count_text: str, lattice_text: str, elements_text: str) -> str:
    count = parse_count_text(count_text)
    lattice = parse_lattice_text(lattice_text)
    elements = parse_elements_text(elements_text, int(count["num_atoms"]))
    return FIXED_PLAIN_COORDS_PROMPT_TEMPLATE.format(
        count_text=count["count_text"],
        lattice_text=lattice["lattice_text"],
        elements_text=elements["elements_text"],
        num_atoms=int(count["num_atoms"]),
    )


def atom_types_to_species(atom_types: Iterable[int]) -> List[str]:
    return [Z_TO_SYMBOL[int(value)] for value in atom_types]


__all__ = [
    "FIXED_PLAIN_PROMPT_VERSION",
    "FIXED_PLAIN_MODULE_TO_ID",
    "FIXED_PLAIN_COUNT_PROMPT",
    "FIXED_PLAIN_LATTICE_PROMPT_TEMPLATE",
    "FIXED_PLAIN_ELEMENTS_PROMPT_TEMPLATE",
    "FIXED_PLAIN_COORDS_PROMPT_TEMPLATE",
    "CifLiteError",
    "FixedSlotError",
    "arrays_to_structure",
    "arrays_to_torch_payload",
    "assemble_plain_answer",
    "atom_types_to_species",
    "coords_prompt",
    "count_prompt",
    "elements_prompt",
    "lattice_prompt",
    "metadata_from_csv_row",
    "parse_count_text",
    "parse_coords_text",
    "parse_elements_text",
    "parse_lattice_text",
    "parse_plain_answer",
    "plain_tokens_to_arrays",
    "split_plain_tokens",
    "strip_fixed_token",
    "structure_to_plain_answer",
    "tokenize_plain_answer",
    "write_json",
]
