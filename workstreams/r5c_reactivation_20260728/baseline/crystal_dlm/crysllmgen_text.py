"""CrysLLMGen-style plain-text crystal representation.

This intentionally mirrors ``reference/crysllmgen/llm_finetune.py``:

    a b c
    alpha beta gamma
    Element
    x y z
    Element
    x y z

There are no crystal special tokens and no invented section headers.  The
parser is a little more defensive than the original CrysLLMGen parser so a DLM
fixed-window sample can be truncated at the first complete crystal-like prefix.
"""

from __future__ import annotations

from collections import Counter
import random
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from crystal_dlm.cif_lite import CifLiteError, assert_no_pbc_duplicate, pbc_coordinate_key
from crystal_dlm.fixed_slot import (
    SYMBOL_TO_Z,
    Z_TO_SYMBOL,
    arrays_to_structure,
    arrays_to_torch_payload,
    metadata_from_csv_row,
    write_json,
)
from crystal_dlm.lattice_geometry import lattice_angle_rad


CRYSLLMGEN_TEXT_PROMPT_VERSION = "crysllmgen_text_v1_single_pass"
CRYSLLMGEN_MODULE_PROMPT_VERSION = "crysllmgen_text_v2_answer_modules"
CRYSLLMGEN_COMPOSITION_MODULE_PROMPT_VERSION = "crysllmgen_text_v3_composition_modules"
CRYSLLMGEN_TEXT_PROMPT = (
    "Below is a description of a bulk material. "
    "Generate a description of the lengths and angles of the lattice vectors "
    "and then the element type and coordinates for each atom within the lattice:\n"
)
CRYSLLMGEN_LATTICE_PROMPT = (
    "Below is a description of a bulk material. "
    "Generate only the first two lines of the crystal description: "
    "one line with three lattice lengths and one line with three lattice angles.\n"
)
CRYSLLMGEN_SPECIES_PROMPT_TEMPLATE = (
    "Below are the lattice lines of a bulk material:\n"
    "{lattice_text}\n"
    "Generate only the element symbol lines for each atom, one symbol per line. "
    "Use at most 20 atoms and no coordinates:\n"
)
CRYSLLMGEN_COMPOSITION_PROMPT_TEMPLATE = (
    "Below are the lattice lines of a bulk material:\n"
    "{lattice_text}\n"
    "Generate only the composition as element-count lines. "
    "Use one element symbol and one integer count per line, total atoms at most 20, and no coordinates:\n"
)
CRYSLLMGEN_COORDS_PROMPT_TEMPLATE = (
    "Below are the lattice lines and element sequence of a bulk material:\n"
    "{lattice_text}\n"
    "{species_text}\n"
    "Generate exactly one fractional coordinate line for each element above, "
    "in the same order. Use three decimals rounded to two places and no element names:\n"
)
CRYSLLMGEN_SITE_COORD_PROMPT_TEMPLATE = (
    "Below are the lattice lines, element sequence, and coordinates generated so far:\n"
    "{partial_text}\n"
    "Generate only the next fractional coordinate line for {symbol}:\n"
)
CRYSLLMGEN_MODULE_TO_ID = {
    "full": 0,
    "lattice": 1,
    "species": 2,
    "coords": 3,
    "site_coord": 4,
    "composition": 5,
}
MAX_MP20_ATOMS = 20
MAX_REASONABLE_LATTICE_LENGTH = 80.0


def _species_symbol(value: Any) -> str:
    if hasattr(value, "symbol"):
        return str(value.symbol)
    if hasattr(value, "specie") and hasattr(value.specie, "symbol"):
        return str(value.specie.symbol)
    return str(value)


def _wrap_frac(value: float) -> float:
    wrapped = float(value) % 1.0
    if abs(wrapped - 1.0) < 1e-8:
        return 0.0
    return wrapped


def _clean_lines(text: str) -> List[str]:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        lines.append(stripped)
    return lines


def _float_triplet(line: str) -> List[float] | None:
    parts = line.split()
    if len(parts) != 3:
        return None
    try:
        return [float(value) for value in parts]
    except ValueError:
        return None


def _is_element_line(line: str) -> str | None:
    parts = line.split()
    if len(parts) != 1:
        return None
    symbol = parts[0]
    if symbol not in SYMBOL_TO_Z:
        return None
    return symbol


def _valid_lattice(lengths: Sequence[float], angles: Sequence[float]) -> None:
    if len(lengths) != 3 or len(angles) != 3:
        raise CifLiteError("Lattice requires three lengths and three angles")
    if any(float(value) <= 0.0 for value in lengths):
        raise CifLiteError(f"Lattice lengths must be positive: {lengths}")
    if any(float(value) > MAX_REASONABLE_LATTICE_LENGTH for value in lengths):
        raise CifLiteError(f"Lattice lengths exceed {MAX_REASONABLE_LATTICE_LENGTH:g} A: {lengths}")
    if any(float(value) <= 0.0 or float(value) >= 180.0 for value in angles):
        raise CifLiteError(f"Lattice angles must be in (0, 180): {angles}")
    if lattice_angle_rad(int(round(angles[0])), int(round(angles[1])), int(round(angles[2]))) <= 1e-4:
        raise CifLiteError(f"Illegal lattice angle triple: {angles}")


def format_crysllmgen_text(
    species: Sequence[str],
    frac_coords: Sequence[Sequence[float]],
    lengths: Sequence[float],
    angles: Sequence[float],
) -> str:
    if len(species) != len(frac_coords):
        raise CifLiteError("Species and coordinate counts do not match")
    _valid_lattice(lengths, angles)
    lines = [
        " ".join(f"{float(value):.1f}" for value in lengths),
        " ".join(str(int(round(float(value)))) for value in angles),
    ]
    for symbol, coord in zip(species, frac_coords):
        symbol = str(symbol)
        if symbol not in SYMBOL_TO_Z:
            raise CifLiteError(f"Unsupported element symbol {symbol!r}")
        if len(coord) != 3:
            raise CifLiteError(f"Site for {symbol} does not have 3 coordinates")
        lines.append(symbol)
        lines.append(" ".join(f"{_wrap_frac(float(value)):.2f}" for value in coord))
    return "\n".join(lines)


def structure_to_crysllmgen_text(
    structure: Any,
    *,
    rng: random.Random | None = None,
    origin_shift: bool = False,
    permute_sites: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    copied = structure.copy()
    if origin_shift:
        if rng is None:
            rng = random.Random()
        copied.translate_sites(
            indices=range(len(copied.sites)),
            vector=[rng.random(), rng.random(), rng.random()],
            frac_coords=True,
            to_unit_cell=True,
        )
    species = [_species_symbol(site.specie) for site in copied.sites]
    coords = copied.frac_coords.tolist()
    order = list(range(len(species)))
    if permute_sites:
        if rng is None:
            rng = random.Random()
        rng.shuffle(order)
    species = [species[idx] for idx in order]
    coords = [coords[idx] for idx in order]
    answer = format_crysllmgen_text(species, coords, copied.lattice.abc, copied.lattice.angles)
    arrays = parse_crysllmgen_text(answer)
    return answer, arrays


def _find_crystal_start(lines: Sequence[str]) -> int:
    for idx in range(min(len(lines), 8)):
        lengths = _float_triplet(lines[idx])
        angles = _float_triplet(lines[idx + 1]) if idx + 1 < len(lines) else None
        if lengths is None or angles is None:
            continue
        if idx + 2 >= len(lines) or _is_element_line(lines[idx + 2]) is None:
            continue
        try:
            _valid_lattice(lengths, angles)
        except CifLiteError:
            continue
        return idx
    raise CifLiteError("Could not find CrysLLMGen lattice prefix")


def parse_crysllmgen_text(
    text: str,
    *,
    require_no_pbc_duplicate: bool = True,
    strict_all_lines: bool = False,
) -> Dict[str, Any]:
    lines = _clean_lines(text)
    start = _find_crystal_start(lines)
    lengths = _float_triplet(lines[start])
    angles = _float_triplet(lines[start + 1])
    if lengths is None or angles is None:
        raise CifLiteError("Missing CrysLLMGen lattice lines")
    _valid_lattice(lengths, angles)

    species: List[str] = []
    atom_types: List[int] = []
    frac_coords: List[List[float]] = []
    idx = start + 2
    while idx + 1 < len(lines):
        symbol = _is_element_line(lines[idx])
        coord = _float_triplet(lines[idx + 1])
        if symbol is None or coord is None:
            break
        if len(species) >= MAX_MP20_ATOMS:
            break
        species.append(symbol)
        atom_types.append(int(SYMBOL_TO_Z[symbol]))
        frac_coords.append([_wrap_frac(value) for value in coord])
        idx += 2

    if not 1 <= len(species) <= MAX_MP20_ATOMS:
        raise CifLiteError(f"Atom count {len(species)} outside 1..{MAX_MP20_ATOMS}")
    if strict_all_lines and idx < len(lines):
        raise CifLiteError(f"Unexpected trailing text after crystal prefix: {lines[idx]!r}")
    if require_no_pbc_duplicate:
        assert_no_pbc_duplicate(frac_coords)

    counts = dict(sorted(Counter(species).items(), key=lambda item: SYMBOL_TO_Z[item[0]]))
    answer = format_crysllmgen_text(species, frac_coords, lengths, angles)
    return {
        "num_atoms": int(len(species)),
        "lengths": [float(value) for value in lengths],
        "angles": [float(value) for value in angles],
        "species": species,
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "composition": counts,
        "answer": answer,
    }


def split_crysllmgen_answer(text: str) -> Dict[str, Any]:
    arrays = parse_crysllmgen_text(text, require_no_pbc_duplicate=False)
    lattice_text = format_lattice_text(arrays["lengths"], arrays["angles"])
    composition_text = format_composition_text(arrays["species"])
    species_text = format_species_text(arrays["species"])
    coords_text = format_coords_text(arrays["frac_coords"])
    return {
        "lattice_text": lattice_text,
        "composition_text": composition_text,
        "species_text": species_text,
        "coords_text": coords_text,
        "arrays": arrays,
    }


def format_lattice_text(lengths: Sequence[float], angles: Sequence[float]) -> str:
    _valid_lattice(lengths, angles)
    return (
        " ".join(f"{float(value):.1f}" for value in lengths)
        + "\n"
        + " ".join(str(int(round(float(value)))) for value in angles)
    )


def parse_lattice_text(text: str) -> Dict[str, Any]:
    lines = _clean_lines(text)
    if len(lines) < 2:
        raise CifLiteError("Could not parse lattice module")
    first_error: str | None = None
    for idx in range(0, len(lines) - 1):
        lengths = _float_triplet(lines[idx])
        angles = _float_triplet(lines[idx + 1])
        if lengths is None or angles is None:
            continue
        try:
            _valid_lattice(lengths, angles)
        except CifLiteError as exc:
            if first_error is None:
                first_error = str(exc)
            continue
        return {"lengths": lengths, "angles": angles, "text": format_lattice_text(lengths, angles)}
    if first_error:
        raise CifLiteError(f"Could not parse valid lattice module: {first_error}")
    raise CifLiteError("Could not parse lattice module")


def format_composition_text(species: Sequence[str]) -> str:
    counts: Counter[str] = Counter()
    for symbol in species:
        symbol = str(symbol)
        if symbol not in SYMBOL_TO_Z:
            raise CifLiteError(f"Unsupported element symbol {symbol!r}")
        counts[symbol] += 1
    if not counts:
        raise CifLiteError("Composition is empty")
    total = sum(counts.values())
    if not 1 <= total <= MAX_MP20_ATOMS:
        raise CifLiteError(f"Composition atom count {total} outside 1..{MAX_MP20_ATOMS}")
    lines = []
    for symbol in sorted(counts, key=lambda item: SYMBOL_TO_Z[item]):
        lines.append(f"{symbol} {int(counts[symbol])}")
    return "\n".join(lines)


def parse_composition_text(text: str) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    total = 0
    for line in _clean_lines(text):
        parts = line.split()
        if len(parts) != 2:
            break
        symbol = parts[0]
        if symbol not in SYMBOL_TO_Z:
            break
        try:
            count = int(parts[1])
        except ValueError:
            break
        if count <= 0:
            break
        if total + int(count) > MAX_MP20_ATOMS:
            break
        counts[symbol] += int(count)
        total += int(count)
    if not counts:
        raise CifLiteError("Could not parse composition module")
    if not 1 <= total <= MAX_MP20_ATOMS:
        raise CifLiteError(f"Composition atom count {total} outside 1..{MAX_MP20_ATOMS}")
    species: List[str] = []
    for symbol in sorted(counts, key=lambda item: SYMBOL_TO_Z[item]):
        species.extend([symbol] * int(counts[symbol]))
    return {
        "composition": dict(sorted(counts.items(), key=lambda item: SYMBOL_TO_Z[item[0]])),
        "species": species,
        "atom_types": [int(SYMBOL_TO_Z[symbol]) for symbol in species],
        "text": format_composition_text(species),
        "species_text": format_species_text(species),
    }


def reorder_coords_by_composition_species(species: Sequence[str], coords: Sequence[Sequence[float]]) -> Tuple[List[str], List[List[float]]]:
    if len(species) != len(coords):
        raise CifLiteError("Species and coordinate counts do not match")
    grouped: Dict[str, List[List[float]]] = {}
    for symbol, coord in zip(species, coords):
        symbol = str(symbol)
        if symbol not in SYMBOL_TO_Z:
            raise CifLiteError(f"Unsupported element symbol {symbol!r}")
        grouped.setdefault(symbol, []).append([_wrap_frac(float(value)) for value in coord])
    ordered_species: List[str] = []
    ordered_coords: List[List[float]] = []
    for symbol in sorted(grouped, key=lambda item: SYMBOL_TO_Z[item]):
        for coord in grouped[symbol]:
            ordered_species.append(symbol)
            ordered_coords.append(coord)
    return ordered_species, ordered_coords


def format_species_text(species: Sequence[str]) -> str:
    out: List[str] = []
    for symbol in species:
        symbol = str(symbol)
        if symbol not in SYMBOL_TO_Z:
            raise CifLiteError(f"Unsupported element symbol {symbol!r}")
        out.append(symbol)
    if not 1 <= len(out) <= MAX_MP20_ATOMS:
        raise CifLiteError(f"Species count {len(out)} outside 1..{MAX_MP20_ATOMS}")
    return "\n".join(out)


def parse_species_text(text: str) -> Dict[str, Any]:
    species: List[str] = []
    for line in _clean_lines(text):
        symbol = _is_element_line(line)
        if symbol is None:
            break
        species.append(symbol)
        if len(species) >= MAX_MP20_ATOMS:
            break
    if not 1 <= len(species) <= MAX_MP20_ATOMS:
        raise CifLiteError(f"Species count {len(species)} outside 1..{MAX_MP20_ATOMS}")
    return {
        "species": species,
        "atom_types": [int(SYMBOL_TO_Z[symbol]) for symbol in species],
        "text": format_species_text(species),
    }


def format_coords_text(frac_coords: Sequence[Sequence[float]]) -> str:
    lines: List[str] = []
    for coord in frac_coords:
        if len(coord) != 3:
            raise CifLiteError(f"Coordinate line must contain 3 values: {coord}")
        lines.append(" ".join(f"{_wrap_frac(float(value)):.2f}" for value in coord))
    if not 1 <= len(lines) <= MAX_MP20_ATOMS:
        raise CifLiteError(f"Coordinate count {len(lines)} outside 1..{MAX_MP20_ATOMS}")
    return "\n".join(lines)


def parse_coords_text(
    text: str,
    expected_count: int | None = None,
    *,
    skip_invalid: bool = False,
    deduplicate_pbc: bool = False,
) -> Dict[str, Any]:
    coords: List[List[float]] = []
    seen_pbc: set[Tuple[int, int, int]] = set()
    for line in _clean_lines(text):
        coord = _float_triplet(line)
        if coord is None:
            if skip_invalid:
                continue
            break
        wrapped = [_wrap_frac(value) for value in coord]
        if deduplicate_pbc:
            key = pbc_coordinate_key(wrapped)
            if key in seen_pbc:
                continue
            seen_pbc.add(key)
        coords.append(wrapped)
        if expected_count is not None and len(coords) >= int(expected_count):
            break
        if len(coords) >= MAX_MP20_ATOMS:
            break
    if expected_count is not None and len(coords) != int(expected_count):
        raise CifLiteError(f"Expected {expected_count} coordinate lines, got {len(coords)}")
    if not 1 <= len(coords) <= MAX_MP20_ATOMS:
        raise CifLiteError(f"Coordinate count {len(coords)} outside 1..{MAX_MP20_ATOMS}")
    return {"frac_coords": coords, "text": format_coords_text(coords)}


def parse_coord_line(text: str) -> List[float]:
    for line in _clean_lines(text):
        coord = _float_triplet(line)
        if coord is not None:
            return [_wrap_frac(value) for value in coord]
    raise CifLiteError("Could not parse coordinate line")


def assemble_crysllmgen_text(
    lattice_text: str,
    species_text: str,
    coords_text: str,
    *,
    require_no_pbc_duplicate: bool = True,
) -> Dict[str, Any]:
    lattice = parse_lattice_text(lattice_text)
    species = parse_species_text(species_text)
    coords = parse_coords_text(coords_text, expected_count=len(species["species"]))
    if require_no_pbc_duplicate:
        assert_no_pbc_duplicate(coords["frac_coords"])
    answer = format_crysllmgen_text(
        species["species"],
        coords["frac_coords"],
        lattice["lengths"],
        lattice["angles"],
    )
    arrays = parse_crysllmgen_text(answer, require_no_pbc_duplicate=require_no_pbc_duplicate)
    return arrays


def lattice_prompt() -> str:
    return CRYSLLMGEN_LATTICE_PROMPT


def species_prompt(lattice_text: str) -> str:
    return CRYSLLMGEN_SPECIES_PROMPT_TEMPLATE.format(lattice_text=parse_lattice_text(lattice_text)["text"])


def composition_prompt(lattice_text: str) -> str:
    return CRYSLLMGEN_COMPOSITION_PROMPT_TEMPLATE.format(lattice_text=parse_lattice_text(lattice_text)["text"])


def coords_prompt(lattice_text: str, species_text: str) -> str:
    lattice = parse_lattice_text(lattice_text)["text"]
    species = parse_species_text(species_text)["text"]
    return CRYSLLMGEN_COORDS_PROMPT_TEMPLATE.format(lattice_text=lattice, species_text=species)


def site_coord_prompt(
    lattice_text: str,
    species_text: str,
    previous_coords: Sequence[Sequence[float]],
    *,
    site_index: int,
) -> str:
    lattice = parse_lattice_text(lattice_text)["text"]
    species = parse_species_text(species_text)["species"]
    if not 0 <= int(site_index) < len(species):
        raise CifLiteError(f"site_index {site_index} outside species count {len(species)}")
    lines = [lattice]
    for idx, symbol in enumerate(species):
        lines.append(symbol)
        if idx < len(previous_coords):
            lines.append(" ".join(f"{_wrap_frac(float(value)):.2f}" for value in previous_coords[idx]))
        if idx == int(site_index):
            break
    return CRYSLLMGEN_SITE_COORD_PROMPT_TEMPLATE.format(
        partial_text="\n".join(lines),
        symbol=species[int(site_index)],
    )


def atom_types_to_species(atom_types: Iterable[int]) -> List[str]:
    return [Z_TO_SYMBOL[int(value)] for value in atom_types]


__all__ = [
    "CRYSLLMGEN_MODULE_PROMPT_VERSION",
    "CRYSLLMGEN_COMPOSITION_MODULE_PROMPT_VERSION",
    "CRYSLLMGEN_TEXT_PROMPT",
    "CRYSLLMGEN_TEXT_PROMPT_VERSION",
    "CRYSLLMGEN_LATTICE_PROMPT",
    "CRYSLLMGEN_COMPOSITION_PROMPT_TEMPLATE",
    "CRYSLLMGEN_MODULE_TO_ID",
    "CifLiteError",
    "assemble_crysllmgen_text",
    "arrays_to_structure",
    "arrays_to_torch_payload",
    "atom_types_to_species",
    "coords_prompt",
    "composition_prompt",
    "format_crysllmgen_text",
    "format_composition_text",
    "format_coords_text",
    "format_lattice_text",
    "format_species_text",
    "lattice_prompt",
    "metadata_from_csv_row",
    "parse_coord_line",
    "parse_coords_text",
    "parse_composition_text",
    "parse_crysllmgen_text",
    "parse_lattice_text",
    "parse_species_text",
    "site_coord_prompt",
    "species_prompt",
    "split_crysllmgen_answer",
    "reorder_coords_by_composition_species",
    "structure_to_crysllmgen_text",
    "write_json",
]
