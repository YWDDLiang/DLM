"""Compressed-token variants of the MP-20 fixed-slot crystal schema.

The representation keeps the same 107 semantic positions as the successful
fixed-slot path, but lets token identity drop axis information for selected
families. Position then determines whether a shared token means x/y/z,
a/b/c, or alpha/beta/gamma.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .fixed_slot import (
    ANSWER_TOKEN_COUNT,
    CHEMICAL_SYMBOLS,
    FixedSlotConfig,
    FixedSlotError,
    SYMBOL_TO_Z,
    Z_TO_SYMBOL,
    EncodeDiagnostics,
    _clip,
    _format_bin,
    _round_half_up,
    answer_token_count,
    normalize_fractional_coord,
)


@dataclass(frozen=True)
class CompressedFixedSlotConfig:
    """Config for fixed-slot compressed-token ablations."""

    max_atoms: int = 20
    length_step: float = 0.1
    length_min_bin: int = 0
    length_max_bin: int = 500
    angle_min_bin: int = 1
    angle_max_bin: int = 179
    coord_min_bin: int = 0
    coord_max_bin: int = 100
    max_atomic_number: int = SYMBOL_TO_Z["Pu"]
    share_coordinates: bool = True
    share_lengths: bool = False
    share_angles: bool = False
    representation: str = "fixed_slot_compressed_v1"

    def to_fixed_config(self) -> FixedSlotConfig:
        return FixedSlotConfig(
            max_atoms=self.max_atoms,
            length_step=self.length_step,
            length_min_bin=self.length_min_bin,
            length_max_bin=self.length_max_bin,
            angle_min_bin=self.angle_min_bin,
            angle_max_bin=self.angle_max_bin,
            coord_min_bin=self.coord_min_bin,
            coord_max_bin=self.coord_max_bin,
            max_atomic_number=self.max_atomic_number,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "representation": self.representation,
            "max_atoms": self.max_atoms,
            "length_step": self.length_step,
            "length_min_bin": self.length_min_bin,
            "length_max_bin": self.length_max_bin,
            "angle_min_bin": self.angle_min_bin,
            "angle_max_bin": self.angle_max_bin,
            "coord_min_bin": self.coord_min_bin,
            "coord_max_bin": self.coord_max_bin,
            "max_atomic_number": self.max_atomic_number,
            "share_coordinates": self.share_coordinates,
            "share_lengths": self.share_lengths,
            "share_angles": self.share_angles,
            "answer_token_count": answer_token_count(self.to_fixed_config()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompressedFixedSlotConfig":
        return cls(
            max_atoms=int(payload.get("max_atoms", 20)),
            length_step=float(payload.get("length_step", 0.1)),
            length_min_bin=int(payload.get("length_min_bin", 0)),
            length_max_bin=int(payload.get("length_max_bin", 500)),
            angle_min_bin=int(payload.get("angle_min_bin", 1)),
            angle_max_bin=int(payload.get("angle_max_bin", 179)),
            coord_min_bin=int(payload.get("coord_min_bin", 0)),
            coord_max_bin=int(payload.get("coord_max_bin", 100)),
            max_atomic_number=int(payload.get("max_atomic_number", SYMBOL_TO_Z["Pu"])),
            share_coordinates=bool(payload.get("share_coordinates", True)),
            share_lengths=bool(payload.get("share_lengths", False)),
            share_angles=bool(payload.get("share_angles", False)),
            representation=str(payload.get("representation", "fixed_slot_compressed_v1")),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "CompressedFixedSlotConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


COMPRESSED_TOKEN_RE = re.compile(
    r"<(?:N_\d{3}|L(?:[ABC])?_\d{3}|A(?:[ABG])?_\d{3}|S\d{2}|"
    r"E_[A-Z][a-z]?|[XYZC]_\d{3}|EMPTY|[XYZC]_PAD)>"
)
COUNT_RE = re.compile(r"^<N_(\d{3})>$")
SLOT_RE = re.compile(r"^<S(\d{2})>$")
ELEMENT_RE = re.compile(r"^<E_([A-Z][a-z]?)>$")
FULL_LENGTH_RE = re.compile(r"^<L([ABC])_(\d{3})>$")
SHARED_LENGTH_RE = re.compile(r"^<L_(\d{3})>$")
FULL_ANGLE_RE = re.compile(r"^<A([ABG])_(\d{3})>$")
SHARED_ANGLE_RE = re.compile(r"^<A_(\d{3})>$")
FULL_COORD_RE = re.compile(r"^<([XYZ])_(\d{3})>$")
SHARED_COORD_RE = re.compile(r"^<C_(\d{3})>$")


def tokenize_compressed_answer_text(text: str) -> List[str]:
    """Extract fixed-slot compressed schema tokens from decoded model text."""

    return COMPRESSED_TOKEN_RE.findall(text)


def build_compressed_special_tokens(
    config: CompressedFixedSlotConfig = CompressedFixedSlotConfig(),
) -> List[str]:
    """Return the crystal special-token vocabulary for a compressed schema."""

    tokens: List[str] = []
    tokens.extend(f"<N_{i:03d}>" for i in range(1, config.max_atoms + 1))

    if config.share_lengths:
        tokens.extend(
            f"<L_{i:03d}>"
            for i in range(config.length_min_bin, config.length_max_bin + 1)
        )
    else:
        for prefix in ("LA", "LB", "LC"):
            tokens.extend(
                f"<{prefix}_{i:03d}>"
                for i in range(config.length_min_bin, config.length_max_bin + 1)
            )

    if config.share_angles:
        tokens.extend(
            f"<A_{i:03d}>"
            for i in range(config.angle_min_bin, config.angle_max_bin + 1)
        )
    else:
        for prefix in ("AA", "AB", "AG"):
            tokens.extend(
                f"<{prefix}_{i:03d}>"
                for i in range(config.angle_min_bin, config.angle_max_bin + 1)
            )

    tokens.extend(f"<S{i:02d}>" for i in range(config.max_atoms))
    tokens.extend(
        f"<E_{Z_TO_SYMBOL[z]}>" for z in range(1, config.max_atomic_number + 1)
    )

    if config.share_coordinates:
        tokens.extend(
            f"<C_{i:03d}>"
            for i in range(config.coord_min_bin, config.coord_max_bin + 1)
        )
        tokens.append("<C_PAD>")
    else:
        for prefix in ("X", "Y", "Z"):
            tokens.extend(
                f"<{prefix}_{i:03d}>"
                for i in range(config.coord_min_bin, config.coord_max_bin + 1)
            )
        tokens.extend(["<X_PAD>", "<Y_PAD>", "<Z_PAD>"])
    tokens.append("<EMPTY>")
    return tokens


def compressed_token_sources(
    config: CompressedFixedSlotConfig = CompressedFixedSlotConfig(),
) -> Dict[str, List[str]]:
    """Map each compressed token to its full-token source(s).

    Tokens unchanged by the ablation map to themselves. Shared tokens map to
    the full-token family rows that should be averaged when converting a
    checkpoint.
    """

    sources: Dict[str, List[str]] = {}
    for token in build_compressed_special_tokens(config):
        sources[token] = [token]

    if config.share_coordinates:
        for i in range(config.coord_min_bin, config.coord_max_bin + 1):
            sources[f"<C_{i:03d}>"] = [f"<X_{i:03d}>", f"<Y_{i:03d}>", f"<Z_{i:03d}>"]
        sources["<C_PAD>"] = ["<X_PAD>", "<Y_PAD>", "<Z_PAD>"]

    if config.share_lengths:
        for i in range(config.length_min_bin, config.length_max_bin + 1):
            sources[f"<L_{i:03d}>"] = [f"<LA_{i:03d}>", f"<LB_{i:03d}>", f"<LC_{i:03d}>"]

    if config.share_angles:
        for i in range(config.angle_min_bin, config.angle_max_bin + 1):
            sources[f"<A_{i:03d}>"] = [f"<AA_{i:03d}>", f"<AB_{i:03d}>", f"<AG_{i:03d}>"]

    return sources


def arrays_to_compressed_tokens(
    lengths: Sequence[float],
    angles: Sequence[float],
    species: Sequence[str],
    frac_coords: Sequence[Sequence[float]],
    config: CompressedFixedSlotConfig = CompressedFixedSlotConfig(),
) -> Tuple[List[str], EncodeDiagnostics]:
    """Convert crystal arrays into compressed fixed-slot tokens."""

    diagnostics = EncodeDiagnostics()
    if len(lengths) != 3:
        raise FixedSlotError(f"Expected 3 lattice lengths, got {len(lengths)}")
    if len(angles) != 3:
        raise FixedSlotError(f"Expected 3 lattice angles, got {len(angles)}")
    if len(species) != len(frac_coords):
        raise FixedSlotError("Species and coordinate counts do not match")
    if not 1 <= len(species) <= config.max_atoms:
        raise FixedSlotError(f"Atom count {len(species)} outside 1..{config.max_atoms}")

    tokens = [f"<N_{len(species):03d}>"]

    for prefix, length in zip(("LA", "LB", "LC"), lengths):
        raw_bin = _round_half_up(float(length) / config.length_step)
        bin_value, clipped = _clip(raw_bin, config.length_min_bin, config.length_max_bin)
        diagnostics.length_clips += int(clipped)
        tokens.append(_format_bin("L" if config.share_lengths else prefix, bin_value))

    for prefix, angle in zip(("AA", "AB", "AG"), angles):
        raw_bin = _round_half_up(float(angle))
        bin_value, clipped = _clip(raw_bin, config.angle_min_bin, config.angle_max_bin)
        diagnostics.angle_clips += int(clipped)
        tokens.append(_format_bin("A" if config.share_angles else prefix, bin_value))

    for slot_index in range(config.max_atoms):
        tokens.append(f"<S{slot_index:02d}>")
        if slot_index >= len(species):
            if config.share_coordinates:
                tokens.extend(["<EMPTY>", "<C_PAD>", "<C_PAD>", "<C_PAD>"])
            else:
                tokens.extend(["<EMPTY>", "<X_PAD>", "<Y_PAD>", "<Z_PAD>"])
            continue

        symbol = str(species[slot_index])
        if symbol not in SYMBOL_TO_Z:
            raise FixedSlotError(f"Unsupported element symbol {symbol!r}")
        if SYMBOL_TO_Z[symbol] > config.max_atomic_number:
            raise FixedSlotError(f"Element {symbol} exceeds configured max Z")
        tokens.append(f"<E_{symbol}>")

        coord = frac_coords[slot_index]
        if len(coord) != 3:
            raise FixedSlotError(f"Atom slot {slot_index} does not have 3 coords")
        for prefix, value in zip(("X", "Y", "Z"), coord):
            wrapped, changed = normalize_fractional_coord(float(value))
            diagnostics.coord_wraps += int(changed)
            raw_bin = _round_half_up(wrapped * config.coord_max_bin)
            bin_value, clipped = _clip(raw_bin, config.coord_min_bin, config.coord_max_bin)
            diagnostics.coord_clips += int(clipped)
            tokens.append(_format_bin("C" if config.share_coordinates else prefix, bin_value))

    expected_tokens = ANSWER_TOKEN_COUNT
    if len(tokens) != expected_tokens:
        raise FixedSlotError(
            f"Internal schema error: produced {len(tokens)} tokens, expected {expected_tokens}"
        )
    return tokens, diagnostics


def _parse_length_token(
    token: str,
    expected_axis: str,
    config: CompressedFixedSlotConfig,
) -> int:
    if config.share_lengths:
        match = SHARED_LENGTH_RE.match(token)
        if not match:
            raise FixedSlotError(f"Expected shared length token at {expected_axis}, got {token!r}")
        bin_value = int(match.group(1))
    else:
        match = FULL_LENGTH_RE.match(token)
        if not match:
            raise FixedSlotError(f"Expected length token at {expected_axis}, got {token!r}")
        axis = match.group(1)
        if axis != expected_axis:
            raise FixedSlotError(f"Expected length axis {expected_axis}, got {axis}")
        bin_value = int(match.group(2))
    if not config.length_min_bin <= bin_value <= config.length_max_bin:
        raise FixedSlotError(f"Length bin {bin_value} outside range")
    return bin_value


def _parse_angle_token(
    token: str,
    expected_axis: str,
    config: CompressedFixedSlotConfig,
) -> int:
    if config.share_angles:
        match = SHARED_ANGLE_RE.match(token)
        if not match:
            raise FixedSlotError(f"Expected shared angle token at {expected_axis}, got {token!r}")
        bin_value = int(match.group(1))
    else:
        match = FULL_ANGLE_RE.match(token)
        if not match:
            raise FixedSlotError(f"Expected angle token at {expected_axis}, got {token!r}")
        axis = match.group(1)
        if axis != expected_axis:
            raise FixedSlotError(f"Expected angle axis {expected_axis}, got {axis}")
        bin_value = int(match.group(2))
    if not config.angle_min_bin <= bin_value <= config.angle_max_bin:
        raise FixedSlotError(f"Angle bin {bin_value} outside range")
    return bin_value


def _parse_coord_token(
    token: str,
    expected_axis: str,
    config: CompressedFixedSlotConfig,
) -> int:
    if config.share_coordinates:
        match = SHARED_COORD_RE.match(token)
        if not match:
            raise FixedSlotError(f"Expected shared coordinate token at {expected_axis}, got {token!r}")
        bin_value = int(match.group(1))
    else:
        match = FULL_COORD_RE.match(token)
        if not match:
            raise FixedSlotError(f"Expected coordinate token at {expected_axis}, got {token!r}")
        axis = match.group(1)
        if axis != expected_axis:
            raise FixedSlotError(f"Expected coord axis {expected_axis}, got {axis}")
        bin_value = int(match.group(2))
    if not config.coord_min_bin <= bin_value <= config.coord_max_bin:
        raise FixedSlotError(f"Coordinate bin {bin_value} outside range")
    return bin_value


def compressed_tokens_to_arrays(
    tokens: Sequence[str],
    config: CompressedFixedSlotConfig = CompressedFixedSlotConfig(),
    strict: bool = True,
) -> Dict[str, Any]:
    """Parse compressed fixed-slot tokens into crystal arrays."""

    token_list = list(tokens)
    expected_tokens = ANSWER_TOKEN_COUNT
    if strict and len(token_list) != expected_tokens:
        raise FixedSlotError(f"Expected {expected_tokens} tokens, got {len(token_list)}")
    if len(token_list) < expected_tokens:
        raise FixedSlotError(f"Expected at least {expected_tokens} tokens, got {len(token_list)}")
    token_list = token_list[:expected_tokens]

    count_match = COUNT_RE.match(token_list[0])
    if not count_match:
        raise FixedSlotError(f"Invalid atom-count token {token_list[0]!r}")
    num_atoms = int(count_match.group(1))
    if not 1 <= num_atoms <= config.max_atoms:
        raise FixedSlotError(f"Atom count {num_atoms} outside schema range")

    lengths = [
        _parse_length_token(token, expected_axis, config) * config.length_step
        for expected_axis, token in zip(("A", "B", "C"), token_list[1:4])
    ]
    angles = [
        float(_parse_angle_token(token, expected_axis, config))
        for expected_axis, token in zip(("A", "B", "G"), token_list[4:7])
    ]

    species: List[str] = []
    atom_types: List[int] = []
    frac_coords: List[List[float]] = []
    position = 7
    empty_coord_tokens = (
        ["<C_PAD>", "<C_PAD>", "<C_PAD>"]
        if config.share_coordinates
        else ["<X_PAD>", "<Y_PAD>", "<Z_PAD>"]
    )
    for slot_index in range(config.max_atoms):
        slot_token = token_list[position]
        slot_match = SLOT_RE.match(slot_token)
        if not slot_match:
            raise FixedSlotError(f"Invalid slot token {slot_token!r}")
        actual_slot = int(slot_match.group(1))
        if actual_slot != slot_index:
            raise FixedSlotError(f"Expected slot {slot_index:02d}, got {actual_slot:02d}")
        position += 1

        field_tokens = token_list[position : position + 4]
        position += 4

        if slot_index >= num_atoms:
            if field_tokens != ["<EMPTY>", *empty_coord_tokens]:
                raise FixedSlotError(f"Slot {slot_index:02d} should be empty, got {field_tokens}")
            continue

        element_match = ELEMENT_RE.match(field_tokens[0])
        if not element_match:
            raise FixedSlotError(
                f"Slot {slot_index:02d} expected element token, got {field_tokens[0]!r}"
            )
        symbol = element_match.group(1)
        if symbol not in SYMBOL_TO_Z:
            raise FixedSlotError(f"Unsupported element token {field_tokens[0]!r}")
        atomic_number = SYMBOL_TO_Z[symbol]
        if atomic_number > config.max_atomic_number:
            raise FixedSlotError(f"Element {symbol} exceeds configured max Z")

        coord_values = [
            _parse_coord_token(token, expected_axis, config) / config.coord_max_bin
            for expected_axis, token in zip(("X", "Y", "Z"), field_tokens[1:])
        ]
        species.append(symbol)
        atom_types.append(atomic_number)
        frac_coords.append(coord_values)

    return {
        "num_atoms": num_atoms,
        "lengths": lengths,
        "angles": angles,
        "species": species,
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "tokens": token_list,
        "answer": " ".join(token_list),
    }


def parse_compressed_fixed_slot_answer(
    text: str,
    config: CompressedFixedSlotConfig = CompressedFixedSlotConfig(),
    strict: bool = False,
) -> Dict[str, Any]:
    tokens = tokenize_compressed_answer_text(text)
    return compressed_tokens_to_arrays(tokens, config=config, strict=strict)


def arrays_to_compressed_answer(
    lengths: Sequence[float],
    angles: Sequence[float],
    species: Sequence[str],
    frac_coords: Sequence[Sequence[float]],
    config: CompressedFixedSlotConfig = CompressedFixedSlotConfig(),
    separator: str = " ",
) -> Tuple[str, EncodeDiagnostics]:
    tokens, diagnostics = arrays_to_compressed_tokens(
        lengths=lengths,
        angles=angles,
        species=species,
        frac_coords=frac_coords,
        config=config,
    )
    return separator.join(tokens), diagnostics


def structure_to_compressed_answer(
    structure: Any,
    config: CompressedFixedSlotConfig = CompressedFixedSlotConfig(),
    separator: str = " ",
) -> Tuple[str, EncodeDiagnostics]:
    lengths = list(structure.lattice.abc)
    angles = list(structure.lattice.angles)
    species = [site.specie.symbol for site in structure.sites]
    frac_coords = structure.frac_coords.tolist()
    return arrays_to_compressed_answer(lengths, angles, species, frac_coords, config=config, separator=separator)


def compressed_tokens_to_full_tokens(
    tokens: Sequence[str],
    config: CompressedFixedSlotConfig = CompressedFixedSlotConfig(),
) -> List[str]:
    """Expand shared family tokens back to full fixed-slot token strings."""

    arrays = compressed_tokens_to_arrays(tokens, config=config, strict=True)
    from .fixed_slot import arrays_to_tokens

    full_tokens, _ = arrays_to_tokens(
        lengths=arrays["lengths"],
        angles=arrays["angles"],
        species=arrays["species"],
        frac_coords=arrays["frac_coords"],
        config=config.to_fixed_config(),
    )
    return full_tokens


def full_tokens_to_compressed_tokens(
    tokens: Sequence[str],
    config: CompressedFixedSlotConfig = CompressedFixedSlotConfig(),
) -> List[str]:
    """Convert full fixed-slot tokens to compressed tokens via arrays."""

    from .fixed_slot import tokens_to_arrays

    arrays = tokens_to_arrays(tokens, config=config.to_fixed_config(), strict=True)
    compressed_tokens, _ = arrays_to_compressed_tokens(
        lengths=arrays["lengths"],
        angles=arrays["angles"],
        species=arrays["species"],
        frac_coords=arrays["frac_coords"],
        config=config,
    )
    return compressed_tokens


def schema_allowed_token_strings(
    config: CompressedFixedSlotConfig = CompressedFixedSlotConfig(),
) -> Dict[int, List[str]]:
    """Allowed special-token strings for each semantic answer position."""

    allowed: Dict[int, List[str]] = {}
    allowed[0] = [f"<N_{i:03d}>" for i in range(1, config.max_atoms + 1)]

    length_positions = (1, 2, 3)
    length_prefixes = ("L", "L", "L") if config.share_lengths else ("LA", "LB", "LC")
    for pos, prefix in zip(length_positions, length_prefixes):
        allowed[pos] = [
            f"<{prefix}_{i:03d}>"
            for i in range(config.length_min_bin, config.length_max_bin + 1)
        ]

    angle_positions = (4, 5, 6)
    angle_prefixes = ("A", "A", "A") if config.share_angles else ("AA", "AB", "AG")
    for pos, prefix in zip(angle_positions, angle_prefixes):
        allowed[pos] = [
            f"<{prefix}_{i:03d}>"
            for i in range(config.angle_min_bin, config.angle_max_bin + 1)
        ]

    coord_prefixes = ("C", "C", "C") if config.share_coordinates else ("X", "Y", "Z")
    element_tokens = [
        f"<E_{symbol}>"
        for symbol in CHEMICAL_SYMBOLS[1 : config.max_atomic_number + 1]
    ]
    for slot_index in range(config.max_atoms):
        base = 7 + slot_index * 5
        allowed[base] = [f"<S{slot_index:02d}>"]
        allowed[base + 1] = element_tokens + ["<EMPTY>"]
        for offset, prefix in enumerate(coord_prefixes, start=2):
            allowed[base + offset] = [
                f"<{prefix}_{i:03d}>"
                for i in range(config.coord_min_bin, config.coord_max_bin + 1)
            ] + [f"<{prefix}_PAD>"]

    return allowed


__all__ = [
    "CompressedFixedSlotConfig",
    "build_compressed_special_tokens",
    "compressed_token_sources",
    "tokenize_compressed_answer_text",
    "arrays_to_compressed_tokens",
    "compressed_tokens_to_arrays",
    "parse_compressed_fixed_slot_answer",
    "arrays_to_compressed_answer",
    "structure_to_compressed_answer",
    "compressed_tokens_to_full_tokens",
    "full_tokens_to_compressed_tokens",
    "schema_allowed_token_strings",
]
