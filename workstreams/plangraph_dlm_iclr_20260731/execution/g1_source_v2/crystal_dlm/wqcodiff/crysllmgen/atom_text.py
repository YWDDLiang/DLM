"""Exact atom-text boundary used by the disabled CrysLLMGen extension.

The active project has a more defensive historical parser in
``crystal_dlm.crysllmgen_text``.  It is intentionally *not* used for upstream
parity: the functions here preserve the line-selection semantics of
``upstream/crysllmgen_sample.py::parse_generated_text``.  Validation beyond
what pymatgen performs belongs to a later attempt stage and must not alter the
upstream reproduction.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class AtomTextFields:
    """Semantic fields extracted by the upstream atom-text grammar."""

    lengths: tuple[float, float, float]
    angles: tuple[float, float, float]
    species: tuple[str, ...]
    frac_coords: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if len(self.species) != len(self.frac_coords):
            raise ValueError("species and fractional-coordinate counts differ")

    @property
    def num_atoms(self) -> int:
        return len(self.species)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lengths": list(self.lengths),
            "angles": list(self.angles),
            "species": list(self.species),
            "frac_coords": [list(values) for values in self.frac_coords],
            "num_atoms": self.num_atoms,
        }


def _float_triplet(line: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in line.split())
    if len(values) != 3:
        raise ValueError("expected exactly three floating-point values")
    return values  # type: ignore[return-value]


def parse_upstream_atom_text_fields(generated_text: str) -> AtomTextFields:
    """Mirror upstream non-empty-line parsing without adding recovery rules.

    The upstream function takes the first two non-empty lines as lattice
    lengths/angles and then consumes every alternating species/coordinate
    line.  An odd trailing species line therefore raises, as it does upstream.
    Code fences, prompt prefixes, trailing prose, wrapping, and truncation are
    not repaired here.
    """

    lines = [line for line in str(generated_text).split("\n") if line.strip()]
    if len(lines) < 4:
        raise ValueError("upstream atom text requires lattice plus at least one site")
    lengths = _float_triplet(lines[0])
    angles = _float_triplet(lines[1])
    species = tuple(lines[2::2])
    coordinate_lines = lines[3::2]
    if len(species) != len(coordinate_lines):
        raise ValueError("upstream atom text has an unmatched species line")
    coordinates = tuple(_float_triplet(line) for line in coordinate_lines)
    return AtomTextFields(
        lengths=lengths,
        angles=angles,
        species=species,
        frac_coords=coordinates,
    )


def atom_text_fields_to_cif(fields: AtomTextFields) -> str:
    """Build the same pymatgen structure/CIF representation as upstream."""

    try:
        from pymatgen.core import Structure
        from pymatgen.core.lattice import Lattice
    except ImportError as exc:  # pragma: no cover - exercised on A800 runtime.
        raise RuntimeError("pymatgen is required for CrysLLMGen parser parity") from exc
    structure = Structure(
        lattice=Lattice.from_parameters(*(fields.lengths + fields.angles)),
        species=fields.species,
        coords=fields.frac_coords,
        coords_are_cartesian=False,
    )
    return structure.to(fmt="cif")


def parse_upstream_atom_text_to_cif(generated_text: str) -> str:
    """Derived disabled-extension entry point corresponding to upstream."""

    return atom_text_fields_to_cif(parse_upstream_atom_text_fields(generated_text))
