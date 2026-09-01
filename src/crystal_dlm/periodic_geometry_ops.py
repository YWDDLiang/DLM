"""Shared bounded triclinic PBC operators and frozen element radii."""

from __future__ import annotations

import hashlib
import json

import torch


# Frozen from pymatgen 2025.6.14 ``atomic_radius`` with
# ``atomic_radius_calculated`` and then 1.5 A as deterministic fallbacks.
# Index zero is padding; indices 1..118 are atomic numbers.
ELEMENT_RADII_ANGSTROM_BY_Z = (
    0.0,
    0.25, 0.31, 1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.38,
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 0.71, 2.20, 1.80,
    1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35,
    1.30, 1.25, 1.15, 1.15, 1.15, 0.88, 2.35, 2.00, 1.80, 1.55,
    1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, 1.55, 1.45,
    1.45, 1.40, 1.40, 1.08, 2.60, 2.15, 1.95, 1.85, 1.85, 1.85,
    1.85, 1.85, 1.85, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50,
    1.90, 1.80, 1.60, 1.90, 1.50, 1.20, 1.50, 2.15, 1.95, 1.80,
    1.80, 1.75, 1.75, 1.75, 1.75, 1.50, 1.50, 1.50, 1.50, 1.50,
    1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50,
    1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50,
)
ELEMENT_RADII_SHA256 = hashlib.sha256(
    json.dumps(
        ELEMENT_RADII_ANGSTROM_BY_Z,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def element_radius(atomic_number: int) -> float:
    atomic_number = int(atomic_number)
    if not 1 <= atomic_number <= 118:
        raise ValueError("atomic number must be in 1..118")
    return float(ELEMENT_RADII_ANGSTROM_BY_Z[atomic_number])


def minimum_image_distances_27(
    fractional_deltas: torch.Tensor,
    lattice: torch.Tensor,
) -> torch.Tensor:
    """Return the minimum norm over the centered bounded 27-image shell.

    ``fractional_deltas`` may be ``[..., 3]`` with one lattice ``[3, 3]``, or
    ``[B, ..., 3]`` with batched lattices ``[B, 3, 3]``.
    """

    if fractional_deltas.shape[-1] != 3:
        raise ValueError("fractional_deltas must end in three coordinates")
    centered = fractional_deltas - torch.round(fractional_deltas)
    values = torch.arange(-1, 2, dtype=centered.dtype, device=centered.device)
    shifts = torch.cartesian_prod(values, values, values).reshape(-1, 3)
    candidates = centered.unsqueeze(-2) + shifts
    if lattice.ndim == 2:
        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape [3, 3]")
        cartesian = torch.einsum("...nc,cd->...nd", candidates, lattice)
    elif lattice.ndim == 3:
        if lattice.shape[0] != candidates.shape[0] or lattice.shape[1:] != (3, 3):
            raise ValueError("batched lattice must have shape [B, 3, 3]")
        cartesian = torch.einsum("b...nc,bcd->b...nd", candidates, lattice)
    else:
        raise ValueError("lattice must be rank two or three")
    squared = cartesian.square().sum(dim=-1)
    return torch.sqrt(squared.min(dim=-1).values.clamp_min(1.0e-12))


__all__ = [
    "ELEMENT_RADII_ANGSTROM_BY_Z",
    "ELEMENT_RADII_SHA256",
    "element_radius",
    "minimum_image_distances_27",
]
