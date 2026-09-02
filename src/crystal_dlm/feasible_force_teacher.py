"""Deterministic periodic feasibility projection for force-teacher targets.

The module is deliberately model-free.  A force provider proposes a local
Cartesian displacement; this code keeps that proposal inside the same
species-aware periodic feasible region used by G2-PBC-R.  Severe collisions
can therefore receive a geometry-only target even when a local energy gradient
is not trustworthy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import itertools
import math
from typing import Sequence

import numpy as np

from crystal_dlm.periodic_geometry_ops import element_radius


@dataclass(frozen=True)
class FeasibilityProjectionReport:
    initial_minimum_distance_A: float
    final_minimum_distance_A: float
    initial_margin_violations: int
    final_margin_violations: int
    iterations: int
    converged: bool
    maximum_atom_displacement_A: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class AdjacentTokenProjectionReport:
    attempted: bool
    resolved: bool
    pair: tuple[int, int] | None
    candidates_checked: int
    changed_coordinate_tokens: int
    final_minimum_distance_A: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def species_pair_margin(
    left_atomic_number: int,
    right_atomic_number: int,
    *,
    scale: float = 0.55,
    floor_A: float = 0.60,
    ceiling_A: float = 1.40,
) -> float:
    """Return the frozen G2-PBC-R species-aware pair margin."""

    if not 0.0 <= float(scale):
        raise ValueError("scale must be non-negative")
    if not 0.0 < float(floor_A) <= float(ceiling_A):
        raise ValueError("invalid pair-margin bounds")
    raw = float(scale) * (
        element_radius(int(left_atomic_number))
        + element_radius(int(right_atomic_number))
    )
    return float(np.clip(raw, float(floor_A), float(ceiling_A)))


@lru_cache(maxsize=2)
def _image_shifts(image_radius: int) -> np.ndarray:
    if int(image_radius) not in (1, 2):
        raise ValueError("image_radius must be one (27) or two (125)")
    shifts = np.asarray(
        list(itertools.product(range(-image_radius, image_radius + 1), repeat=3)),
        dtype=float,
    )
    shifts.setflags(write=False)
    return shifts


def minimum_image_vector(
    left_fractional: np.ndarray,
    right_fractional: np.ndarray,
    lattice: np.ndarray,
    *,
    image_radius: int = 2,
) -> tuple[np.ndarray, float]:
    """Return the shortest Cartesian vector from ``left`` to an image of ``right``."""

    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("lattice must be a finite 3x3 matrix")
    left = np.asarray(left_fractional, dtype=float).reshape(3)
    right = np.asarray(right_fractional, dtype=float).reshape(3)
    centered = right - left
    centered -= np.round(centered)
    candidates = (centered[None, :] + _image_shifts(int(image_radius))) @ lattice
    squared = np.einsum("ij,ij->i", candidates, candidates)
    selected = int(np.argmin(squared))
    vector = candidates[selected]
    return vector, float(math.sqrt(max(float(squared[selected]), 0.0)))


def periodic_pair_summary(
    fractional_coordinates: np.ndarray,
    lattice: np.ndarray,
    atomic_numbers: Sequence[int],
    *,
    image_radius: int = 2,
    margin_scale: float = 0.55,
    margin_floor_A: float = 0.60,
    margin_ceiling_A: float = 1.40,
) -> tuple[float, int]:
    """Return minimum PBC distance and count of species-margin violations."""

    coordinates = np.asarray(fractional_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("fractional_coordinates must have shape [N, 3]")
    if len(atomic_numbers) != len(coordinates):
        raise ValueError("atomic-number count does not match coordinates")
    minimum = math.inf
    violations = 0
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            _vector, distance = minimum_image_vector(
                coordinates[left],
                coordinates[right],
                lattice,
                image_radius=int(image_radius),
            )
            minimum = min(minimum, distance)
            margin = species_pair_margin(
                int(atomic_numbers[left]),
                int(atomic_numbers[right]),
                scale=float(margin_scale),
                floor_A=float(margin_floor_A),
                ceiling_A=float(margin_ceiling_A),
            )
            violations += int(distance + 1.0e-6 < margin)
    return minimum, violations


def adjacent_token_feasible_projection(
    quantized_fractional_coordinates: np.ndarray,
    continuous_target_fractional_coordinates: np.ndarray,
    lattice: np.ndarray,
    atomic_numbers: Sequence[int],
    *,
    coordinate_token_step: float = 0.01,
    accept_minimum_distance_A: float = 0.50,
    image_radius: int = 2,
) -> tuple[np.ndarray, AdjacentTokenProjectionReport]:
    """Resolve a quantization trap in the one-token neighborhood of one pair.

    Only the two atoms forming the shortest PBC pair are changed.  The search
    examines the fixed ``{-1, 0, +1}`` neighborhood for their six coordinate
    tokens, rejects geometrically invalid candidates, and chooses the candidate
    closest to the continuous projected target.  No energy is queried.
    """

    quantized = np.asarray(quantized_fractional_coordinates, dtype=float).copy()
    target = np.asarray(continuous_target_fractional_coordinates, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    if quantized.shape != target.shape or quantized.ndim != 2 or quantized.shape[1] != 3:
        raise ValueError("quantized and target coordinates must share shape [N, 3]")
    if len(atomic_numbers) != len(quantized):
        raise ValueError("atomic-number count does not match coordinates")
    if float(coordinate_token_step) <= 0.0:
        raise ValueError("coordinate_token_step must be positive")
    initial_minimum, _violations = periodic_pair_summary(
        quantized,
        lattice,
        atomic_numbers,
        image_radius=int(image_radius),
        margin_scale=0.0,
        margin_floor_A=float(accept_minimum_distance_A),
        margin_ceiling_A=float(accept_minimum_distance_A),
    )
    if initial_minimum >= float(accept_minimum_distance_A):
        return quantized, AdjacentTokenProjectionReport(
            attempted=False,
            resolved=True,
            pair=None,
            candidates_checked=0,
            changed_coordinate_tokens=0,
            final_minimum_distance_A=float(initial_minimum),
        )

    shortest_pair: tuple[int, int] | None = None
    shortest_distance = math.inf
    for left in range(len(quantized)):
        for right in range(left + 1, len(quantized)):
            _vector, distance = minimum_image_vector(
                quantized[left],
                quantized[right],
                lattice,
                image_radius=int(image_radius),
            )
            if distance < shortest_distance:
                shortest_pair = (left, right)
                shortest_distance = distance
    if shortest_pair is None:
        raise ValueError("adjacent token projection requires at least two atoms")

    best: tuple[tuple[float, int, tuple[int, ...]], np.ndarray, float] | None = None
    checked = 0
    for offsets in itertools.product((-1, 0, 1), repeat=6):
        checked += 1
        candidate = quantized.copy()
        candidate[shortest_pair[0]] = (
            candidate[shortest_pair[0]]
            + float(coordinate_token_step) * np.asarray(offsets[:3], dtype=float)
        ) % 1.0
        candidate[shortest_pair[1]] = (
            candidate[shortest_pair[1]]
            + float(coordinate_token_step) * np.asarray(offsets[3:], dtype=float)
        ) % 1.0
        minimum, violations = periodic_pair_summary(
            candidate,
            lattice,
            atomic_numbers,
            image_radius=int(image_radius),
            margin_scale=0.0,
            margin_floor_A=float(accept_minimum_distance_A),
            margin_ceiling_A=float(accept_minimum_distance_A),
        )
        if violations != 0:
            continue
        delta = candidate - target
        delta -= np.round(delta)
        score = (
            float(np.square(delta).sum()),
            int(sum(abs(value) for value in offsets)),
            tuple(int(value) for value in offsets),
        )
        if best is None or score < best[0]:
            best = (score, candidate, float(minimum))
    if best is None:
        return quantized, AdjacentTokenProjectionReport(
            attempted=True,
            resolved=False,
            pair=shortest_pair,
            candidates_checked=checked,
            changed_coordinate_tokens=0,
            final_minimum_distance_A=float(initial_minimum),
        )
    selected = best[1]
    changed = int(
        np.count_nonzero(
            np.abs((selected - quantized) - np.round(selected - quantized)) > 1.0e-9
        )
    )
    return selected, AdjacentTokenProjectionReport(
        attempted=True,
        resolved=True,
        pair=shortest_pair,
        candidates_checked=checked,
        changed_coordinate_tokens=changed,
        final_minimum_distance_A=best[2],
    )


def project_periodic_feasible(
    fractional_coordinates: np.ndarray,
    lattice: np.ndarray,
    atomic_numbers: Sequence[int],
    *,
    image_radius: int = 2,
    margin_scale: float = 0.55,
    margin_floor_A: float = 0.60,
    margin_ceiling_A: float = 1.40,
    max_iterations: int = 16,
    max_pair_atom_step_A: float = 0.25,
    tolerance_A: float = 1.0e-3,
) -> tuple[np.ndarray, FeasibilityProjectionReport]:
    """Project coordinates into a deterministic periodic pair-feasible region.

    A Gauss--Seidel position projection is used instead of optimizing a scalar
    energy.  Each violating pair is separated along its exact triclinic
    minimum-image vector.  The operation is a detached teacher constructor and
    is not used during inference.
    """

    coordinates = np.asarray(fractional_coordinates, dtype=float).copy()
    lattice = np.asarray(lattice, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("fractional_coordinates must have shape [N, 3]")
    if len(atomic_numbers) != len(coordinates):
        raise ValueError("atomic-number count does not match coordinates")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")
    if float(max_pair_atom_step_A) <= 0.0:
        raise ValueError("max_pair_atom_step_A must be positive")
    determinant = float(np.linalg.det(lattice))
    if not math.isfinite(determinant) or determinant <= 1.0e-8:
        raise ValueError("lattice must have positive finite volume")
    inverse_lattice = np.linalg.inv(lattice)
    coordinates %= 1.0
    initial = coordinates.copy()
    initial_minimum, initial_violations = periodic_pair_summary(
        coordinates,
        lattice,
        atomic_numbers,
        image_radius=int(image_radius),
        margin_scale=float(margin_scale),
        margin_floor_A=float(margin_floor_A),
        margin_ceiling_A=float(margin_ceiling_A),
    )

    completed_iterations = 0
    for iteration in range(int(max_iterations)):
        completed_iterations = iteration + 1
        changed = False
        for left in range(len(coordinates)):
            for right in range(left + 1, len(coordinates)):
                vector, distance = minimum_image_vector(
                    coordinates[left],
                    coordinates[right],
                    lattice,
                    image_radius=int(image_radius),
                )
                margin = species_pair_margin(
                    int(atomic_numbers[left]),
                    int(atomic_numbers[right]),
                    scale=float(margin_scale),
                    floor_A=float(margin_floor_A),
                    ceiling_A=float(margin_ceiling_A),
                )
                violation = margin + float(tolerance_A) - distance
                if violation <= 0.0:
                    continue
                changed = True
                if distance <= 1.0e-10:
                    # Deterministic fallback for coincident sites.
                    direction = np.zeros(3, dtype=float)
                    direction[(left + right) % 3] = 1.0
                else:
                    direction = vector / distance
                per_atom = min(0.5 * violation, float(max_pair_atom_step_A))
                cartesian = per_atom * direction
                fractional = cartesian @ inverse_lattice
                coordinates[left] = (coordinates[left] - fractional) % 1.0
                coordinates[right] = (coordinates[right] + fractional) % 1.0
        if not changed:
            break

    final_minimum, final_violations = periodic_pair_summary(
        coordinates,
        lattice,
        atomic_numbers,
        image_radius=int(image_radius),
        margin_scale=float(margin_scale),
        margin_floor_A=float(margin_floor_A),
        margin_ceiling_A=float(margin_ceiling_A),
    )
    fractional_delta = coordinates - initial
    fractional_delta -= np.round(fractional_delta)
    cartesian_delta = fractional_delta @ lattice
    maximum_displacement = float(np.linalg.norm(cartesian_delta, axis=1).max())
    return coordinates, FeasibilityProjectionReport(
        initial_minimum_distance_A=float(initial_minimum),
        final_minimum_distance_A=float(final_minimum),
        initial_margin_violations=int(initial_violations),
        final_margin_violations=int(final_violations),
        iterations=int(completed_iterations),
        converged=bool(final_violations == 0),
        maximum_atom_displacement_A=maximum_displacement,
    )


def bounded_force_displacement(
    forces_eV_per_A: np.ndarray,
    *,
    eta_A2_per_eV: float = 0.03,
    max_atom_step_A: float = 0.15,
) -> np.ndarray:
    """Convert forces to a translation-free bounded Cartesian displacement."""

    forces = np.asarray(forces_eV_per_A, dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 3 or not np.isfinite(forces).all():
        raise ValueError("forces must be a finite [N, 3] array")
    centered = forces - forces.mean(axis=0, keepdims=True)
    displacement = float(eta_A2_per_eV) * centered
    norms = np.linalg.norm(displacement, axis=1)
    selectors = norms > float(max_atom_step_A)
    if bool(selectors.any()):
        displacement[selectors] *= (
            float(max_atom_step_A) / norms[selectors]
        )[:, None]
    return displacement


__all__ = [
    "AdjacentTokenProjectionReport",
    "FeasibilityProjectionReport",
    "adjacent_token_feasible_projection",
    "bounded_force_displacement",
    "minimum_image_vector",
    "periodic_pair_summary",
    "project_periodic_feasible",
    "species_pair_margin",
]
