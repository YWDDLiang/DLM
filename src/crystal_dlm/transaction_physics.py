"""Deterministic physics proposals for complete SPAD crystal transactions.

This module only proposes actions.  It never consumes an energy and therefore
cannot select a candidate by its outcome.  Cartesian forces propose complete
XYZ transactions; stresses propose complete six-token lattice transactions.
Every proposal is quantized through the deployed dynamic ``7 + 4N`` schema and
is then checked in that quantized (i.e. actually executable) geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from crystal_dlm.dynamic_crystal import (
    arrays_to_dynamic_tokens,
    dynamic_tokens_to_arrays,
)
from crystal_dlm.feasible_force_teacher import minimum_image_vector
from crystal_dlm.fixed_slot import FixedSlotConfig


ProposalStatus = Literal["accepted", "noop", "duplicate", "invalid"]

DEFAULT_CARTESIAN_STEPS_A = (0.05, 0.10, 0.15, 0.20)
DEFAULT_STRAIN_STEPS = (0.0025, 0.005, 0.010, 0.020)


@dataclass(frozen=True)
class TransactionProposal:
    """One direction of a deterministic complete-transaction proposal."""

    kind: Literal["site_xyz", "lattice"]
    direction: str
    status: ProposalStatus
    reason: str
    step: float | None
    transaction_tokens: tuple[str, ...]
    full_tokens: tuple[str, ...]
    lengths: tuple[float, float, float]
    angles: tuple[float, float, float]
    species: tuple[str, ...]
    frac_coords: tuple[tuple[float, float, float], ...]
    minimum_distance_A: float | None


def _as_vector(value: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite length-three vector")
    return array


def _as_matrix(value: Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return array


def _fixed_positive_steps(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    steps = tuple(float(value) for value in values)
    if not steps or any(not math.isfinite(value) or value <= 0.0 for value in steps):
        raise ValueError(f"{name} must contain finite positive values")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return steps


def lattice_matrix_from_dynamic_arrays(arrays: Mapping[str, Any]) -> np.ndarray:
    """Build the row-vector lattice used by pymatgen from lengths and angles."""

    lengths = _as_vector(arrays["lengths"], name="lengths")
    angles = _as_vector(arrays["angles"], name="angles")
    if np.any(lengths <= 0.0) or np.any(angles <= 0.0) or np.any(angles >= 180.0):
        raise ValueError("lattice lengths and angles lie outside the physical domain")
    alpha, beta, gamma = np.deg2rad(angles)
    sin_gamma = math.sin(float(gamma))
    if abs(sin_gamma) <= 1.0e-10:
        raise ValueError("lattice gamma angle is singular")
    a, b, c = lengths
    cx = c * math.cos(float(beta))
    cy = c * (
        math.cos(float(alpha))
        - math.cos(float(beta)) * math.cos(float(gamma))
    ) / sin_gamma
    cz_squared = c * c - cx * cx - cy * cy
    if cz_squared <= 1.0e-12:
        raise ValueError("lattice angle triple has non-positive volume")
    return np.asarray(
        [
            [a, 0.0, 0.0],
            [b * math.cos(float(gamma)), b * sin_gamma, 0.0],
            [cx, cy, math.sqrt(float(cz_squared))],
        ],
        dtype=np.float64,
    )


def lattice_parameters_from_matrix(
    lattice: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ``(a,b,c)`` and pymatgen ``(alpha,beta,gamma)`` from row vectors."""

    matrix = _as_matrix(lattice, name="lattice")
    determinant = float(np.linalg.det(matrix))
    if determinant <= 1.0e-10:
        raise ValueError("lattice must have positive volume")
    lengths = np.linalg.norm(matrix, axis=1)
    if np.any(lengths <= 1.0e-10):
        raise ValueError("lattice vector has zero length")

    def angle(left: int, right: int) -> float:
        cosine = float(
            np.dot(matrix[left], matrix[right]) / (lengths[left] * lengths[right])
        )
        return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))

    angles = np.asarray((angle(1, 2), angle(0, 2), angle(0, 1)))
    if np.any(angles <= 0.0) or np.any(angles >= 180.0):
        raise ValueError("lattice matrix produces an illegal angle")
    return tuple(float(value) for value in lengths), tuple(float(value) for value in angles)


def _matrix_exp_symmetric(direction: np.ndarray, step: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (direction + direction.T))
    return (vectors * np.exp(float(step) * values)) @ vectors.T


def _quantized_arrays(
    lengths: Sequence[float],
    angles: Sequence[float],
    species: Sequence[str],
    frac_coords: Sequence[Sequence[float]],
    *,
    config: FixedSlotConfig,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    tokens, _diagnostics = arrays_to_dynamic_tokens(
        lengths,
        angles,
        species,
        frac_coords,
        config=config,
    )
    decoded = dynamic_tokens_to_arrays(tokens, config=config, strict=True)
    return tuple(tokens), decoded


def _minimum_distance(
    frac_coords: Sequence[Sequence[float]],
    lattice: np.ndarray,
    *,
    image_radius: int,
) -> float | None:
    fractional = np.asarray(frac_coords, dtype=np.float64)
    count = int(fractional.shape[0])
    if count < 2:
        return None
    minimum = math.inf
    for left in range(count):
        for right in range(left + 1, count):
            _vector, distance = minimum_image_vector(
                fractional[left],
                fractional[right],
                lattice,
                image_radius=int(image_radius),
            )
            minimum = min(minimum, float(distance))
    return minimum


def _validate_quantized_candidate(
    decoded: Mapping[str, Any],
    *,
    expected_species: tuple[str, ...],
    minimum_distance_A: float,
    image_radius: int,
) -> tuple[bool, str, float | None]:
    if tuple(decoded["species"]) != expected_species:
        return False, "species_or_order_changed", None
    try:
        lattice = lattice_matrix_from_dynamic_arrays(decoded)
    except (KeyError, TypeError, ValueError) as error:
        return False, f"invalid_lattice:{error}", None
    if float(np.linalg.det(lattice)) <= 1.0e-10:
        return False, "non_positive_volume", None
    distance = _minimum_distance(
        np.asarray(decoded["frac_coords"], dtype=np.float64) % 1.0,
        lattice,
        image_radius=int(image_radius),
    )
    if distance is not None and distance < float(minimum_distance_A) - 1.0e-10:
        return False, "pbc_minimum_distance", distance
    return True, "accepted", distance


def _record(
    *,
    kind: Literal["site_xyz", "lattice"],
    direction: str,
    status: ProposalStatus,
    reason: str,
    step: float | None,
    transaction_tokens: Sequence[str],
    full_tokens: Sequence[str],
    decoded: Mapping[str, Any],
    minimum_distance_A: float | None,
) -> TransactionProposal:
    return TransactionProposal(
        kind=kind,
        direction=direction,
        status=status,
        reason=reason,
        step=None if step is None else float(step),
        transaction_tokens=tuple(transaction_tokens),
        full_tokens=tuple(full_tokens),
        lengths=tuple(float(value) for value in decoded["lengths"]),
        angles=tuple(float(value) for value in decoded["angles"]),
        species=tuple(str(value) for value in decoded["species"]),
        frac_coords=tuple(
            tuple(float(value) for value in coordinate)
            for coordinate in decoded["frac_coords"]
        ),
        minimum_distance_A=minimum_distance_A,
    )


def _finish_direction(
    *,
    kind: Literal["site_xyz", "lattice"],
    direction: str,
    attempts: Sequence[tuple[float, tuple[str, ...], Mapping[str, Any]]],
    no_op_transaction: tuple[str, ...],
    seen_transactions: set[tuple[str, ...]],
    transaction_slice: slice,
    expected_species: tuple[str, ...],
    minimum_distance_A: float,
    image_radius: int,
) -> TransactionProposal:
    selected: tuple[float, tuple[str, ...], Mapping[str, Any]] | None = None
    for attempt in attempts:
        transaction = tuple(attempt[1][transaction_slice])
        if transaction != no_op_transaction:
            selected = attempt
            break
    if selected is None:
        step, full_tokens, decoded = attempts[-1]
        return _record(
            kind=kind,
            direction=direction,
            status="noop",
            reason="all_fixed_steps_quantize_to_noop",
            step=step,
            transaction_tokens=no_op_transaction,
            full_tokens=full_tokens,
            decoded=decoded,
            minimum_distance_A=None,
        )

    step, full_tokens, decoded = selected
    transaction = tuple(full_tokens[transaction_slice])
    if transaction in seen_transactions:
        return _record(
            kind=kind,
            direction=direction,
            status="duplicate",
            reason="quantized_transaction_duplicate",
            step=step,
            transaction_tokens=transaction,
            full_tokens=full_tokens,
            decoded=decoded,
            minimum_distance_A=None,
        )
    seen_transactions.add(transaction)
    valid, reason, distance = _validate_quantized_candidate(
        decoded,
        expected_species=expected_species,
        minimum_distance_A=float(minimum_distance_A),
        image_radius=int(image_radius),
    )
    return _record(
        kind=kind,
        direction=direction,
        status="accepted" if valid else "invalid",
        reason=reason,
        step=step,
        transaction_tokens=transaction,
        full_tokens=full_tokens,
        decoded=decoded,
        minimum_distance_A=distance,
    )


def propose_force_site_transactions(
    crystal: Mapping[str, Any],
    site_index: int,
    cartesian_force: Sequence[float],
    *,
    step_sizes_A: Sequence[float] = DEFAULT_CARTESIAN_STEPS_A,
    minimum_distance_A: float = 0.50,
    image_radius: int = 2,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> tuple[TransactionProposal, TransactionProposal]:
    """Propose ``+F`` and ``-F`` complete XYZ actions for one site.

    The force is used only as a direction.  Each direction walks the fixed,
    predeclared step sequence until the 101-bin tokenizer produces the first
    non-noop transaction.  No energy is accepted by this API.
    """

    steps = _fixed_positive_steps(step_sizes_A, name="step_sizes_A")
    if int(image_radius) not in (1, 2):
        raise ValueError("image_radius must be one (27) or two (125)")
    if not math.isfinite(minimum_distance_A) or minimum_distance_A < 0.0:
        raise ValueError("minimum_distance_A must be finite and non-negative")
    species = tuple(str(value) for value in crystal["species"])
    coordinates = np.asarray(crystal["frac_coords"], dtype=np.float64)
    if coordinates.shape != (len(species), 3) or not np.isfinite(coordinates).all():
        raise ValueError("frac_coords must be a finite [N,3] array matching species")
    site = int(site_index)
    if not 0 <= site < len(species):
        raise IndexError("site_index lies outside the crystal")
    force = _as_vector(cartesian_force, name="cartesian_force")
    force_norm = float(np.linalg.norm(force))
    lattice = lattice_matrix_from_dynamic_arrays(crystal)
    base_tokens, base_decoded = _quantized_arrays(
        crystal["lengths"], crystal["angles"], species, coordinates, config=config
    )
    transaction_slice = slice(8 + 4 * site, 11 + 4 * site)
    no_op = tuple(base_tokens[transaction_slice])
    if force_norm <= 1.0e-12:
        return tuple(
            _record(
                kind="site_xyz",
                direction=direction,
                status="invalid",
                reason="zero_force_has_no_direction",
                step=None,
                transaction_tokens=no_op,
                full_tokens=base_tokens,
                decoded=base_decoded,
                minimum_distance_A=None,
            )
            for direction in ("positive_force", "negative_force")
        )

    inverse_lattice = np.linalg.inv(lattice)
    unit_force = force / force_norm
    seen: set[tuple[str, ...]] = {no_op}
    results: list[TransactionProposal] = []
    for label, sign in (("positive_force", 1.0), ("negative_force", -1.0)):
        attempts = []
        for step in steps:
            candidate_coords = coordinates.copy()
            delta_cart = sign * float(step) * unit_force
            # Pymatgen uses row vectors: r_cart = r_frac @ lattice.
            delta_frac = delta_cart @ inverse_lattice
            candidate_coords[site] = (candidate_coords[site] + delta_frac) % 1.0
            full_tokens, decoded = _quantized_arrays(
                crystal["lengths"],
                crystal["angles"],
                species,
                candidate_coords,
                config=config,
            )
            attempts.append((step, full_tokens, decoded))
        results.append(
            _finish_direction(
                kind="site_xyz",
                direction=label,
                attempts=attempts,
                no_op_transaction=no_op,
                seen_transactions=seen,
                transaction_slice=transaction_slice,
                expected_species=species,
                minimum_distance_A=float(minimum_distance_A),
                image_radius=int(image_radius),
            )
        )
    return results[0], results[1]


def propose_stress_lattice_transactions(
    crystal: Mapping[str, Any],
    stress: Sequence[Sequence[float]],
    *,
    strain_steps: Sequence[float] = DEFAULT_STRAIN_STEPS,
    minimum_distance_A: float = 0.50,
    image_radius: int = 2,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> tuple[TransactionProposal, TransactionProposal]:
    """Propose ``-sym(stress)`` and its reverse as complete lattice actions.

    CHGNet stress is interpreted as ``(1/V) dE/dstrain``.  For row-vector
    lattices the update is exactly ``L' = L @ exp(epsilon * D)`` while
    fractional coordinates remain fixed.
    """

    steps = _fixed_positive_steps(strain_steps, name="strain_steps")
    if int(image_radius) not in (1, 2):
        raise ValueError("image_radius must be one (27) or two (125)")
    if not math.isfinite(minimum_distance_A) or minimum_distance_A < 0.0:
        raise ValueError("minimum_distance_A must be finite and non-negative")
    stress_matrix = _as_matrix(stress, name="stress")
    symmetric = 0.5 * (stress_matrix + stress_matrix.T)
    stress_norm = float(np.linalg.norm(symmetric))
    species = tuple(str(value) for value in crystal["species"])
    coordinates = np.asarray(crystal["frac_coords"], dtype=np.float64)
    if coordinates.shape != (len(species), 3) or not np.isfinite(coordinates).all():
        raise ValueError("frac_coords must be a finite [N,3] array matching species")
    lattice = lattice_matrix_from_dynamic_arrays(crystal)
    base_tokens, base_decoded = _quantized_arrays(
        crystal["lengths"], crystal["angles"], species, coordinates, config=config
    )
    transaction_slice = slice(1, 7)
    no_op = tuple(base_tokens[transaction_slice])
    if stress_norm <= 1.0e-12:
        return tuple(
            _record(
                kind="lattice",
                direction=direction,
                status="invalid",
                reason="zero_symmetric_stress_has_no_direction",
                step=None,
                transaction_tokens=no_op,
                full_tokens=base_tokens,
                decoded=base_decoded,
                minimum_distance_A=None,
            )
            for direction in ("negative_stress", "positive_stress")
        )

    unit_stress = symmetric / stress_norm
    seen: set[tuple[str, ...]] = {no_op}
    results: list[TransactionProposal] = []
    for label, direction in (
        ("negative_stress", -unit_stress),
        ("positive_stress", unit_stress),
    ):
        attempts = []
        for step in steps:
            candidate_lattice = lattice @ _matrix_exp_symmetric(direction, step)
            lengths, angles = lattice_parameters_from_matrix(candidate_lattice)
            full_tokens, decoded = _quantized_arrays(
                lengths,
                angles,
                species,
                coordinates,
                config=config,
            )
            attempts.append((step, full_tokens, decoded))
        results.append(
            _finish_direction(
                kind="lattice",
                direction=label,
                attempts=attempts,
                no_op_transaction=no_op,
                seen_transactions=seen,
                transaction_slice=transaction_slice,
                expected_species=species,
                minimum_distance_A=float(minimum_distance_A),
                image_radius=int(image_radius),
            )
        )
    return results[0], results[1]


__all__ = [
    "DEFAULT_CARTESIAN_STEPS_A",
    "DEFAULT_STRAIN_STEPS",
    "TransactionProposal",
    "lattice_matrix_from_dynamic_arrays",
    "lattice_parameters_from_matrix",
    "propose_force_site_transactions",
    "propose_stress_lattice_transactions",
]
