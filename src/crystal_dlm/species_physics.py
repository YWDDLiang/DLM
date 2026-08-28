"""Deterministic Aufbau features for element/oxidation semantic nodes.

The representation borrows the useful physical idea of initializing ionic
species from electronic configurations, but is independent of any external
token table.  It is deliberately a feature generator, not a chemistry
validator; C³FD legality remains the exact atom/charge state machine.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from crystal_dlm.composition_pair_prior import ValenceNode


# Madelung/Aufbau filling order.  Known isolated-atom exceptions are not
# hand-patched because the feature is a smooth prior rather than a ground-state
# spectroscopy claim.
ORBITALS: tuple[tuple[int, str, int], ...] = (
    (1, "s", 2),
    (2, "s", 2),
    (2, "p", 6),
    (3, "s", 2),
    (3, "p", 6),
    (4, "s", 2),
    (3, "d", 10),
    (4, "p", 6),
    (5, "s", 2),
    (4, "d", 10),
    (5, "p", 6),
    (6, "s", 2),
    (4, "f", 14),
    (5, "d", 10),
    (6, "p", 6),
    (7, "s", 2),
    (5, "f", 14),
    (6, "d", 10),
    (7, "p", 6),
)

SHELL_CAPACITY = {n: 2 * n * n for n in range(1, 8)}


def aufbau_occupancy(electron_count: int) -> tuple[int, ...]:
    electrons = int(electron_count)
    if electrons < 0 or electrons > sum(capacity for _n, _kind, capacity in ORBITALS):
        raise ValueError(f"electron count {electrons} outside supported Aufbau range")
    remaining = electrons
    occupancy: list[int] = []
    for _n, _kind, capacity in ORBITALS:
        filled = min(int(capacity), remaining)
        occupancy.append(filled)
        remaining -= filled
    if remaining != 0:
        raise ValueError(f"failed to place {remaining} electrons")
    return tuple(occupancy)


def feature_names() -> tuple[str, ...]:
    orbital_names = tuple(f"occ_{n}{kind}" for n, kind, _capacity in ORBITALS)
    shell_names = tuple(f"shell_{n}_fraction" for n in range(1, 8))
    return (
        "atomic_number_fraction",
        "oxidation_state_scaled",
        "electron_count_fraction",
        "highest_occupied_shell_fraction",
        *orbital_names,
        *shell_names,
    )


def species_physics_vector(node: ValenceNode) -> tuple[float, ...]:
    atomic_number = int(node.atomic_number)
    oxidation = int(node.oxidation_state)
    if atomic_number <= 0 or atomic_number > 118:
        raise ValueError(f"atomic number {atomic_number} outside 1..118")
    electron_count = atomic_number - oxidation
    occupancy = aufbau_occupancy(electron_count)
    highest_shell = max(
        (n for (n, _kind, _capacity), value in zip(ORBITALS, occupancy) if value > 0),
        default=0,
    )
    orbital_features = [
        float(value) / float(capacity)
        for value, (_n, _kind, capacity) in zip(occupancy, ORBITALS)
    ]
    shell_totals = {n: 0 for n in range(1, 8)}
    for value, (n, _kind, _capacity) in zip(occupancy, ORBITALS):
        shell_totals[int(n)] += int(value)
    shell_features = [
        float(shell_totals[n]) / float(SHELL_CAPACITY[n]) for n in range(1, 8)
    ]
    vector = (
        float(atomic_number) / 118.0,
        max(-1.0, min(1.0, float(oxidation) / 8.0)),
        float(electron_count) / 118.0,
        float(highest_shell) / 7.0,
        *orbital_features,
        *shell_features,
    )
    if len(vector) != len(feature_names()) or not all(math.isfinite(value) for value in vector):
        raise RuntimeError("invalid species physics feature vector")
    return tuple(float(value) for value in vector)


def species_physics_matrix(
    nodes: Sequence[ValenceNode] | Iterable[ValenceNode],
) -> tuple[tuple[float, ...], ...]:
    return tuple(species_physics_vector(node) for node in nodes)


__all__ = [
    "ORBITALS",
    "aufbau_occupancy",
    "feature_names",
    "species_physics_matrix",
    "species_physics_vector",
]
