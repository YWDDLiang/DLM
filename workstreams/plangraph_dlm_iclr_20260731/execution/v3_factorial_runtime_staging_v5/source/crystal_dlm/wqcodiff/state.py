"""Semantic dynamic-multiset state for the stratified Wyckoff quotient."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Iterable, Sequence


def _finite(values: Iterable[float]) -> bool:
    import math

    return all(math.isfinite(float(value)) for value in values)


@dataclasses.dataclass(frozen=True, slots=True)
class OrbitState:
    orbit_id: str
    wyckoff_type: int
    species: int
    multiplicity: int
    chart_dimension: int
    free_coordinate: tuple[float, ...]
    primitive_multiplicity: int | None = None

    def __post_init__(self) -> None:
        if not self.orbit_id:
            raise ValueError("orbit_id is required")
        if self.wyckoff_type < 0 or self.species <= 0:
            raise ValueError("wyckoff_type and atomic number must be positive-domain values")
        if self.multiplicity <= 0:
            raise ValueError("multiplicity must be positive")
        if self.primitive_multiplicity is None:
            object.__setattr__(self, "primitive_multiplicity", self.multiplicity)
        if int(self.primitive_multiplicity) <= 0 or self.multiplicity % int(self.primitive_multiplicity) != 0:
            raise ValueError("primitive multiplicity must be a positive divisor of conventional multiplicity")
        if self.chart_dimension not in {0, 1, 2, 3}:
            raise ValueError("orbit chart dimension must be in {0,1,2,3}")
        if len(self.free_coordinate) != self.chart_dimension:
            raise ValueError("free-coordinate length does not match chart dimension")
        if not _finite(self.free_coordinate):
            raise ValueError("free coordinates must be finite")

    def storage_key(self) -> tuple[Any, ...]:
        rounded = tuple(round(value % 1.0, 12) for value in self.free_coordinate)
        return self.wyckoff_type, self.species, rounded, self.orbit_id

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["free_coordinate"] = list(self.free_coordinate)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrbitState":
        payload = dict(data)
        payload["free_coordinate"] = tuple(float(v) for v in payload["free_coordinate"])
        return cls(**payload)


@dataclasses.dataclass(frozen=True, slots=True)
class StratifiedState:
    space_group: int
    lattice_system: str
    lattice_chart: tuple[float, ...]
    orbits: tuple[OrbitState, ...]
    attempt_id: str = ""
    timestep: float = 0.0
    space_group_committed: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.space_group <= 230:
            raise ValueError("space_group must be in [1, 230]")
        if not self.lattice_system:
            raise ValueError("lattice_system is required")
        if not 1 <= len(self.lattice_chart) <= 6 or not _finite(self.lattice_chart):
            raise ValueError("lattice chart must contain 1-6 finite values")
        if not self.orbits:
            raise ValueError("a semantic crystal state must contain at least one orbit")
        ids = [orbit.orbit_id for orbit in self.orbits]
        if len(ids) != len(set(ids)):
            raise ValueError("orbit IDs must be unique")
        if not 0.0 <= float(self.timestep) <= 1.0:
            raise ValueError("timestep must be in [0,1]")
        if not 1 <= self.atom_count <= 20:
            raise ValueError("MP20 state atom count must be in [1,20]")

    @property
    def atom_count(self) -> int:
        return sum(int(orbit.primitive_multiplicity) for orbit in self.orbits)

    @property
    def conventional_atom_count(self) -> int:
        return sum(orbit.multiplicity for orbit in self.orbits)

    @property
    def continuous_dimension(self) -> int:
        return len(self.lattice_chart) + sum(orbit.chart_dimension for orbit in self.orbits)

    @property
    def field_count(self) -> int:
        # existence, Wyckoff type, and species per orbit; SG is fixed once committed.
        return 3 * len(self.orbits) + (0 if self.space_group_committed else 1)

    def canonical_orbits(self) -> tuple[OrbitState, ...]:
        return tuple(sorted(self.orbits, key=OrbitState.storage_key))

    def permuted(self, rng: random.Random) -> "StratifiedState":
        values = list(self.orbits)
        rng.shuffle(values)
        return dataclasses.replace(self, orbits=tuple(values))

    def replace_orbits(self, orbits: Sequence[OrbitState]) -> "StratifiedState":
        return dataclasses.replace(self, orbits=tuple(orbits))

    def topology_hash(self, *, include_geometry: bool = False) -> str:
        payload: dict[str, Any] = {
            "space_group": self.space_group,
            "orbits": [
                {
                    "wyckoff_type": orbit.wyckoff_type,
                    "species": orbit.species,
                    "multiplicity": orbit.multiplicity,
                    "primitive_multiplicity": orbit.primitive_multiplicity,
                    **(
                        {"free_coordinate": [round(v % 1.0, 10) for v in orbit.free_coordinate]}
                        if include_geometry
                        else {}
                    ),
                }
                for orbit in self.canonical_orbits()
            ],
        }
        if include_geometry:
            payload["lattice_system"] = self.lattice_system
            payload["lattice_chart"] = [round(v, 10) for v in self.lattice_chart]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, canonical_storage: bool = True) -> dict[str, Any]:
        orbits = self.canonical_orbits() if canonical_storage else self.orbits
        return {
            "space_group": self.space_group,
            "lattice_system": self.lattice_system,
            "lattice_chart": list(self.lattice_chart),
            "orbits": [orbit.to_dict() for orbit in orbits],
            "attempt_id": self.attempt_id,
            "timestep": self.timestep,
            "space_group_committed": self.space_group_committed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StratifiedState":
        return cls(
            space_group=int(data["space_group"]),
            lattice_system=str(data["lattice_system"]),
            lattice_chart=tuple(float(v) for v in data["lattice_chart"]),
            orbits=tuple(OrbitState.from_dict(item) for item in data["orbits"]),
            attempt_id=str(data.get("attempt_id", "")),
            timestep=float(data.get("timestep", 0.0)),
            space_group_committed=bool(data.get("space_group_committed", True)),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryEvidence:
    collision_deficit: float
    coordination_anomaly: float
    lattice_strain: float
    symmetry_residual: float
    score_norm: float
    basin_uncertainty: float

    def as_tuple(self) -> tuple[float, ...]:
        return dataclasses.astuple(self)

    def __post_init__(self) -> None:
        if not _finite(self.as_tuple()):
            raise ValueError("geometry evidence must be finite")
