"""Explicit orbit-level topology-event semantics."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


class TopologyEventType(str, enum.Enum):
    NONE = "none"
    BIRTH = "orbit_birth"
    DEATH = "orbit_death"
    WYCKOFF_CHANGE = "wyckoff_type_change"
    SPECIES_CHANGE = "species_change"


@dataclasses.dataclass(frozen=True, slots=True)
class TopologyEvent:
    event_type: TopologyEventType
    orbit_id: str | None = None
    target_wyckoff_type: int | None = None
    target_species: int | None = None
    new_orbit_id: str | None = None

    def __post_init__(self) -> None:
        kind = self.event_type
        if kind is TopologyEventType.NONE:
            if any(
                value is not None
                for value in (
                    self.orbit_id,
                    self.target_wyckoff_type,
                    self.target_species,
                    self.new_orbit_id,
                )
            ):
                raise ValueError("NONE event cannot carry a payload")
        elif kind is TopologyEventType.BIRTH:
            if self.orbit_id is not None or None in (
                self.target_wyckoff_type,
                self.target_species,
                self.new_orbit_id,
            ):
                raise ValueError("birth requires target type/species/new_orbit_id only")
        elif kind is TopologyEventType.DEATH:
            if not self.orbit_id or any(
                value is not None
                for value in (
                    self.target_wyckoff_type,
                    self.target_species,
                    self.new_orbit_id,
                )
            ):
                raise ValueError("death requires only orbit_id")
        elif kind is TopologyEventType.WYCKOFF_CHANGE:
            if not self.orbit_id or self.target_wyckoff_type is None:
                raise ValueError("Wyckoff change requires orbit_id and target type")
            if self.target_species is not None:
                raise ValueError("Wyckoff change preserves species")
        elif kind is TopologyEventType.SPECIES_CHANGE:
            if not self.orbit_id or self.target_species is None:
                raise ValueError("species change requires orbit_id and target species")
            if self.target_wyckoff_type is not None or self.new_orbit_id is not None:
                raise ValueError("species change preserves the chart")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["event_type"] = self.event_type.value
        return result
