"""Data and exact-position helpers for SGTC-DLM-v1."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping


SGTC_FORBIDDEN_TRAIN_KEYS = frozenset(
    {
        "e_above_hull",
        "ehull",
        "energy_above_hull",
        "official_e_above_hull",
        "formation_energy_per_atom",
        "energy",
        "target_energy",
        "stable",
        "stability_hint",
        "target_stability",
    }
)


def dynamic_geometry_relative_positions(num_atoms: int) -> tuple[int, ...]:
    atoms = int(num_atoms)
    if not 1 <= atoms <= 20:
        raise ValueError("SGTC requires N in 1..20")
    positions = list(range(1, 7))
    for slot in range(atoms):
        start = 7 + 4 * slot
        positions.extend((start + 1, start + 2, start + 3))
    if len(positions) != 6 + 3 * atoms:
        raise RuntimeError("SGTC dynamic geometry cardinality changed")
    return tuple(positions)


def source_ehull(row: Mapping[str, Any]) -> float:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("e_above_hull") is None:
        raise ValueError("SGTC source row lacks e_above_hull")
    value = float(metadata["e_above_hull"])
    if not math.isfinite(value):
        raise ValueError("SGTC source e_above_hull is not finite")
    return value


def strip_training_outcomes(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): strip_training_outcomes(item)
            for key, item in value.items()
            if str(key).lower() not in SGTC_FORBIDDEN_TRAIN_KEYS
        }
    if isinstance(value, list):
        return [strip_training_outcomes(item) for item in value]
    if isinstance(value, tuple):
        return [strip_training_outcomes(item) for item in value]
    return deepcopy(value)


def forbidden_training_paths(value: Any, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in SGTC_FORBIDDEN_TRAIN_KEYS:
                found.append(path)
            found.extend(forbidden_training_paths(item, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(forbidden_training_paths(item, f"{prefix}[{index}]"))
    return found


__all__ = [
    "SGTC_FORBIDDEN_TRAIN_KEYS",
    "dynamic_geometry_relative_positions",
    "forbidden_training_paths",
    "source_ehull",
    "strip_training_outcomes",
]
