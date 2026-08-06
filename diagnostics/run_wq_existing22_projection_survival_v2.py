#!/usr/bin/env python3
"""Run the existing-22 survival audit with identity-only orbit order alignment.

The frozen projection artifact uses ``canonical_storage=True``.  Because the
canonical storage key contains species, a whole-orbit species reassignment can
permute the serialized orbit list even when every orbit ID and every geometric
field is unchanged.  This amendment validates topology by orbit ID and restores
the source panel order before rendering.  It does not change any species,
geometry, sample, denominator, metric, or acceptance threshold.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from diagnostics import run_wq_existing22_projection_survival as _v1


CONTRACT_SCHEMA = (
    "wqcodiff_existing22_projection_survival_contract_v2_order_alignment"
)
EXPECTED_REORDERED_ORDINALS = (
    276,
    295,
    298,
    316,
    323,
    324,
    328,
    354,
    380,
    383,
    445,
    457,
    502,
)
GEOMETRY_FIELDS = (
    "orbit_id",
    "wyckoff_type",
    "multiplicity",
    "primitive_multiplicity",
    "chart_dimension",
    "free_coordinate",
)
EXACT_STATE_FIELDS = (
    "attempt_id",
    "space_group",
    "lattice_system",
    "lattice_chart",
    "timestep",
    "space_group_committed",
)
_ORIGINAL_SELECT = _v1.select_frozen_projections


def _orbit_by_id(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    orbits = list(state.get("orbits", ()))
    by_id = {str(orbit.get("orbit_id", "")): orbit for orbit in orbits}
    if (
        len(by_id) != len(orbits)
        or "" in by_id
        or not orbits
    ):
        raise _v1.SurvivalAuditError(
            "state has missing or duplicate orbit IDs"
        )
    return by_id


def validate_topology_invariants_by_id(
    original: Mapping[str, Any],
    projected: Mapping[str, Any],
    declared_changed_orbits: Sequence[str],
) -> None:
    """Validate exact topology while treating JSON list order as representation."""

    for field in EXACT_STATE_FIELDS:
        if original.get(field) != projected.get(field):
            raise _v1.SurvivalAuditError(
                f"projected state changed frozen field {field}"
            )
    original_orbits = list(original.get("orbits", ()))
    projected_orbits = list(projected.get("orbits", ()))
    if len(original_orbits) != len(projected_orbits):
        raise _v1.SurvivalAuditError("projected state changed orbit count")
    original_by_id = _orbit_by_id(original)
    projected_by_id = _orbit_by_id(projected)
    if set(original_by_id) != set(projected_by_id):
        raise _v1.SurvivalAuditError("projected state changed orbit ID set")

    changed: list[str] = []
    for old in original_orbits:
        orbit_id = str(old["orbit_id"])
        new = projected_by_id[orbit_id]
        if {
            field: old[field] for field in GEOMETRY_FIELDS
        } != {
            field: new[field] for field in GEOMETRY_FIELDS
        }:
            raise _v1.SurvivalAuditError(
                "projected state changed topology/geometry of orbit "
                f"{orbit_id}"
            )
        if int(old["species"]) != int(new["species"]):
            changed.append(orbit_id)
    if sorted(changed) != sorted(str(value) for value in declared_changed_orbits):
        raise _v1.SurvivalAuditError(
            "declared changed_orbit_ids do not match states"
        )
    if not changed:
        raise _v1.SurvivalAuditError(
            "a projected row must change at least one orbit"
        )
    if _v1._atom_count(original) != _v1._atom_count(projected):
        raise _v1.SurvivalAuditError(
            "projected state changed primitive atom count"
        )
    if _v1._element_set(original) != _v1._element_set(projected):
        raise _v1.SurvivalAuditError(
            "projected state changed the original element set"
        )


def align_projected_state_to_source_order(
    original: Mapping[str, Any],
    projected: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return a byte-new state whose orbit list follows the source panel order."""

    projected_by_id = _orbit_by_id(projected)
    original_order = [
        str(orbit["orbit_id"]) for orbit in original.get("orbits", ())
    ]
    projected_order = [
        str(orbit["orbit_id"]) for orbit in projected.get("orbits", ())
    ]
    if set(original_order) != set(projected_order):
        raise _v1.SurvivalAuditError(
            "cannot align states with different orbit ID sets"
        )
    aligned = dict(projected)
    aligned["orbits"] = [
        dict(projected_by_id[orbit_id]) for orbit_id in original_order
    ]
    return aligned, original_order != projected_order


def select_frozen_projections_order_aligned(
    panel_rows: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
    *,
    expected_panel_rows: int,
    expected_projection_rows: int,
    expected_ordinals: Sequence[int],
) -> list[_v1.FrozenProjection]:
    """Run the v1 identity selection, then apply a checked list-order alignment."""

    selected = _ORIGINAL_SELECT(
        panel_rows,
        projection_rows,
        expected_panel_rows=expected_panel_rows,
        expected_projection_rows=expected_projection_rows,
        expected_ordinals=expected_ordinals,
    )
    panel_by_id = {
        str(row["attempt_id"]): row for row in panel_rows
    }
    aligned_rows: list[_v1.FrozenProjection] = []
    reordered_ordinals: list[int] = []
    for value in selected:
        source_state = panel_by_id[value.attempt_id].get("state")
        if not isinstance(source_state, Mapping):
            raise _v1.SurvivalAuditError(
                "selected source panel state is absent"
            )
        aligned, reordered = align_projected_state_to_source_order(
            source_state,
            value.projected_state,
        )
        if reordered:
            reordered_ordinals.append(value.ordinal)
        aligned_rows.append(
            dataclasses.replace(
                value,
                projected_state=aligned,
                projected_state_sha256=_v1.canonical_sha256(aligned),
            )
        )
    if tuple(reordered_ordinals) != EXPECTED_REORDERED_ORDINALS:
        raise _v1.SurvivalAuditError(
            "canonical-storage reorder identity changed: "
            f"{reordered_ordinals}"
        )
    return aligned_rows


def install_amendment() -> None:
    """Install the narrow v2 behavior into the already-tested v1 engine."""

    _v1.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    _v1.validate_topology_invariants = validate_topology_invariants_by_id
    _v1.select_frozen_projections = select_frozen_projections_order_aligned
    # The v1 engine reads this module global when recording script provenance.
    _v1.__file__ = str(Path(__file__).resolve())


def main(argv: Sequence[str] | None = None) -> int:
    install_amendment()
    return _v1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
