"""H1-preserving PlanGraph schedule with a fail-closed XYZ ordering invariant."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from crystal_dlm.plangraph_v1 import (
    ensure_valid_plangraph,
    plangraph_from_plan_state,
)


def _coordinate_axis(position: int, num_atoms: int) -> str | None:
    for slot in range(int(num_atoms)):
        base = 7 + 4 * slot
        if position == base + 1:
            return "x"
        if position == base + 2:
            return "y"
        if position == base + 3:
            return "z"
    return None


def analyze_axis_schedule(
    schedule: Sequence[Sequence[int]],
    *,
    num_atoms: int,
) -> dict[str, Any]:
    """Prove coverage and that every X/Y group strictly precedes every Z group."""

    num_atoms = int(num_atoms)
    answer_length = 7 + 4 * num_atoms
    normalized = [[int(position) for position in group] for group in schedule]
    flattened = [position for group in normalized for position in group]
    coverage_exact = (
        len(flattened) == answer_length
        and len(set(flattened)) == answer_length
        and sorted(flattened) == list(range(answer_length))
    )
    position_to_group = {
        position: group_index
        for group_index, group in enumerate(normalized)
        for position in group
    }
    z_before_xy: list[dict[str, int]] = []
    for slot in range(num_atoms):
        x_position = 8 + 4 * slot
        y_position = 9 + 4 * slot
        z_position = 10 + 4 * slot
        x_group = position_to_group.get(x_position, -1)
        y_group = position_to_group.get(y_position, -1)
        z_group = position_to_group.get(z_position, -1)
        if z_group <= x_group or z_group <= y_group:
            z_before_xy.append(
                {
                    "slot": slot,
                    "x_group": x_group,
                    "y_group": y_group,
                    "z_group": z_group,
                }
            )
    coordinate_group_axes: list[list[str]] = []
    mixed_axis_coordinate_groups = 0
    for group in normalized:
        axes = sorted(
            {
                axis
                for position in group
                if (axis := _coordinate_axis(position, num_atoms)) is not None
            }
        )
        coordinate_group_axes.append(axes)
        if len(axes) > 1:
            mixed_axis_coordinate_groups += 1
    x_groups = [
        position_to_group[8 + 4 * slot] for slot in range(num_atoms)
    ]
    y_groups = [
        position_to_group[9 + 4 * slot] for slot in range(num_atoms)
    ]
    z_groups = [
        position_to_group[10 + 4 * slot] for slot in range(num_atoms)
    ]
    all_xy_precede_all_z = bool(
        coverage_exact
        and x_groups
        and y_groups
        and z_groups
        and max([*x_groups, *y_groups]) < min(z_groups)
    )
    return {
        "schema": "h1_safe_axis_schedule_invariant_v1",
        "num_atoms": num_atoms,
        "answer_length": answer_length,
        "group_count": len(normalized),
        "coverage_exact": coverage_exact,
        "mixed_axis_coordinate_groups": mixed_axis_coordinate_groups,
        "coordinate_group_axes": coordinate_group_axes,
        "z_before_xy_count": len(z_before_xy),
        "z_before_xy": z_before_xy,
        "all_xy_precede_all_z": all_xy_precede_all_z,
        "gate_passed": bool(
            coverage_exact
            and mixed_axis_coordinate_groups == 0
            and not z_before_xy
            and all_xy_precede_all_z
        ),
    }


def require_safe_axis_schedule(
    schedule: Sequence[Sequence[int]],
    *,
    num_atoms: int,
) -> dict[str, Any]:
    report = analyze_axis_schedule(schedule, num_atoms=num_atoms)
    if not report["gate_passed"]:
        raise ValueError(f"safe-axis schedule invariant failed: {report}")
    return report


def h1a2_safe_axis_generation_schedule(
    plan_state: Mapping[str, Any],
) -> list[list[int]]:
    """Compile composition/lattice, grouped X, grouped Y, then grouped Z."""

    graph = plangraph_from_plan_state(plan_state)
    ensure_valid_plangraph(graph)
    num_atoms = int(graph["composition"]["N"])
    element_positions = [7 + 4 * slot for slot in range(num_atoms)]
    schedule: list[list[int]] = [
        [0, *element_positions],
        [1, 2, 3, 4, 5, 6],
    ]
    for axis_offset in (1, 2, 3):
        for site_group in graph["site_groups"]:
            positions = sorted(
                7 + 4 * int(slot) + axis_offset
                for slot in site_group["slot_indices"]
            )
            if positions:
                schedule.append(positions)
    require_safe_axis_schedule(schedule, num_atoms=num_atoms)
    return schedule


__all__ = [
    "analyze_axis_schedule",
    "h1a2_safe_axis_generation_schedule",
    "require_safe_axis_schedule",
]
