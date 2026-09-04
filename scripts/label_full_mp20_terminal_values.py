#!/usr/bin/env python3
"""Attach two CHGNet values to one shared pool of terminal DLM actions.

Every input candidate is preserved.  A valid terminal crystal receives both
the instantaneous CHGNet value ``E0`` and the value after one fixed short
cell-and-coordinate relaxation ``EK``.  Consequently, downstream single-point
and basin objectives differ only in the value they read, never in their action
support.

The optional 500-step calibration mode is intended for an already frozen,
train-only subset.  It reports within-group pairwise ordering agreement for E0
and EK against E500.  Calibration outcomes are diagnostics only: they never
filter or reorder candidates.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_BATCH_SIZE = 16
DEFAULT_FMAX_EV_PER_A = 0.1
DEFAULT_SHORT_STEPS = 64
CALIBRATION_FULL_STEPS = 500
ROUTE_B_PREFLIGHT_SHORT_STEPS = (3, 5, 10, 20)
ROUTE_B_PREFLIGHT_FULL_STEPS = 50
ROUTE_B_PREFLIGHT_CRYSTALS = 100
ROUTE_B_PREFLIGHT_SELECTION_SEED = 20260904
DEFAULT_TIE_TOLERANCE_EV_PER_ATOM = 1.0e-6
EV_PER_A3_TO_GPA = 160.21766208
ELBOW_RANK_TOLERANCE = 0.02
ELBOW_MIN_RELATIVE_NON_TIED_COVERAGE = 0.80
ELBOW_MIN_VARIANCE_RETENTION = 0.50
ELBOW_MIN_VARIED_GROUP_RETENTION = 0.80
TRAJECTORY_INITIAL_ENERGY_TOLERANCE_EV_PER_ATOM = 1.0e-5


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from *path* without assuming an upstream schema name."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield value


def validate_groups(groups: Sequence[Mapping[str, Any]]) -> None:
    """Validate only the semantic fields required by this labeler."""

    seen: set[int] = set()
    for position, group in enumerate(groups):
        if "group_idx" not in group or isinstance(group["group_idx"], bool):
            raise ValueError(f"group {position} lacks an integer group_idx")
        try:
            group_idx = int(group["group_idx"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"group {position} has an invalid group_idx") from error
        if group_idx < 0 or group_idx in seen:
            raise ValueError("group_idx values must be unique and non-negative")
        seen.add(group_idx)
        if "source" not in group:
            raise ValueError(f"group {group_idx} lacks source")
        if not isinstance(group.get("stage"), str) or not str(group["stage"]).strip():
            raise ValueError(f"group {group_idx} lacks a non-empty stage")
        if not isinstance(group.get("state"), Mapping):
            raise ValueError(f"group {group_idx} lacks a state object")
        candidates = group.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"group {group_idx} requires at least one candidate")
        for candidate_idx, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"group {group_idx} candidate {candidate_idx} is not an object"
                )
            if not isinstance(candidate.get("valid_terminal"), bool):
                raise ValueError(
                    f"group {group_idx} candidate {candidate_idx} lacks bool valid_terminal"
                )
            answer = candidate.get("terminal_answer")
            cif = candidate.get("terminal_cif")
            has_answer = isinstance(answer, str) and bool(answer.strip())
            has_cif = isinstance(cif, str) and bool(cif.strip())
            if candidate["valid_terminal"] and has_answer == has_cif:
                raise ValueError(
                    f"group {group_idx} candidate {candidate_idx} must contain exactly "
                    "one terminal_answer or terminal_cif"
                )


def select_groups_for_shard(
    groups: Sequence[Mapping[str, Any]], *, shard_rank: int, shard_count: int
) -> list[Mapping[str, Any]]:
    """Return the mutually exclusive ``group_idx % shard_count`` partition."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_rank < shard_count:
        raise ValueError("shard_rank must lie in [0, shard_count)")
    return [
        group
        for group in groups
        if int(group["group_idx"]) % int(shard_count) == int(shard_rank)
    ]


def _declared_split(group: Mapping[str, Any]) -> str | None:
    for container in (group, group.get("source"), group.get("state")):
        if not isinstance(container, Mapping):
            continue
        for key in ("split", "dataset_split", "source_split"):
            if key in container and container[key] is not None:
                return str(container[key]).strip().lower()
    return None


def validate_calibration_is_train_only(
    groups: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed when 500-step calibration is not explicitly train-only."""

    accepted = {"train", "training", "mp20_train", "mp20-train"}
    for group in groups:
        split = _declared_split(group)
        if split not in accepted:
            raise ValueError(
                f"calibration group {group['group_idx']} is not explicitly train-only"
            )


def validate_route_b_preflight(groups: Sequence[Mapping[str, Any]]) -> None:
    """Require the preregistered 100-crystal, shared-candidate train subset."""

    validate_groups(groups)
    validate_calibration_is_train_only(groups)
    if len(groups) != ROUTE_B_PREFLIGHT_CRYSTALS:
        raise ValueError("Route-B preflight requires exactly 100 train crystals")
    source_ids = {
        json.dumps(group["source"], ensure_ascii=False, sort_keys=True)
        for group in groups
    }
    if len(source_ids) != ROUTE_B_PREFLIGHT_CRYSTALS:
        raise ValueError("Route-B preflight requires 100 unique sources")
    if any(group.get("shared_terminal_pool") is not True for group in groups):
        raise ValueError("Route-B preflight requires shared terminal candidates")


def freeze_route_b_preflight_groups(
    groups: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Freeze 100 train sources by group index before reading any value field."""

    validate_groups(groups)
    validate_calibration_is_train_only(groups)
    if any(group.get("shared_terminal_pool") is not True for group in groups):
        raise ValueError("Route-B preflight requires shared terminal candidates")
    if len(groups) < ROUTE_B_PREFLIGHT_CRYSTALS:
        raise ValueError("Route-B preflight source pool contains fewer than 100 groups")
    ordered = sorted(groups, key=lambda group: int(group["group_idx"]))
    rng = random.Random(ROUTE_B_PREFLIGHT_SELECTION_SEED)
    selected_positions = sorted(
        rng.sample(range(len(ordered)), ROUTE_B_PREFLIGHT_CRYSTALS)
    )
    selected = [ordered[position] for position in selected_positions]
    validate_route_b_preflight(selected)
    return selected


def preflight_prefix(steps: int) -> str:
    if int(steps) not in (
        *ROUTE_B_PREFLIGHT_SHORT_STEPS,
        ROUTE_B_PREFLIGHT_FULL_STEPS,
    ):
        raise ValueError("unregistered Route-B preflight horizon")
    return f"terminal_preflight_k{int(steps)}"


def default_structure_loader(candidate: Mapping[str, Any]) -> Any:
    """Parse one valid terminal candidate through the deployed representations."""

    answer = candidate.get("terminal_answer")
    if isinstance(answer, str) and answer.strip():
        from crystal_dlm.dynamic_crystal import arrays_to_structure, parse_dynamic_answer

        return arrays_to_structure(parse_dynamic_answer(answer, strict=True))
    cif = candidate.get("terminal_cif")
    if isinstance(cif, str) and cif.strip():
        from pymatgen.core import Structure

        return Structure.from_str(cif, fmt="cif")
    raise ValueError("valid terminal candidate lacks a terminal structure")


def predict_batches(
    predictor: Any, structures: Sequence[Any], *, batch_size: int
) -> list[Mapping[str, Any] | None]:
    """Run fixed-size EFSM batches, retaining per-structure failures as unknown."""

    if int(batch_size) != DEFAULT_BATCH_SIZE:
        raise ValueError("terminal single-point labels fix CHGNet batch size to 16")
    output: list[Mapping[str, Any] | None] = []
    for start in range(0, len(structures), int(batch_size)):
        chunk = list(structures[start : start + int(batch_size)])
        try:
            values = predictor.predict_structure(
                chunk, task="efsm", batch_size=int(batch_size)
            )
            if isinstance(values, Mapping):
                values = [values]
            values = list(values)
            if len(values) != len(chunk):
                raise RuntimeError("CHGNet prediction count changed within a batch")
            output.extend(values)
        except Exception:
            for structure in chunk:
                try:
                    output.append(predictor.predict_structure(structure, task="efsm"))
                except Exception:
                    output.append(None)
    if len(output) != len(structures):
        raise RuntimeError("CHGNet prediction count changed")
    return output


def _finite_array(value: Any, *, shape_tail: tuple[int, ...]) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim < len(shape_tail) or tuple(array.shape[-len(shape_tail) :]) != shape_tail:
        return None
    if not np.isfinite(array).all():
        return None
    return array


def single_point_fields(
    prediction: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize one CHGNet EFSM prediction into serializable terminal fields."""

    empty = {
        "terminal_single_point_known": False,
        "terminal_single_point_energy_eV_per_atom": None,
        "terminal_single_point_forces_eV_per_A": None,
        "terminal_single_point_force_rms_eV_per_A": None,
        "terminal_single_point_force_max_eV_per_A": None,
        "terminal_single_point_stress_GPa": None,
        "terminal_single_point_stress_frobenius_GPa": None,
    }
    if prediction is None:
        return empty
    try:
        energy = float(np.asarray(prediction["e"], dtype=np.float64).reshape(()))
        forces = _finite_array(prediction["f"], shape_tail=(3,))
        stress = np.asarray(prediction["s"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return empty
    if (
        not math.isfinite(energy)
        or forces is None
        or forces.ndim != 2
        or stress.shape not in ((3, 3), (6,))
        or not np.isfinite(stress).all()
    ):
        return empty
    force_norms = np.linalg.norm(forces, axis=1)
    return {
        "terminal_single_point_known": True,
        "terminal_single_point_energy_eV_per_atom": energy,
        "terminal_single_point_forces_eV_per_A": forces.tolist(),
        "terminal_single_point_force_rms_eV_per_A": float(
            np.sqrt(np.mean(force_norms * force_norms))
        ),
        "terminal_single_point_force_max_eV_per_A": float(np.max(force_norms)),
        "terminal_single_point_stress_GPa": stress.tolist(),
        "terminal_single_point_stress_frobenius_GPa": float(
            np.linalg.norm(_stress_matrix(stress))
        ),
    }


def _stress_matrix(stress: np.ndarray) -> np.ndarray:
    if stress.shape == (3, 3):
        return stress
    if stress.shape != (6,):
        raise ValueError("stress must be a 3x3 matrix or ASE six-vector")
    xx, yy, zz, yz, xz, xy = (float(value) for value in stress)
    return np.asarray(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))


def _num_sites(structure: Any) -> int:
    try:
        count = len(structure)
    except TypeError:
        count = getattr(structure, "num_sites", None)
    if count is None or int(count) <= 0:
        raise ValueError("terminal structure has no sites")
    return int(count)


def _trajectory_steps(trajectory: Any, *, maximum_steps: int) -> int:
    for name in ("steps_taken", "n_steps", "nsteps"):
        value = getattr(trajectory, name, None)
        if value is not None:
            return max(0, min(int(value), int(maximum_steps)))
    energies = list(getattr(trajectory, "energies", ()))
    # StructOptimizer observes step zero and explicitly appends the final frame.
    return max(0, min(len(energies) - 2, int(maximum_steps)))


def relaxation_fields(
    relaxer: Any,
    structure: Any,
    *,
    steps: int,
    fmax: float,
    prefix: str,
) -> dict[str, Any]:
    """Relax one candidate and report its terminal basin state without dropping it."""

    base = {
        f"{prefix}_known": False,
        f"{prefix}_energy_eV_per_atom": None,
        f"{prefix}_final_forces_eV_per_A": None,
        f"{prefix}_final_force_rms_eV_per_A": None,
        f"{prefix}_final_force_max_eV_per_A": None,
        f"{prefix}_final_stress_eV_per_A3": None,
        f"{prefix}_final_stress_GPa": None,
        f"{prefix}_final_stress_frobenius_GPa": None,
        f"{prefix}_steps_requested": int(steps),
        f"{prefix}_steps_taken": None,
        f"{prefix}_converged_fmax": None,
        f"{prefix}_early_stopped": None,
        f"{prefix}_error": None,
    }
    try:
        result = relaxer.relax(
            structure,
            relax_cell=True,
            fmax=float(fmax),
            steps=int(steps),
            verbose=False,
        )
        trajectory = result["trajectory"]
        final_structure = result["final_structure"]
        energies = list(trajectory.energies)
        forces_history = list(trajectory.forces)
        stresses_history = list(trajectory.stresses)
        if not energies or not forces_history or not stresses_history:
            raise ValueError("relaxation trajectory is empty")
        total_energy = float(energies[-1])
        forces = np.asarray(forces_history[-1], dtype=np.float64)
        stress_eV_A3 = np.asarray(stresses_history[-1], dtype=np.float64)
        if (
            not math.isfinite(total_energy)
            or forces.ndim != 2
            or forces.shape[1] != 3
            or not np.isfinite(forces).all()
            or stress_eV_A3.shape not in ((3, 3), (6,))
            or not np.isfinite(stress_eV_A3).all()
        ):
            raise ValueError("relaxation terminal E/F/stress is non-finite or malformed")
        num_sites = _num_sites(final_structure)
        force_norms = np.linalg.norm(forces, axis=1)
        maximum_force = float(np.max(force_norms))
        steps_taken = _trajectory_steps(trajectory, maximum_steps=int(steps))
        converged = maximum_force <= float(fmax) + 1.0e-12
        stress_GPa = stress_eV_A3 * EV_PER_A3_TO_GPA
        return {
            **base,
            f"{prefix}_known": True,
            f"{prefix}_energy_eV_per_atom": total_energy / num_sites,
            f"{prefix}_final_forces_eV_per_A": forces.tolist(),
            f"{prefix}_final_force_rms_eV_per_A": float(
                np.sqrt(np.mean(force_norms * force_norms))
            ),
            f"{prefix}_final_force_max_eV_per_A": maximum_force,
            f"{prefix}_final_stress_eV_per_A3": stress_eV_A3.tolist(),
            f"{prefix}_final_stress_GPa": stress_GPa.tolist(),
            f"{prefix}_final_stress_frobenius_GPa": float(
                np.linalg.norm(_stress_matrix(stress_GPa))
            ),
            f"{prefix}_steps_taken": steps_taken,
            f"{prefix}_converged_fmax": converged,
            f"{prefix}_early_stopped": bool(converged and steps_taken < int(steps)),
        }
    except Exception as error:
        return {**base, f"{prefix}_error": f"{type(error).__name__}: {error}"}


def relaxation_path_fields(
    relaxer: Any,
    structure: Any,
    *,
    horizons: Sequence[int],
    fmax: float,
) -> dict[str, Any]:
    """Read several K values from one deterministic full-horizon trajectory."""

    steps = tuple(int(value) for value in horizons)
    expected = (*ROUTE_B_PREFLIGHT_SHORT_STEPS, ROUTE_B_PREFLIGHT_FULL_STEPS)
    if steps != expected:
        raise ValueError(f"Route-B preflight horizons are fixed at {expected}")
    unknown: dict[str, Any] = {}
    for horizon in steps:
        unknown.update(
            _unknown_relaxation_fields(
                steps=horizon,
                prefix=preflight_prefix(horizon),
                reason="relaxation_path_failed",
            )
        )
    try:
        result = relaxer.relax(
            structure,
            relax_cell=True,
            fmax=float(fmax),
            steps=ROUTE_B_PREFLIGHT_FULL_STEPS,
            verbose=False,
        )
        trajectory = result["trajectory"]
        final_structure = result["final_structure"]
        energies = list(trajectory.energies)
        forces_history = list(trajectory.forces)
        stresses_history = list(trajectory.stresses)
        if not energies or not forces_history or not stresses_history:
            raise ValueError("relaxation trajectory is empty")
        if not (len(energies) == len(forces_history) == len(stresses_history)):
            raise ValueError("relaxation trajectory E/F/stress lengths differ")
        actual_steps = _trajectory_steps(
            trajectory, maximum_steps=ROUTE_B_PREFLIGHT_FULL_STEPS
        )
        num_sites = _num_sites(final_structure)
        initial_energy = float(energies[0]) / num_sites
        index_contract_observed = (
            len(energies) >= actual_steps + 1
            and len(forces_history) >= actual_steps + 1
            and len(stresses_history) >= actual_steps + 1
        )
        output: dict[str, Any] = {
            "terminal_preflight_trajectory_initial_energy_eV_per_atom": initial_energy,
            "terminal_preflight_trajectory_frame_count": len(energies),
            "terminal_preflight_optimizer_steps_taken": int(actual_steps),
            "terminal_preflight_trajectory_index_contract_observed": bool(
                index_contract_observed
            ),
            "terminal_preflight_trajectory_index_definition": (
                "frame0_is_unrelaxed; frameK_is_after_K_optimizer_steps; "
                "early_stop_reuses_converged_endpoint"
            ),
        }
        for horizon in steps:
            # StructOptimizer observes the unrelaxed step zero, every optimizer
            # step, and one duplicate final frame.  Reading index K therefore
            # gives R_K; an early-stopped path stays at its converged endpoint.
            frame_index = min(int(horizon), int(actual_steps), len(energies) - 1)
            total_energy = float(energies[frame_index])
            forces = np.asarray(forces_history[frame_index], dtype=np.float64)
            stress_eV_A3 = np.asarray(
                stresses_history[frame_index], dtype=np.float64
            )
            if (
                not math.isfinite(total_energy)
                or forces.ndim != 2
                or forces.shape[1] != 3
                or not np.isfinite(forces).all()
                or stress_eV_A3.shape not in ((3, 3), (6,))
                or not np.isfinite(stress_eV_A3).all()
            ):
                raise ValueError("relaxation path E/F/stress is non-finite or malformed")
            force_norms = np.linalg.norm(forces, axis=1)
            maximum_force = float(np.max(force_norms))
            converged = maximum_force <= float(fmax) + 1.0e-12
            stress_GPa = stress_eV_A3 * EV_PER_A3_TO_GPA
            prefix = preflight_prefix(horizon)
            output.update(
                {
                    f"{prefix}_known": True,
                    f"{prefix}_energy_eV_per_atom": total_energy / num_sites,
                    f"{prefix}_final_forces_eV_per_A": forces.tolist(),
                    f"{prefix}_final_force_rms_eV_per_A": float(
                        np.sqrt(np.mean(force_norms * force_norms))
                    ),
                    f"{prefix}_final_force_max_eV_per_A": maximum_force,
                    f"{prefix}_final_stress_eV_per_A3": stress_eV_A3.tolist(),
                    f"{prefix}_final_stress_GPa": stress_GPa.tolist(),
                    f"{prefix}_final_stress_frobenius_GPa": float(
                        np.linalg.norm(_stress_matrix(stress_GPa))
                    ),
                    f"{prefix}_steps_requested": int(horizon),
                    f"{prefix}_steps_taken": min(int(horizon), int(actual_steps)),
                    f"{prefix}_converged_fmax": converged,
                    f"{prefix}_early_stopped": bool(
                        actual_steps < int(horizon) and converged
                    ),
                    f"{prefix}_error": None,
                }
            )
        return output
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return {
            key: (message if key.endswith("_error") else value)
            for key, value in unknown.items()
        }


def _unknown_relaxation_fields(*, steps: int, prefix: str, reason: str) -> dict[str, Any]:
    return {
        f"{prefix}_known": False,
        f"{prefix}_energy_eV_per_atom": None,
        f"{prefix}_final_forces_eV_per_A": None,
        f"{prefix}_final_force_rms_eV_per_A": None,
        f"{prefix}_final_force_max_eV_per_A": None,
        f"{prefix}_final_stress_eV_per_A3": None,
        f"{prefix}_final_stress_GPa": None,
        f"{prefix}_final_stress_frobenius_GPa": None,
        f"{prefix}_steps_requested": int(steps),
        f"{prefix}_steps_taken": None,
        f"{prefix}_converged_fmax": None,
        f"{prefix}_early_stopped": None,
        f"{prefix}_error": reason,
    }


def _blank_single_point(reason: str) -> dict[str, Any]:
    return {**single_point_fields(None), "terminal_single_point_error": reason}


def label_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    predictor: Any,
    relaxer: Any,
    structure_loader: Callable[[Mapping[str, Any]], Any] = default_structure_loader,
    batch_size: int = DEFAULT_BATCH_SIZE,
    short_steps: int = DEFAULT_SHORT_STEPS,
    fmax: float = DEFAULT_FMAX_EV_PER_A,
    calibration_full_steps: int | None = None,
    preflight_steps: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Label shared candidates with E0/EK, E500, or fixed preflight horizons."""

    validate_groups(groups)
    if int(batch_size) != DEFAULT_BATCH_SIZE:
        raise ValueError("batch_size is scientifically fixed at 16")
    if int(short_steps) <= 0:
        raise ValueError("short_steps must be positive")
    horizons = None if preflight_steps is None else tuple(int(v) for v in preflight_steps)
    if horizons is not None:
        expected = (*ROUTE_B_PREFLIGHT_SHORT_STEPS, ROUTE_B_PREFLIGHT_FULL_STEPS)
        if horizons != expected:
            raise ValueError(f"Route-B preflight horizons are fixed at {expected}")
        if calibration_full_steps is not None:
            raise ValueError("Route-B preflight and E500 calibration are separate modes")
        validate_route_b_preflight(groups)
    if calibration_full_steps is not None:
        if int(calibration_full_steps) != CALIBRATION_FULL_STEPS:
            raise ValueError("calibration_full_steps is fixed at 500")
        validate_calibration_is_train_only(groups)
    if not math.isfinite(float(fmax)) or float(fmax) <= 0.0:
        raise ValueError("fmax must be finite and positive")

    relaxation_specs = (
        [(steps, preflight_prefix(steps)) for steps in horizons]
        if horizons is not None
        else [(int(short_steps), "terminal_basin")]
    )
    if calibration_full_steps is not None:
        relaxation_specs.append(
            (int(calibration_full_steps), "terminal_calibration")
        )

    output = copy.deepcopy(list(groups))
    parsed: dict[tuple[int, int], Any] = {}
    parse_errors: dict[tuple[int, int], str] = {}
    flat_structures: list[Any] = []
    flat_keys: list[tuple[int, int]] = []
    for group_position, group in enumerate(output):
        for candidate_position, candidate in enumerate(group["candidates"]):
            key = (group_position, candidate_position)
            if candidate["valid_terminal"] is not True:
                continue
            try:
                structure = structure_loader(candidate)
                _num_sites(structure)
                parsed[key] = structure
                flat_structures.append(structure)
                flat_keys.append(key)
            except Exception as error:
                parse_errors[key] = f"{type(error).__name__}: {error}"

    predictions = predict_batches(
        predictor, flat_structures, batch_size=int(batch_size)
    )
    prediction_by_key = dict(zip(flat_keys, predictions, strict=True))

    for group_position, group in enumerate(output):
        for candidate_position, candidate in enumerate(group["candidates"]):
            key = (group_position, candidate_position)
            if candidate["valid_terminal"] is not True:
                reason = "invalid_terminal_preserved"
                candidate.update(_blank_single_point(reason))
                for steps, prefix in relaxation_specs:
                    candidate.update(
                        _unknown_relaxation_fields(
                            steps=int(steps), prefix=prefix, reason=reason
                        )
                    )
                continue
            if key not in parsed:
                reason = f"terminal_parse_failed:{parse_errors[key]}"
                candidate.update(_blank_single_point(reason))
                for steps, prefix in relaxation_specs:
                    candidate.update(
                        _unknown_relaxation_fields(
                            steps=int(steps), prefix=prefix, reason=reason
                        )
                    )
                continue
            single = single_point_fields(prediction_by_key[key])
            single["terminal_single_point_error"] = (
                None if single["terminal_single_point_known"] else "efsm_prediction_failed"
            )
            candidate.update(single)
            if horizons is not None:
                path_fields = relaxation_path_fields(
                    relaxer,
                    parsed[key],
                    horizons=horizons,
                    fmax=float(fmax),
                )
                candidate.update(path_fields)
                initial = path_fields.get(
                    "terminal_preflight_trajectory_initial_energy_eV_per_atom"
                )
                e0 = candidate.get("terminal_single_point_energy_eV_per_atom")
                candidate["terminal_preflight_trajectory_initial_matches_E0"] = (
                    None
                    if initial is None or e0 is None
                    else abs(float(initial) - float(e0))
                    <= TRAJECTORY_INITIAL_ENERGY_TOLERANCE_EV_PER_ATOM
                )
            else:
                for steps, prefix in relaxation_specs:
                    candidate.update(
                        relaxation_fields(
                            relaxer,
                            parsed[key],
                            steps=int(steps),
                            fmax=float(fmax),
                            prefix=prefix,
                        )
                    )
        group["terminal_value_labels_shared_candidates"] = True
        if horizons is None:
            group["terminal_short_steps"] = int(short_steps)
        else:
            group["route_b_preflight_horizons"] = list(horizons)
            group["route_b_preflight_reference_steps"] = (
                ROUTE_B_PREFLIGHT_FULL_STEPS
            )
        group["terminal_relax_cell"] = True
        group["terminal_fmax_eV_per_A"] = float(fmax)
        group["candidate_selection_or_filtering"] = False
    return output


def _relation(left: float, right: float, tolerance: float) -> int:
    delta = float(left) - float(right)
    if abs(delta) <= float(tolerance):
        return 0
    return -1 if delta < 0.0 else 1


def pairwise_agreement(
    groups: Sequence[Mapping[str, Any]],
    *,
    metric_field: str,
    reference_field: str = "terminal_calibration_energy_eV_per_atom",
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE_EV_PER_ATOM,
) -> dict[str, Any]:
    """Compare all within-group candidate pairs; ties are disclosed, not hidden."""

    possible = reference_known = jointly_known = reference_ties = 0
    reference_non_ties = 0
    agreements = disagreements = metric_ties = tie_agreements = 0
    for group in groups:
        candidates = list(group["candidates"])
        for left_idx in range(len(candidates)):
            for right_idx in range(left_idx + 1, len(candidates)):
                possible += 1
                left = candidates[left_idx]
                right = candidates[right_idx]
                ref_left, ref_right = left.get(reference_field), right.get(reference_field)
                if ref_left is None or ref_right is None:
                    continue
                reference_known += 1
                reference_relation = _relation(ref_left, ref_right, tie_tolerance)
                if reference_relation == 0:
                    reference_ties += 1
                else:
                    reference_non_ties += 1
                metric_left, metric_right = left.get(metric_field), right.get(metric_field)
                if metric_left is None or metric_right is None:
                    continue
                jointly_known += 1
                metric_relation = _relation(metric_left, metric_right, tie_tolerance)
                if reference_relation == 0:
                    tie_agreements += int(metric_relation == 0)
                elif metric_relation == 0:
                    metric_ties += 1
                elif metric_relation == reference_relation:
                    agreements += 1
                else:
                    disagreements += 1
    directional = agreements + disagreements
    reference_non_tie_covered = agreements + disagreements + metric_ties
    return {
        "metric_field": metric_field,
        "reference_field": reference_field,
        "tie_tolerance_eV_per_atom": float(tie_tolerance),
        "possible_pairs": possible,
        "reference_known_pairs": reference_known,
        "reference_non_tie_pairs": reference_non_ties,
        "jointly_known_pairs": jointly_known,
        "coverage_of_reference_known": (
            None if reference_known == 0 else jointly_known / reference_known
        ),
        "reference_ties": reference_ties,
        "reference_tie_agreements": tie_agreements,
        "metric_ties_on_reference_non_ties": metric_ties,
        "non_tied_directional_pairs": directional,
        "non_tied_pair_coverage": (
            None if reference_non_ties == 0 else directional / reference_non_ties
        ),
        "agreements": agreements,
        "disagreements": disagreements,
        "directional_accuracy_excluding_metric_ties": (
            None if directional == 0 else agreements / directional
        ),
        "agreement_rate_counting_metric_ties_as_unresolved": (
            None
            if reference_non_tie_covered == 0
            else agreements / reference_non_tie_covered
        ),
    }


def _kendall_counts(
    candidates: Sequence[Mapping[str, Any]],
    *,
    metric_field: str,
    reference_field: str,
    tie_tolerance: float,
) -> dict[str, int | float | None]:
    concordant = discordant = metric_only_ties = reference_only_ties = both_ties = 0
    known_pairs = 0
    for left_idx in range(len(candidates)):
        for right_idx in range(left_idx + 1, len(candidates)):
            left = candidates[left_idx]
            right = candidates[right_idx]
            metric_left, metric_right = left.get(metric_field), right.get(metric_field)
            ref_left, ref_right = left.get(reference_field), right.get(reference_field)
            if None in (metric_left, metric_right, ref_left, ref_right):
                continue
            known_pairs += 1
            metric_relation = _relation(metric_left, metric_right, tie_tolerance)
            reference_relation = _relation(ref_left, ref_right, tie_tolerance)
            if metric_relation == reference_relation == 0:
                both_ties += 1
            elif metric_relation == 0:
                metric_only_ties += 1
            elif reference_relation == 0:
                reference_only_ties += 1
            elif metric_relation == reference_relation:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + metric_only_ties)
        * (concordant + discordant + reference_only_ties)
    )
    return {
        "known_pairs": known_pairs,
        "concordant": concordant,
        "discordant": discordant,
        "metric_only_ties": metric_only_ties,
        "reference_only_ties": reference_only_ties,
        "both_ties": both_ties,
        "tau_b": (
            None
            if denominator == 0.0
            else (concordant - discordant) / denominator
        ),
    }


def kendall_tau_b_by_group(
    groups: Sequence[Mapping[str, Any]],
    *,
    metric_field: str,
    reference_field: str,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE_EV_PER_ATOM,
) -> dict[str, Any]:
    """Report pooled and macro within-group Kendall tau-b with explicit ties."""

    group_values: list[float] = []
    pooled = {
        "known_pairs": 0,
        "concordant": 0,
        "discordant": 0,
        "metric_only_ties": 0,
        "reference_only_ties": 0,
        "both_ties": 0,
    }
    for group in groups:
        counts = _kendall_counts(
            list(group["candidates"]),
            metric_field=metric_field,
            reference_field=reference_field,
            tie_tolerance=float(tie_tolerance),
        )
        for key in pooled:
            pooled[key] += int(counts[key])
        if counts["tau_b"] is not None:
            group_values.append(float(counts["tau_b"]))
    denominator = math.sqrt(
        (pooled["concordant"] + pooled["discordant"] + pooled["metric_only_ties"])
        * (
            pooled["concordant"]
            + pooled["discordant"]
            + pooled["reference_only_ties"]
        )
    )
    pooled_tau = (
        None
        if denominator == 0.0
        else (pooled["concordant"] - pooled["discordant"]) / denominator
    )
    return {
        "metric_field": metric_field,
        "reference_field": reference_field,
        "tie_tolerance_eV_per_atom": float(tie_tolerance),
        "groups": len(groups),
        "groups_with_defined_tau_b": len(group_values),
        "macro_mean_tau_b": (
            None if not group_values else float(np.mean(group_values))
        ),
        "macro_median_tau_b": (
            None if not group_values else float(np.median(group_values))
        ),
        "pooled_tau_b": pooled_tau,
        **pooled,
    }


def _describe(values: Sequence[float]) -> dict[str, int | float | None]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "q10": float(np.quantile(finite, 0.10)),
        "q90": float(np.quantile(finite, 0.90)),
    }


def energy_variation_report(
    groups: Sequence[Mapping[str, Any]],
    *,
    field: str,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE_EV_PER_ATOM,
) -> dict[str, Any]:
    """Measure within-group candidate-energy variance and pairwise gap scale."""

    group_variances: list[float] = []
    pairwise_gaps: list[float] = []
    varied_groups = 0
    for group in groups:
        values = [
            float(candidate[field])
            for candidate in group["candidates"]
            if candidate.get(field) is not None
        ]
        if len(values) < 2:
            continue
        variance = float(np.var(np.asarray(values, dtype=np.float64)))
        group_variances.append(variance)
        varied_groups += int(max(values) - min(values) > float(tie_tolerance))
        pairwise_gaps.extend(
            abs(values[left] - values[right])
            for left in range(len(values))
            for right in range(left + 1, len(values))
        )
    return {
        "field": field,
        "groups_with_at_least_two_known": len(group_variances),
        "groups_with_variation": varied_groups,
        "varied_group_fraction": (
            None
            if not group_variances
            else varied_groups / len(group_variances)
        ),
        "within_group_energy_variance_eV2_per_atom2": _describe(group_variances),
        "within_group_absolute_pair_gap_eV_per_atom": _describe(pairwise_gaps),
    }


def variance_retention_report(
    groups: Sequence[Mapping[str, Any]],
    *,
    field: str,
    baseline_field: str,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE_EV_PER_ATOM,
) -> dict[str, Any]:
    """Quantify whether a longer relaxation erases candidate energy variation."""

    ratios: list[float] = []
    baseline_varied = current_varied = paired_groups = 0
    floor = float(tie_tolerance) ** 2
    for group in groups:
        paired = [
            (float(candidate[baseline_field]), float(candidate[field]))
            for candidate in group["candidates"]
            if candidate.get(baseline_field) is not None
            and candidate.get(field) is not None
        ]
        if len(paired) < 2:
            continue
        paired_groups += 1
        baseline_values = np.asarray([value[0] for value in paired], dtype=np.float64)
        current_values = np.asarray([value[1] for value in paired], dtype=np.float64)
        baseline_variance = float(np.var(baseline_values))
        current_variance = float(np.var(current_values))
        baseline_is_varied = (
            float(np.max(baseline_values) - np.min(baseline_values))
            > float(tie_tolerance)
        )
        current_is_varied = (
            float(np.max(current_values) - np.min(current_values))
            > float(tie_tolerance)
        )
        baseline_varied += int(baseline_is_varied)
        current_varied += int(baseline_is_varied and current_is_varied)
        if baseline_variance > floor:
            ratios.append(current_variance / baseline_variance)
    ratio_summary = _describe(ratios)
    median_ratio = ratio_summary["median"]
    varied_retention = (
        None if baseline_varied == 0 else current_varied / baseline_varied
    )
    washout = (
        median_ratio is not None
        and float(median_ratio) < ELBOW_MIN_VARIANCE_RETENTION
    ) or (
        varied_retention is not None
        and float(varied_retention) < ELBOW_MIN_VARIED_GROUP_RETENTION
    )
    return {
        "field": field,
        "baseline_field": baseline_field,
        "paired_groups": paired_groups,
        "baseline_varied_groups": baseline_varied,
        "current_varied_among_baseline_varied": current_varied,
        "varied_group_retention": varied_retention,
        "per_group_variance_ratio": ratio_summary,
        "variance_washout": bool(washout),
    }


def _horizon_report(
    groups: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    field = f"{preflight_prefix(steps)}_energy_eV_per_atom"
    reference = (
        f"{preflight_prefix(ROUTE_B_PREFLIGHT_FULL_STEPS)}_energy_eV_per_atom"
    )
    baseline = (
        f"{preflight_prefix(ROUTE_B_PREFLIGHT_SHORT_STEPS[0])}_energy_eV_per_atom"
    )
    return {
        "steps": int(steps),
        "energy_field": field,
        "pairwise_ordering_vs_full50": pairwise_agreement(
            groups,
            metric_field=field,
            reference_field=reference,
            tie_tolerance=float(tie_tolerance),
        ),
        "within_group_kendall_tau_b_vs_full50": kendall_tau_b_by_group(
            groups,
            metric_field=field,
            reference_field=reference,
            tie_tolerance=float(tie_tolerance),
        ),
        "energy_variation": energy_variation_report(
            groups, field=field, tie_tolerance=float(tie_tolerance)
        ),
        "variance_retention_vs_k3": variance_retention_report(
            groups,
            field=field,
            baseline_field=baseline,
            tie_tolerance=float(tie_tolerance),
        ),
    }


def minimum_elbow_k(
    horizons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the smallest near-saturated K with adequate non-tied coverage."""

    candidates: list[tuple[int, float, float, float]] = []
    for steps in ROUTE_B_PREFLIGHT_SHORT_STEPS:
        report = horizons[str(steps)]
        pair_report = report["pairwise_ordering_vs_full50"]
        pair = pair_report[
            "agreement_rate_counting_metric_ties_as_unresolved"
        ]
        tau = report["within_group_kendall_tau_b_vs_full50"]["macro_mean_tau_b"]
        coverage = pair_report["non_tied_pair_coverage"]
        if pair is not None and tau is not None and coverage is not None:
            candidates.append((steps, float(pair), float(tau), float(coverage)))
    if not candidates:
        return {
            "selected_short_steps": None,
            "approved": False,
            "reason": "no_short_horizon_has_comparable_rank_statistics",
            "per_horizon": {},
        }
    best_pair = max(value[1] for value in candidates)
    best_tau = max(value[2] for value in candidates)
    best_non_tied_coverage = max(value[3] for value in candidates)
    decisions: dict[str, Any] = {}
    selected: int | None = None
    for steps, pair, tau, coverage in candidates:
        criteria = {
            "pairwise_within_0p02_of_best": pair >= best_pair - ELBOW_RANK_TOLERANCE,
            "kendall_within_0p02_of_best": tau >= best_tau - ELBOW_RANK_TOLERANCE,
            "non_tied_pair_coverage_ge_0p80_of_best_short_k": (
                coverage
                >= ELBOW_MIN_RELATIVE_NON_TIED_COVERAGE
                * best_non_tied_coverage
            ),
        }
        eligible = all(criteria.values())
        decisions[str(steps)] = {
            "pairwise_agreement": pair,
            "macro_mean_kendall_tau_b": tau,
            "non_tied_pair_coverage": coverage,
            "non_tied_pair_coverage_relative_to_best_short_k": (
                None
                if best_non_tied_coverage == 0.0
                else coverage / best_non_tied_coverage
            ),
            "variance_vs_k3_is_diagnostic_only": horizons[str(steps)][
                "variance_retention_vs_k3"
            ],
            "criteria": criteria,
            "eligible": eligible,
        }
        if selected is None and eligible:
            selected = steps
    return {
        "selected_short_steps": selected,
        "approved": selected is not None,
        "reason": (
            "smallest_near_saturated_horizon_with_non_tied_coverage"
            if selected is not None
            else "no_horizon_satisfies_rank_coverage_and_variation_rules"
        ),
        "rank_tolerance": ELBOW_RANK_TOLERANCE,
        "minimum_non_tied_coverage_relative_to_best_short_k": (
            ELBOW_MIN_RELATIVE_NON_TIED_COVERAGE
        ),
        "best_short_horizon_pairwise_agreement": best_pair,
        "best_short_horizon_macro_mean_kendall_tau_b": best_tau,
        "best_short_horizon_non_tied_pair_coverage": best_non_tied_coverage,
        "per_horizon": decisions,
        "prospective_or_sun_used": False,
    }


def route_b_preflight_report(
    groups: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE_EV_PER_ATOM,
    real_chgnet_runtime: bool = False,
    runtime_source_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit K={3,5,10,20} against full50 on shared terminal candidates."""

    validate_route_b_preflight(groups)

    def summarize(partition: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        horizons = {
            str(steps): _horizon_report(
                partition, steps=steps, tie_tolerance=float(tie_tolerance)
            )
            for steps in (
                *ROUTE_B_PREFLIGHT_SHORT_STEPS,
                ROUTE_B_PREFLIGHT_FULL_STEPS,
            )
        }
        full50 = (
            f"{preflight_prefix(ROUTE_B_PREFLIGHT_FULL_STEPS)}_energy_eV_per_atom"
        )
        return {
            "groups": len(partition),
            "horizons": horizons,
            "E0_baseline_vs_full50": {
                "pairwise_ordering": pairwise_agreement(
                    partition,
                    metric_field="terminal_single_point_energy_eV_per_atom",
                    reference_field=full50,
                    tie_tolerance=float(tie_tolerance),
                ),
                "within_group_kendall_tau_b": kendall_tau_b_by_group(
                    partition,
                    metric_field="terminal_single_point_energy_eV_per_atom",
                    reference_field=full50,
                    tie_tolerance=float(tie_tolerance),
                ),
            },
        }

    pooled = summarize(groups)
    stages = sorted({str(group["stage"]) for group in groups})
    elbow = minimum_elbow_k(pooled["horizons"])
    e0_pairwise = pooled["E0_baseline_vs_full50"]["pairwise_ordering"]
    e0_kendall = pooled["E0_baseline_vs_full50"]["within_group_kendall_tau_b"]
    selected_steps = elbow["selected_short_steps"]
    selected_pairwise = (
        None
        if selected_steps is None
        else pooled["horizons"][str(selected_steps)][
            "pairwise_ordering_vs_full50"
        ]
    )
    selected_agreement = (
        None
        if selected_pairwise is None
        else selected_pairwise[
            "agreement_rate_counting_metric_ties_as_unresolved"
        ]
    )
    e0_agreement = e0_pairwise[
        "agreement_rate_counting_metric_ties_as_unresolved"
    ]
    selected_beats_e0 = bool(
        selected_agreement is not None
        and e0_agreement is not None
        and float(selected_agreement) > float(e0_agreement)
    )
    runtime_candidates = [
        candidate
        for group in groups
        for candidate in group["candidates"]
        if candidate.get("valid_terminal") is True
        and candidate.get("terminal_single_point_known") is True
        and candidate.get(
            f"{preflight_prefix(ROUTE_B_PREFLIGHT_FULL_STEPS)}_known"
        )
        is True
    ]
    runtime_initial_matches = sum(
        candidate.get("terminal_preflight_trajectory_initial_matches_E0") is True
        for candidate in runtime_candidates
    )
    runtime_index_contracts = sum(
        candidate.get("terminal_preflight_trajectory_index_contract_observed")
        is True
        for candidate in runtime_candidates
    )
    runtime_passed = bool(
        real_chgnet_runtime
        and runtime_source_contract is not None
        and runtime_source_contract.get("passed") is True
        and runtime_candidates
        and runtime_initial_matches == len(runtime_candidates)
        and runtime_index_contracts == len(runtime_candidates)
    )
    route_b_approved = bool(
        elbow["approved"] and selected_beats_e0 and runtime_passed
    )
    return {
        "calibration_scope": "frozen_train_only_100_crystals",
        "shared_terminal_candidates": True,
        "short_horizons": list(ROUTE_B_PREFLIGHT_SHORT_STEPS),
        "full_reference_steps": ROUTE_B_PREFLIGHT_FULL_STEPS,
        "tie_tolerance_eV_per_atom": float(tie_tolerance),
        "candidate_selection_or_filtering": False,
        "prospective_or_final_sun_read": False,
        "pooled": pooled,
        "by_stage": {
            stage: summarize(
                [group for group in groups if str(group["stage"]) == stage]
            )
            for stage in stages
        },
        "minimum_elbow_rule": elbow,
        "E0_baseline_vs_full50": {
            "pairwise_ordering": e0_pairwise,
            "within_group_kendall_tau_b": e0_kendall,
        },
        "selected_EK_vs_E0": {
            "selected_short_steps": selected_steps,
            "selected_EK_pairwise_agreement": selected_agreement,
            "E0_pairwise_agreement": e0_agreement,
            "selected_EK_strictly_better_than_E0": selected_beats_e0,
        },
        "remote_runtime_preflight": {
            "required": True,
            "real_chgnet_runtime": bool(real_chgnet_runtime),
            "installed_source_contract": (
                None if runtime_source_contract is None else dict(runtime_source_contract)
            ),
            "eligible_candidates": len(runtime_candidates),
            "trajectory_initial_energy_matches_E0": runtime_initial_matches,
            "trajectory_index_contract_observed": runtime_index_contracts,
            "initial_energy_tolerance_eV_per_atom": (
                TRAJECTORY_INITIAL_ENERGY_TOLERANCE_EV_PER_ATOM
            ),
            "required_index_semantics": (
                "frame0 is the unrelaxed terminal crystal; frame K is after K "
                "optimizer steps; an early stop reuses the converged endpoint"
            ),
            "fake_runtime_tests_are_not_sufficient": True,
            "passed": runtime_passed,
        },
        "route_b_approved": route_b_approved,
        "route_b_approval_requirements": {
            "minimum_elbow_selected": bool(elbow["approved"]),
            "selected_EK_beats_E0_pairwise": selected_beats_e0,
            "real_chgnet_trajectory_index_preflight": runtime_passed,
        },
    }


def _variation_summary(
    groups: Sequence[Mapping[str, Any]], *, field: str, tolerance: float
) -> dict[str, int]:
    known_two = varied = 0
    for group in groups:
        values = [
            float(candidate[field])
            for candidate in group["candidates"]
            if candidate.get(field) is not None
        ]
        if len(values) >= 2:
            known_two += 1
            varied += int(max(values) - min(values) > float(tolerance))
    return {"groups_with_at_least_two_known": known_two, "groups_with_variation": varied}


def calibration_report(
    groups: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE_EV_PER_ATOM,
) -> dict[str, Any]:
    """Summarize E0/E64 ordering relative to E500, pooled and by stage."""

    metrics = {
        "E0": "terminal_single_point_energy_eV_per_atom",
        "E64": "terminal_basin_energy_eV_per_atom",
    }
    value_fields = {
        **metrics,
        "E500": "terminal_calibration_energy_eV_per_atom",
    }

    def summarize(partition: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "groups": len(partition),
            "pairwise_ordering": {
                name: pairwise_agreement(
                    partition,
                    metric_field=field,
                    tie_tolerance=float(tie_tolerance),
                )
                for name, field in metrics.items()
            },
            "variation": {
                name: _variation_summary(
                    partition, field=field, tolerance=float(tie_tolerance)
                )
                for name, field in value_fields.items()
            },
        }

    stages = sorted({str(group["stage"]) for group in groups})
    return {
        "calibration_full_steps": CALIBRATION_FULL_STEPS,
        "train_only": True,
        "candidate_selection_or_filtering": False,
        "pooled": summarize(groups),
        "by_stage": {
            stage: summarize(
                [group for group in groups if str(group["stage"]) == stage]
            )
            for stage in stages
        },
    }


def summarize_labels(
    groups: Sequence[Mapping[str, Any]],
    *,
    input_group_count: int,
    shard_rank: int,
    shard_count: int,
    short_steps: int,
    fmax: float,
    calibration_full_steps: int | None,
    tie_tolerance: float,
) -> dict[str, Any]:
    candidates = [candidate for group in groups for candidate in group["candidates"]]
    valid = [candidate for candidate in candidates if candidate["valid_terminal"] is True]
    report: dict[str, Any] = {
        "input_groups": int(input_group_count),
        "labelled_groups_in_shard": len(groups),
        "labelled_candidates_in_shard": len(candidates),
        "valid_terminal_candidates_in_shard": len(valid),
        "single_point_known": sum(
            candidate["terminal_single_point_known"] is True for candidate in candidates
        ),
        "short_basin_known": sum(
            candidate["terminal_basin_known"] is True for candidate in candidates
        ),
        "short_steps": int(short_steps),
        "fmax_eV_per_A": float(fmax),
        "relax_cell": True,
        "efsm_batch_size": DEFAULT_BATCH_SIZE,
        "shard_rank": int(shard_rank),
        "shard_count": int(shard_count),
        "shard_rule": "group_idx modulo shard_count",
        "source_or_candidate_filtering": False,
        "shared_candidates_for_E0_and_EK": True,
    }
    if calibration_full_steps is not None:
        report["full_basin_known"] = sum(
            candidate["terminal_calibration_known"] is True for candidate in candidates
        )
        report["calibration"] = calibration_report(
            groups, tie_tolerance=float(tie_tolerance)
        )
    return report


def load_runtime(device: str) -> tuple[Any, Any]:
    """Load CHGNet once and share it with predictor and StructOptimizer."""

    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer

    model = CHGNet.load(
        use_device=str(device), check_cuda_mem=False, verbose=False
    )
    return model, StructOptimizer(model=model, use_device=str(device))


def inspect_chgnet_trajectory_runtime() -> dict[str, Any]:
    """Verify frame-index semantics from the actually installed remote packages."""

    from ase.optimize.optimize import Dynamics
    from chgnet.model.dynamics import StructOptimizer, TrajectoryObserver

    relax_source = inspect.getsource(StructOptimizer.relax)
    dynamics_source = inspect.getsource(Dynamics.irun)
    observer_source = inspect.getsource(TrajectoryObserver.__call__)
    attach = relax_source.find("optimizer.attach(obs")
    run = relax_source.find("optimizer.run(")
    final_observe = relax_source.find("obs()", run)
    initial_branch = dynamics_source.find("if self.nsteps == 0:")
    initial_observe = dynamics_source.find("self.call_observers()", initial_branch)
    loop = dynamics_source.find("while not is_converged")
    increment = dynamics_source.find("self.nsteps += 1", loop)
    loop_observe = dynamics_source.find("self.call_observers()", increment)
    observer_records_efstress = all(
        marker in observer_source
        for marker in (
            "self.energies.append",
            "self.forces.append",
            "self.stresses.append",
        )
    )
    checks = {
        "observer_attached_before_optimizer_run": 0 <= attach < run,
        "explicit_final_observation_after_run": 0 <= run < final_observe,
        "step_zero_observed_before_loop": (
            0 <= initial_branch < initial_observe < loop
        ),
        "each_incremented_step_observed": 0 <= loop < increment < loop_observe,
        "trajectory_observer_records_energy_force_stress": observer_records_efstress,
    }
    return {
        "source": "installed_remote_chgnet_and_ase_runtime",
        "struct_optimizer_module": inspect.getfile(StructOptimizer),
        "ase_dynamics_module": inspect.getfile(Dynamics),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--short-steps", type=int, default=DEFAULT_SHORT_STEPS)
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX_EV_PER_A)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--route-b-preflight",
        action="store_true",
        help="Run fixed K=3,5,10,20 versus full50 on exactly 100 train groups.",
    )
    parser.add_argument(
        "--calibration-full-steps",
        type=int,
        choices=(CALIBRATION_FULL_STEPS,),
        default=None,
    )
    parser.add_argument(
        "--tie-tolerance-eV-per-atom",
        type=float,
        default=DEFAULT_TIE_TOLERANCE_EV_PER_ATOM,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    groups = list(iter_jsonl(args.candidate_groups.resolve()))
    validate_groups(groups)
    if args.route_b_preflight:
        if args.calibration_full_steps is not None:
            raise ValueError("Route-B preflight is separate from E500 calibration")
        if int(args.shard_count) != 1 or int(args.shard_rank) != 0:
            raise ValueError("Route-B 100-crystal report currently requires one complete shard")
        selected = freeze_route_b_preflight_groups(groups)
    else:
        selected = select_groups_for_shard(
            groups, shard_rank=int(args.shard_rank), shard_count=int(args.shard_count)
        )
    if args.calibration_full_steps is not None:
        validate_calibration_is_train_only(selected)
    predictor, relaxer = load_runtime(str(args.device))
    labelled = label_groups(
        selected,
        predictor=predictor,
        relaxer=relaxer,
        batch_size=int(args.batch_size),
        short_steps=int(args.short_steps),
        fmax=float(args.fmax),
        calibration_full_steps=args.calibration_full_steps,
        preflight_steps=(
            (*ROUTE_B_PREFLIGHT_SHORT_STEPS, ROUTE_B_PREFLIGHT_FULL_STEPS)
            if args.route_b_preflight
            else None
        ),
    )
    if args.route_b_preflight:
        report = {
            "input_groups": len(groups),
            "labelled_groups": len(labelled),
            "frozen_group_indices": [int(group["group_idx"]) for group in labelled],
            "selection_seed": ROUTE_B_PREFLIGHT_SELECTION_SEED,
            "selection_uses_values_or_outcomes": False,
            "efsm_batch_size": DEFAULT_BATCH_SIZE,
            "fmax_eV_per_A": float(args.fmax),
            "relax_cell": True,
            "source_or_candidate_filtering": False,
            "shared_candidates_for_all_horizons": True,
            "route_b_preflight": route_b_preflight_report(
                labelled,
                tie_tolerance=float(args.tie_tolerance_eV_per_atom),
                real_chgnet_runtime=True,
                runtime_source_contract=inspect_chgnet_trajectory_runtime(),
            ),
        }
    else:
        report = summarize_labels(
            labelled,
            input_group_count=len(groups),
            shard_rank=int(args.shard_rank),
            shard_count=int(args.shard_count),
            short_steps=int(args.short_steps),
            fmax=float(args.fmax),
            calibration_full_steps=args.calibration_full_steps,
            tie_tolerance=float(args.tie_tolerance_eV_per_atom),
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "labelled_groups.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        for group in labelled:
            handle.write(json.dumps(group, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
