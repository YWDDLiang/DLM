#!/usr/bin/env python3
"""Label the frozen 128-group SPAD basin preflight with CHGNet basin values.

This module is deliberately independent of the older fixed-100 Route-B
labeler.  Every retained action remains in the output.  A terminal-legal
action receives one batched CHGNet EFSM single point and one deterministic
20-step ``StructOptimizer`` trajectory.  Frames K=3,5,10,20 are read from
that same trajectory; if the optimizer stops early, its endpoint is reused.

The resulting data are a train-only headroom audit.  They do not query hull
data, run model494 or Direct, select states by outcome, or delete candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import json
import math
from itertools import combinations
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


INPUT_SCHEMA = "spad_basin_preflight_action_group_v1"
OUTPUT_SCHEMA = "spad_basin_preflight_labelled_group_v1"
CANDIDATE_LABEL_SCHEMA = "spad_basin_preflight_candidate_value_v1"
REPORT_SCHEMA = "spad_basin_preflight_value_report_v1"
FINAL_REPORT_SCHEMA = "spad_basin_preflight_value_final_v1"
EXPECTED_GROUPS = 128
HORIZONS = (3, 5, 10, 20)
MAX_RELAX_STEPS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_FMAX_EV_PER_A = 0.1
DEFAULT_TIE_TOLERANCE = 1.0e-6
EV_PER_A3_TO_GPA = 160.21766208
HEADROOM_THRESHOLDS_MEV = (5, 10, 20, 50)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects and attach useful line numbers to parse errors."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _candidate_source(candidate: Mapping[str, Any]) -> str | None:
    """Read the action source without coupling to one builder field spelling."""

    if candidate.get("is_no_op") is True:
        return "no_op"
    values: list[Any] = [
        candidate.get("action_source"),
        candidate.get("candidate_source"),
        candidate.get("source"),
    ]
    action = candidate.get("action")
    if isinstance(action, Mapping):
        values.extend(action.get(key) for key in ("source", "kind", "type"))
    for value in values:
        if isinstance(value, Mapping):
            for key in ("source", "kind", "type", "name"):
                nested = value.get(key)
                if nested is not None:
                    return str(nested)
        elif value is not None and str(value).strip():
            return str(value)
    return None


def _normalized_source(candidate: Mapping[str, Any]) -> str:
    value = (_candidate_source(candidate) or "unknown").strip().lower()
    return value.replace("-", "_").replace(" ", "_")


def _is_no_op(candidate: Mapping[str, Any]) -> bool:
    return _normalized_source(candidate) in {"no_op", "noop"}


def _action_id(candidate: Mapping[str, Any], fallback: int) -> Any:
    for key in (
        "action_id",
        "action_ids",
        "action_token_ids",
        "candidate_id",
        "candidate_idx",
    ):
        if key in candidate and candidate[key] is not None:
            return candidate[key]
    action = candidate.get("action")
    if isinstance(action, Mapping):
        for key in ("id", "ids", "token_ids"):
            if key in action and action[key] is not None:
                return action[key]
    return fallback


def _compact_identity(value: Any) -> str:
    """Return a short deterministic identity without serializing a whole body."""

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "b2s:" + hashlib.blake2s(encoded, digest_size=8).hexdigest()


def _terminal_identity(candidate: Mapping[str, Any]) -> dict[str, str]:
    for key in (
        "terminal_body_identity",
        "terminal_token_identity",
        "terminal_body_id",
        "terminal_body_token_ids",
        "terminal_token_ids",
        "terminal_tokens",
        "terminal_body",
    ):
        if key in candidate and candidate[key] is not None:
            value = candidate[key]
            if isinstance(value, str) and 0 < len(value) <= 80:
                identity = value
            else:
                identity = _compact_identity(value)
            return {"field": key, "identity": identity}
    for key in ("terminal_structure", "terminal_arrays", "terminal_cif"):
        if candidate.get(key) is not None:
            return {"field": key, "identity": _compact_identity(candidate[key])}
    return {
        "field": "failed_action",
        "identity": _compact_identity(
            {
                "action": _action_id(candidate, -1),
                "failure": _candidate_failure(candidate),
            }
        ),
    }


def validate_action_groups(groups: Sequence[Mapping[str, Any]]) -> None:
    """Validate the frozen, ordered 128-group action contract."""

    if len(groups) != EXPECTED_GROUPS:
        raise ValueError(f"preflight requires exactly {EXPECTED_GROUPS} groups")
    indices: list[int] = []
    for position, group in enumerate(groups):
        if group.get("schema") != INPUT_SCHEMA:
            raise ValueError(f"group {position}: action schema changed")
        if isinstance(group.get("sample_idx"), bool):
            raise ValueError(f"group {position}: invalid sample_idx")
        try:
            sample_idx = int(group["sample_idx"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"group {position}: invalid sample_idx") from error
        indices.append(sample_idx)
        state_type = str(group.get("state_type", "")).lower()
        if state_type not in {"cell", "xyz"}:
            raise ValueError(f"group {sample_idx}: state_type must be cell or xyz")
        state = group.get("state")
        cursor = group.get("cursor")
        if cursor is None and isinstance(state, Mapping):
            cursor = state.get("cursor")
        if state_type == "xyz" and not isinstance(cursor, Mapping):
            raise ValueError(f"group {sample_idx}: XYZ cursor is missing")
        candidates = group.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
            raise ValueError(f"group {sample_idx}: candidate K must lie in [1,4]")
        no_ops = 0
        for candidate_idx, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"group {sample_idx}: candidate {candidate_idx} is not an object"
                )
            if not isinstance(candidate.get("terminal_legal"), bool):
                raise ValueError(
                    f"group {sample_idx}: candidate {candidate_idx} lacks terminal_legal"
                )
            if _candidate_source(candidate) is None:
                raise ValueError(
                    f"group {sample_idx}: candidate {candidate_idx} lacks action source"
                )
            _action_id(candidate, candidate_idx)
            no_ops += int(_is_no_op(candidate))
            if candidate["terminal_legal"]:
                if not any(
                    isinstance(candidate.get(key), Mapping)
                    for key in ("terminal_structure", "terminal_arrays")
                ):
                    raise ValueError(
                        f"group {sample_idx}: legal candidate {candidate_idx} lacks "
                        "terminal_structure/terminal_arrays"
                    )
            elif not any(
                candidate.get(key) is not None
                for key in (
                    "failure",
                    "error",
                    "terminal_error",
                    "terminal_failure",
                    "failure_reason",
                )
            ):
                raise ValueError(
                    f"group {sample_idx}: illegal candidate {candidate_idx} lacks failure"
                )
        if no_ops != 1:
            raise ValueError(f"group {sample_idx}: exactly one no_op source is required")
    if indices != list(range(EXPECTED_GROUPS)):
        raise ValueError("sample_idx must be ordered and contiguous from 0 through 127")


def select_groups_for_shard(
    groups: Sequence[Mapping[str, Any]], *, shard_rank: int, shard_count: int
) -> list[Mapping[str, Any]]:
    """Return a deterministic, disjoint sample-index shard."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_rank < shard_count:
        raise ValueError("shard_rank must lie in [0, shard_count)")
    return [
        group
        for group in groups
        if int(group["sample_idx"]) % shard_count == shard_rank
    ]


def default_structure_loader(candidate: Mapping[str, Any]) -> Any:
    """Build a pymatgen Structure from the action builder's explicit arrays."""

    payload = candidate.get("terminal_structure")
    if not isinstance(payload, Mapping):
        payload = candidate.get("terminal_arrays")
    if not isinstance(payload, Mapping):
        raise ValueError("terminal_structure is missing")
    try:
        lengths = np.asarray(payload["lengths"], dtype=np.float64)
        angles = np.asarray(payload["angles"], dtype=np.float64)
        species = list(payload["species"])
        frac_coords = np.asarray(payload["frac_coords"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("terminal_structure arrays are malformed") from error
    if lengths.shape != (3,) or angles.shape != (3,):
        raise ValueError("lengths and angles must each contain three values")
    if frac_coords.ndim != 2 or frac_coords.shape[1] != 3:
        raise ValueError("frac_coords must have shape [N,3]")
    if len(species) == 0 or len(species) != len(frac_coords):
        raise ValueError("species and frac_coords lengths differ or are empty")
    if not (np.isfinite(lengths).all() and np.isfinite(angles).all()):
        raise ValueError("lattice contains non-finite values")
    if not np.isfinite(frac_coords).all():
        raise ValueError("fractional coordinates contain non-finite values")
    if np.any(lengths <= 0.0) or np.any(angles <= 0.0) or np.any(angles >= 180.0):
        raise ValueError("lattice lengths or angles are outside their physical domain")

    from pymatgen.core import Lattice, Structure

    lattice = Lattice.from_parameters(*lengths.tolist(), *angles.tolist())
    return Structure(
        lattice,
        species,
        frac_coords,
        coords_are_cartesian=False,
        validate_proximity=False,
    )


def _num_sites(structure: Any) -> int:
    try:
        count = len(structure)
    except TypeError:
        count = getattr(structure, "num_sites", None)
    if count is None or int(count) <= 0:
        raise ValueError("terminal structure has no sites")
    return int(count)


def _stress_matrix(stress: np.ndarray) -> np.ndarray:
    if stress.shape == (3, 3):
        return stress
    if stress.shape != (6,):
        raise ValueError("stress must be a 3x3 matrix or ASE six-vector")
    xx, yy, zz, yz, xz, xy = (float(value) for value in stress)
    return np.asarray(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:500]


def predict_batches(
    predictor: Any, structures: Sequence[Any], *, batch_size: int
) -> list[tuple[Mapping[str, Any] | None, str | None]]:
    """Batch EFSM at 16 and fall back one-by-one while retaining failures."""

    if int(batch_size) != DEFAULT_BATCH_SIZE:
        raise ValueError("CHGNet EFSM batch size is fixed at 16")
    output: list[tuple[Mapping[str, Any] | None, str | None]] = []
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
            output.extend((value, None) for value in values)
        except Exception as batch_error:
            for structure in chunk:
                try:
                    value = predictor.predict_structure(structure, task="efsm")
                    output.append((value, None))
                except Exception as item_error:
                    output.append(
                        (
                            None,
                            "batch="
                            + _error_text(batch_error)
                            + "; item="
                            + _error_text(item_error),
                        )
                    )
    if len(output) != len(structures):
        raise RuntimeError("CHGNet prediction count changed")
    return output


def unknown_single_point(error: str) -> dict[str, Any]:
    return {
        "known": False,
        "energy_eV_per_atom": None,
        "forces_eV_per_A": None,
        "force_rms_eV_per_A": None,
        "force_max_eV_per_A": None,
        "stress_GPa": None,
        "stress_frobenius_GPa": None,
        "error": error,
    }


def single_point_fields(
    prediction: Mapping[str, Any] | None, *, error: str | None = None
) -> dict[str, Any]:
    """Normalize CHGNet's per-atom E and per-site F/stress prediction."""

    if prediction is None:
        return unknown_single_point(error or "efsm_prediction_failed")
    try:
        energy = float(np.asarray(prediction["e"], dtype=np.float64).reshape(()))
        forces = np.asarray(prediction["f"], dtype=np.float64)
        stress = np.asarray(prediction["s"], dtype=np.float64)
        if forces.ndim != 2 or forces.shape[1] != 3:
            raise ValueError("force array must have shape [N,3]")
        stress_matrix = _stress_matrix(stress)
        if not (
            math.isfinite(energy)
            and np.isfinite(forces).all()
            and np.isfinite(stress_matrix).all()
        ):
            raise ValueError("EFSM output contains non-finite values")
        force_norms = np.linalg.norm(forces, axis=1)
        return {
            "known": True,
            "energy_eV_per_atom": energy,
            "forces_eV_per_A": forces.tolist(),
            "force_rms_eV_per_A": float(
                np.sqrt(np.mean(force_norms * force_norms))
            ),
            "force_max_eV_per_A": float(np.max(force_norms)),
            "stress_GPa": stress.tolist(),
            "stress_frobenius_GPa": float(np.linalg.norm(stress_matrix)),
            "error": None,
        }
    except Exception as normalize_error:
        return unknown_single_point(
            "malformed_efsm_prediction:" + _error_text(normalize_error)
        )


def unknown_horizon(horizon: int, error: str) -> dict[str, Any]:
    return {
        "known": False,
        "energy_eV_per_atom": None,
        "forces_eV_per_A": None,
        "force_rms_eV_per_A": None,
        "force_max_eV_per_A": None,
        "stress_eV_per_A3": None,
        "stress_GPa": None,
        "stress_frobenius_GPa": None,
        "steps_requested": int(horizon),
        "frame_index": None,
        "endpoint_reused": None,
        "converged_fmax": None,
        "error": error,
    }


def _trajectory_steps(trajectory: Any, *, maximum_steps: int) -> int:
    for name in ("steps_taken", "n_steps", "nsteps"):
        value = getattr(trajectory, name, None)
        if value is not None:
            return max(0, min(int(value), int(maximum_steps)))
    energies = list(getattr(trajectory, "energies", ()))
    # CHGNet 0.3/ASE records frame zero, each step, and a duplicate final frame.
    return max(0, min(len(energies) - 2, int(maximum_steps)))


def _trajectory_frame(
    *,
    total_energy: Any,
    forces_value: Any,
    stress_value: Any,
    num_sites: int,
    horizon: int,
    frame_index: int,
    actual_steps: int,
    fmax: float,
) -> dict[str, Any]:
    energy = float(total_energy) / int(num_sites)
    forces = np.asarray(forces_value, dtype=np.float64)
    stress_eV_A3 = np.asarray(stress_value, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[1] != 3:
        raise ValueError("trajectory force array must have shape [N,3]")
    stress_matrix_eV_A3 = _stress_matrix(stress_eV_A3)
    if not (
        math.isfinite(energy)
        and np.isfinite(forces).all()
        and np.isfinite(stress_matrix_eV_A3).all()
    ):
        raise ValueError("trajectory frame contains non-finite values")
    force_norms = np.linalg.norm(forces, axis=1)
    max_force = float(np.max(force_norms))
    stress_GPa = stress_eV_A3 * EV_PER_A3_TO_GPA
    return {
        "known": True,
        "energy_eV_per_atom": energy,
        "forces_eV_per_A": forces.tolist(),
        "force_rms_eV_per_A": float(np.sqrt(np.mean(force_norms * force_norms))),
        "force_max_eV_per_A": max_force,
        "stress_eV_per_A3": stress_eV_A3.tolist(),
        "stress_GPa": stress_GPa.tolist(),
        "stress_frobenius_GPa": float(np.linalg.norm(_stress_matrix(stress_GPa))),
        "steps_requested": int(horizon),
        "frame_index": int(frame_index),
        "endpoint_reused": bool(actual_steps < int(horizon)),
        "converged_fmax": bool(max_force <= float(fmax) + 1.0e-12),
        "error": None,
    }


def unknown_trajectory(error: str) -> dict[str, Any]:
    return {
        "known": False,
        "steps_requested": MAX_RELAX_STEPS,
        "steps_taken": None,
        "frame_count": None,
        "relax_cell": True,
        "fmax_eV_per_A": DEFAULT_FMAX_EV_PER_A,
        "frame_definition": (
            "frame0_is_unrelaxed; frameK_is_after_K_optimizer_steps; "
            "early_stop_reuses_endpoint"
        ),
        "frame0_energy_eV_per_atom": None,
        "frame0_minus_E0_eV_per_atom": None,
        "horizons": {
            str(horizon): unknown_horizon(horizon, error) for horizon in HORIZONS
        },
        "error": error,
    }


def relaxation_path(
    relaxer: Any,
    structure: Any,
    *,
    e0_energy: float | None,
    fmax: float = DEFAULT_FMAX_EV_PER_A,
) -> dict[str, Any]:
    """Read K=3/5/10/20 from exactly one deterministic 20-step trajectory."""

    try:
        result = relaxer.relax(
            structure,
            relax_cell=True,
            fmax=float(fmax),
            steps=MAX_RELAX_STEPS,
            verbose=False,
        )
        trajectory = result["trajectory"]
        energies = list(trajectory.energies)
        forces = list(trajectory.forces)
        stresses = list(trajectory.stresses)
        if not energies or not (len(energies) == len(forces) == len(stresses)):
            raise ValueError("trajectory E/F/stress arrays are empty or misaligned")
        actual_steps = _trajectory_steps(
            trajectory, maximum_steps=MAX_RELAX_STEPS
        )
        if len(energies) < actual_steps + 1:
            raise ValueError("trajectory does not contain all declared optimizer frames")
        num_sites = _num_sites(structure)
        frame0_energy = float(energies[0]) / num_sites
        if not math.isfinite(frame0_energy):
            raise ValueError("trajectory frame-zero energy is non-finite")
        horizon_values: dict[str, Any] = {}
        for horizon in HORIZONS:
            frame_index = min(int(horizon), int(actual_steps), len(energies) - 1)
            try:
                horizon_values[str(horizon)] = _trajectory_frame(
                    total_energy=energies[frame_index],
                    forces_value=forces[frame_index],
                    stress_value=stresses[frame_index],
                    num_sites=num_sites,
                    horizon=int(horizon),
                    frame_index=frame_index,
                    actual_steps=actual_steps,
                    fmax=float(fmax),
                )
            except Exception as frame_error:
                horizon_values[str(horizon)] = unknown_horizon(
                    int(horizon), "trajectory_frame_failed:" + _error_text(frame_error)
                )
        return {
            "known": all(value["known"] for value in horizon_values.values()),
            "steps_requested": MAX_RELAX_STEPS,
            "steps_taken": int(actual_steps),
            "frame_count": len(energies),
            "relax_cell": True,
            "fmax_eV_per_A": float(fmax),
            "frame_definition": (
                "frame0_is_unrelaxed; frameK_is_after_K_optimizer_steps; "
                "early_stop_reuses_endpoint"
            ),
            "frame0_energy_eV_per_atom": frame0_energy,
            "frame0_minus_E0_eV_per_atom": (
                None if e0_energy is None else frame0_energy - float(e0_energy)
            ),
            "horizons": horizon_values,
            "error": (
                None
                if all(value["known"] for value in horizon_values.values())
                else "one_or_more_trajectory_frames_failed"
            ),
        }
    except Exception as error:
        return unknown_trajectory("relaxation_failed:" + _error_text(error))


def _candidate_failure(candidate: Mapping[str, Any]) -> str:
    for key in (
        "failure",
        "error",
        "terminal_error",
        "terminal_failure",
        "failure_reason",
    ):
        if candidate.get(key) is not None:
            return f"terminal_illegal:{candidate[key]}"[:500]
    return "terminal_illegal"


def _cache_identity(
    candidate: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    terminal = _terminal_identity(candidate)
    package_version = str(runtime.get("chgnet_package_version", "unknown"))
    model = str(runtime.get("chgnet_model", "CHGNet-0.3.0"))
    key = (
        f"{terminal['identity']}|chgnet={package_version}/{model}"
        "|K=3,5,10,20|fmax=0.1|relax_cell=1"
    )
    return {
        "key": key,
        "terminal": terminal,
        "chgnet_package_version": package_version,
        "chgnet_model": model,
        "horizons": list(HORIZONS),
        "fmax_eV_per_A": DEFAULT_FMAX_EV_PER_A,
        "relax_cell": True,
    }


def _metric_value(candidate: Mapping[str, Any], metric: str) -> float | None:
    label = candidate.get("basin_value")
    if not isinstance(label, Mapping):
        return None
    if metric == "E0":
        item = label.get("E0")
    else:
        trajectory = label.get("trajectory")
        item = (
            trajectory.get("horizons", {}).get(metric.removeprefix("K"))
            if isinstance(trajectory, Mapping)
            else None
        )
    if not isinstance(item, Mapping) or item.get("known") is not True:
        return None
    value = item.get("energy_eV_per_atom")
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _tie_pair_count(values: Sequence[float], *, tolerance: float) -> int:
    return sum(
        abs(float(left) - float(right)) <= float(tolerance)
        for left, right in combinations(values, 2)
    )


def group_headroom(
    group: Mapping[str, Any], *, tie_tolerance: float = DEFAULT_TIE_TOLERANCE
) -> dict[str, Any]:
    """Compare each legal-known candidate against the retained no-op."""

    candidates = list(group["candidates"])
    no_op_indices = [index for index, row in enumerate(candidates) if _is_no_op(row)]
    by_metric: dict[str, Any] = {}
    for metric in ("E0", *(f"K{value}" for value in HORIZONS)):
        known = [
            (index, row, value)
            for index, row in enumerate(candidates)
            if row.get("terminal_legal") is True
            and (value := _metric_value(row, metric)) is not None
        ]
        no_op_value = (
            _metric_value(candidates[no_op_indices[0]], metric)
            if len(no_op_indices) == 1
            else None
        )
        if not known or no_op_value is None:
            by_metric[metric] = {
                "known": False,
                "legal_known_candidates": len(known),
                "no_op_known": no_op_value is not None,
                "best_energy_eV_per_atom": None,
                "best_minus_no_op_eV_per_atom": None,
                "headroom_meV_per_atom": None,
                "best_candidate_indices": [],
                "winning_action_sources": [],
                "best_tie_count": 0,
                "rank_tie_pairs": _tie_pair_count(
                    [value for _, _, value in known], tolerance=tie_tolerance
                ),
            }
            continue
        best = min(value for _, _, value in known)
        winners = [
            (index, row)
            for index, row, value in known
            if abs(value - best) <= float(tie_tolerance)
        ]
        delta = float(best - no_op_value)
        by_metric[metric] = {
            "known": True,
            "legal_known_candidates": len(known),
            "no_op_known": True,
            "best_energy_eV_per_atom": float(best),
            "no_op_energy_eV_per_atom": float(no_op_value),
            "best_minus_no_op_eV_per_atom": delta,
            "headroom_meV_per_atom": float(-1000.0 * delta),
            "best_candidate_indices": [index for index, _ in winners],
            "winning_action_sources": [
                _normalized_source(row) for _, row in winners
            ],
            "best_tie_count": len(winners),
            "rank_tie_pairs": _tie_pair_count(
                [value for _, _, value in known], tolerance=tie_tolerance
            ),
        }
    return {
        "no_op_candidate_indices": no_op_indices,
        "no_op_count": len(no_op_indices),
        "by_metric": by_metric,
    }


def label_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    predictor: Any,
    relaxer: Any,
    structure_loader: Callable[[Mapping[str, Any]], Any] = default_structure_loader,
    runtime_identity: Mapping[str, Any] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fmax: float = DEFAULT_FMAX_EV_PER_A,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> list[dict[str, Any]]:
    """Attach E0 and four same-trajectory basin values without filtering rows."""

    if int(batch_size) != DEFAULT_BATCH_SIZE:
        raise ValueError("batch_size is scientifically fixed at 16")
    if not math.isfinite(float(fmax)) or float(fmax) != DEFAULT_FMAX_EV_PER_A:
        raise ValueError("fmax is scientifically fixed at 0.1 eV/A")
    runtime = dict(
        runtime_identity
        or {
            "chgnet_package_version": "fake-or-unspecified",
            "chgnet_model": "CHGNet-0.3.0",
        }
    )
    output = deepcopy(list(groups))
    parsed: dict[tuple[int, int], Any] = {}
    parse_errors: dict[tuple[int, int], str] = {}
    flat_structures: list[Any] = []
    flat_keys: list[tuple[int, int]] = []
    for group_position, group in enumerate(output):
        candidates = group.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
            raise ValueError("each shard group must retain candidate K in [1,4]")
        for candidate_position, candidate in enumerate(candidates):
            key = (group_position, candidate_position)
            if candidate.get("terminal_legal") is not True:
                continue
            try:
                structure = structure_loader(candidate)
                _num_sites(structure)
                parsed[key] = structure
                flat_structures.append(structure)
                flat_keys.append(key)
            except Exception as error:
                parse_errors[key] = _error_text(error)

    predictions = predict_batches(
        predictor, flat_structures, batch_size=int(batch_size)
    )
    prediction_by_key = dict(zip(flat_keys, predictions, strict=True))

    for group_position, group in enumerate(output):
        for candidate_position, candidate in enumerate(group["candidates"]):
            key = (group_position, candidate_position)
            cache = _cache_identity(candidate, runtime)
            if candidate.get("terminal_legal") is not True:
                reason = _candidate_failure(candidate)
                e0 = unknown_single_point(reason)
                trajectory = unknown_trajectory(reason)
            elif key not in parsed:
                reason = "terminal_structure_parse_failed:" + parse_errors[key]
                e0 = unknown_single_point(reason)
                trajectory = unknown_trajectory(reason)
            else:
                prediction, prediction_error = prediction_by_key[key]
                e0 = single_point_fields(prediction, error=prediction_error)
                trajectory = relaxation_path(
                    relaxer,
                    parsed[key],
                    e0_energy=(
                        float(e0["energy_eV_per_atom"]) if e0["known"] else None
                    ),
                    fmax=float(fmax),
                )
            candidate["basin_value"] = {
                "schema": CANDIDATE_LABEL_SCHEMA,
                "E0": e0,
                "trajectory": trajectory,
                "cache_identity": cache,
            }
            # Compact scalar aliases make downstream training and audits unambiguous.
            candidate["E0_known"] = bool(e0["known"])
            candidate["E0_energy_eV_per_atom"] = e0["energy_eV_per_atom"]
            candidate["terminal_single_point_energy_eV_per_atom"] = e0[
                "energy_eV_per_atom"
            ]
            for horizon in HORIZONS:
                frame = trajectory["horizons"][str(horizon)]
                candidate[f"K{horizon}_known"] = bool(frame["known"])
                candidate[f"K{horizon}_energy_eV_per_atom"] = frame[
                    "energy_eV_per_atom"
                ]
            candidate["terminal_relax_k10_energy_eV_per_atom"] = candidate[
                "K10_energy_eV_per_atom"
            ]
        group["input_schema"] = group.get("schema")
        group["schema"] = OUTPUT_SCHEMA
        group["K"] = len(group["candidates"])
        group["basin_headroom"] = group_headroom(
            group, tie_tolerance=float(tie_tolerance)
        )
        group["candidates_filtered"] = False
        group["outcome_based_state_selection"] = False
        group["hull_model494_direct_read"] = False
    return output


def kendall_tau_b(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Compute tie-aware Kendall tau-b and expose every pair category."""

    if len(left) != len(right):
        raise ValueError("Kendall inputs must have equal length")
    concordant = discordant = left_ties = right_ties = joint_ties = 0
    for (left_a, right_a), (left_b, right_b) in combinations(
        zip(left, right), 2
    ):
        delta_left = float(left_a) - float(left_b)
        delta_right = float(right_a) - float(right_b)
        tied_left = abs(delta_left) <= float(tie_tolerance)
        tied_right = abs(delta_right) <= float(tie_tolerance)
        if tied_left and tied_right:
            joint_ties += 1
        elif tied_left:
            left_ties += 1
        elif tied_right:
            right_ties += 1
        elif delta_left * delta_right > 0.0:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    tau = None if denominator == 0.0 else (concordant - discordant) / denominator
    return {
        "tau_b": tau,
        "candidates": len(left),
        "pairs": len(left) * (len(left) - 1) // 2,
        "concordant": concordant,
        "discordant": discordant,
        "left_only_ties": left_ties,
        "right_only_ties": right_ties,
        "joint_ties": joint_ties,
    }


def _describe(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def _aggregate_headroom(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for metric in ("E0", *(f"K{value}" for value in HORIZONS)):
        entries = [group["basin_headroom"]["by_metric"][metric] for group in groups]
        known = [entry for entry in entries if entry["known"]]
        headrooms = [float(entry["headroom_meV_per_atom"]) for entry in known]
        winners = Counter(
            source
            for entry in known
            for source in entry["winning_action_sources"]
        )
        report[metric] = {
            "groups": len(entries),
            "groups_with_known_no_op_and_candidate": len(known),
            "best_minus_no_op_eV_per_atom": _describe(
                float(entry["best_minus_no_op_eV_per_atom"]) for entry in known
            ),
            "headroom_meV_per_atom": _describe(headrooms),
            "groups_above_headroom_threshold_meV": {
                str(threshold): sum(value > threshold for value in headrooms)
                for threshold in HEADROOM_THRESHOLDS_MEV
            },
            "groups_with_best_ties": sum(entry["best_tie_count"] > 1 for entry in known),
            "best_tied_candidates_total": sum(entry["best_tie_count"] for entry in known),
            "rank_tie_pairs": sum(entry["rank_tie_pairs"] for entry in entries),
            "winning_action_source_histogram": dict(sorted(winners.items())),
        }
    return report


def _aggregate_kendall(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for left_horizon, right_horizon in combinations(HORIZONS, 2):
        rows: list[dict[str, Any]] = []
        for group in groups:
            paired = [
                (left, right)
                for candidate in group["candidates"]
                if candidate.get("terminal_legal") is True
                and (left := _metric_value(candidate, f"K{left_horizon}")) is not None
                and (right := _metric_value(candidate, f"K{right_horizon}")) is not None
            ]
            if len(paired) < 2:
                continue
            rows.append(
                kendall_tau_b(
                    [value[0] for value in paired],
                    [value[1] for value in paired],
                )
            )
        defined = [float(row["tau_b"]) for row in rows if row["tau_b"] is not None]
        sums = {
            key: sum(int(row[key]) for row in rows)
            for key in (
                "concordant",
                "discordant",
                "left_only_ties",
                "right_only_ties",
                "joint_ties",
            )
        }
        denominator = math.sqrt(
            (sums["concordant"] + sums["discordant"] + sums["left_only_ties"])
            * (sums["concordant"] + sums["discordant"] + sums["right_only_ties"])
        )
        output[f"K{left_horizon}_vs_K{right_horizon}"] = {
            "groups_with_two_or_more_jointly_known": len(rows),
            "groups_with_defined_tau_b": len(defined),
            "median_tau_b": None if not defined else float(statistics.median(defined)),
            "mean_tau_b": None if not defined else float(statistics.fmean(defined)),
            "pooled_tau_b": (
                None
                if denominator == 0.0
                else (sums["concordant"] - sums["discordant"]) / denominator
            ),
            **sums,
        }
    return output


def _display_action_id(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value)
        return text if len(text) <= 80 else _compact_identity(value)
    return _compact_identity(value)


def summarize_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    shard_rank: int | None,
    shard_count: int,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = [candidate for group in groups for candidate in group["candidates"]]
    legal = [candidate for candidate in candidates if candidate.get("terminal_legal") is True]
    metric_names = ("E0", *(f"K{value}" for value in HORIZONS))
    coverage = {}
    for metric in metric_names:
        known = sum(_metric_value(candidate, metric) is not None for candidate in legal)
        coverage[metric] = {
            "known": known,
            "eligible_terminal_legal": len(legal),
            "coverage": None if not legal else known / len(legal),
        }
    failures = Counter()
    for candidate in candidates:
        label = candidate["basin_value"]
        if candidate.get("terminal_legal") is not True:
            failures["terminal_illegal"] += 1
        e0_error = label["E0"].get("error")
        if e0_error:
            failures[str(e0_error).split(":", 1)[0]] += 1
        trajectory_error = label["trajectory"].get("error")
        if trajectory_error:
            failures[str(trajectory_error).split(":", 1)[0]] += 1
    source_histogram = Counter(_normalized_source(candidate) for candidate in candidates)
    action_histogram = Counter(
        _display_action_id(_action_id(candidate, index))
        for group in groups
        for index, candidate in enumerate(group["candidates"])
    )
    candidate_index_histogram = Counter(
        str(candidate.get("candidate_idx", index))
        for group in groups
        for index, candidate in enumerate(group["candidates"])
    )
    cache_rows = [candidate["basin_value"]["cache_identity"] for candidate in candidates]
    cache_keys = [str(value["key"]) for value in cache_rows]
    cache_contract = {
        "label_set_key": _compact_identity(cache_keys),
        "candidate_keys": len(cache_keys),
        "unique_candidate_keys": len(set(cache_keys)),
        "terminal_identity_field_histogram": dict(
            sorted(Counter(value["terminal"]["field"] for value in cache_rows).items())
        ),
        "terminal_identity_in_every_candidate_key": all(
            value.get("terminal", {}).get("identity") for value in cache_rows
        ),
        "chgnet_package_version": (
            None if not cache_rows else cache_rows[0]["chgnet_package_version"]
        ),
        "chgnet_model": None if not cache_rows else cache_rows[0]["chgnet_model"],
        "horizons": list(HORIZONS),
        "fmax_eV_per_A": DEFAULT_FMAX_EV_PER_A,
        "relax_cell": True,
    }
    return {
        "schema": REPORT_SCHEMA,
        "scope": scope,
        "groups": len(groups),
        "sample_indices": [int(group["sample_idx"]) for group in groups],
        "candidates": len(candidates),
        "terminal_legal": len(legal),
        "terminal_illegal_retained": len(candidates) - len(legal),
        "shard_rank": shard_rank,
        "shard_count": int(shard_count),
        "K_histogram": dict(
            sorted(Counter(str(len(group["candidates"])) for group in groups).items())
        ),
        "candidate_index_histogram": dict(sorted(candidate_index_histogram.items())),
        "action_id_histogram": dict(sorted(action_histogram.items())),
        "action_source_histogram": dict(sorted(source_histogram.items())),
        "state_type_histogram": dict(
            sorted(Counter(str(group.get("state_type")) for group in groups).items())
        ),
        "coverage": coverage,
        "failure_histogram": dict(sorted(failures.items())),
        "headroom": _aggregate_headroom(groups),
        "kendall_tau_b": _aggregate_kendall(groups),
        "cache_identity_contract": cache_contract,
        "runtime": dict(runtime or {}),
        "scientific_contract": {
            "expected_complete_groups": EXPECTED_GROUPS,
            "single_point_batch_size": DEFAULT_BATCH_SIZE,
            "one_trajectory_per_terminal_legal_candidate": True,
            "horizons": list(HORIZONS),
            "maximum_relax_steps": MAX_RELAX_STEPS,
            "fmax_eV_per_A": DEFAULT_FMAX_EV_PER_A,
            "relax_cell": True,
            "invalid_candidates_retained": True,
            "K1_and_zero_headroom_groups_retained": True,
            "source_or_candidate_replacement": False,
            "outcome_based_state_selection": False,
            "hull_query": False,
            "model494": False,
            "direct": False,
            "reranking": False,
        },
    }


def inspect_chgnet_trajectory_runtime() -> dict[str, Any]:
    """Inspect the installed CHGNet/ASE observer order used by frame indexing."""

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
    checks = {
        "observer_attached_before_run": 0 <= attach < run,
        "explicit_final_observation": 0 <= run < final_observe,
        "frame_zero_observed": 0 <= initial_branch < initial_observe < loop,
        "each_optimizer_step_observed": 0 <= loop < increment < loop_observe,
        "observer_records_EFS": all(
            marker in observer_source
            for marker in (
                "self.energies.append",
                "self.forces.append",
                "self.stresses.append",
            )
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def runtime_identity(model: Any) -> dict[str, Any]:
    try:
        package_version = importlib.metadata.version("chgnet")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    model_value = None
    for name in ("model_name", "version", "__version__"):
        value = getattr(model, name, None)
        if value is not None and isinstance(value, (str, int, float)):
            model_value = str(value)
            break
    return {
        "chgnet_package_version": package_version,
        "chgnet_model": model_value or "default-pretrained-0.3.0",
        "trajectory_runtime": inspect_chgnet_trajectory_runtime(),
    }


def load_runtime(device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load one CHGNet 0.3 predictor shared with one StructOptimizer."""

    from chgnet.model.dynamics import StructOptimizer
    from chgnet.model.model import CHGNet

    model = CHGNet.load(
        use_device=str(device), check_cuda_mem=False, verbose=False
    )
    return (
        model,
        StructOptimizer(model=model, use_device=str(device)),
        runtime_identity(model),
    )


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_shard(args: argparse.Namespace) -> None:
    groups = list(iter_jsonl(args.candidate_groups.resolve(strict=True)))
    validate_action_groups(groups)
    selected = select_groups_for_shard(
        groups,
        shard_rank=int(args.shard_rank),
        shard_count=int(args.shard_count),
    )
    predictor, relaxer, runtime = load_runtime(str(args.device))
    labelled = label_groups(
        selected,
        predictor=predictor,
        relaxer=relaxer,
        runtime_identity=runtime,
        batch_size=int(args.batch_size),
        fmax=float(args.fmax),
        tie_tolerance=float(args.tie_tolerance),
    )
    report = summarize_groups(
        labelled,
        scope="shard",
        shard_rank=int(args.shard_rank),
        shard_count=int(args.shard_count),
        runtime=runtime,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_path = output_dir / f"labelled_groups_rank{args.shard_rank}.jsonl"
    report_path = output_dir / f"report_rank{args.shard_rank}.json"
    _write_jsonl_exclusive(groups_path, labelled)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def merge_shards(output_dir: Path, *, shard_count: int) -> dict[str, Any]:
    """Merge complete shard outputs by sample_idx and recompute all statistics."""

    output_dir = output_dir.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    for rank in range(int(shard_count)):
        groups_path = output_dir / f"labelled_groups_rank{rank}.jsonl"
        report_path = output_dir / f"report_rank{rank}.json"
        rows.extend(iter_jsonl(groups_path.resolve(strict=True)))
        shard_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    rows.sort(key=lambda group: int(group["sample_idx"]))
    if len(rows) != EXPECTED_GROUPS:
        raise ValueError(f"merged shards contain {len(rows)} rather than 128 groups")
    if [int(group["sample_idx"]) for group in rows] != list(range(EXPECTED_GROUPS)):
        raise ValueError("merged shards duplicate or omit sample_idx values")
    for group in rows:
        if group.get("schema") != OUTPUT_SCHEMA:
            raise ValueError("merged labelled group schema changed")
    for rank, report in enumerate(shard_reports):
        expected = [index for index in range(EXPECTED_GROUPS) if index % shard_count == rank]
        if report.get("sample_indices") != expected:
            raise ValueError(f"rank {rank} report does not match its deterministic shard")
    runtime = {
        "rank_runtime_identities": [report.get("runtime", {}) for report in shard_reports]
    }
    summary = summarize_groups(
        rows,
        scope="merged",
        shard_rank=None,
        shard_count=int(shard_count),
        runtime=runtime,
    )
    final = {
        **summary,
        "schema": FINAL_REPORT_SCHEMA,
        "complete": True,
        "shard_reports": [f"report_rank{rank}.json" for rank in range(shard_count)],
        "headroom_audit_only": True,
        "training_authorization_or_selection_decision": False,
    }
    labelled_path = output_dir / "labelled_groups.jsonl"
    final_path = output_dir / "PRELIGHT_VALUE_FINAL.json"
    success_path = output_dir / "_SUCCESS"
    if any(path.exists() for path in (labelled_path, final_path, success_path)):
        raise FileExistsError("merged output or _SUCCESS already exists")
    _write_jsonl_exclusive(labelled_path, rows)
    final_path.write_text(
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    success_path.touch(exist_ok=False)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-groups", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX_EV_PER_A)
    parser.add_argument("--tie-tolerance", type=float, default=DEFAULT_TIE_TOLERANCE)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.shard_count) <= 0:
        raise ValueError("shard_count must be positive")
    if args.merge_shards:
        final = merge_shards(args.output_dir, shard_count=int(args.shard_count))
        print(json.dumps(final, ensure_ascii=False, sort_keys=True))
        return
    if args.candidate_groups is None:
        raise ValueError("--candidate-groups is required for shard labelling")
    run_shard(args)


if __name__ == "__main__":
    main()
