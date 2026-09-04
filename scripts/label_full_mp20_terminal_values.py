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
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_BATCH_SIZE = 16
DEFAULT_FMAX_EV_PER_A = 0.1
DEFAULT_SHORT_STEPS = 64
CALIBRATION_FULL_STEPS = 500
DEFAULT_TIE_TOLERANCE_EV_PER_ATOM = 1.0e-6
EV_PER_A3_TO_GPA = 160.21766208


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
) -> list[dict[str, Any]]:
    """Label shared candidates with E0/EK and optionally E500."""

    validate_groups(groups)
    if int(batch_size) != DEFAULT_BATCH_SIZE:
        raise ValueError("batch_size is scientifically fixed at 16")
    if int(short_steps) <= 0:
        raise ValueError("short_steps must be positive")
    if calibration_full_steps is not None:
        if int(calibration_full_steps) != CALIBRATION_FULL_STEPS:
            raise ValueError("calibration_full_steps is fixed at 500")
        validate_calibration_is_train_only(groups)
    if not math.isfinite(float(fmax)) or float(fmax) <= 0.0:
        raise ValueError("fmax must be finite and positive")

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
                candidate.update(
                    _unknown_relaxation_fields(
                        steps=int(short_steps), prefix="terminal_basin", reason=reason
                    )
                )
                if calibration_full_steps is not None:
                    candidate.update(
                        _unknown_relaxation_fields(
                            steps=int(calibration_full_steps),
                            prefix="terminal_calibration",
                            reason=reason,
                        )
                    )
                continue
            if key not in parsed:
                reason = f"terminal_parse_failed:{parse_errors[key]}"
                candidate.update(_blank_single_point(reason))
                candidate.update(
                    _unknown_relaxation_fields(
                        steps=int(short_steps), prefix="terminal_basin", reason=reason
                    )
                )
                if calibration_full_steps is not None:
                    candidate.update(
                        _unknown_relaxation_fields(
                            steps=int(calibration_full_steps),
                            prefix="terminal_calibration",
                            reason=reason,
                        )
                    )
                continue
            single = single_point_fields(prediction_by_key[key])
            single["terminal_single_point_error"] = (
                None if single["terminal_single_point_known"] else "efsm_prediction_failed"
            )
            candidate.update(single)
            candidate.update(
                relaxation_fields(
                    relaxer,
                    parsed[key],
                    steps=int(short_steps),
                    fmax=float(fmax),
                    prefix="terminal_basin",
                )
            )
            if calibration_full_steps is not None:
                candidate.update(
                    relaxation_fields(
                        relaxer,
                        parsed[key],
                        steps=int(calibration_full_steps),
                        fmax=float(fmax),
                        prefix="terminal_calibration",
                    )
                )
        group["terminal_value_labels_shared_candidates"] = True
        group["terminal_short_steps"] = int(short_steps)
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
    )
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
