#!/usr/bin/env python3
"""Evaluate CHGNet force/stress directions and dynamic-token retention."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np
from pymatgen.core import Lattice, Structure

from chgnet.model.model import CHGNet
from crystal_dlm.dynamic_crystal import (
    arrays_to_structure,
    parse_dynamic_answer,
    structure_to_dynamic_answer,
)
from crystal_dlm.fixed_slot import tokenize_answer_text


FORCE_ETA_A2_PER_EV = 0.03
MAX_ATOM_STEP_A = 0.15
STRESS_STRAIN_NORM = 0.01


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def minimum_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return math.inf
    matrix = structure.distance_matrix
    return min(
        float(matrix[left, right])
        for left in range(len(structure))
        for right in range(left + 1, len(structure))
    )


def structure_validity(structure: Structure) -> bool:
    return minimum_distance(structure) >= 0.5 and float(structure.volume) >= 0.1


def prediction(model: CHGNet, structures: list[Structure]) -> list[dict | None]:
    result: list[dict | None] = []
    for start in range(0, len(structures), 16):
        chunk = structures[start : start + 16]
        try:
            values = model.predict_structure(chunk, task="efsm", batch_size=16)
            if isinstance(values, dict):
                values = [values]
            result.extend(values)
        except Exception:
            for structure in chunk:
                try:
                    result.append(model.predict_structure(structure, task="efsm"))
                except Exception:
                    result.append(None)
    if len(result) != len(structures):
        raise RuntimeError("CHGNet prediction count changed")
    return result


def force_step(structure: Structure, forces: np.ndarray, sign: float = 1.0):
    centered = np.asarray(forces, dtype=float) - np.asarray(forces, dtype=float).mean(
        axis=0, keepdims=True
    )
    step = float(sign) * FORCE_ETA_A2_PER_EV * centered
    norms = np.linalg.norm(step, axis=1)
    scale = np.ones_like(norms)
    positive = norms > MAX_ATOM_STEP_A
    scale[positive] = MAX_ATOM_STEP_A / norms[positive]
    step *= scale[:, None]
    candidate = Structure(
        structure.lattice,
        structure.species,
        structure.cart_coords + step,
        coords_are_cartesian=True,
    )
    return candidate, float(np.sqrt(np.mean(np.sum(step * step, axis=1))))


def stress_step(structure: Structure, stress: np.ndarray, sign: float):
    tensor = 0.5 * (np.asarray(stress, dtype=float) + np.asarray(stress, dtype=float).T)
    norm = float(np.linalg.norm(tensor))
    if not math.isfinite(norm) or norm < 1.0e-12:
        return None
    strain = float(sign) * STRESS_STRAIN_NORM * tensor / norm
    matrix = np.asarray(structure.lattice.matrix) @ (np.eye(3) + strain)
    if np.linalg.det(matrix) <= 0:
        return None
    return Structure(
        Lattice(matrix),
        structure.species,
        structure.frac_coords,
        coords_are_cartesian=False,
    )


def quantize(structure: Structure):
    answer, diagnostics = structure_to_dynamic_answer(structure)
    return (
        arrays_to_structure(parse_dynamic_answer(answer, strict=True)),
        tokenize_answer_text(answer),
        diagnostics.to_dict(),
    )


def circular_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(right) - np.asarray(left) + 0.5, 1.0) - 0.5


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q10": float(np.quantile(values, 0.1)),
        "q90": float(np.quantile(values, 0.9)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row["teacher_known"]]
    return {
        "rows": len(rows),
        "teacher_known": len(known),
        "teacher_coverage": len(known) / len(rows) if rows else None,
        "initial_valid": sum(row["initial_valid"] for row in rows),
        "continuous_force_valid": sum(row.get("force_valid") is True for row in known),
        "quantized_force_valid": sum(
            row.get("quantized_force_valid") is True for row in known
        ),
        "continuous_energy_lower_fraction": (
            None
            if not known
            else sum(row["force_delta_eV_per_atom"] < 0 for row in known) / len(known)
        ),
        "quantized_energy_lower_fraction": (
            None
            if not known
            else sum(row["quantized_force_delta_eV_per_atom"] < 0 for row in known)
            / len(known)
        ),
        "force_delta_eV_per_atom": describe(
            row["force_delta_eV_per_atom"] for row in known
        ),
        "quantized_force_delta_eV_per_atom": describe(
            row["quantized_force_delta_eV_per_atom"] for row in known
        ),
        "minimum_distance_delta_A": describe(
            row["force_minimum_distance_A"] - row["initial_minimum_distance_A"]
            for row in known
        ),
        "invalid_to_valid_continuous": sum(
            not row["initial_valid"] and row["force_valid"] for row in known
        ),
        "invalid_to_valid_quantized": sum(
            not row["initial_valid"] and row["quantized_force_valid"]
            for row in known
        ),
        "valid_to_invalid_continuous": sum(
            row["initial_valid"] and not row["force_valid"] for row in known
        ),
        "valid_to_invalid_quantized": sum(
            row["initial_valid"] and not row["quantized_force_valid"]
            for row in known
        ),
        "coordinate_axes": sum(row.get("coordinate_axes", 0) for row in known),
        "sub_half_bin_axes": sum(row.get("sub_half_bin_axes", 0) for row in known),
        "hard_changed_geometry_tokens": sum(
            row.get("hard_changed_geometry_tokens", 0) for row in known
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = read_jsonl(args.data_jsonl.resolve())
    if len(rows) != 512:
        raise ValueError("Force-Score preflight requires 512 rows")
    structures = [Structure.from_dict(row["perturbed_structure"]) for row in rows]
    model = CHGNet.load(use_device=args.device, check_cuda_mem=True, verbose=False)
    initial_predictions = prediction(model, structures)

    force_candidates: list[Structure] = []
    force_owner: list[int] = []
    force_steps: dict[int, float] = {}
    negative_candidates: list[Structure] = []
    negative_owner: list[int] = []
    stress_candidates: list[Structure] = []
    stress_owner: list[tuple[int, str]] = []
    result_rows: list[dict[str, Any]] = []
    for index, (source, structure, predicted) in enumerate(
        zip(rows, structures, initial_predictions, strict=True)
    ):
        result = {
            "schema": "force_score_preflight_result_v1",
            "row_index": index,
            "base_index": int(source["base_index"]),
            "perturbation": str(source["perturbation"]),
            "stratum": str(source["stratum"]),
            "initial_valid": structure_validity(structure),
            "initial_minimum_distance_A": minimum_distance(structure),
            "initial_volume_A3": float(structure.volume),
            "teacher_known": predicted is not None,
        }
        if predicted is not None:
            energy = float(to_numpy(predicted["e"]).reshape(()))
            forces = to_numpy(predicted["f"])
            stress = to_numpy(predicted["s"])
            finite = (
                math.isfinite(energy)
                and np.isfinite(forces).all()
                and np.isfinite(stress).all()
            )
            result["teacher_known"] = bool(finite)
            if finite:
                result["initial_energy_eV_per_atom"] = energy
                result["force_max_eV_per_A"] = float(
                    np.linalg.norm(forces, axis=1).max()
                )
                result["stress_frobenius"] = float(np.linalg.norm(stress))
                candidate, rms_step = force_step(structure, forces, sign=1.0)
                force_owner.append(index)
                force_candidates.append(candidate)
                force_steps[index] = rms_step
                if index % 8 == 0:
                    control, _ = force_step(structure, forces, sign=-1.0)
                    negative_owner.append(index)
                    negative_candidates.append(control)
                if source["perturbation"] in {
                    "isotropic_compression_6pct",
                    "anisotropic_strain_6pct",
                    "shear_4pct",
                }:
                    for sign, label in ((1.0, "stress_plus"), (-1.0, "stress_minus")):
                        candidate_stress = stress_step(structure, stress, sign)
                        if candidate_stress is not None:
                            stress_owner.append((index, label))
                            stress_candidates.append(candidate_stress)
        result_rows.append(result)

    force_predictions = prediction(model, force_candidates)
    quantized_candidates = []
    quantized_owner = []
    for owner, candidate, predicted in zip(
        force_owner, force_candidates, force_predictions, strict=True
    ):
        row = result_rows[owner]
        if predicted is None:
            row["teacher_known"] = False
            continue
        quantized, quantized_tokens, diagnostics = quantize(candidate)
        initial_tokens = tokenize_answer_text(str(rows[owner]["dynamic_answer"]))
        row["force_rms_step_A"] = force_steps[owner]
        row["force_energy_eV_per_atom"] = float(to_numpy(predicted["e"]).reshape(()))
        row["force_delta_eV_per_atom"] = (
            row["force_energy_eV_per_atom"] - row["initial_energy_eV_per_atom"]
        )
        row["force_minimum_distance_A"] = minimum_distance(candidate)
        row["force_valid"] = structure_validity(candidate)
        row["hard_changed_geometry_tokens"] = sum(
            left != right for left, right in zip(initial_tokens, quantized_tokens, strict=True)
        )
        fractional_delta = circular_delta(
            structures[owner].frac_coords, candidate.frac_coords
        )
        row["coordinate_axes"] = int(fractional_delta.size)
        row["sub_half_bin_axes"] = int((np.abs(fractional_delta) * 100.0 < 0.5).sum())
        row["quantized_encoding_diagnostics"] = diagnostics
        quantized_owner.append(owner)
        quantized_candidates.append(quantized)

    quantized_predictions = prediction(model, quantized_candidates)
    for owner, candidate, predicted in zip(
        quantized_owner, quantized_candidates, quantized_predictions, strict=True
    ):
        row = result_rows[owner]
        if predicted is None:
            row["teacher_known"] = False
            continue
        row["quantized_force_energy_eV_per_atom"] = float(
            to_numpy(predicted["e"]).reshape(())
        )
        row["quantized_force_delta_eV_per_atom"] = (
            row["quantized_force_energy_eV_per_atom"]
            - row["initial_energy_eV_per_atom"]
        )
        row["quantized_force_minimum_distance_A"] = minimum_distance(candidate)
        row["quantized_force_valid"] = structure_validity(candidate)

    for owner, predicted in zip(
        negative_owner, prediction(model, negative_candidates), strict=True
    ):
        if predicted is not None:
            result_rows[owner]["negative_force_delta_eV_per_atom"] = float(
                to_numpy(predicted["e"]).reshape(())
            ) - result_rows[owner]["initial_energy_eV_per_atom"]
    for (owner, label), predicted in zip(
        stress_owner, prediction(model, stress_candidates), strict=True
    ):
        if predicted is not None:
            result_rows[owner][f"{label}_delta_eV_per_atom"] = float(
                to_numpy(predicted["e"]).reshape(())
            ) - result_rows[owner]["initial_energy_eV_per_atom"]

    complete = [
        row
        for row in result_rows
        if row.get("teacher_known")
        and "force_delta_eV_per_atom" in row
        and "quantized_force_delta_eV_per_atom" in row
    ]
    by_class = defaultdict(list)
    for row in complete:
        by_class[row["perturbation"]].append(row)
    overall = summarize(complete)
    collision = summarize(
        [row for row in complete if row["perturbation"].startswith("collision_")]
    )
    control_deltas = [
        row["negative_force_delta_eV_per_atom"]
        for row in result_rows
        if "negative_force_delta_eV_per_atom" in row
    ]
    stress = {
        label: describe(
            row[f"{label}_delta_eV_per_atom"]
            for row in result_rows
            if f"{label}_delta_eV_per_atom" in row
        )
        for label in ("stress_plus", "stress_minus")
    }
    report = {
        "schema": "force_score_teacher_preflight_v1",
        "status": "complete",
        "constants": {
            "force_eta_A2_per_eV": FORCE_ETA_A2_PER_EV,
            "max_atom_step_A": MAX_ATOM_STEP_A,
            "stress_strain_norm": STRESS_STRAIN_NORM,
        },
        "rows_requested": len(rows),
        "rows_complete": len(complete),
        "overall": overall,
        "collision": collision,
        "by_perturbation": {
            name: summarize(values) for name, values in sorted(by_class.items())
        },
        "negative_force_control_delta_eV_per_atom": describe(control_deltas),
        "stress_sign_audit": stress,
        "supports_force_score_student_preflight": bool(
            overall["teacher_coverage"] is not None
            and overall["teacher_coverage"] >= 0.9
            and overall["continuous_energy_lower_fraction"] >= 0.7
            and overall["quantized_energy_lower_fraction"] >= 0.55
            and overall["force_delta_eV_per_atom"]["median"] < 0.0
        ),
    }
    output.mkdir(parents=True)
    (output / "force_score_teacher_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows)
    )
    (output / "FORCE_SCORE_PREFLIGHT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
