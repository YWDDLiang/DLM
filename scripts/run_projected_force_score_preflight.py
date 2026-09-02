#!/usr/bin/env python3
"""Evaluate a feasibility-projected, quantization-aware force teacher."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure

from chgnet.model.model import CHGNet
from crystal_dlm.feasible_force_teacher import (
    bounded_force_displacement,
    periodic_pair_summary,
    project_periodic_feasible,
)
from crystal_dlm.dynamic_crystal import arrays_to_structure, parse_dynamic_answer
from run_force_score_preflight import (
    describe,
    minimum_distance,
    prediction,
    quantize,
    read_jsonl,
    structure_validity,
    to_numpy,
)


MAX_QUANTIZATION_PROJECTIONS = 1
IMAGE_RADIUS = 2
MARGIN_SCALE = 0.0
MARGIN_FLOOR_A = 0.55
MARGIN_CEILING_A = 0.55
SPECIES_MARGIN_SCALE = 0.55
SPECIES_MARGIN_FLOOR_A = 0.60
SPECIES_MARGIN_CEILING_A = 1.40


def atomic_numbers(structure: Structure) -> list[int]:
    values: list[int] = []
    for site in structure:
        if not getattr(site, "is_ordered", True):
            raise ValueError("projected force teacher requires ordered sites")
        values.append(int(site.specie.Z))
    return values


def structure_from_fractional(source: Structure, coordinates: np.ndarray) -> Structure:
    return Structure(
        source.lattice,
        source.species,
        np.asarray(coordinates, dtype=float),
        coords_are_cartesian=False,
    )


def pair_summary(structure: Structure) -> tuple[float, int]:
    return periodic_pair_summary(
        np.asarray(structure.frac_coords, dtype=float),
        np.asarray(structure.lattice.matrix, dtype=float),
        atomic_numbers(structure),
        image_radius=IMAGE_RADIUS,
        margin_scale=MARGIN_SCALE,
        margin_floor_A=MARGIN_FLOOR_A,
        margin_ceiling_A=MARGIN_CEILING_A,
    )


def species_prior_violations(structure: Structure) -> int:
    _minimum, violations = periodic_pair_summary(
        np.asarray(structure.frac_coords, dtype=float),
        np.asarray(structure.lattice.matrix, dtype=float),
        atomic_numbers(structure),
        image_radius=IMAGE_RADIUS,
        margin_scale=SPECIES_MARGIN_SCALE,
        margin_floor_A=SPECIES_MARGIN_FLOOR_A,
        margin_ceiling_A=SPECIES_MARGIN_CEILING_A,
    )
    return int(violations)


def project_structure(source: Structure) -> tuple[Structure, dict[str, Any]]:
    coordinates, report = project_periodic_feasible(
        np.asarray(source.frac_coords, dtype=float),
        np.asarray(source.lattice.matrix, dtype=float),
        atomic_numbers(source),
        image_radius=IMAGE_RADIUS,
        margin_scale=MARGIN_SCALE,
        margin_floor_A=MARGIN_FLOOR_A,
        margin_ceiling_A=MARGIN_CEILING_A,
    )
    return structure_from_fractional(source, coordinates), report.to_dict()


def quantize_feasible(source: Structure) -> tuple[Structure, dict[str, Any]]:
    """Alternate exact token quantization and analytic feasibility projection."""

    candidate = source
    rounds: list[dict[str, Any]] = []
    tokens: list[str] = []
    encoding: dict[str, Any] = {}
    for quantization_round in range(MAX_QUANTIZATION_PROJECTIONS + 1):
        quantized, tokens, encoding = quantize(candidate)
        _minimum, violations = pair_summary(quantized)
        rounds.append(
            {
                "round": quantization_round,
                "direct_valid": structure_validity(quantized),
                "projection_buffer_violations": int(violations),
                "minimum_distance_A": minimum_distance(quantized),
            }
        )
        if structure_validity(quantized):
            return quantized, {
                "converged": True,
                "rounds": rounds,
                "tokens": tokens,
                "encoding": encoding,
            }
        candidate, projection = project_structure(quantized)
        rounds[-1]["projection"] = projection
    return quantized, {
        "converged": False,
        "rounds": rounds,
        "tokens": tokens,
        "encoding": encoding,
    }


def force_proposal(structure: Structure, forces: np.ndarray) -> Structure:
    displacement = bounded_force_displacement(forces)
    proposed = Structure(
        structure.lattice,
        structure.species,
        np.asarray(structure.cart_coords, dtype=float) + displacement,
        coords_are_cartesian=True,
    )
    projected, _report = project_structure(proposed)
    return projected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("teacher_complete")]
    force_candidates = [
        row
        for row in complete
        if row["initial_valid"] and row.get("force_candidate_known") is True
    ]
    force_active = [row for row in complete if row.get("teacher_mode") == "force_projected"]
    initially_valid = [row for row in complete if row["initial_valid"]]
    return {
        "rows": len(rows),
        "complete": len(complete),
        "initial_valid": sum(row["initial_valid"] for row in complete),
        "selected_valid": sum(row["selected_valid"] for row in complete),
        "invalid_to_valid": sum(
            not row["initial_valid"] and row["selected_valid"] for row in complete
        ),
        "valid_to_invalid": sum(
            row["initial_valid"] and not row["selected_valid"] for row in complete
        ),
        "selected_projection_buffer_violations": sum(
            int(row["selected_projection_buffer_violations"]) for row in complete
        ),
        "selected_species_prior_violations": sum(
            int(row["selected_species_prior_violations"]) for row in complete
        ),
        "force_candidate_rows": len(force_candidates),
        "force_candidate_energy_lower_fraction": (
            None
            if not force_candidates
            else sum(row["force_candidate_delta_eV_per_atom"] < 0 for row in force_candidates)
            / len(force_candidates)
        ),
        "force_active_rows": len(force_active),
        "force_active_energy_lower_fraction": (
            None
            if not force_active
            else sum(row["selected_delta_eV_per_atom"] < 0 for row in force_active)
            / len(force_active)
        ),
        "selected_delta_eV_per_atom": describe(
            row["selected_delta_eV_per_atom"] for row in complete
        ),
        "force_active_delta_eV_per_atom": describe(
            row["selected_delta_eV_per_atom"] for row in force_active
        ),
        "initially_valid_energy_worsen": sum(
            row["selected_delta_eV_per_atom"] > 1.0e-8 for row in initially_valid
        ),
        "teacher_modes": {
            name: sum(row.get("teacher_mode") == name for row in complete)
            for name in ("force_projected", "barrier_only", "identity", "unresolved")
        },
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
        raise ValueError("projected Force-Score preflight requires 512 rows")
    # q0 is the exact token state seen by the DLM.  Comparing a continuous
    # perturbation against a quantized candidate would incorrectly attribute
    # ordinary round-trip error to the force teacher.
    structures = [
        arrays_to_structure(
            parse_dynamic_answer(str(row["dynamic_answer"]), strict=True)
        )
        for row in rows
    ]
    model = CHGNet.load(use_device=args.device, check_cuda_mem=True, verbose=False)
    initial_predictions = prediction(model, structures)

    barrier_candidates: list[Structure] = []
    force_candidates: list[Structure] = []
    result_rows: list[dict[str, Any]] = []
    for index, (source, structure, predicted) in enumerate(
        zip(rows, structures, initial_predictions, strict=True)
    ):
        initial_minimum, initial_violations = pair_summary(structure)
        result: dict[str, Any] = {
            "schema": "projected_force_score_teacher_row_v2",
            "row_index": index,
            "base_index": int(source["base_index"]),
            "perturbation": str(source["perturbation"]),
            "stratum": str(source["stratum"]),
            "initial_valid": structure_validity(structure),
            "initial_minimum_distance_A": float(initial_minimum),
            "initial_projection_buffer_violations": int(initial_violations),
            "initial_species_prior_violations": species_prior_violations(structure),
            "initial_teacher_known": predicted is not None,
        }
        barrier, barrier_projection = project_structure(structure)
        barrier, barrier_quantization = quantize_feasible(barrier)
        result["barrier_projection"] = barrier_projection
        result["barrier_quantization"] = barrier_quantization
        barrier_candidates.append(barrier)

        force_trusted = bool(result["initial_valid"])
        result["force_trusted_region"] = force_trusted
        if predicted is not None:
            energy = float(to_numpy(predicted["e"]).reshape(()))
            forces = to_numpy(predicted["f"])
            finite = math.isfinite(energy) and np.isfinite(forces).all()
            result["initial_teacher_known"] = bool(finite)
            if finite:
                result["initial_energy_eV_per_atom"] = energy
                force_candidate = force_proposal(structure, forces)
                force_candidate, force_quantization = quantize_feasible(force_candidate)
                result["force_quantization"] = force_quantization
                force_candidates.append(force_candidate)
            else:
                force_candidates.append(barrier)
        else:
            force_candidates.append(barrier)
        result_rows.append(result)

    barrier_predictions = prediction(model, barrier_candidates)
    force_predictions = prediction(model, force_candidates)
    selected_candidates: list[Structure] = []
    for index, (structure, barrier, force_candidate, barrier_pred, force_pred) in enumerate(
        zip(
            structures,
            barrier_candidates,
            force_candidates,
            barrier_predictions,
            force_predictions,
            strict=True,
        )
    ):
        result = result_rows[index]
        initial_known = bool(result.get("initial_teacher_known"))
        barrier_known = barrier_pred is not None
        force_known = force_pred is not None and initial_known
        barrier_minimum, barrier_violations = pair_summary(barrier)
        force_minimum, force_violations = pair_summary(force_candidate)
        barrier_valid = structure_validity(barrier)
        force_valid = structure_validity(force_candidate)
        barrier_feasible = barrier_valid
        force_feasible = force_valid
        result.update(
            {
                "barrier_candidate_known": barrier_known,
                "barrier_candidate_valid": barrier_valid,
                "barrier_candidate_feasible": barrier_feasible,
                "barrier_candidate_minimum_distance_A": float(barrier_minimum),
                "barrier_candidate_projection_buffer_violations": int(barrier_violations),
                "barrier_candidate_species_prior_violations": species_prior_violations(barrier),
                "force_candidate_known": force_known,
                "force_candidate_valid": force_valid,
                "force_candidate_feasible": force_feasible,
                "force_candidate_minimum_distance_A": float(force_minimum),
                "force_candidate_projection_buffer_violations": int(force_violations),
                "force_candidate_species_prior_violations": species_prior_violations(
                    force_candidate
                ),
            }
        )
        if barrier_known and initial_known:
            result["barrier_candidate_delta_eV_per_atom"] = float(
                to_numpy(barrier_pred["e"]).reshape(())
                - result["initial_energy_eV_per_atom"]
            )
        if force_known:
            result["force_candidate_delta_eV_per_atom"] = float(
                to_numpy(force_pred["e"]).reshape(())
                - result["initial_energy_eV_per_atom"]
            )

        use_force = bool(
            result["force_trusted_region"]
            and force_known
            and force_feasible
            and result["force_candidate_delta_eV_per_atom"] < 0.0
        )
        if use_force:
            mode = "force_projected"
            selected = force_candidate
        elif not result["initial_valid"] and barrier_known and barrier_feasible:
            mode = "barrier_only"
            selected = barrier
        elif result["initial_valid"]:
            mode = "identity"
            # The continuous source can cross the 0.5 A Direct boundary after
            # exact token quantization.  Reuse the feasibility-checked no-force
            # candidate rather than blindly re-emitting the source tokens.
            selected = barrier
        else:
            mode = "unresolved"
            selected = barrier
        result["teacher_mode"] = mode
        selected_candidates.append(selected)

    selected_predictions = prediction(model, selected_candidates)
    for result, selected, predicted in zip(
        result_rows, selected_candidates, selected_predictions, strict=True
    ):
        minimum, violations = pair_summary(selected)
        result["selected_minimum_distance_A"] = float(minimum)
        result["selected_projection_buffer_violations"] = int(violations)
        result["selected_species_prior_violations"] = species_prior_violations(selected)
        result["selected_valid"] = bool(structure_validity(selected))
        result["teacher_complete"] = bool(
            result.get("initial_teacher_known") and predicted is not None
        )
        if result["teacher_complete"]:
            result["selected_energy_eV_per_atom"] = float(
                to_numpy(predicted["e"]).reshape(())
            )
            result["selected_delta_eV_per_atom"] = (
                result["selected_energy_eV_per_atom"]
                - result["initial_energy_eV_per_atom"]
            )

    by_perturbation = defaultdict(list)
    for row in result_rows:
        by_perturbation[row["perturbation"]].append(row)
    overall = summarize(result_rows)
    near_threshold = summarize(by_perturbation["near_threshold_0p60A"])
    collision_summaries = {
        name: summarize(values)
        for name, values in sorted(by_perturbation.items())
        if name.startswith("collision_")
    }
    initially_valid_complete = [
        row
        for row in result_rows
        if row.get("teacher_complete") and row["initial_valid"]
    ]
    force_active_coverage = (
        sum(
            row.get("teacher_mode") == "force_projected"
            for row in initially_valid_complete
        )
        / len(initially_valid_complete)
        if initially_valid_complete
        else 0.0
    )
    collision_barrier_decrease = [
        row
        for row in result_rows
        if row["perturbation"].startswith("collision_")
        and row.get("barrier_candidate_projection_buffer_violations", 1)
        < row.get("initial_projection_buffer_violations", 0)
    ]
    report = {
        "schema": "projected_force_score_teacher_preflight_v2",
        "status": "complete",
        "rows_requested": len(rows),
        "rows_complete": overall["complete"],
        "teacher_coverage": overall["complete"] / len(rows),
        "initial_invalid_rows": len(rows) - overall["initial_valid"],
        "constants": {
            "image_radius": IMAGE_RADIUS,
            "hard_accept_minimum_distance_A": 0.50,
            "hard_project_minimum_distance_A": MARGIN_FLOOR_A,
            "species_prior_scale": SPECIES_MARGIN_SCALE,
            "species_prior_floor_A": SPECIES_MARGIN_FLOOR_A,
            "species_prior_ceiling_A": SPECIES_MARGIN_CEILING_A,
            "maximum_quantization_projection_rounds": MAX_QUANTIZATION_PROJECTIONS,
        },
        "overall": overall,
        "near_threshold": near_threshold,
        "collisions": collision_summaries,
        "force_active_coverage_initially_valid": force_active_coverage,
        "collision_barrier_decrease_fraction": len(collision_barrier_decrease) / 256.0,
    }
    report["supports_microstudent"] = bool(
        report["teacher_coverage"] == 1.0
        and overall["selected_valid"] == 512
        and overall["invalid_to_valid"] == report["initial_invalid_rows"]
        and overall["valid_to_invalid"] == 0
        and overall["teacher_modes"]["unresolved"] == 0
        and force_active_coverage >= 0.25
        and overall["force_active_rows"] >= 64
        and overall["force_active_delta_eV_per_atom"]["median"] <= -0.005
        and overall["initially_valid_energy_worsen"] <= 2
        and all(summary["selected_valid"] == 64 for summary in collision_summaries.values())
    )
    output.mkdir(parents=True)
    (output / "projected_force_score_teacher_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows)
    )
    (output / "PROJECTED_FORCE_SCORE_PREFLIGHT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
