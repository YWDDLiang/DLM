"""Small, synthetic CPU counterexamples; never a trained-model result.

Run from the project root with its CPU Python and PYTHONPATH=src. The frozen
label source is imported without invoking its GPU worker or command-line entry.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_frozen(name: str, relative: str):
    path = EVIDENCE / "frozen" / (relative + ".txt")
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def main():
    labels = load_frozen("frozen_physics_labels", "scripts/label_programmed_paths.py")
    evaluate = load_frozen("frozen_physics_evaluation", "scripts/evaluate_programmed_paths.py")

    # The frozen [−2,2]^3 shell misses b−3a in a legal numeric-token lattice.
    gamma = math.radians(8.)
    lattice = np.array([[1., 0., 0.], [3. * math.cos(gamma), 3. * math.sin(gamma), 0.], [0., 0., 4.]])
    structure = SimpleNamespace(lattice=SimpleNamespace(matrix=lattice),
                                frac_coords=np.zeros((1, 3)), num_sites=1)
    shell_minimum = labels.validate_structure_geometry(structure)
    shifts = np.stack(np.meshgrid(*([np.arange(-5, 6)] * 3), indexing="ij"), -1).reshape(-1, 3)
    shifts = shifts[np.any(shifts != 0, axis=-1)]
    distances = np.linalg.norm(shifts @ structure.lattice.matrix, axis=-1)
    index = int(np.argmin(distances))
    assert shell_minimum >= .5 and distances[index] < .5

    # Analytic reference-state FrechetCellFilter criterion, not an ASE run:
    # at deformation I and exp_cell_factor=N, cell gradient = −V sigma/N.
    # ASE is not installed in this CPU runtime, and no package is installed here.
    volume_per_atom = 18.
    stress_eV_A3 = .6 / labels.EV_A3_TO_GPA
    filter_fmax = volume_per_atom * stress_eV_A3
    force_criterion_passes = filter_fmax < .1
    measured = labels.force_and_stress(np.zeros((1, 3)),
        np.array([stress_eV_A3, stress_eV_A3, stress_eV_A3, 0., 0., 0.]), stress_unit="eV/A3")
    assert force_criterion_passes and measured["stress_max_GPa"] > .5

    # The retained threshold classifier deliberately does not require verification.
    threshold_only = evaluate.classify_stability(
        verified=False, energy=-5., hull_energy=-4.9, novel=True, unique=True)
    missing_hull = evaluate.classify_stability(
        verified=True, energy=-5., hull_energy=None, novel=True, unique=True)
    assert threshold_only["strict_sun"] and not threshold_only["verified_strict_sun"]
    assert not missing_hull["strict_sun"] and missing_hull["e_above_hull_eV_atom"] is None

    # Entrywise maximum stress is an explicit convention, not a rotational norm.
    stress = np.diag([.6, 0., 0.])
    angle = math.pi / 4
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                         [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    rotated = rotation @ stress @ rotation.T
    original_max, rotated_max = float(abs(stress).max()), float(abs(rotated).max())
    assert original_max > .5 and rotated_max < .5

    # Pure bookkeeping identity: reducing B alone can increase A.
    original_e0, original_eR = -4., -5.
    new_e0, new_eR = -4., -5.1
    delta_A = (new_e0 - new_eR) - (original_e0 - original_eR)
    delta_B = new_eR - original_eR
    assert np.isclose(delta_A, .1) and np.isclose(delta_B, -.1)

    report = {
        "schema": "synthetic_cpu_physics_counterexamples_v1",
        "frozen_code_commit": "2e904c260bafb6750c9a5dbb0d920c4ff8c3a868",
        "MLIP_or_DLM_executed": False,
        "scope": "Synthetic counterexamples establish contract boundaries, not prevalence or model effects.",
        "finite_shell": {
            "lattice_parameters": [1., 3., 4., 90., 90., 8.],
            "declared_shell_radius": 2, "reported_minimum_A": shell_minimum,
            "larger_shell_radius": 5, "shortest_found_A": float(distances[index]),
            "shortest_found_image": shifts[index].tolist(),
        },
        "filter_stop": {
            "verification_type": "analytic_reference_deformation_formula_not_executed_ASE",
            "ASE_executed": False,
            "volume_per_atom_A3": volume_per_atom,
            "entrywise_stress_GPa": measured["stress_max_GPa"],
            "initial_generalized_fmax_eV_A": filter_fmax,
            "FIRE_fmax": .1, "generalized_force_criterion_passes": force_criterion_passes,
            "external_stress_threshold_GPa": .5,
            "external_verification_passes": measured["stress_max_GPa"] <= .5,
        },
        "unverified_energy_threshold_SUN": threshold_only,
        "missing_hull_not_relabelled_as_energy": missing_hull,
        "stress_convention": {
            "entrywise_max_before_GPa": original_max,
            "entrywise_max_after_rotation_GPa": rotated_max,
            "spectral_norm_before_GPa": float(np.linalg.norm(stress, 2)),
            "spectral_norm_after_GPa": float(np.linalg.norm(rotated, 2)),
        },
        "A_B_identity": {"delta_e0": new_e0-original_e0, "delta_A": delta_A, "delta_B": delta_B},
    }
    (EVIDENCE / "CPU_PHYSICS_PROBES.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
