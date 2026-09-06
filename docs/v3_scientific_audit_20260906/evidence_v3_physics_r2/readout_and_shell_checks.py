"""Analytic H-P33 endpoint and finite-shell review cases; no model calls."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def wrapped_mixture_score(value: float, centers, weights, sigma: float) -> float:
    delta = value - np.asarray(centers)[:, None] + np.arange(-8, 9)[None, :]
    log_weight = -0.5 * (delta / sigma) ** 2 + np.log(weights)[:, None]
    probability = np.exp(log_weight - log_weight.max())
    probability /= probability.sum()
    return float(-(probability * delta).sum() / sigma ** 2)


def normal_cdf(value: float) -> float:
    return .5 * math.erfc(-value / math.sqrt(2))


def shifts(radius: int) -> np.ndarray:
    values = np.asarray(list(itertools.product(range(-radius, radius + 1), repeat=3)), dtype=float)
    return values[np.any(values != 0, axis=1)]


def prior_cell_shell_probe(normalizer: dict, draws: int = 256) -> dict:
    """Cell-input probe only: no sites, crystal policy, model or energy calls."""
    basis = np.zeros((6, 3, 3))
    basis[0] = np.diag([1., -1., 0.]) / math.sqrt(2)
    basis[1] = np.diag([1., 1., -2.]) / math.sqrt(6)
    for index, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)), 2):
        basis[index, a, b] = basis[index, b, a] = 1 / math.sqrt(2)
    basis[5] = np.eye(3) / math.sqrt(3)
    normalized = np.random.default_rng(20260906033).normal(size=(draws, 6))
    fields = np.asarray(normalizer["mean"]) + normalized * np.asarray(normalizer["std"])
    shell = shifts(2)
    reports = []
    for count in (1, 10, 20):
        misses = floor_misses = completed = budget_exceeded = 0
        maximum_images = 0
        for field in fields:
            coefficients = field.copy()
            coefficients[5] += math.log(count) / math.sqrt(3)
            symmetric = np.einsum("k,kij->ij", coefficients, basis)
            values, vectors = np.linalg.eigh(symmetric)
            gram = (vectors * np.exp(2 * values)) @ vectors.T
            lattice = np.linalg.cholesky(gram)
            shell_distances = np.linalg.norm(shell @ lattice, axis=1)
            bounds = np.floor(6 * np.linalg.norm(np.linalg.inv(lattice), axis=0) + 1e-12).astype(int)
            images = int(np.prod(2 * bounds + 1))
            maximum_images = max(maximum_images, images)
            if images > 200_000:
                budget_exceeded += 1
                continue
            candidates = np.asarray(list(itertools.product(*(range(-int(b), int(b) + 1) for b in bounds))), dtype=float)
            candidates = candidates[np.any(candidates != 0, axis=1)]
            distances = np.linalg.norm(candidates @ lattice, axis=1)
            outside = np.any(np.abs(candidates) > 2, axis=1)
            misses += int(np.any(outside & (distances < 6)))
            complete_minimum = float(distances.min()) if distances.size else math.inf
            floor_misses += int(shell_distances.min() >= .5 and complete_minimum < .5)
            completed += 1
        reports.append({"N": count, "draws": draws, "complete_image_boxes": completed,
                        "image_budget_exceeded": budget_exceeded, "maximum_box_images": maximum_images,
                        "cells_with_self_images_inside_6A_omitted_by_radius2": misses,
                        "radius2_false_pass_of_05A_self_image_floor": floor_misses})
    return {"normalizer_source": "GEOMETRY_NUMERICS_40013.json, train-only fitted normalizer",
            "seed": 20260906033, "per_N": reports,
            "interpretation": "Fixed small CPU lattice-prior probe. Same Gaussian draws across N; checks self-image omissions only, not non-self pairs or generated endpoint frequencies."}


def main() -> None:
    epsilon = .002
    alpha, sigma_l = math.cos(math.pi * epsilon / 2), math.sin(math.pi * epsilon / 2)
    z0 = np.array([1., -.5, .2, 2., 0., -2.3])
    noise = np.array([.3, -1.2, .2, 0., 1.4, .8])
    z_epsilon = alpha * z0 + sigma_l * noise
    v_oracle = alpha * noise - sigma_l * z0
    z_readout = alpha * z_epsilon - sigma_l * v_oracle
    assert np.max(np.abs(z_readout - z0)) < 1e-13
    point_cases = []
    for noise_value in (-3., -.7, 0., .7, 2.):
        clean = .0049
        state = (clean + epsilon * noise_value) % 1
        u = epsilon * wrapped_mixture_score(state, [clean], [.1 + .9], epsilon)
        result = (state + epsilon * u) % 1
        delta = (result - clean + .5) % 1 - .5
        assert abs(delta) < 1e-12
        point_cases.append({"clean": clean, "noise": noise_value, "state_at_epsilon": state,
                            "u_oracle": u, "after_33rd_readout": result,
                            "canonical_Q_bin": int(math.floor(result * 100 + .5)) % 100})
    assert all(row["canonical_Q_bin"] == 0 for row in point_cases)

    half_separation = .006
    centers = [.5 - half_separation, .5 + half_separation]
    midpoint = .5
    u_midpoint = epsilon * wrapped_mixture_score(midpoint, centers, [.5, .5], epsilon)
    output_midpoint = (midpoint + epsilon * u_midpoint) % 1
    assert abs(output_midpoint - midpoint) < 1e-14
    radius = epsilon ** 2 / half_separation * math.atanh(.5)
    central_mass = normal_cdf((radius - half_separation) / epsilon) - normal_cdf((-radius - half_separation) / epsilon)
    assert 0 < central_mass < .01
    barrier = .05

    theta = math.radians(8.)
    lattice = np.array([[1., 0., 0.], [3 * math.cos(theta), 3 * math.sin(theta), 0.], [0., 0., 4.]])
    shell = shifts(2)
    shell_lengths = np.linalg.norm(shell @ lattice, axis=1)
    missed_shift = np.array([-3., 1., 0.])
    missed_length = float(np.linalg.norm(missed_shift @ lattice))
    assert shell_lengths.min() >= .5 and missed_length < .5
    cutoff = 6.0
    coordinate_bounds = np.floor(cutoff * np.linalg.norm(np.linalg.inv(lattice), axis=0) + 1e-12).astype(int)
    full = np.asarray(list(itertools.product(*(range(-int(b), int(b) + 1) for b in coordinate_bounds))), dtype=float)
    full = full[np.any(full != 0, axis=1)]
    full_lengths = np.linalg.norm(full @ lattice, axis=1)
    # If ||nL||<r then |n_j|<r||column_j(L^-1)||, so this box covers all in-cutoff images.
    all_in_cutoff = int((full_lengths < cutoff).sum())
    shell_in_cutoff = int((shell_lengths < cutoff).sum())
    assert all_in_cutoff > shell_in_cutoff

    float_sites = np.array([[.9951, .9951, .9951], [.0049, .0049, .0049]])
    delta = float_sites[0] - float_sites[1]
    delta -= np.round(delta)
    minimum = float(np.linalg.norm(delta * 30.))
    old_bin_keys = [tuple(int(round(float(v) * 100)) % 100 for v in row) for row in float_sites]
    assert minimum > .5 and old_bin_keys[0] == old_bin_keys[1]

    root = Path(__file__).resolve().parents[3]
    source_file = root / "docs/v3_scientific_audit_20260906/evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json"
    source = json.loads(source_file.read_text(encoding="utf-8"))
    numeric_path = root / "docs/v3_scientific_audit_20260906/evidence/GEOMETRY_NUMERICS_40013.json"
    numerics = json.loads(numeric_path.read_text(encoding="utf-8"))
    assert numerics["success"] and numerics["summary"]["model_forwards"] == 0
    gate = {}
    for split, count in (("train", 27136), ("val", 9047)):
        item = source[split + "_summary"]
        counts = item["counts"]
        assert item["original_cif_identity_gate_passed"] and item["audit_complete"]
        assert counts["training_records"] == counts["verified_training_records"] == count
        assert counts["csv_parse_or_encode_errors"] == counts["csv_full_answer_duplicate_groups"] == counts["unverified_training_records"] == 0
        assert not item["remaining_csv_rows_not_assigned"]
        gate[split] = {"records": count, "identity_method": item["identity_methods"],
                       "source_csv_sha256": item["input_receipts"]["source_csv"]["sha256"],
                       "training_jsonl_sha256": item["input_receipts"]["training_jsonl"]["sha256"]}

    paths = ["docs/v3_scientific_audit_20260906/V3_H_P33_SPECIFICATION.md",
             "docs/v3_scientific_audit_20260906/evidence/CONTINUOUS_SOURCE_IDENTITY_40009.json",
             "docs/v3_scientific_audit_20260906/evidence/V2_ACTUAL_EVALUATION_SEEDS.json",
             "docs/v3_scientific_audit_20260906/evidence/GEOMETRY_NUMERICS_40013.json",
             "scripts/export_programmed_path_artifacts.py", "src/scripts/sample_llada_dynamic_crystals.py",
             "src/crystal_dlm/periodic_state_conditioning.py"]
    report = {
        "schema": "hp33_physics_r2_analytic_checks_v1", "model_calls": 0, "GPU_calls": 0, "MLIP_calls": 0,
        "source_identity_gate_closed": gate,
        "registered_prior_cell_shell_probe": prior_cell_shell_probe(numerics["summary"]["normalizer"]),
        "point_oracle_readout": {"lattice_max_error": float(np.max(np.abs(z_readout - z0))), "coordinate_cases": point_cases,
            "interpretation": "The old p_epsilon point-target bin-flip example does not apply to final H-P33 under a correct point oracle."},
        "Gaussian_clean_distribution_readout": {"ideal_probability_flow_variance_at_epsilon": 1.,
            "variance_after_conditional_mean_readout": alpha ** 2, "variance_deficit": 1 - alpha ** 2,
            "interpretation": "A clean conditional mean is not a posterior sample, even though this Gaussian deficit is tiny at epsilon=.002."},
        "two_mode_readout": {"clean_centers": centers, "sigma": epsilon, "observed_state": midpoint,
            "u_at_midpoint": u_midpoint, "readout": output_midpoint,
            "clean_Q_bins": [int(math.floor(v * 100 + .5)) % 100 for v in centers], "readout_Q_bin": 50,
            "toy_energy_formula": "E(f)=.05*(1-((f-.5)/.006)^2)^2 eV/atom",
            "energy_at_modes_eV_atom": 0., "energy_at_readout_eV_atom": barrier,
            "first_derivative_at_readout": 0., "second_derivative_per_fractional_squared": -4 * barrier / half_separation ** 2,
            "true_noisy_mixture_mass_mapping_between_half_modes": central_mass,
            "interpretation": "Multimodal mean can lie on a low-density energy maximum; no actual crystal energy or frequency was measured."},
        "finite_shell": {"lattice_A": lattice.tolist(), "radius2_minimum_self_image_A": float(shell_lengths.min()),
            "missed_shift": missed_shift.tolist(), "missed_image_A": missed_length,
            "complete_box_for_6A_cutoff": coordinate_bounds.tolist(), "radius2_self_images_in_6A": shell_in_cutoff,
            "all_self_images_in_6A": all_in_cutoff,
            "interpretation": "Shows lack of a universal finite-shell guarantee, not its frequency under the as-yet unprofiled registered normalizer/prior."},
        "legacy_bin_guard_float_counterexample": {"lattice_edge_A": 30., "fractional_sites": float_sites.tolist(),
            "exact_periodic_min_distance_A": minimum, "old_graph_bin_keys": old_bin_keys,
            "interpretation": "Distinct float geometry can pass the .5A floor yet be rejected by the old .01-bin duplicate guard; this is not a stability example."},
        "source_receipts": [{"path": p, "sha256": hashlib.sha256((root / p).read_bytes()).hexdigest()} for p in paths],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "point_oracle_passed": True,
                      "gaussian_readout_variance": alpha ** 2, "multimodal_central_mass": central_mass,
                      "shell_count": [shell_in_cutoff, all_in_cutoff], "bin_guard_min_distance_A": minimum}, ensure_ascii=False))


if __name__ == "__main__":
    main()
