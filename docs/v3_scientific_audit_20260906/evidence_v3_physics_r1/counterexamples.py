"""Small analytic/NumPy V3 review examples; no model, data generation, or MLIP."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def wrapped_bin_stay_probability(clean: float, sigma: float, width: float = 0.01) -> float:
    # The reported cases have sigma <= .001; images beyond these five cells
    # contribute less than representable floating-point precision.
    return sum(normal_cdf((n + width / 2 - clean) / sigma)
               - normal_cdf((n - width / 2 - clean) / sigma) for n in range(-2, 3))


def circle_density_score(grid: np.ndarray, sigma: float):
    delta = grid[:, None] + np.arange(-12, 13, dtype=float)[None, :]
    log_weight = -0.5 * (delta / sigma) ** 2
    offset = log_weight.max(axis=1, keepdims=True)
    weight = np.exp(log_weight - offset)
    total = weight.sum(axis=1)
    density = np.exp(offset[:, 0]) * total / (math.sqrt(2 * math.pi) * sigma)
    score = -(delta * weight).sum(axis=1) / (sigma * sigma * total)
    return density, score


def half_log_metric(lattice: np.ndarray) -> np.ndarray:
    eigenvalue, eigenvector = np.linalg.eigh(lattice @ lattice.T)
    assert np.all(eigenvalue > 0)
    return (eigenvector * (0.5 * np.log(eigenvalue))) @ eigenvector.T


def gaussian_ddpm_variance(steps: int, rate: float) -> dict:
    # Clean z0~N(0,1), so all forward marginals and the terminal prior are
    # exactly N(0,1). The optimal epsilon predictor is sqrt(1-abar)*z.
    abar = np.exp(-rate * (np.arange(steps + 1, dtype=float) / steps) ** 2)
    variance = 1.0
    largest_beta = 0.0
    for k in range(steps, 0, -1):
        alpha = abar[k] / abar[k - 1]
        beta = 1 - alpha
        posterior_variance = beta * (1 - abar[k - 1]) / (1 - abar[k])
        variance = alpha * variance + posterior_variance
        largest_beta = max(largest_beta, beta)
    assert 0 < variance < 1
    return {"steps": steps, "lambda_z": rate, "terminal_abar": float(abar[-1]),
            "largest_beta": largest_beta, "clean_target_variance": 1.0,
            "generated_variance_with_perfect_epsilon_and_beta_tilde": float(variance),
            "relative_variance_deficit": float(1 - variance),
            "reason": "beta_tilde omits posterior uncertainty in z0; exact Gaussian-data reverse variance is beta"}


def main() -> None:
    result = {"schema": "v3_physics_r1_counterexamples_v1", "model_calls": 0,
              "new_crystal_samples": 0, "MLIP_calls": 0, "relaxations": 0,
              "claim_boundary": "Analytic toys and local source inspection, not candidate performance or original-CIF coverage."}
    codec = []
    for gamma in (1.0, 1.5):
        sigma = 1e-3 ** gamma
        crossing = 1 - wrapped_bin_stay_probability(0.0049, sigma)
        codec.append({"clean_fractional_coordinate": 0.0049, "canonical_clean_bin": 0,
                      "upper_bin_boundary": 0.005, "gamma": gamma,
                      "last_step_sigma_fractional": sigma,
                      "oracle_last_Euler_step_probability_of_different_codec_bin": crossing,
                      "per_axis_last_noise_A_at_cell_length_20A": 20 * sigma})
    assert abs(codec[0]["oracle_last_Euler_step_probability_of_different_codec_bin"] - 0.460172162722971) < 1e-12
    result["last_euler_step_codec_boundary"] = codec
    result["H_probability_flow_finite_epsilon_endpoint"] = {
        "epsilon_time": .002, "sigma_fractional": .002, "clean_fractional_coordinate": .0049,
        "perfect_marginal_probability_of_different_codec_bin": 1 - wrapped_bin_stay_probability(.0049, .002),
        "cartesian_axis_std_at_20A": .04,
        "interpretation": "A perfect ODE ends at p_epsilon, not p_0; this is marginal residual blur, not newly injected terminal Euler noise."}
    result["gaussian_VP_perfect_predictor_finite_kernel"] = [gaussian_ddpm_variance(t, 18.0) for t in (32, 64, 128)]

    grid = np.arange(32768, dtype=float) / 32768 - 0.5
    information = []
    for sigma in (0.001, 0.05, 0.2, 0.5, 1.0):
        density, score = circle_density_score(grid, sigma)
        normalizer = float(density.mean())
        scaled_target_second_moment = float((density * (sigma * score) ** 2).mean())
        assert abs(normalizer - 1) < 1e-10
        assert math.isfinite(scaled_target_second_moment)
        information.append({"sigma": sigma, "integrated_density": normalizer,
                            "E_q_of_squared_sigma_times_conditional_score": scaled_target_second_moment})
    result["dense_target_information_is_not_component_count"] = information

    scaling = []
    for count in (1, 4, 10, 20):
        edge = (20 * count) ** (1 / 3)
        scaling.append({"N": count, "VPA_A3": 20, "cubic_edge_A": edge,
                        "fractional_sigma": 0.05, "cartesian_axis_std_A": 0.05 * edge,
                        "V2_construct_and_repair_NFE": 2 * (6 + 3 * count), "C_NFE": 64})
    result["N_and_A_scale"] = scaling
    result["anisotropic_noise_example"] = {"lattice_A": np.diag([2., 20., 5.]).tolist(),
        "fractional_sigma": .05, "cartesian_covariance_A2": (.05 ** 2 * np.diag([2., 20., 5.]).T @ np.diag([2., 20., 5.])).tolist()}

    lattice = np.eye(3) * 3
    basis_change = np.array([[1., 3., 0.], [0., 1., 0.], [0., 0., 1.]])
    sheared = basis_change @ lattice
    shape0, shape1 = half_log_metric(lattice), half_log_metric(sheared)
    shape0 -= np.trace(shape0) / 3 * np.eye(3)
    shape1 -= np.trace(shape1) / 3 * np.eye(3)
    assert abs(np.linalg.det(lattice) - np.linalg.det(sheared)) < 1e-12
    result["same_lattice_different_chart"] = {"row_lattice_original": lattice.tolist(), "unimodular_M": basis_change.tolist(),
        "row_lattice_changed": sheared.tolist(), "volume_both_A3": float(np.linalg.det(lattice)),
        "shape_distance_Frobenius": float(np.linalg.norm(shape1 - shape0)),
        "angle_gamma_changed_degrees": math.degrees(math.acos(np.dot(sheared[0], sheared[1]) / np.linalg.norm(sheared[0]) / np.linalg.norm(sheared[1]))),
        "interpretation": "Same simple-cubic translation lattice under a determinant-one basis change; metric-cell classifier may say monoclinic while conventional symmetry remains cubic."}

    centered_covariances = []
    for count in (1, 2, 4):
        projection = np.eye(count) - np.ones((count, count)) / count
        centered_covariances.append({"N": count, "projected_unit_noise_covariance": projection.tolist(),
                                     "rank": int(np.linalg.matrix_rank(projection)),
                                     "per_site_variance": (1 - 1 / count)})
    result["translation_gauge_changes_forward_measure"] = centered_covariances
    base = np.array([.99, .01])
    shifted = np.mod(base + .02, 1)
    result["arithmetic_fractional_centering_discontinuity"] = {
        "F": base.tolist(), "wrap_F_plus_002": shifted.tolist(),
        "centered_F": np.mod(base - base.mean(), 1).tolist(),
        "centered_shifted_F": np.mod(shifted - shifted.mean(), 1).tolist(),
        "interpretation": "Representatives jump by half a cell despite a small common translation; physical relative geometry remains equivalent."}

    exact = np.array([0., 1 / 3, 2 / 3])
    quantized = np.floor(exact * 100 + .5) / 100
    transformed = np.mod(quantized + 1 / 3, 1)
    distance = np.abs(transformed[:, None] - quantized[None, :])
    distance = np.minimum(distance, 1 - distance)
    result["codec_does_not_close_third_translation"] = {"exact_orbit": exact.tolist(), "quantized_orbit": quantized.tolist(),
        "max_residual_fractional_after_translation_one_third": float(distance.min(axis=1).max()),
        "max_residual_A_if_axis_6A": float(6 * distance.min(axis=1).max()),
        "interpretation": "Geometric group-closure counterexample; no spglib SG classification was run."}

    kernel = np.array([[1., 0.], [.5, .5]])
    uniform = np.array([.5, .5])
    result["rollback_selection_can_change_stationary_distribution"] = {
        "target": uniform.tolist(), "D_ok": [.5, 0.], "explicit_failure_probability": .5,
        "rollback_kernel": kernel.tolist(), "one_step_from_target": (uniform @ kernel).tolist(),
        "limiting_distribution": [1., 0.], "interpretation": "An otherwise uniform denoiser with output-dependent execution failure loses one mode under rollback."}
    result["A_independent_marginals_miss_joint_support"] = {"observed_tuples": [[0, 0, 0], [1, 1, 1]],
        "data_mass_each": .5, "perfect_marginal_probability_each_bit": .5,
        "independent_prior_mass_on_observed_joint_support": .25,
        "independent_prior_mass_on_unobserved_combinations": .75,
        "interpretation": "Unobserved is not necessarily physically impossible; exact conditional likelihood on data leaves these requests unidentified."}
    result["chart_score_is_not_energy_gradient"] = {"physical_density": "p(V)=exp(-V), V>0", "chart": "v=log(V)",
        "chart_density": "p(v)=exp(v-exp(v))", "chart_score_at_v0": 0., "negative_energy_gradient_at_v0": -1.,
        "interpretation": "Jacobian changes physical-density score; this does not demand adding a Jacobian to an already declared empirical dz dF training measure."}
    result["ordinal_source_join_counterexample"] = {"csv_rows": ["A", "B", "C"], "failed_parse_rows": ["B"],
        "written_SFT_rows": ["A", "C"], "later_assigned_source_row_idx": [0, 1],
        "unsafe_join_second_source_to_csv_row": "B", "true_second_source": "C",
        "interpretation": "Possible under available skip-failure and assign-ordinal code paths, not an observed MP20 misalignment."}
    result["program_clock_endpoint_boundary"] = {
        "theorem": "If clean conditional data are independent of P and each P-specific forward process has its exact reverse and correct terminal marginal, all clocks recover the same clean distribution.",
        "implication": "P can change computational paths and finite-step error without creating identifiable target semantics; compare gamma=1 before making the 0.5 rank-clock mandatory."}
    result["supervision_exposure"] = {"T": 64, "views_per_source_after_2_epochs": 4,
        "expected_distinct_time_levels_per_source": 64 * (1 - (63 / 64) ** 4),
        "interpretation": "Dense coordinates increase target components, not independent structures, time coverage, or a SUN guarantee."}
    high_time_probability = math.log(1 / .9) / math.log(1 / .002)
    result["H_log_uniform_time_exposure"] = {
        "epsilon": .002, "G_views_per_source_after_two_additional_epochs": 2,
        "probability_t_above_09": high_time_probability,
        "expected_t_above_09_views_across_27136_sources": 27136 * 2 * high_time_probability,
        "probability_a_source_has_no_t_above_09_G_view": (1 - high_time_probability) ** 2,
        "interpretation": "A coverage expectation, not actual training counts or a theorem requiring every source at every time."}
    result["zero_initialized_shared_head"] = {
        "x": 2., "frozen_base_weight": 3., "adapter_weight": .5, "target": 1.,
        "initial_head_weight": 0., "first_head_gradient": -7., "first_adapter_gradient": 0.,
        "head_weight_after_one_SGD_step_lr_001": .07, "next_adapter_gradient": -.0714,
        "interpretation": "A zero first hidden/adapter gradient can be expected; persistent detach remains a different failure."}

    root = Path(__file__).resolve().parents[3]
    sources = ["docs/v3_scientific_audit_20260906/V3_DLM_CANDIDATES_AND_ATTACKS.md",
               "docs/v3_scientific_audit_20260906/V3_GEOMETRIC_HEAD_COUNTERPROPOSAL.md",
               "src/crystal_dlm/programmed_path_runtime.py", "src/crystal_dlm/r5_plan_state.py",
               "src/crystal_dlm/canonical_site_order.py", "src/scripts/build_r5_exact_length_sft_data.py",
               "scripts/build_c3fd_native_sft_data.py", "scripts/canonicalize_c3fd_native_teacher_sft.py",
               "src/crystal_dlm/periodic_repair_model.py", "src/crystal_dlm/periodic_repair_initialization.py"]
    result["source_receipts"] = [{"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()} for path in sources]
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "codec": codec, "VP": result["gaussian_VP_perfect_predictor_finite_kernel"],
                      "information": information, "shape_distance": result["same_lattice_different_chart"]["shape_distance_Frobenius"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
