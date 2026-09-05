"""CPU-only numerical illustrations for notes 23 and 24; no model changes.

These verify algebra and counterexamples, not crystal-generation performance.
Run with the project virtualenv; output is adjacent JSON.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def kl(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return float(np.sum(np.where(p > 0, p * np.log(p / q), 0.0)))


def wrapped_density(y, mu, sigma):
    displacement = np.asarray(y)[..., None] - mu + np.arange(-12, 13)
    return np.exp(-0.5 * (displacement / sigma) ** 2).sum(axis=-1) / (
        math.sqrt(2 * math.pi) * sigma
    )


def two_endpoint_noisy_density_and_score(z, weights, sigma=0.45):
    """A toy clean law on {-1, +1}, followed by Gaussian forward corruption."""
    z = np.asarray(z, dtype=float)
    displacement = z[..., None] - np.array([-1.0, 1.0])
    density_by_endpoint = np.exp(-0.5 * (displacement / sigma) ** 2) / (
        math.sqrt(2 * math.pi) * sigma
    )
    weighted_density = density_by_endpoint * np.asarray(weights)
    density = weighted_density.sum(axis=-1)
    score = (weighted_density * (-displacement / sigma**2)).sum(axis=-1) / density
    return density, score


def main():
    n = 100_000
    grid = (np.arange(n) + 0.5) / n
    integral = float(wrapped_density(grid, 0.98, 0.05).mean())
    actual = float(wrapped_density(0.02, 0.98, 0.05))
    shifted = float(wrapped_density(0.02 + 0.371, (0.98 + 0.371) % 1, 0.05))
    log_naive = -0.5 * ((0.02 - 0.98) / 0.05) ** 2 - math.log(
        math.sqrt(2 * math.pi) * 0.05
    )
    assert abs(integral - 1) < 1e-12
    assert abs(actual - shifted) < 1e-11

    proposal0 = np.array([0.5, 0.5])
    proposal1 = np.array([0.4, 0.6])
    refine0 = np.array([[0.3, 0.3, 0.4], [0.25, 0.25, 0.5]])
    refine1 = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
    joint0 = proposal0[:, None] * refine0
    joint1 = proposal1[:, None] * refine1
    joint_kl = kl(joint1, joint0)
    chain_kl = kl(proposal1, proposal0) + sum(
        proposal1[i] * kl(refine1[i], refine0[i]) for i in range(2)
    )
    # One fixed deterministic endpoint map merges trajectory classes 0 and 1.
    endpoint0 = np.array([joint0[:, :2].sum(), joint0[:, 2].sum()])
    endpoint1 = np.array([joint1[:, :2].sum(), joint1[:, 2].sum()])
    endpoint_kl = kl(endpoint1, endpoint0)
    assert abs(joint_kl - chain_kl) < 1e-14
    assert endpoint_kl <= joint_kl

    baseline_b = np.array([-0.01, 0.30])
    improved_b = np.array([0.14, 0.14])
    assert improved_b.mean() < baseline_b.mean()
    assert (improved_b <= 0).mean() < (baseline_b <= 0).mean()

    # Same mathematical schedules as the inspected implementation, float64 here.
    times = np.arange(1001, dtype=float)
    cosine = np.cos(((times / 1000 + 0.008) / 1.008) * math.pi * 0.5) ** 2
    cosine /= cosine[0]
    beta = np.clip(1 - cosine[1:] / cosine[:-1], 0.0001, 0.9999)
    alpha_bar_800 = float(np.prod(1 - beta[:800]))
    sigma_x_800 = float(np.exp(np.linspace(np.log(0.005), np.log(0.5), 1000))[799])
    corrector_std_800 = math.sqrt(2e-5) * sigma_x_800 / 0.005

    # Constant-drift Euler chains: fixed covariance gives a finite SDE KL limit.
    horizon, drift_change, sigma0, sigma1 = 1.0, 0.4, 0.6, 0.7
    girsanov_value = horizon * drift_change**2 / (2 * sigma0**2)
    discretized_kl = {}
    covariance_change_kl = {}
    variance_ratio = (sigma1 / sigma0) ** 2
    for steps in [16, 64, 256, 1024]:
        dt = horizon / steps
        discretized_kl[str(steps)] = steps * (drift_change * dt) ** 2 / (
            2 * sigma0**2 * dt
        )
        covariance_change_kl[str(steps)] = steps * 0.5 * (
            variance_ratio - 1 - math.log(variance_ratio)
        )
        assert abs(discretized_kl[str(steps)] - girsanov_value) < 1e-14

    # A one-step fixed-variance Gaussian cannot realize every finite-KL teacher.
    real_grid = np.linspace(-10, 10, 200_001)
    spacing = float(real_grid[1] - real_grid[0])
    mode, teacher_sigma = 0.6, 0.4
    normal0 = np.exp(-0.5 * real_grid**2) / math.sqrt(2 * math.pi)
    teacher = sum(
        0.5 * np.exp(-0.5 * ((real_grid - center) / teacher_sigma) ** 2)
        / (math.sqrt(2 * math.pi) * teacher_sigma)
        for center in [-mode, mode]
    )
    teacher_kl = float(np.sum(teacher * np.log(teacher / normal0)) * spacing)
    teacher_mean = float(np.sum(teacher * real_grid) * spacing)
    cost0 = 3 - 2 * mode**2 + mode**4
    cost_teacher = 4 * mode**2 * teacher_sigma**2 + 3 * teacher_sigma**4
    assert teacher_kl < 0.2
    assert abs(teacher_mean) < 1e-13
    assert cost_teacher < cost0

    # Weighted denoising learns the score of the reweighted, then corrupted law.
    score_points = np.array([-1.7, -0.7, 0.0, 0.3, 1.4])
    reference_weights = [0.5, 0.5]
    target_weights = [0.2, 0.8]
    _, reference_score = two_endpoint_noisy_density_and_score(
        score_points, reference_weights
    )
    _, target_score = two_endpoint_noisy_density_and_score(
        score_points, target_weights
    )
    finite_difference_step = 1e-5

    def log_noise_posterior_weight(z):
        reference_density, _ = two_endpoint_noisy_density_and_score(
            z, reference_weights
        )
        target_density, _ = two_endpoint_noisy_density_and_score(z, target_weights)
        return np.log(target_density / reference_density)

    noise_h_gradient = (
        log_noise_posterior_weight(score_points + finite_difference_step)
        - log_noise_posterior_weight(score_points - finite_difference_step)
    ) / (2 * finite_difference_step)
    weighted_score_residual = float(
        np.max(np.abs(target_score - reference_score - noise_h_gradient))
    )
    assert weighted_score_residual < 2e-8

    # Global terminal tilting changes a random initial law unless H(0,z) is constant.
    initial_law = np.array([0.5, 0.5])
    # Columns: good endpoint, ordinary endpoint.
    endpoint_kernel = np.array([[0.9, 0.1], [0.1, 0.9]])
    terminal_weights = np.array([2.0, 1.0])
    reference_joint = initial_law[:, None] * endpoint_kernel
    h_initial = endpoint_kernel @ terminal_weights
    globally_tilted = reference_joint * terminal_weights
    globally_tilted /= globally_tilted.sum()
    fixed_initial_tilted = (
        reference_joint * terminal_weights / h_initial[:, None]
    )
    assert np.allclose(fixed_initial_tilted.sum(axis=1), initial_law)
    assert not np.allclose(globally_tilted.sum(axis=1), initial_law)
    assert abs(globally_tilted[:, 0].sum() - 2 / 3) < 1e-14
    assert abs(
        fixed_initial_tilted[:, 0].sum() - (18 / 19 + 2 / 11) / 2
    ) < 1e-14

    # Nondifferentiable terminal basin labels have a smooth future H for s<T.
    brownian_sigma, remaining_time = 0.6, 0.25
    future_noise_std = brownian_sigma * math.sqrt(remaining_time)
    basin_probe_points = np.array([-0.2, 0.1, 0.4])
    scaled_basin_points = basin_probe_points / future_noise_std
    future_h = 1 + ndtr(scaled_basin_points)
    future_log_h_gradient = (
        np.exp(-0.5 * scaled_basin_points**2)
        / (math.sqrt(2 * math.pi) * future_noise_std * future_h)
    )
    future_log_h_fd = (
        np.log(1 + ndtr((basin_probe_points + finite_difference_step) / future_noise_std))
        - np.log(1 + ndtr((basin_probe_points - finite_difference_step) / future_noise_std))
    ) / (2 * finite_difference_step)
    future_gradient_residual = float(
        np.max(np.abs(future_log_h_gradient - future_log_h_fd))
    )
    assert future_gradient_residual < 2e-8
    assert np.all(future_log_h_gradient > 0)

    # Holding the corrector rule fixed while refining its grid adds MCMC time.
    corrector_time_by_grid = {}
    for grid_size in [250, 1000, 4000]:
        grid_sigmas = np.exp(
            np.linspace(np.log(0.005), np.log(0.5), grid_size)
        )
        corrector_time_by_grid[str(grid_size)] = float(
            np.sum(1e-5 * (grid_sigmas / 0.005) ** 2)
        )
    corrector_growth_ratio = (
        corrector_time_by_grid["4000"] / corrector_time_by_grid["1000"]
    )
    assert 3.9 < corrector_growth_ratio < 4.1

    rare_events = []
    for probability in [1e-4, 1e-3, 1e-2]:
        def bernoulli_kl(value):
            return kl([value, 1 - value], [probability, 1 - probability])

        max_probability = brentq(lambda value: bernoulli_kl(value) - 0.2,
                                 probability, 1 - 1e-12)
        rare_events.append({
            "hypothetical_reference_basin_probability": probability,
            "at_least_one_hit_in_K8": 1 - (1 - probability)**8,
            "max_new_probability_under_population_KL_0_2": max_probability,
        })

    result = {
        "scope": "algebra_and_toy_counterexamples_only_not_crystal_experiments",
        "wrapped_transition": {
            "integral_on_unit_torus": integral,
            "common_translation_density_error": abs(actual - shifted),
            "boundary_correct_log_density": math.log(actual),
            "boundary_naive_gaussian_log_density": log_naive,
        },
        "kl_chain_and_fixed_endpoint_map": {
            "joint_kl": joint_kl,
            "chain_rule_rhs": chain_kl,
            "chain_rule_residual": abs(joint_kl - chain_kl),
            "endpoint_kl": endpoint_kl,
        },
        "deterministic_limit": {
            "mean_shift": 0.01,
            "gaussian_kl_by_sigma": {
                str(s): 0.01**2 / (2 * s**2) for s in [0.1, 0.01, 0.001]
            },
            "sigma_zero_different_dirac_means_kl": "infinite",
        },
        "mean_objectives_do_not_guarantee_sun": {
            "baseline_A_mean": 0.02,
            "improved_A_mean": 0.01,
            "baseline_B_mean": float(baseline_b.mean()),
            "improved_B_mean": float(improved_b.mean()),
            "baseline_stable_fraction": float((baseline_b <= 0).mean()),
            "improved_stable_fraction": float((improved_b <= 0).mean()),
            "assumption": "all_valid_unique_novel_for_this_toy_example",
        },
        "sampler_schedule_800_of_1000_float64": {
            "alpha_bar": alpha_bar_800,
            "lattice_clean_coefficient": math.sqrt(alpha_bar_800),
            "lattice_noise_coefficient": math.sqrt(1 - alpha_bar_800),
            "fractional_forward_sigma": sigma_x_800,
            "fractional_corrector_std": corrector_std_800,
            "actual_initial_state": "unnoised_DLM_proposal",
        },
        "sde_kl_limit": {
            "fixed_noise_girsanov_kl": girsanov_value,
            "fixed_noise_euler_kl_by_steps": discretized_kl,
            "changed_noise_euler_kl_by_steps": covariance_change_kl,
            "changed_noise_continuous_path_laws": "singular_by_quadratic_variation",
        },
        "finite_KL_teacher_not_realizable_by_one_gaussian_mean_update": {
            "teacher": "0.5*N(-0.6,0.4^2)+0.5*N(0.6,0.4^2)",
            "reference_and_student_family": "reference=N(0,1); student=N(mu,1)",
            "teacher_kl_to_reference": teacher_kl,
            "optimal_student_mean": teacher_mean,
            "terminal_cost": "(x^2-0.6^2)^2",
            "teacher_expected_cost": cost_teacher,
            "reference_and_projected_student_expected_cost": cost0,
            "scope": "one_step_counterexample_not_impossibility_for_a_full_SDE",
        },
        "weighted_endpoint_denoising_score_identity": {
            "clean_endpoint_support": [-1, 1],
            "reference_weights": reference_weights,
            "target_weights": target_weights,
            "forward_noise_sigma": 0.45,
            "probe_points": score_points.tolist(),
            "reference_noisy_score": reference_score.tolist(),
            "target_noisy_score": target_score.tolist(),
            "finite_difference_log_noise_H_gradient": noise_h_gradient.tolist(),
            "max_identity_residual": weighted_score_residual,
            "scope": "positive_noise_toy_law_not_a_population_crystal_KL_certificate",
        },
        "terminal_tilt_and_initial_distribution_boundary": {
            "reference_initial_law": initial_law.tolist(),
            "reference_good_endpoint_probability": float(reference_joint[:, 0].sum()),
            "terminal_weights_good_ordinary": terminal_weights.tolist(),
            "H_at_initial_states": h_initial.tolist(),
            "global_tilt_initial_law": globally_tilted.sum(axis=1).tolist(),
            "global_tilt_good_probability": float(globally_tilted[:, 0].sum()),
            "fixed_initial_law": fixed_initial_tilted.sum(axis=1).tolist(),
            "fixed_initial_conditional_tilt_good_probability": float(
                fixed_initial_tilted[:, 0].sum()
            ),
            "scope": "random_initial_law_counterexample_not_a_fixed_xD_failure",
        },
        "discontinuous_terminal_cost_smooth_future_H": {
            "reference": "one_dimensional_Brownian_motion",
            "terminal_weight": "2_if_Y_positive_else_1",
            "remaining_time": remaining_time,
            "sigma": brownian_sigma,
            "probe_points_off_basin_boundary": basin_probe_points.tolist(),
            "local_terminal_cost_gradient": [0, 0, 0],
            "future_H": future_h.tolist(),
            "future_log_H_gradient": future_log_h_gradient.tolist(),
            "optimal_drift_change": (
                brownian_sigma**2 * future_log_h_gradient
            ).tolist(),
            "max_gradient_finite_difference_residual": future_gradient_residual,
        },
        "unscaled_corrector_rule_is_not_fixed_horizon_grid_refinement": {
            "full_sigma_range": [0.005, 0.5],
            "step_lr": 1e-5,
            "sum_corrector_step_sizes_by_grid_size": corrector_time_by_grid,
            "4000_to_1000_ratio": corrector_growth_ratio,
            "scope": "schedule_algebra_only_not_a_model494_convergence_experiment",
        },
        "wrapped_prior_not_exactly_uniform_at_finite_sigma": {
            "sigma_T": 0.5,
            "first_fourier_mode_magnitude": math.exp(-2 * math.pi**2 * 0.5**2),
            "scope": "one_coordinate_mode_not_a_joint_total_variation_bound",
        },
        "rare_basin_illustrations_not_measured_crystal_probabilities": rare_events,
        "all_assertions_passed": True,
    }
    target = Path(__file__).with_name("continuous_math_checks.json")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
