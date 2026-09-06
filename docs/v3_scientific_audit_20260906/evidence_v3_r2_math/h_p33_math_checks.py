"""Independent H-P33 finite-algorithm checks, NumPy/stdlib only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

EPSILON = .002
INTERVALS = 32


def normal_parameters(t):
    return np.cos(np.pi*t/2), np.sin(np.pi*t/2)


def grid(intervals=INTERVALS):
    return EPSILON+(1-EPSILON)*(np.arange(intervals+1)/intervals)**2


def wrapped_u(delta, t, radius=8):
    difference = np.asarray(delta, dtype=np.float64)[..., None]+np.arange(-radius, radius+1)
    log_weights = -.5*(difference/t)**2
    maximum = log_weights.max(-1, keepdims=True)
    weights = np.exp(log_weights-maximum)
    total = weights.sum(-1)
    u = -(weights*difference).sum(-1)/(t*total)
    log_density = maximum[..., 0]+np.log(total)-math.log(math.sqrt(2*math.pi)*t)
    return u, log_density


def gaussian_variance(variance, intervals=INTERVALS):
    times = grid(intervals)
    current_variance = 1.
    for index in range(intervals, 0, -1):
        t, previous = times[index], times[index-1]
        alpha, sigma = normal_parameters(t)
        noisy_variance = alpha**2*variance+sigma**2
        v_coefficient = alpha*sigma*(1-variance)/noisy_variance
        factor = 1-(np.pi/2)*(t-previous)*v_coefficient
        current_variance *= factor**2
    alpha, sigma = normal_parameters(EPSILON)
    noisy_variance = alpha**2*variance+sigma**2
    v_coefficient = alpha*sigma*(1-variance)/noisy_variance
    readout_factor = alpha-sigma*v_coefficient
    return {"clean_variance": variance, "exact_variance_at_epsilon": float(noisy_variance),
            "Euler32_variance_at_epsilon": float(current_variance),
            "exact_flow_then_readout_variance": float(readout_factor**2*noisy_variance),
            "H_P33_variance": float(readout_factor**2*current_variance),
            "relative_H_P33_variance_error": float(readout_factor**2*current_variance/variance-1)}


def fourier_u(fractional, t, amplitude=.8, noise_factor=1):
    amplitude_t = amplitude*math.exp(-2*math.pi**2*noise_factor*t*t)
    phase = 2*math.pi*fractional
    return -2*math.pi*noise_factor*t*amplitude_t*np.sin(phase)/(1+amplitude_t*np.cos(phase))


def fourier_case(points, noise_factor=1):
    # p_0(f)=1+.8 cos(2pi f). Quantile integration is deterministic.
    uniform = (np.arange(points, dtype=np.float64)+.5)/points
    sampled = uniform.copy()
    times = grid()
    for index in range(INTERVALS, 0, -1):
        sampled = (sampled+(times[index]-times[index-1])*fourier_u(sampled,times[index],noise_factor=noise_factor)) % 1
    euler_moment = float(np.cos(2*np.pi*sampled).mean())
    final = (sampled+EPSILON*fourier_u(sampled,EPSILON,noise_factor=noise_factor)) % 1
    # Isolate readout bias by drawing the exact p_epsilon via its analytic CDF.
    amplitude_e = .8*math.exp(-2*math.pi**2*noise_factor*EPSILON**2)
    left, right = np.zeros(points), np.ones(points)
    for _ in range(55):
        midpoint = .5*(left+right)
        cdf = midpoint+amplitude_e*np.sin(2*np.pi*midpoint)/(2*np.pi)
        lower = cdf < uniform
        left = np.where(lower,midpoint,left)
        right = np.where(lower,right,midpoint)
    exact = .5*(left+right)
    readout = (exact+EPSILON*fourier_u(exact,EPSILON,noise_factor=noise_factor)) % 1
    return {"points": points, "noise_factor": noise_factor, "true_clean_cosine_moment": .4,
            "exact_epsilon_cosine_moment": amplitude_e/2,
            "Euler32_epsilon_cosine_moment": euler_moment,
            "exact_flow_then_readout_cosine_moment": float(np.cos(2*np.pi*readout).mean()),
            "H_P33_cosine_moment": float(np.cos(2*np.pi*final).mean())}


def point_and_multimodal_readout():
    clean = .0049
    deviations = np.linspace(-6*EPSILON,6*EPSILON,2401)
    noisy = (clean+deviations) % 1
    u, _ = wrapped_u(noisy-clean,EPSILON)
    readout = (noisy+EPSILON*u) % 1
    error = (readout-clean+.5) % 1-.5
    assert np.max(np.abs(error)) < 1e-14
    # Two valid labelled two-site geometries (same-species permutations).
    modes = np.array([[.2,.3],[.3,.2]])
    observation = np.array([.25,.25])
    conditional_u, log_density = wrapped_u(observation[None]-modes,EPSILON)
    joint_logs = log_density.sum(-1)
    posterior = np.exp(joint_logs-joint_logs.max())
    posterior /= posterior.sum()
    joint_u = (posterior[:, None]*conditional_u).sum(0)
    result = (observation+EPSILON*joint_u) % 1
    assert np.max(np.abs(result-.25)) < 1e-14
    return {"point_clean_coordinate": clean, "local_noisy_extent_in_sigmas": 6,
            "point_max_periodic_readout_error": float(np.max(np.abs(error))),
            "point_Q_bin_change_count_in_test_grid": int(np.sum(np.floor(readout*100+.5)%100 != 0)),
            "multimodal_clean_modes": modes.tolist(), "observation": observation.tolist(),
            "mode_posterior": posterior.tolist(), "readout": result.tolist(),
            "clean_pair_distances_A_at_lattice6": [float(abs(x[0]-x[1])*6) for x in modes],
            "readout_pair_distance_A_at_lattice6": float(abs(result[0]-result[1])*6),
            "observation_log_density_each_mode": joint_logs.tolist(),
            "caveat": "The ambiguous observation is extraordinarily unlikely under exact low-noise data; no real-crystal failure rate is claimed."}


def truncation_checks():
    errors = []
    for t in (EPSILON,.01,.1,.5,1.):
        delta = np.array([-.999999,-.5,-.1234,0.,.1234,.4999,.999999])
        a, la = wrapped_u(delta,t,8)
        b, lb = wrapped_u(delta,t,32)
        errors.append({"t": t, "max_u_error": float(np.max(abs(a-b))), "max_log_density_error": float(np.max(abs(la-lb)))})
    # Nearest included image has |delta+n|<=.5; omitted |n|>=9 has
    # |delta+n|>=|n|-1. The worst bound over t<=1 is attained at t=1.
    omitted_relative_mass_bound = 2*sum(math.exp(-((n-1)**2-.25)/2) for n in range(9,50))
    normalized_u_error_bound = 2*sum((n+10)*math.exp(-((n-1)**2-.25)/2) for n in range(9,50))
    return {"window8_vs32": errors,
            "relative_tail_mass_bound_all_abs_delta_le1_t_le1": omitted_relative_mass_bound,
            "normalized_u_error_bound_all_abs_delta_le1_t_le1": normalized_u_error_bound}


def accumulation_budget():
    rows = []
    for world, accumulation in ((6,4),(4,6)):
        per_mode_window = world*accumulation//2
        updates = math.ceil(27136/per_mode_window)
        rows.append({"world": world, "microbatch": 1, "accumulation": accumulation,
                     "global_batch": world*accumulation, "per_mode_examples_per_window": per_mode_window,
                     "updates_per_epoch": updates, "padded_per_mode_per_epoch": updates*per_mode_window,
                     "zero_weight_padding_per_mode": updates*per_mode_window-27136,
                     "real_T_and_G_in_last_window": 27136-per_mode_window*(updates-1)})
    return {"layouts": rows, "view_average_objective": "(mean(L_T)+mean(L_G))/2",
            "microbatch_scalar_multiplier": "1/accumulation, with DDP rank averaging; zero-weight padding retains the registered denominator",
            "clip_counterexample": {"micro_gradients": [2.,-1.], "clip_after_sum": 1., "sum_individually_clipped": 0.}}


def main():
    spec = Path(__file__).resolve().with_name("reviewed_spec.md")
    fourier, quadrature = {}, {}
    for factor in (1,2):
        small, large = fourier_case(32768,factor), fourier_case(65536,factor)
        error = abs(small["H_P33_cosine_moment"]-large["H_P33_cosine_moment"])
        assert error < 1e-10
        fourier[str(factor)], quadrature[str(factor)] = large, error
    result = {"scope": "independent mathematical CPU examples, no neural-model or GPU execution",
              "reviewed_spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
              "epsilon": EPSILON, "intervals": INTERVALS, "grid": grid().tolist(),
              "gaussian_oracle_cases": [gaussian_variance(value) for value in (.2,1.,1.8)],
              "gaussian_interpretation": "Eigenvalues .2 and 1.8 form covariance [[1,.8],[.8,1]], consistent with per-channel unit-variance normalization.",
              "fourier_oracle": fourier,
              "fourier_factor2_interpretation": "Joint p0(f1,f2)=1+.8*cos(2pi*(f2-f1)); both coordinates have noise std t, so the relative coordinate has variance 2t^2 and update u2-u1=2t*d_delta log p_t.",
              "fourier_quadrature_change_32768_to65536": quadrature,
              "readout_examples": point_and_multimodal_readout(), "wrapped_truncation": truncation_checks(),
              "T_G_budget": accumulation_budget()}
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
