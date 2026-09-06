"""Small mathematical checks for the V3 design note; no model/data/MLIP calls."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def circle_density_and_score(x: float, mean: float, sigma: float) -> tuple[float, float]:
    # The tested sigma is <= 1; omitted |n| > 12 images are negligible here.
    delta = x - mean + np.arange(-12, 13, dtype=np.float64)
    weights = np.exp(-0.5 * np.square(delta / sigma))
    density = float(weights.sum() / (math.sqrt(2 * math.pi) * sigma))
    score = float(-(delta * weights).sum() / (sigma * sigma * weights.sum()))
    return density, score


def reversible_kernel(prior: np.ndarray, clock: float, *, circle: bool) -> np.ndarray:
    count = len(prior)
    rates = np.zeros((count, count), dtype=np.float64)
    for a in range(count):
        for b in range(count):
            adjacent = abs(a - b) == 1 or (circle and abs(a - b) == count - 1)
            if adjacent:
                rates[a, b] = math.sqrt(prior[b] / prior[a])
        rates[a, a] = -rates[a].sum()
    roots = np.sqrt(prior)
    symmetric = roots[:, None] * rates / roots[None, :]
    assert np.max(np.abs(symmetric - symmetric.T)) < 1e-12
    values, vectors = np.linalg.eigh(symmetric)
    return ((vectors * np.exp(clock * values)) @ vectors.T) / roots[:, None] * roots[None, :]


def run() -> dict:
    score_cases = []
    for x, mean, sigma in ((0.49, 0.0, 0.2), (0.5, 0.0, 0.2), (0.97, 0.01, 0.04), (0.31, 0.83, 1.0)):
        density, score = circle_density_and_score(x, mean, sigma)
        h = 1e-6
        plus = circle_density_and_score(x + h, mean, sigma)[0]
        minus = circle_density_and_score(x - h, mean, sigma)[0]
        finite_difference = (math.log(plus) - math.log(minus)) / (2 * h)
        assert abs(score - finite_difference) < 1e-7
        assert abs(density - circle_density_and_score(x + 1, mean, sigma)[0]) < 1e-12
        nearest = (x - mean + 0.5) % 1 - 0.5
        score_cases.append(dict(x=x, mean=mean, sigma=sigma, wrapped_score=score,
                                finite_difference=finite_difference, nearest_image_score=-nearest / sigma**2))
    delta = 2 * sum(math.exp(-2 * math.pi**2 * k**2) for k in range(1, 20))
    torus_tv_bound = 0.5 * math.expm1(60 * math.log1p(delta))
    assert torus_tv_bound < 2e-7

    prior = np.arange(1, 10, dtype=np.float64)
    prior /= prior.sum()
    q1 = reversible_kernel(prior, 0.2, circle=False)
    q2 = reversible_kernel(prior, 0.7, circle=False)
    total = reversible_kernel(prior, 0.9, circle=False)
    assert np.min(q1) > 0
    assert np.max(abs(q1.sum(1) - 1)) < 1e-12
    assert np.max(abs(prior @ q1 - prior)) < 1e-12
    assert np.max(abs(q1 @ q2 - total)) < 1e-12
    posterior_errors = []
    for x0 in range(len(prior)):
        for xt in range(len(prior)):
            posterior = q1[x0] * q2[:, xt] / total[x0, xt]
            posterior_errors.append(abs(posterior.sum() - 1))
    assert max(posterior_errors) < 1e-10

    density = []
    for sites in (1, 10, 20):
        d = 6 + 3 * sites
        expected_v2_targets = 0.75 + 0.25 * 0.55 * d
        density.append(dict(N=sites, geometry_dimensions=d,
                            expected_V2_supervised_scalars_per_view=expected_v2_targets,
                            dense_geometry_scalars_per_view=d,
                            ratio=d / expected_v2_targets,
                            V2_construct_and_one_repair_nominal_NFE=2*d,
                            one_Cmix_then_D_nominal_NFE=2*d,
                            two_Cmix_then_D_nominal_NFE=3*d,
                            proposed_dense_reverse_steps=64))
    # A joint distribution can carry semantic information that independent
    # randomized annotations cannot identify, even with the same marginals.
    aligned = np.diag([0.5, 0.5])
    independent = np.full((2, 2), 0.25)
    def mutual_information(joint: np.ndarray) -> float:
        denominator = joint.sum(1)[:, None] * joint.sum(0)[None, :]
        positive = joint > 0
        return float((joint[positive] * np.log(joint[positive] / denominator[positive])).sum())
    return {
        "scope": "mathematical_toys_only_not_model_performance",
        "wrapped_score_checks": score_cases,
        "uniform_prior_sigma_fractional": 1.0,
        "uniform_prior_worst_case_TV_bound_for_60_fractional_components": torus_tv_bound,
        "reversible_Q": {
            "row_sum_error": float(np.max(abs(q1.sum(1)-1))),
            "stationary_error": float(np.max(abs(prior @ q1-prior))),
            "semigroup_error": float(np.max(abs(q1@q2-total))),
            "posterior_sum_error": float(max(posterior_errors)),
        },
        "target_density_and_nominal_NFE": density,
        "semantic_mutual_information_nats": {
            "aligned_requested_plan": mutual_information(aligned),
            "independent_prediction_with_same_marginals": mutual_information(independent),
        },
        "extra_lattice_chart_129_bins_six_fields_untied_input_output_parameters_at_width4096": 6*129*4096*2,
        "linear_geometry_heads_6_and_3_outputs_at_width4096_including_bias": 4096*9+9,
    }


if __name__ == "__main__":
    result = run()
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
