"""Small numerical counterexamples; no project model or downloaded code is run."""
from pathlib import Path
import json
import math
import numpy as np


def half_log_gram(lattice):
    eigenvalues, vectors = np.linalg.eigh(lattice @ lattice.T)
    return 0.5 * (vectors * np.log(eigenvalues)) @ vectors.T


angle = 0.71
rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
lattice = np.diag([3.0, 4.0, 5.0])
basis_change = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
wrapped = {}
for sigma in [0.1, 0.5]:
    for delta in [0.49, 0.5]:
        offsets = delta + np.arange(-8, 9)
        logw = -offsets ** 2 / (2 * sigma ** 2)
        weights = np.exp(logw - np.max(logw))
        exact = -float(np.sum(weights * offsets) / weights.sum()) / sigma ** 2
        nearest = -float(offsets[np.argmin(np.abs(offsets))]) / sigma ** 2
        wrapped[f'sigma={sigma},delta={delta}'] = {'wrapped_score': exact, 'nearest_image_score': nearest}

fallback = 0.095
p_g, p_s = 0.8, 0.6
joint11 = (1 - fallback) * p_g * p_s + fallback * p_g
joint10 = (1 - fallback) * p_g * (1 - p_s)
denom1 = (1 - fallback) * p_s + fallback * p_g
denom0 = 1 - denom1
frac_rotation = np.array([[0.0, -1.0], [1.0, -1.0]])
hex_metric = np.array([[1.0, -0.5], [-0.5, 1.0]])
center = np.array([0.13, 0.24])
query = np.array([0.18, 0.26])
orbit_centers = np.stack([center, frac_rotation @ center, frac_rotation @ frac_rotation @ center])
translations = np.array([(i, j) for i in range(-8, 9) for j in range(-8, 9)], dtype=float)


def orbit_gaussian_sum(point, metric):
    delta = point[None, None, :] - orbit_centers[:, None, :] + translations[None, :, :]
    squared = np.einsum('...i,ij,...j->...', delta, metric, delta)
    return float(np.exp(-squared / (2 * 0.04 ** 2)).sum())


out = {
    'purpose': 'Numerical illustrations of derived limits, not model or physical evaluations.',
    'log_spd_global_rotation_error': float(np.linalg.norm(half_log_gram(lattice @ rotation) - half_log_gram(lattice))),
    'log_spd_equivalent_integer_basis_difference': float(np.linalg.norm(half_log_gram(basis_change @ lattice) - half_log_gram(lattice))),
    'fractional_third_grid100_residual': abs(1 / 3 - round(100 / 3) / 100),
    'old_vpa_46_4117_to_12_7358_log_change': math.log(12.7358 / 46.4117),
    'same_log_change_in_largest_v2_sigma_units': abs(math.log(12.7358 / 46.4117)) / 0.2,
    'coordinate_quantization_scalar_unchanged_probability_sigma_A_0_05': {
        str(length): math.erf((0.01 * length / 2) / (math.sqrt(2) * 0.05)) for length in [5, 10, 20]
    },
    'wrapped_score_counterexamples': wrapped,
    'hard_mirror_constraint_saddle': {'E_x0': 1.0, 'E_xplusminus1': 0.0, 'second_derivative_at_x0': -4.0},
    'fractional_sgwn_metric_counterexample': {
        'fractional_rotation': frac_rotation.tolist(),
        'hexagonal_metric': hex_metric.tolist(),
        'R_transpose_G_R_error': float(np.linalg.norm(frac_rotation.T @ hex_metric @ frac_rotation - hex_metric)),
        'R_not_orthogonal_error': float(np.linalg.norm(frac_rotation.T @ frac_rotation - np.eye(2))),
        'q_euclidean_x': orbit_gaussian_sum(query, np.eye(2)),
        'q_euclidean_Rx': orbit_gaussian_sum(frac_rotation @ query, np.eye(2)),
        'q_metric_x': orbit_gaussian_sum(query, hex_metric),
        'q_metric_Rx': orbit_gaussian_sum(frac_rotation @ query, hex_metric),
        'scope': '2D plane embeds in a 3D trigonal crystal; independent formula check, not execution of SGEquiDiff.'
    },
    'independent_soft_condition_demo': {'P_geometry1_given_soft1': p_g, 'P_geometry1_given_soft0': p_g},
    'illustrative_teacher_fallback_mixture_demo': {
        'fallback_fraction': fallback,
        'P_geometry1_given_soft1': joint11 / denom1,
        'P_geometry1_given_soft0': joint10 / denom0,
        'warning': 'Synthetic Bernoulli example; not a fit to project data.'
    },
    'training_exposure_context': {
        'v2_registered_source_views': 27136 * 2 * 2,
        'diffcsppp_mp20_epoch_upper_budget_structure_exposures': 27136 * 1000,
        'flowllm_rfm_upper_budget_pairs': 3_300_000 * 20,
        'mattergen_base_structure_exposures': 1_740_000 * 64 * 8,
        'warning': 'Different objectives, early stopping, parameter counts and pretraining; not FLOP-equivalent.'
    }
}
assert out['log_spd_global_rotation_error'] < 1e-12
assert abs(wrapped['sigma=0.5,delta=0.5']['wrapped_score']) < 1e-12
sg_check = out['fractional_sgwn_metric_counterexample']
assert abs(sg_check['q_metric_x'] - sg_check['q_metric_Rx']) < 1e-12
assert abs(sg_check['q_euclidean_x'] - sg_check['q_euclidean_Rx']) > 0.2
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
