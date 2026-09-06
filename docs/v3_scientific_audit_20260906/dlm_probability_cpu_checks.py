"""Independent finite-state counterexamples and tiny CPU checks; no model scoring."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_tokens
from crystal_dlm.fixed_slot import build_special_tokens
from crystal_dlm.periodic_repair_model import PeriodicRepairConfig, task_features
from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
from crystal_dlm.periodic_v2_model import PeriodicV2AttentionBias, PeriodicV2Config, numeric_noise_component_features
from crystal_dlm.periodic_v2_objective import fold_typed_logits
from crystal_dlm.spad_generation import _transaction_candidate_tokens
from crystal_dlm.spad_program import program_from_element_order
from crystal_dlm.state_conditioned_model import StateConditionedDLM, context_from_programs


def posterior(data, corruption):
    noisy = data @ corruption
    return (corruption.T * data[None]) / noisy[:, None]


def stationary(kernel):
    value = np.ones(kernel.shape[0]) / kernel.shape[0]
    for _ in range(10000):
        update = value @ kernel
        if np.max(np.abs(update - value)) < 1e-15:
            break
        value = update
    return update


def mutual_information(joint):
    product = joint.sum(1)[:, None] * joint.sum(0)[None]
    nonzero = joint > 0
    return float(np.sum(joint[nonzero] * np.log(joint[nonzero] / product[nonzero])))


def wrapped_rounded_gaussian(sigma, bins=8):
    def cdf(value):
        return .5 * (1 + math.erf(value / (sigma * math.sqrt(2))))
    matrix = np.zeros((bins, bins))
    for i in range(bins):
        for j in range(bins):
            matrix[i, j] = sum(cdf(j - i + .5 + shift * bins) - cdf(j - i - .5 + shift * bins)
                               for shift in range(-4, 5))
    return matrix


class Tokenizer:
    def __init__(self):
        self.vocab = {token: index for index, token in enumerate(build_special_tokens())}
        self.mask_id = len(self.vocab)
        self.mask_token_id = self.mask_id
        self.vocab["<MASK>"] = self.mask_id

    def get_vocab(self):
        return self.vocab


class TinyBase(nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 16)
        self.output = nn.Linear(16, vocabulary)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output


@torch.no_grad()
def geometry_visibility_checks():
    torch.manual_seed(193)
    tokenizer = Tokenizer()
    decoder = StateConditionedDLM(TinyBase(len(tokenizer.vocab)), tokenizer,
                                  PeriodicStateConfig(16, width=12, radial_basis_count=4, max_sites=2))
    attention = PeriodicV2AttentionBias(
        PeriodicRepairConfig(16, width=12, radial_bins=6, fourier_modes=2),
        PeriodicV2Config(16, heads=4, width=12, radial_bins=6, fourier_modes=2,
                         species_width=4, cell_slot_width=3, max_sites=2),
    )
    with torch.no_grad():
        for module in attention.modules():
            if isinstance(module, nn.Linear):
                module.weight.fill_(.023)
                if module.bias is not None:
                    module.bias.fill_(.017)
    tokens, _ = arrays_to_dynamic_tokens([4., 4., 4.], [90., 90., 90.], ["H", "H"],
                                         [[0., 0., 0.], [.3, .2, .1]])
    clean = [tokenizer.vocab[token] for token in tokens]
    program = program_from_element_order({"N": 2, "elements": ["H"], "counts": [2]}, ["H"], order_source="CPU_fixture")
    positions = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    observations = []
    for step, target in enumerate(positions):
        current = torch.tensor([[0] + clean])
        current[0, [1 + p for p in positions[step:]]] = tokenizer.mask_id
        context = context_from_programs(current.clone(), prompt_length=1, num_sites=2, programs=[program],
                                        active_positions={0: positions[step:]}, task_id=0, max_sites=2)
        geometry = decoder.geometry_inputs(context)
        tasks = task_features(context, current, geometry, tokenizer.mask_id)
        bias = attention(geometry, context, current.shape[1], tasks)
        observations.append({"position": target, "lattice_known": bool(geometry["lattice_known"][0]),
                             "known_sites": int(geometry["site_known"].sum()),
                             "global_bias_abs_sum": float(bias.abs().sum()),
                             "active_query_bias_abs_sum": float(bias[0, :, 1 + target].abs().sum())})
    old = torch.tensor([[0] + clean])
    current = old.clone()
    current[0, [1 + p for p in positions]] = tokenizer.mask_id
    context = context_from_programs(old, prompt_length=1, num_sites=2, programs=[program],
                                    active_positions={0: positions}, task_id=3, max_sites=2)
    geometry = decoder.geometry_inputs(context)
    bias = attention(geometry, context, current.shape[1], task_features(context, current, geometry, tokenizer.mask_id))
    missing = numeric_noise_component_features(context)
    explicit_unknown = numeric_noise_component_features(replace(context, numeric_noise_components=torch.full((1, 3), -1.)))
    assert all(row["global_bias_abs_sum"] == 0 for row in observations[:6])
    assert float(bias.abs().sum()) > 0
    assert torch.equal(missing, explicit_unknown)
    return {"construction_prefix": observations, "repair_global_bias_abs_sum": float(bias.abs().sum()),
            "missing_and_explicit_unknown_sigma_features_equal": True,
            "scope": "nonzero artificial weights prove routing availability; not trained-model effectiveness"}


def main():
    torch.set_num_threads(1)
    data = np.array([.9, .1])
    corruption = np.array([[.8, .2], [.2, .8]])
    denoiser = posterior(data, corruption)
    augmented = corruption @ denoiser
    direct_stationary = stationary(denoiser)
    low = np.array([[.95, .05], [.05, .95]])
    wrong_sigma_kernel = low @ denoiser
    assert np.allclose(data @ augmented, data, atol=1e-14)
    assert not np.allclose(data @ denoiser, data)
    assert not np.allclose(data @ wrong_sigma_kernel, data)

    data_joint = np.array([[.5, 0.], [0., .5]])
    parallel = np.outer(data_joint.sum(1), data_joint.sum(0))
    assert parallel[0, 1] + parallel[1, 0] == .5

    geometry_prior = np.array([.7, .3])
    plan_prior = np.array([.2, .8])
    independent = np.outer(geometry_prior, plan_prior)
    fallback_mixture = .905 * independent + .095 * np.diag(geometry_prior)

    ids = torch.arange(101)
    values = torch.linspace(0, 1, 101)
    folded, _ = fold_typed_logits(torch.zeros(1, 101), ids, values, "coord")
    at_point7 = torch.softmax(folded / .7, -1)[0]
    corrected = torch.zeros(1, 101)
    corrected[0, [0, 100]] = -math.log(2.)
    corrected, _ = fold_typed_logits(corrected, ids, values, "coord")
    corrected_probabilities = torch.softmax(corrected / .7, -1)[0]
    assert float(at_point7[0]) > .02
    torch.testing.assert_close(corrected_probabilities, torch.full((100,), .01), atol=1e-7, rtol=0)

    sampler_values = {}
    for shift in (0., 1000., -1000.):
        logits = torch.tensor([[[-torch.finfo(torch.float64).max, 0. + shift, 2. + shift]]], dtype=torch.float64)
        selected = _transaction_candidate_tokens(logits, active_absolute_positions={0: 0}, temperature=.7,
                                                 remasking="low_confidence", sampling_seeds_by_batch=[19], salt=3)
        generator = torch.Generator().manual_seed(22)
        uniform = torch.rand((3,), dtype=torch.float64, generator=generator).clamp_min(torch.finfo(torch.float64).tiny)
        stable = torch.argmax(logits[0, 0] - .7 * torch.log(-torch.log(uniform)))
        sampler_values[str(shift)] = {"current_exp_sampler": int(selected[0, 0]),
                                      "equivalent_log_space_sampler": int(stable),
                                      "stable_softmax": torch.softmax(logits[0, 0] / .7, -1).tolist()}
    assert sampler_values["-1000.0"]["current_exp_sampler"] == 0
    assert len({row["equivalent_log_space_sampler"] for row in sampler_values.values()}) == 1

    q1, q2 = wrapped_rounded_gaussian(.2), wrapped_rounded_gaussian(.4)
    qtotal = wrapped_rounded_gaussian(math.sqrt(.2 ** 2 + .4 ** 2))
    assert np.allclose(q1.sum(1), 1.) and np.allclose(q2.sum(1), 1.) and np.allclose(qtotal.sum(1), 1.)
    composition_error = float(np.max(np.abs(q1 @ q2 - qtotal)))
    assert composition_error > 1e-3

    source_files = ["periodic_v2_training_data.py", "periodic_v2_objective.py", "periodic_v2_corruption.py",
                    "periodic_v2_model.py", "programmed_path_runtime.py", "spad_generation.py"]
    report = {
        "scope": "tiny CPU mathematics/implementation fixtures; no trained model inference, MLIP, relaxation, or SUN evaluation",
        "execution_commit_reference": "2e904c260bafb6750c9a5dbb0d920c4ff8c3a868",
        "source_sha256": {name: hashlib.sha256((ROOT / "src" / "crystal_dlm" / name).read_bytes()).hexdigest() for name in source_files},
        "denoise_vs_data_augmentation": {
            "data": data.tolist(), "forward_C": corruption.tolist(), "Bayes_D": denoiser.tolist(),
            "data_after_direct_D": (data @ denoiser).tolist(), "stationary_of_direct_D": direct_stationary.tolist(),
            "data_after_C_then_D": (data @ augmented).tolist(),
            "data_after_low_C_but_mixture_D": (data @ wrong_sigma_kernel).tolist(),
            "toy_energy": [0., -10.], "data_mean_toy_energy": float(data @ np.array([0., -10.])),
            "repeated_direct_D_mean_toy_energy": float(direct_stationary @ np.array([0., -10.])),
        },
        "prefix_admission_counterexample": {
            "data_states": ["00", "01", "11"], "data_probabilities": [.1, .4, .5],
            "fully_legal_states": ["00", "11"], "locally_fitted_legal_chain": [.5, .5],
            "globally_restricted_data": [1 / 6, 5 / 6],
        },
        "parallel_marginals_are_not_joint_posterior": {
            "true_joint": data_joint.tolist(), "one_shot_independent_marginals": parallel.tolist(),
            "unsupported_probability": float(parallel[0, 1] + parallel[1, 0]),
            "sequential_oracle_conditional_unsupported_probability": 0.,
        },
        "random_soft_plan_independence": {
            "joint_G_S": independent.tolist(), "p_G_given_each_S": (independent / independent.sum(0)).T.tolist(),
            "mutual_information_nats": mutual_information(independent),
            "illustrative_9p5_teacher_fallback_MI_nats": mutual_information(fallback_mixture),
            "scope": "population toy, not measured V2 conditional independence",
        },
        "alias_measure_and_temperature": {
            "raw_equal_logits_physical_zero_probability_at_T0p7": float(at_point7[0]),
            "other_physical_class_probability": float(at_point7[1]),
            "uniform_physical_class_probability": .01,
            "alias_split_mass_correction_uniform_max_error": float((corrected_probabilities - .01).abs().max()),
            "scope": "a prior/measure effect, not an alias implementation bug",
        },
        "current_sampler_shift_invariance_edge_case": sampler_values,
        "quantized_gaussian_marginals_not_naive_semigroup": {"max_abs_Q0p2_Q0p4_minus_Qsqrt0p2": composition_error},
        "representation_limits": {
            "fractional_bin": .01, "worst_coordinate_error_bound_cube4_A": .005 * 12,
            "coordinate_rounding_rms_cube4_A": math.sqrt(3 * (4 * .01) ** 2 / 12),
            "isotropic_cartesian_noise_fractional_variance_ratio_cell_1_10_10": 100.,
            "legal_distance_repulsive_proxy_energy_ratio_r0p6_to_r1p2": (1.2 / .6) ** 12,
        },
        "two_epoch_repair_coordinate_supervision": {
            str(n): {"no_coordinate_prefix_probability": (1 - .75 * .5) ** 2,
                     "no_coordinate_target_any_branch_probability_before_admission":
                         (.75 * .5 + .25 * .9 ** (3 * n) / (3 * n + 1)) ** 2}
            for n in (1, 5, 20)
        },
        "geometry_routing": geometry_visibility_checks(),
        "all_assertions_passed": True,
    }
    destination = Path(__file__).with_name("dlm_probability_cpu_checks.json")
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
