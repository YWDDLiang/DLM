"""Independent algebra/budget checks for the mixed-space counterproposal.

No model import, checkpoint, GPU, MLIP, or external data generation.
"""
from hashlib import sha256
from pathlib import Path
import json
import math
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SHA = '2e904c260bafb6750c9a5dbb0d920c4ff8c3a868'


def alpha_bar(t):
    return 0.0 if t == 1.0 else math.cos(math.pi * t / 2) ** 2


def budget(n):
    d = 6 + 3 * n
    scalar_per_view = 0.75 + 0.25 * 0.55 * d
    return {
        'N': n, 'geometric_channels': d,
        'v2_expected_targets_per_forward': scalar_per_view,
        'v2_expected_targets_per_source_two_epochs': 4 * scalar_per_view,
        'hybrid_token_targets_per_source_two_epochs': 2 * scalar_per_view,
        'hybrid_geometric_targets_per_source_two_epochs': 2 * d,
        'dense_to_v2_repair_target_ratio_not_information_ratio': d / scalar_per_view,
        'nfe_v2_construction_and_repair': 2 * d,
        'nfe_prior_euler32': 32, 'nfe_prior_heun32': 64,
        'nfe_token_construction_plus_euler32': d + 32,
        'nfe_token_construction_plus_heun32': d + 64,
    }


def main():
    rng = np.random.default_rng(173)
    z0, eps = rng.normal(size=(2, 6))
    inversions = []
    for t in [0.002, 0.1, 0.5, 1.0]:
        a = alpha_bar(t)
        alpha, sigma = math.sqrt(a), math.sqrt(1 - a)
        zt = alpha * z0 + sigma * eps
        v = alpha * eps - sigma * z0
        recovered_z0 = alpha * zt - sigma * v
        recovered_eps = sigma * zt + alpha * v
        score_from_v = -zt - alpha / sigma * v
        inversions.append({'t': t, 'z0_error': float(np.max(np.abs(recovered_z0 - z0))),
                           'eps_error': float(np.max(np.abs(recovered_eps - eps))),
                           'score_error': float(np.max(np.abs(score_from_v + eps / sigma)))})
    data = json.loads((ROOT / 'results/remote_screens/C3FD_V21_STEP1_DATA_AUDIT.json').read_text())
    train = next(d for d in data['datasets'] if d['name'] == 'train')
    source_count = sum(train['N'].values())
    atom_count = sum(int(n) * count for n, count in train['N'].items())
    channel_count = 6 * source_count + 3 * atom_count
    aT = alpha_bar(1.0)
    torus_1d_tv_upper = sum(math.exp(-2 * math.pi ** 2 * k ** 2) for k in range(1, 9))
    F, delta = rng.uniform(size=(5, 3)), rng.normal(size=(5, 3)) * 0.1
    first, second = F.copy(), F.copy()
    for i in [0, 1, 2, 3, 4]:
        first[i] = (first[i] + delta[i]) % 1
    for i in [4, 2, 0, 3, 1]:
        second[i] = (second[i] + delta[i]) % 1
    result = {
        'scope': 'Algebra and expected-label/NFE accounting, not model performance or measured runtime.',
        'vp_v_parameterization': inversions,
        'zero_head_initialization': {
            'alpha_bar_terminal': aT,
            'naive_epsilon_to_x0_at_exact_prior': 'undefined at alpha=0; use the regular v parameterization',
            'zero_v_probability_flow_drift': 0.0,
            'meaning': 'Zero v is a stationary Gaussian-reference probability flow, not a trained crystal denoiser.'
        },
        'prior_approximations': {
            'lattice_terminal_prior_is_exact_standard_Gaussian': True,
            'torus_1d_TV_upper_sigma1': torus_1d_tv_upper,
            'torus_60d_TV_upper_sigma1': 60 * torus_1d_tv_upper,
            'boundary': 'Coordinate prior is an explicitly bounded uniform approximation; these are prior bounds only.'
        },
        'write_order_of_precomputed_independent_site_updates': {'max_difference': float(np.max(np.abs(first - second)))},
        'parameter_accounting': {
            'v2_trainable_inventory': 36993232,
            'input_delta_table': 20324352 // 2,
            'output_delta_table': 20324352 // 2,
            'numeric_adapter': (6 * 18 + 3 * 17) * 4096,
            'legacy_repair_task_projection': 9 * 4096,
            'new_geometry_heads': 9 * (4096 + 1),
            'new_time_mlp': 64 * 128 + 128 + 128 * 4096 + 4096,
            'new_task_vector': 4096,
            'G_graph_eligible_existing_inventory': 36993232 - 20324352 // 2 - (6 * 18 + 3 * 17) * 4096 - 9 * 4096,
            'T_plus_G_optimizer_union_candidate': 36993232 + 9 * (4096 + 1) + 64 * 128 + 128 + 128 * 4096 + 4096 + 4096,
            'boundary': 'Static module inventory, not measured nonzero gradients. G bypasses legacy logits and task projection; whole input_delta tensor remains eligible but absent-token rows have zero direct G gradient.',
        },
        'example_budgets': [budget(n) for n in [1, 10, 20]],
        'MP20_train_aggregate': {
            'source_count': source_count, 'atom_count': atom_count,
            'mean_N': atom_count / source_count, 'channels_per_full_view': channel_count,
            'v2_two_epoch_forward_examples': 4 * source_count,
            'v2_two_epoch_expected_numeric_labels': 3 * source_count + 0.55 * channel_count,
            'hybrid_two_epoch_forward_examples': 4 * source_count,
            'hybrid_two_epoch_token_labels': 1.5 * source_count + 0.275 * channel_count,
            'hybrid_two_epoch_geometric_labels': 2 * channel_count,
            'all_geometry_two_epoch_labels': 4 * channel_count,
            'geometry_views_if_one_token_and_one_geometry_per_source_epoch': 2 * source_count,
            'updates_two_epochs_at_global24': 2 * math.ceil(2 * source_count / 24),
        },
    }
    assert all(x['z0_error'] < 1e-12 and x['eps_error'] < 1e-12 and x['score_error'] < 1e-10 for x in inversions)
    assert result['write_order_of_precomputed_independent_site_updates']['max_difference'] == 0.0
    (OUT / 'geometric_head_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    records = []
    sources = ['src/crystal_dlm/pmtr_training.py', 'src/scripts/train_pmtr.py',
               'src/crystal_dlm/manifold_repair_head.py', 'src/crystal_dlm/manifold_token_transport.py',
               'src/crystal_dlm/manifold_repair_objective.py', 'src/crystal_dlm/periodic_v2_initialization.py',
               'src/crystal_dlm/periodic_v2_model.py', 'src/crystal_dlm/periodic_state_conditioning.py',
               'src/crystal_dlm/state_conditioned_model.py', 'src/crystal_dlm/periodic_v2_training_data.py',
               'src/crystal_dlm/periodic_v2_objective.py',
               'src/crystal_dlm/periodic_repair_initialization.py', 'src/crystal_dlm/periodic_repair_model.py',
               'src/crystal_dlm/canonical_site_order.py', 'src/scripts/train_periodic_dlm_v2.py',
               'docs/teacher_feedback_unified_v1/15_PMTR_SCIENTIFIC_METHOD_AND_EXECUTION.md',
               'docs/teacher_feedback_unified_v1/16_PMTR_CODE_GROUNDED_ARCHITECTURE_AUDIT.md']
    for source in sources:
        raw = subprocess.check_output(['git', 'show', SHA + ':' + source], cwd=ROOT)
        target = OUT / 'frozen' / (source + '.txt')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        records.append({'source': source, 'git_commit': SHA, 'sha256': sha256(raw).hexdigest()})
    (OUT / 'source_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
