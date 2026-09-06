"""Freeze read-only upstream evidence and verify independent probability examples.

No checkpoint, GPU, MLIP, downloaded code, or project training module is run.
"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import math
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FREEZE = '2e904c260bafb6750c9a5dbb0d920c4ff8c3a868'
SOURCES = [
    'src/crystal_dlm/c3fd_calibration.py',
    'src/crystal_dlm/c3fd_planner_model.py',
    'src/crystal_dlm/semantic_composition_head.py',
    'src/crystal_dlm/species_physics.py',
    'src/crystal_dlm/family_reachability.py',
    'src/crystal_dlm/ccfd_v2.py',
    'src/crystal_dlm/composition_validity.py',
    'src/crystal_dlm/c3fd_llama_typed_planner.py',
    'src/crystal_dlm/c3fd_llama_fused_plan.py',
    'src/crystal_dlm/c3fd_native_plan.py',
    'src/crystal_dlm/species_program_pointer.py',
    'src/crystal_dlm/spad_program.py',
    'src/crystal_dlm/periodic_v2_plan_data.py',
    'src/scripts/train_c3fd_llama_typed_planner.py',
    'src/scripts/sample_c3fd_llama_typed_planner.py',
    'src/scripts/train_spad_species_pointer.py',
    'src/scripts/prepare_periodic_v2_plans.py',
    'scripts/build_c3fd_planner_data.py',
    'scripts/extract_c3fd_planner_context.py',
    'scripts/build_c3fd_llama_fused_data.py',
    'scripts/build_spad_species_pointer_data.py',
    'scripts/sample_c3fd_plans.py',
]


def softmax(values):
    values = np.asarray(values, dtype=float)
    result = np.exp(values - np.max(values))
    return result / result.sum()


def load_json(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    return json.loads(raw), {'path': relative, 'sha256': sha256(raw).hexdigest()}


def main():
    source_records = []
    for relative in SOURCES:
        raw = subprocess.check_output(['git', 'show', FREEZE + ':' + relative], cwd=ROOT)
        target = OUT / 'frozen' / (relative + '.txt')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        source_records.append({'path': relative, 'git_commit': FREEZE,
                               'sha256': sha256(raw).hexdigest(), 'lines': len(raw.splitlines())})

    source_metrics = []
    history = {}
    names = [
        'C3FD_PLANNER_FINAL.json', 'C3FD_V21_PILOT_FINAL.json',
        'C3FD_V21_STEP1_DATA_AUDIT.json', 'C3FD_V21_STEP3_CALIBRATION.json',
        'C3FD_V21_STEP4_PROPOSAL_SIM.json', 'C3FD_DRIFT_DIAGNOSTIC.json',
        'C3FD_V25_PILOT_FINAL.json', 'C3FD_V25_REQUESTED1000_FINAL.json',
        'C3FD_V25_TEACHER_WITNESS_AUDIT.json', 'PLANNER_COUNTVALENCE_FACTORIAL_FINAL.json',
        'C3FD_V22_AUDIT_V1_ENGINEERING_FAILURE.json', 'C3FD_V22_AUDIT_V2_ENGINEERING_FAILURE.json',
        'C3FD_V23_AUDIT_ENGINEERING_FAILURE.json', 'C3FD_V24_AUDIT_ENGINEERING_FAILURE.json',
    ]
    for name in names:
        value, receipt = load_json('results/remote_screens/' + name)
        source_metrics.append(receipt)
        if 'datasets' in value:
            value = {**value, 'datasets': [{k: v for k, v in d.items() if k not in ['strata', 'N', 'arity', 'family']} for d in value['datasets']]}
        history[name] = value

    current, receipt = load_json('docs/v3_scientific_audit_20260906/evidence/FROZEN_CONDITIONING_AUDIT.json')
    source_metrics.append(receipt)
    current_summary = {
        'source_files': current['source_files'], 'train_val_overlap': current['train_val_overlap'],
        'splits': {s: {k: v for k, v in data.items() if k not in ['confusion_predicted_only', 'source_order_examples_of_soft_mismatch']}
                   for s, data in current['splits'].items()},
    }
    for split, values in current_summary['splits'].items():
        n = values['checks']['predicted']
        values['all_three_soft_agreement_fraction'] = values['checks']['all_three_soft_match'] / n
        values['single_field_agreement_fraction'] = {k: v / n for k, v in values['soft_agreement_predicted_only'].items()}
        values['product_of_aggregate_field_agreements_not_joint_probability'] = math.prod(values['single_field_agreement_fraction'].values())

    checklist, receipt = load_json('docs/C3FD_LLAMA_DLM_SUN_CHECKLIST_V6.json')
    source_metrics.append(receipt)
    historical_fused = checklist['post_fm']['compact_v2_pivot']

    base = np.array([0.8, 0.2])
    target = np.array([0.35, 0.65])
    residual = np.log(target) - np.log(base)
    fitted = softmax(np.log(base) + residual)
    duplicated = base * base / np.sum(base * base)
    pfull = np.array([0.2, 0.3, 0.5])
    plegal = pfull[:2] / pfull[:2].sum()
    local_first = np.array([0.5, 0.5])
    valid_completion_probability = np.array([0.1, 1.0])
    global_first = local_first * valid_completion_probability
    global_first /= global_first.sum()
    # Two identical path experts with a prefix-dependent product normalizer.
    path_expert = np.array([0.45, 0.05, 0.25, 0.25])
    global_path_poe = path_expert ** 2
    global_path_poe /= global_path_poe.sum()
    local_path_poe = np.array([0.5 * 0.81 / 0.82, 0.5 * 0.01 / 0.82, 0.25, 0.25])
    toy = {
        'unit_poe_step0': softmax(np.log(base)).tolist(),
        'unit_poe_residual_can_fit_target': {'target': target.tolist(), 'fitted': fitted.tolist()},
        'independently_relearning_same_base_squares_prior': duplicated.tolist(),
        'full_vs_legal_support_nll': {'full': -math.log(pfull[0]), 'legal': -math.log(plegal[0]),
                                    'difference': -math.log(pfull[:2].sum())},
        'viability_mask_vs_global_conditioning': {'local_first': local_first.tolist(), 'global_conditioned_first': global_first.tolist()},
        'local_poe_vs_global_path_poe': {'local': local_path_poe.tolist(), 'global': global_path_poe.tolist()},
        'independent_soft_heads_two_mode_example': {'data': {'00': 0.5, '11': 0.5}, 'product_marginal_mass_off_data_modes': 0.5},
        'conditional_independence_does_not_imply_unconditional_independence': {
            'mean_field_success': 0.5, 'product_of_means': 0.5 ** 3,
            'mean_product_given_shared_easy_hard_context': (0.9 ** 3 + 0.1 ** 3) / 2,
        },
        'single_valence_grammar_example': {
            'mixed_valence_charge': 1 * 2 + 2 * 3 + 4 * (-2),
            'single_Fe_valence_charges_for_catalog_2_3': [3 * q + 4 * (-2) for q in [2, 3]],
            'scope': 'Finite-alphabet algebra example; no assertion about DFT stability or the complete production vocabulary.'
        },
        'scope': 'Independent mathematical examples; not actual model outputs, performance, or physical simulations.',
    }
    assert np.max(np.abs(fitted - target)) < 1e-12
    assert abs((-math.log(pfull[0])) - (-math.log(plegal[0])) + math.log(pfull[:2].sum())) < 1e-12
    assert np.max(np.abs(local_path_poe - global_path_poe)) > 0.1
    (OUT / 'source_manifest.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(), 'frozen_source': source_records, 'source_metrics': source_metrics}, indent=2), encoding='utf-8')
    (OUT / 'history_summary.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'conditioning_summary.json').write_text(json.dumps(current_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'fused_planner_historical_summary.json').write_text(json.dumps(historical_fused, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'mathematical_checks.json').write_text(json.dumps(toy, indent=2), encoding='utf-8')
    print(json.dumps({'frozen_sources': len(source_records), 'metric_sources': len(source_metrics), 'math_checks_passed': True,
                      'all_three_agreement': {s: v['all_three_soft_agreement_fraction'] for s, v in current_summary['splits'].items()}}, indent=2))


if __name__ == '__main__':
    main()
