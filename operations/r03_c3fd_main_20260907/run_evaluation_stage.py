"""Score completed trial components using their actual common official cache."""
import argparse
import json
import os
from pathlib import Path

from evaluate_trial import evaluate_trial, evaluate_formal
from run_component import PROJECT

parser = argparse.ArgumentParser()
parser.add_argument('--run-root', type=Path, required=True)
parser.add_argument('--phase', choices=['canary', 'pilot256', 'formal'], required=True)
parser.add_argument('--preview-roles', nargs='+', choices=['R', 'I', 'G', 'P'])
args = parser.parse_args()
if args.preview_roles and args.phase != 'pilot256':
    parser.error('preview roles apply only to the pilot')
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('evaluation requires the registered CPU allocation')
root = args.run_root.resolve()
if args.phase == 'formal':
    print(json.dumps(evaluate_formal(root), sort_keys=True), flush=True)
    raise SystemExit(0)
if args.phase == 'canary':
    marker = json.loads((root / 'GEOMETRY_CANARY_COMPLETE.json').read_text())
    components = Path(marker['directory'])
    roles, count, seed = ('I', 'G', 'P'), 16, 202609070
else:
    components = root / 'pilot_256'
    roles, count, seed = ('R', 'I', 'G', 'P'), 256, 202609071
manifest = {
    'schema': 'r03_trial_evaluation_manifest_v1', 'phase': args.phase,
    'hull_phase': args.phase,
    'expected_requests': count, 'registered_construction_geometry': True,
    'include_matched_interface_reference': args.phase == 'canary',
    'validity_artifact': 'legacy_existing_direct_read_only' if args.phase == 'canary' else 'basic_comp_struct_only',
    'frozen_config': str(PROJECT / 'workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json'),
    'hull_run_root': str(root / ('hull_' + args.phase)),
    'methods': [{'method_id': 'R03_' + role, 'role': role,
                 'component_dir': str(components / ('I_batch1_reference' if args.phase == 'canary' and role == 'I' else role)),
                 'planner_seed': seed} for role in roles],
}
output_name = args.phase
if args.preview_roles:
    manifest['preview_roles'] = args.preview_roles
    manifest['methods'] = [item for item in manifest['methods'] if item['role'] in args.preview_roles]
    output_name += '_preview_' + '_'.join(args.preview_roles)
    manifest['phase'] = output_name
elif args.phase == 'pilot256':
    preview = root / 'evaluation_pilot256_preview_R_I/TRIAL_EVALUATION_FINAL.json'
    if (preview.parent / '_SUCCESS').is_file():
        manifest['reuse_report'] = str(preview)
manifest_path = root / ('EVALUATION_INPUTS_' + output_name + '.json')
if args.phase == 'pilot256':
    manifest['method_freeze'] = str(root / 'METHOD_FREEZE.json')
with manifest_path.open('x') as stream:
    json.dump(manifest, stream, indent=2)
    stream.write('\n')
report = evaluate_trial(manifest_path, root / ('evaluation_' + output_name))
print(json.dumps(report, sort_keys=True), flush=True)
