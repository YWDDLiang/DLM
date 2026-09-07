"""Score completed trial components using their actual common official cache."""
import argparse
import json
import os
from pathlib import Path

from evaluate_trial import evaluate_trial
from run_component import PROJECT

parser = argparse.ArgumentParser()
parser.add_argument('--run-root', type=Path, required=True)
parser.add_argument('--phase', choices=['canary', 'pilot256'], required=True)
args = parser.parse_args()
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('evaluation requires the registered CPU allocation')
root = args.run_root.resolve()
if args.phase == 'canary':
    marker = json.loads((root / 'GEOMETRY_CANARY_COMPLETE.json').read_text())
    components = Path(marker['directory'])
    roles, count, seed = ('I', 'G', 'P'), 16, 202609070
else:
    components = root / 'pilot_256'
    roles, count, seed = ('R', 'I', 'G', 'P'), 256, 202609071
manifest = {
    'schema': 'r03_trial_evaluation_manifest_v1', 'phase': args.phase,
    'expected_requests': count, 'registered_construction_geometry': True,
    'include_matched_interface_reference': args.phase == 'canary',
    'validity_artifact': 'legacy_existing_direct_read_only' if args.phase == 'canary' else 'basic_comp_struct_only',
    'frozen_config': str(PROJECT / 'workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json'),
    'hull_run_root': str(root / ('hull_' + args.phase)),
    'methods': [{'method_id': 'R03_' + role, 'role': role,
                 'component_dir': str(components / ('I_batch1_reference' if args.phase == 'canary' and role == 'I' else role)),
                 'planner_seed': seed} for role in roles],
}
manifest_path = root / ('EVALUATION_INPUTS_' + args.phase + '.json')
with manifest_path.open('x') as stream:
    json.dump(manifest, stream, indent=2)
    stream.write('\n')
report = evaluate_trial(manifest_path, root / ('evaluation_' + args.phase))
print(json.dumps(report, sort_keys=True), flush=True)
