"""Freeze a phase's common candidate chemistry and query its missing official references."""
import argparse
import json
import os
from pathlib import Path
import subprocess as sp
import sys
import time
from run_component import SOURCE, PROJECT, execute, write_json

CONFIG = PROJECT / 'workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json'
BASE_CACHE = PROJECT / 'workstreams/proposal_realization_candidates_20260826/grounding/runs/spad_basin_closure_official_20260904_v1/official_mp_cache'

parser = argparse.ArgumentParser()
parser.add_argument('--run-root', type=Path, required=True)
parser.add_argument('--phase', choices=['canary', 'pilot256', 'formal'], required=True)
parser.add_argument('--login-node-query', action='store_true', help='Run official HTTP queries from the existing login session')
args = parser.parse_args()
if args.login_node_query and os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('official queries must run on the login node, not an offline compute node')
if not args.login_node_query and not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('official hull preparation requires its allocated CPUs')
root = args.run_root
count, seed = (16, 202609070) if args.phase == 'canary' else (256, 202609071)
execution = root / ('hull_execution_' + args.phase)
execution.mkdir(exist_ok=False)
hull = root / ('hull_' + args.phase)
if args.phase == 'canary':
    planners = [('R', root / 'canary_40400/R/planner/plans_for_dlm.jsonl'),
                ('I', root / 'canary_40400/I/planner/plans_for_dlm.jsonl')]
elif args.phase == 'pilot256':
    planners = [('R', root / 'pilot_256/R/planner/plans_for_dlm.jsonl'),
                ('I', root / 'pilot_256/shared_I/planner/plans_for_dlm.jsonl')]
else:
    registration = json.loads((root / 'FORMAL_REGISTRATION.json').read_text())
    count = registration['requests_per_seed']
    formal_root = root / f'formal_{2 * count}'
    planners = [(role, formal_root / role / f'seed_{planner_seed}' / 'planner/plans_for_dlm.jsonl')
                for role in ('R', registration['selected_role']) for planner_seed in registration['planner_seeds']]
inputs = {'schema': 'r03_hull_union_inputs_v1', 'purpose': 'evaluation', 'phase': args.phase,
          'inputs': [{'cell_id': role + '_planner_' + str(index), 'arm': role,
                      'seed': registration['planner_seeds'][index % 2] if args.phase == 'formal' else seed, 'type': 'planner',
                      'path': str(path), 'expected_requests': count} for index, (role, path) in enumerate(planners)]}
write_json(execution / 'INPUTS.json', inputs)
arguments = ['prepare', '--inputs-manifest', execution / 'INPUTS.json', '--known-cache', BASE_CACHE,
             '--query-config', CONFIG, '--query-source-dir', SOURCE / 'eval_runtime', '--run-root', hull]
canary_fresh = root / 'hull_canary/missing_query/official_mp_cache'
if args.phase in ('pilot256', 'formal') and (canary_fresh / 'completion_SUCCESS').is_file():
    arguments += ['--known-cache', canary_fresh]
pilot_fresh = root / 'hull_pilot256/missing_query/official_mp_cache'
if args.phase == 'formal' and (pilot_fresh / 'completion_SUCCESS').is_file():
    arguments += ['--known-cache', pilot_fresh]
try:
    execute(execution, 'prepare', 'operations/r03_c3fd_main_20260907/prepare_hull_union.py', arguments)
    query = json.loads((hull / 'QUERY_COMMAND.json').read_text())
    if query['needed']:
        if not args.login_node_query:
            raise RuntimeError('missing official references require --login-node-query; compute nodes are offline')
        command = [query['argv_without_credential'][0],
                   str(SOURCE / 'operations/r03_c3fd_main_20260907/run_hull_query.py'), '--hull-root', str(hull)]
        began = time.monotonic()
        with (execution / 'query.out').open('x') as stdout, (execution / 'query.err').open('x') as stderr:
            result = sp.run(command, stdout=stdout, stderr=stderr)
        write_json(execution / 'query.stage.json', {'returncode': result.returncode, 'seconds': time.monotonic() - began,
                    'credential_serialized': False, 'credential_provider_modified': False})
        if result.returncode:
            raise RuntimeError('official query failed; errors preserved and no scientific unknown substituted')
    execute(execution, 'finalize', 'operations/r03_c3fd_main_20260907/prepare_hull_union.py', ['finalize', '--run-root', hull])
    write_json(execution / 'STAGE_FINAL.json', {'phase': args.phase, 'requests_per_arm': 2 * count if args.phase == 'formal' else count,
               'cache': str(hull / 'official_mp_cache'), 'actual_endpoint_verification_still_required': True})
    (execution / '_SUCCESS').touch()
except BaseException as error:
    write_json(execution / 'FAILURE.json', {'type': type(error).__name__, 'message': str(error)})
    (execution / '_FAILED').touch()
    raise
