"""Run one registered R03 seed component on one already-allocated GPU.

All commands use immutable source and fresh outputs. Labels are for evaluation;
pooled novelty/uniqueness and hull scoring are a later shared stage.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess as sp
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]
PROJECT = Path('/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion')
P0 = PROJECT / 'runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final'
B0 = PROJECT / 'runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final'
LLAMA = Path('/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B')
LLADA = Path('/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct')
CRYS = Path('/public/home/jiaosz/hengzhang/Code/crysllmgen-main')
FROZEN = PROJECT / 'workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1'
REFINER = CRYS / 'out/mp_20/22042026/203930/model_494.pt'
REFINER_SHA256 = '573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e'
DIRECT_SNAPSHOT = PROJECT / ('runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/'
    'frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1/'
    'runtime/crystal_dlm/wqcodiff/crysllmgen/upstream')


def verify_refiner(output):
    digest = hashlib.sha256()
    with REFINER.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != REFINER_SHA256:
        raise ValueError('actual model494 contents changed before refinement')
    write_json(output / 'REFINER_IDENTITY.json', {'checkpoint': str(REFINER),
               'checkpoint_sha256': digest.hexdigest(), 'diff_steps': 800, 'num_evals': 1,
               'verified_utc': dt.datetime.now(dt.timezone.utc).isoformat()})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def execute(output: Path, name: str, relative_script: str, arguments, *, env=None):
    command = [sys.executable, str(SOURCE / relative_script), *map(str, arguments)]
    started = time.monotonic()
    record = {'stage': name, 'command': command,
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(output / (name + '.command.json'), record)
    print(json.dumps({'event': 'stage_start', **record}), flush=True)
    with (output / (name + '.out')).open('x') as stdout, (output / (name + '.err')).open('x') as stderr:
        result = sp.run(command, env=env, stdout=stdout, stderr=stderr)
    record.update(returncode=result.returncode, seconds=time.monotonic() - started)
    write_json(output / (name + '.stage.json'), record)
    print(json.dumps({'event': 'stage_end', 'stage': name,
                      'returncode': result.returncode, 'seconds': record['seconds']}), flush=True)
    if result.returncode:
        raise RuntimeError(f'{name} failed with exit {result.returncode}; see preserved stage logs')


def sample_plans(output, assets, *, count, seed, offset, constrained):
    arguments = ['--model-path', LLAMA, '--checkpoint-path', P0,
                 '--native-prompt-file', assets / 'P0_NATIVE_PROMPT.txt',
                 '--output-dir', output / 'planner', '--num-samples', count,
                 '--seed', seed, '--sample-index-offset', offset, '--batch-size', 4]
    if constrained:
        arguments.extend(['--c3fd-domain', assets / 'C3FD_DOMAIN.json'])
    execute(output, 'planner', 'src/scripts/sample_r03_h1a2_plans.py', arguments)
    return output / 'planner' / 'plans_for_dlm.jsonl'


def export_programs(output, assets, pointer, plans):
    destination = output / 'plans_with_programs.jsonl'
    execute(output, 'control', 'src/scripts/export_r03_control_programs.py', [
        '--llama-model', LLAMA, '--planner-checkpoint', P0, '--pointer-checkpoint', pointer,
        '--plans-jsonl', plans, '--output-jsonl', destination,
        '--prompt-style', 'h1_rich_plan_v1', '--no-include-sample-id',
        '--prompt-text-file', assets / 'P0_NATIVE_PROMPT.txt', '--batch-size', 16, '--device', 'cuda'])
    return destination


def validate_label_accounting(output, endpoint, expected):
    directory = output / (endpoint + '_labels')
    report = json.loads((directory / 'LABEL_FINAL.json').read_text())
    if report['completed'] != expected or report['requested'] != expected or not (directory / '_SUCCESS').is_file():
        raise RuntimeError('evaluation label request accounting is incomplete')
    if report.get('statuses', {}).get('worker_error', 0):
        raise RuntimeError('evaluation worker errors need engineering recovery; cannot score as model failures')
    # OOM or an environment/model failure is not evidence of a bad structure.
    for line in (directory / 'labels.jsonl').read_text().splitlines():
        row = json.loads(line)
        error = str(row.get('error') or '').lower()
        if any(token in error for token in ('out of memory', 'cuda error', 'brokenprocesspool', 'modulenotfounderror')):
            raise RuntimeError('evaluation environment failure needs recovery before scientific scoring')


def export_and_label(output, *, count, method_id, endpoint, refined=None):
    arguments = ['--body-dir', output / 'body', '--output-dir', output / endpoint,
                 '--endpoint', endpoint, '--expected-requests', count, '--method-id', method_id]
    if refined is not None:
        arguments.extend(['--refined-pt', refined])
    execute(output, endpoint + '_inputs', 'src/scripts/export_r03_evaluation_inputs.py', arguments)
    execute(output, endpoint + '_direct_view', 'operations/r03_c3fd_main_20260907/export_direct_view.py', [
        '--paths-jsonl', output / endpoint / 'paths.jsonl', '--expected-denominator', count,
        '--snapshot-root', DIRECT_SNAPSHOT, '--output-dir', output / (endpoint + '_direct_view')])
    execute(output, endpoint + '_direct', 'scripts/run_direct_validity_fast.py', [
        '--generation-jsonl', output / (endpoint + '_direct_view') / 'generation.jsonl',
        '--snapshot-root', DIRECT_SNAPSHOT, '--expected-denominator', count,
        '--output-dir', output / (endpoint + '_direct')])
    execute(output, endpoint + '_labels', 'scripts/label_programmed_paths.py', [
        '--input-jsonl', output / endpoint / 'paths.jsonl', '--output-dir', output / (endpoint + '_labels'),
        '--purpose', 'evaluation', '--gpu-count', 1, '--workers-per-gpu', 2])
    validate_label_accounting(output, endpoint, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--role', choices=['R', 'I', 'G', 'P'], required=True)
    parser.add_argument('--method-id', required=True)
    parser.add_argument('--requests', type=int, required=True)
    parser.add_argument('--planner-seed', type=int, required=True)
    parser.add_argument('--body-seed', type=int, required=True)
    parser.add_argument('--refiner-seed', type=int, required=True)
    parser.add_argument('--sample-index-offset', type=int, default=0)
    parser.add_argument('--plans-jsonl', type=Path)
    parser.add_argument('--plans-include-programs', action='store_true')
    parser.add_argument('--repair-checkpoint', type=Path)
    parser.add_argument('--body-batch-size', type=int, default=8)
    parser.add_argument('--construction-geometry', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 1:
        raise ValueError('one assigned Slurm GPU is required for each component')
    if args.requests < 1 or args.sample_index_offset < 0 or (args.role == 'P') != (args.repair_checkpoint is not None):
        raise ValueError('invalid registered component or repair checkpoint role')
    if not 1 <= args.body_batch_size <= 8 or (args.construction_geometry and args.body_batch_size != 1):
        raise ValueError('invalid body batch size for construction constraint accounting')
    if args.construction_geometry and args.role not in ('G', 'P'):
        raise ValueError('registered R/I controls do not enable the new construction geometry')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'COMPONENT_CONFIG.json', {key: str(value) if isinstance(value, Path) else value
                                               for key, value in vars(args).items()})
    assets = args.run_root / 'pointer_40395/assets'
    started = time.monotonic()
    try:
        plans = args.plans_jsonl or sample_plans(output, assets, count=args.requests, seed=args.planner_seed,
                                                offset=args.sample_index_offset, constrained=args.role != 'R')
        rows = [json.loads(line) for line in plans.read_text().splitlines() if line.strip()]
        if (len(rows) != args.requests or [row['sample_idx'] for row in rows] !=
                list(range(args.sample_index_offset, args.sample_index_offset + args.requests))):
            raise ValueError('registered input request IDs changed')
        if any(row['seed'] != args.planner_seed or row['c3fd_enabled'] != (args.role != 'R') for row in rows):
            raise ValueError('Planner seed or original formula constraint mode changed')
        if args.role in ('G', 'P') and not args.plans_include_programs:
            plans = export_programs(output, assets, args.run_root / 'pointer_40395/train/r03_control_pointer.pt', plans)
        arguments = ['--frozen-runtime-root', FROZEN, '--base-model', LLADA, '--b0-checkpoint', B0,
                     '--plans-jsonl', plans, '--output-dir', output / 'body', '--expected-requests', args.requests,
                     '--seed', args.body_seed, '--batch-size', args.body_batch_size, '--crysllmgen-dir', PROJECT / 'reference/crysllmgen']
        if args.construction_geometry:
            arguments.append('--construction-geometry')
        if args.role in ('G', 'P'):
            arguments.extend(['--repair', '--geometry-support'])
        if args.repair_checkpoint:
            arguments.extend(['--repair-checkpoint', args.repair_checkpoint])
        execute(output, 'body', 'src/scripts/run_r03_integrated_body.py', arguments)
        export_and_label(output, count=args.requests, method_id=args.method_id, endpoint='native')
        verify_refiner(output)
        execute(output, 'refine', 'src/scripts/refine_dlm_with_crysllmgen.py', [
            '--proposal-graphs', output / 'body/proposal_graphs.pt', '--checkpoint', REFINER,
            '--crysllmgen-dir', CRYS, '--output-dir', output / 'refine', '--batch-size', 1,
            '--diff-steps', 800, '--num-evals', 1, '--max-proposals', args.requests,
            '--seed-by-sample-index', '--seed', args.refiner_seed])
        refinement = json.loads((output / 'refine/refinement_metrics.json').read_text())
        export_and_label(output, count=args.requests, method_id=args.method_id, endpoint='tau800',
                         refined=Path(refinement['output_file']))
        write_json(output / 'COMPONENT_FINAL.json', {'role': args.role, 'method_id': args.method_id,
                  'requests': args.requests, 'planner_seed': args.planner_seed, 'body_seed': args.body_seed,
                  'refiner_seed': args.refiner_seed, 'seconds': time.monotonic() - started,
                  'sample_index_offset': args.sample_index_offset, 'body_dir': str(output / 'body'),
                  'repair_checkpoint': str(args.repair_checkpoint) if args.repair_checkpoint else None,
                  'construction_geometry': args.construction_geometry, 'body_batch_size': args.body_batch_size,
                  'refined_pt': refinement['output_file'], 'plans_jsonl': str(plans),
                  'labels_purpose': 'evaluation', 'pooled_NU_scored_here': False})
        (output / '_SUCCESS').touch()
    except BaseException as error:
        write_json(output / 'FAILURE.json', {'error_type': type(error).__name__, 'message': str(error),
                  'seconds': time.monotonic() - started, 'unstarted_requests_are_model_failures': False})
        (output / '_FAILED').touch()
        raise


if __name__ == '__main__':
    main()
