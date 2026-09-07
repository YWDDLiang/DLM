"""Two-GPU coordinator for the complete repair canary and user-requested 256 trial."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import subprocess as sp
import sys
import time

from run_component import SOURCE, execute, export_programs, sample_plans, write_json
from trial_preflight import require_geometry_canary_acceptance


def component(run_root, directory, role, gpu, *, count, seed, plans=None, programs=False, repair=None,
              construction_geometry=False, body_batch_size=1):
    command = [sys.executable, str(SOURCE / 'operations/r03_c3fd_main_20260907/run_component.py'),
               '--run-root', str(run_root), '--output-dir', str(directory), '--role', role,
               '--method-id', 'R03_' + role, '--requests', str(count), '--planner-seed', str(seed),
               '--body-seed', str(seed + 100), '--refiner-seed', str(seed + 200),
               '--body-batch-size', str(body_batch_size)]
    if construction_geometry:
        command.append('--construction-geometry')
    if plans:
        command.extend(['--plans-jsonl', str(plans)])
    if programs:
        command.append('--plans-include-programs')
    if repair:
        command.extend(['--repair-checkpoint', str(repair)])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    with directory.with_suffix('.out').open('x') as stdout, directory.with_suffix('.err').open('x') as stderr:
        result = sp.run(command, env=env, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise RuntimeError(f'{role} component failed with {result.returncode}; source and partial outputs retained')
    return json.loads((directory / 'COMPONENT_FINAL.json').read_text())


def physical_checkpoint(run_root, deadline):
    training = run_root / 'physics_40399' / 'train'
    while not (training / '_SUCCESS').is_file():
        if (training / '_FAILED').exists() or (training.parent / 'FAILURE.txt').exists():
            raise RuntimeError('registered physical-transfer training failed')
        if dt.datetime.now(dt.timezone.utc) >= deadline:
            raise TimeoutError('physical-transfer availability deadline reached')
        time.sleep(10)
    checkpoint = Path((training / 'POLICY_PATH').read_text().strip())
    if not (checkpoint / 'R03_REPAIR_TRANSFER.json').is_file():
        raise RuntimeError('completed physical training is missing its repair-only receipt')
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--stage', choices=['canary_repair', 'canary_geometry', 'pilot_RI', 'pilot_GP'], required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('an existing Slurm allocation is required')
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(gpus) != 2 or any(not value for value in gpus):
        raise ValueError('exactly two allocated GPUs are required')
    run_root = args.run_root
    assets = run_root / 'pointer_40395/assets'
    pointer = run_root / 'pointer_40395/train/r03_control_pointer.pt'
    is_canary = args.stage.startswith('canary_')
    seed, count = (202609070, 16) if is_canary else (202609071, 256)
    if not is_canary:
        require_geometry_canary_acceptance(run_root, SOURCE)
    root = (run_root / (args.stage + '_' + os.environ['SLURM_JOB_ID']) if is_canary
            else run_root / 'pilot_256')
    root.mkdir(parents=True, exist_ok=not is_canary)
    lane = root / ('LANE_' + args.stage)
    lane.mkdir(exist_ok=False)
    write_json(lane / 'REGISTERED.json', {'stage': args.stage, 'requests_per_arm': count, 'planner_seed': seed,
               'body_seed': seed + 100, 'refiner_seed': seed + 200, 'source': str(SOURCE),
               'job_id': os.environ['SLURM_JOB_ID'], 'role': 'engineering_canary' if count == 16 else 'user_requested_fixed_development',
               'formal_1000_started': False, 'same_generated_plans_for_I_G_P': True,
               'construction_geometry_roles': ['G', 'P'] if args.stage != 'canary_repair' else [],
               'body_batch_size': 1, 'failed_or_empty_support_requests_retained': True})
    results, failures = {}, {}
    started = time.monotonic()
    try:
        if args.stage == 'canary_geometry':
            # Replay the same actual 16 Plans, pointer outputs and paired seeds.
            plans = run_root / 'canary_repair_40403/shared_I/plans_with_programs.jsonl'
            if not plans.is_file():
                raise RuntimeError('previous 16-condition pointer ledger is missing')
            repair = physical_checkpoint(run_root, dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=1))
            def geometric_with_matched_reference():
                result = component(run_root, root / 'G', 'G', gpus[0], count=count, seed=seed,
                                   plans=plans, programs=True, construction_geometry=True)
                # Replay the original interface at the same batch size as G/P.
                # This is a complete, separate 16-request canary control, never
                # a replacement of unsuccessful requests in another method.
                reference = component(run_root, root / 'I_batch1_reference', 'I', gpus[0], count=count, seed=seed,
                                      plans=plans, programs=True)
                write_json(root / 'MATCHED_INTERFACE_REFERENCE.json', reference)
                return result
            functions = {'G': geometric_with_matched_reference,
                         'P': lambda: component(run_root, root / 'P', 'P', gpus[1], count=count, seed=seed,
                                                plans=plans, programs=True, repair=repair, construction_geometry=True)}
        elif args.stage == 'canary_repair':
            source_plans = run_root / 'canary_40400/I/planner/plans_for_dlm.jsonl'
            shared = root / 'shared_I'
            shared.mkdir()
            saved_visibility = os.environ['CUDA_VISIBLE_DEVICES']
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus[0]
            try:
                plans = export_programs(shared, assets, pointer, source_plans)
            finally:
                os.environ['CUDA_VISIBLE_DEVICES'] = saved_visibility
            def run_physical():
                repair = physical_checkpoint(run_root, dt.datetime(2026, 9, 7, 6, 15, 26, tzinfo=dt.timezone.utc))
                return component(run_root, root / 'P', 'P', gpus[1], count=count, seed=seed,
                                 plans=plans, programs=True, repair=repair)
            functions = {'G': lambda: component(run_root, root / 'G', 'G', gpus[0], count=count, seed=seed,
                                                plans=plans, programs=True), 'P': run_physical}
        elif args.stage == 'pilot_RI':
            # Each function changes only its child environment; never mutate the
            # coordinator's visible GPUs while both workers are running.
            def candidate():
                shared = root / 'shared_I'
                shared.mkdir()
                command = [sys.executable, str(SOURCE / 'operations/r03_c3fd_main_20260907/prepare_trial_plans.py'),
                           '--run-root', str(run_root), '--output-dir', str(shared), '--requests', str(count), '--seed', str(seed)]
                with (lane / 'prepare_I.out').open('x') as stdout, (lane / 'prepare_I.err').open('x') as stderr:
                    result = sp.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[1]), stdout=stdout, stderr=stderr)
                if result.returncode:
                    raise RuntimeError('shared candidate Planner/control export failed')
                return component(run_root, root / 'I', 'I', gpus[1], count=count, seed=seed,
                                 plans=shared / 'planner/plans_for_dlm.jsonl')
            functions = {'R': lambda: component(run_root, root / 'R', 'R', gpus[0], count=count, seed=seed), 'I': candidate}
        else:
            shared = root / 'shared_I'
            if not (shared / '_SUCCESS').is_file():
                raise RuntimeError('shared original candidate Plans/programs are not complete')
            repair = physical_checkpoint(run_root, dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=1))
            plans = shared / 'plans_with_programs.jsonl'
            functions = {'G': lambda: component(run_root, root / 'G', 'G', gpus[0], count=count, seed=seed,
                                                plans=plans, programs=True, construction_geometry=True),
                         'P': lambda: component(run_root, root / 'P', 'P', gpus[1], count=count, seed=seed,
                                                plans=plans, programs=True, repair=repair, construction_geometry=True)}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(function) for name, function in functions.items()}
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except BaseException as error:
                    failures[name] = {'type': type(error).__name__, 'message': str(error)}
        write_json(lane / 'STAGE_FINAL.json', {'stage': args.stage, 'requests_per_arm': count, 'results': results,
                   'failures': failures, 'seconds': time.monotonic() - started, 'pooled_NU_scored_here': False})
        if failures:
            raise RuntimeError('one or more registered components need engineering attention')
        (lane / '_SUCCESS').touch()
        if is_canary:
            (root / '_SUCCESS').touch()
            marker = 'GEOMETRY_CANARY_COMPLETE.json' if args.stage == 'canary_geometry' else 'REPAIR_CANARY_COMPLETE.json'
            write_json(run_root / marker, {'directory': str(root), 'source': str(SOURCE),
                       'job_id': os.environ['SLURM_JOB_ID'], 'requests_per_arm': count,
                       'physics_checkpoint': results['P'].get('repair_checkpoint'), 'roles': sorted(results),
                       'matched_interface_reference': str(root / 'I_batch1_reference') if args.stage == 'canary_geometry' else None,
                       'scientific_acceptance_is_separate': True})
    except BaseException as error:
        write_json(lane / 'FAILURE.json', {'type': type(error).__name__, 'message': str(error)})
        (lane / '_FAILED').touch()
        raise


if __name__ == '__main__':
    main()
