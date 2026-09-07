"""Idempotent stage dispatch inside the existing A800 pane, after archive deployment."""
import argparse
import datetime as dt
import json
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess as sp
from trial_preflight import require_geometry_canary_acceptance


def job_belongs_to_run(job_fields, run_root):
    run_root = Path(run_root).resolve()
    paths = [Path(job_fields[key]) for key in ('Command', 'WorkDir', 'StdOut') if job_fields.get(key)]
    return any(path == run_root or run_root in path.parents for path in paths)


def configured_dispatch(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.config.read_text())
    root = Path(spec['run_root']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    guard = root / '.dispatch_guard.lock'
    with guard.open('x') as handle:
        json.dump({'pid': os.getpid(), 'job': args.job,
                   'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()}, handle)
    try:
        return _configured_dispatch_locked(argv)
    finally:
        guard.unlink()


def _configured_dispatch_locked(argv=None):
    """Submit one manifest job with an explicit dated resource contract.

    Generated sbatch files are run artifacts; the reusable execution logic
    stays in run_component.py. A lock records uncertain dispatches instead of
    allowing a retry to create a duplicate job.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.config.read_text(encoding='utf-8'))
    if spec.get('schema') != 'crystal_pipeline_run_v1':
        raise ValueError('unknown pipeline manifest')
    root, source = Path(spec['run_root']).resolve(), Path(spec['source_root']).resolve()
    if not (source / '_CODE_READY').is_file():
        raise ValueError('source has not completed immutable deployment')
    from run_component import verify_deployed_source
    if verify_deployed_source(source) != spec.get('source_identity'):
        raise ValueError('source does not match the configured immutable deployment')
    job = spec['jobs'][args.job]
    indices = list(job['component_indices'])
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(spec['components']) for i in indices)):
        raise ValueError('job must contain distinct component indices')
    components = [spec['components'][i] for i in indices]
    if (len({row['id'] for row in components}) != len(components)
            or len({row['output_dir'] for row in components}) != len(components)):
        raise ValueError('parallel component IDs and output directories must be distinct')
    for component in components:
        directory = (root / component['output_dir']).resolve()
        names = [stage['name'] for stage in component['stages']]
        if root not in directory.parents or len(names) != len(set(names)):
            raise ValueError('invalid component directory or duplicate stage names')
    directories = [(root / row['output_dir']).resolve() for row in components]
    if any(a in b.parents or b in a.parents for i, a in enumerate(directories) for b in directories[i + 1:]):
        raise ValueError('parallel component directories must not overlap')
    records = root / 'submissions'
    records.mkdir(parents=True, exist_ok=True)
    receipt, lock = records / (args.job + '.json'), records / (args.job + '.lock')
    fingerprint = hashlib.sha256(json.dumps(
        {'job': job, 'components': components, 'purpose': spec['purpose'],
         'source_root': str(source), 'source_identity': spec['source_identity'],
         'environment': spec.get('environment', {}),
         'resources': spec['resources'], 'python': spec['python']}, sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        existing = json.loads(receipt.read_text())
        if existing.get('fingerprint') != fingerprint:
            raise ValueError('this job name already refers to a different frozen configuration')
        print(json.dumps(existing))
        return
    unresolved = [path for path in records.glob('*.lock') if not path.with_suffix('.json').exists()]
    if unresolved:
        raise ValueError('uncertain dispatch reservations require reconciliation before further submissions')
    gpus, cpus = int(job['gpus_per_task']), int(job['cpus_per_task'])
    parallel = min(len(indices), int(job.get('parallel_tasks', 1)))
    single_allocation = job.get('allocation_mode') == 'single'
    minutes = int(job['wall_minutes'])
    if (gpus < 0 or min(cpus, parallel, minutes) < 1 or (gpus and cpus > 6 * gpus)
            or any(int(row['gpus']) != gpus for row in components)):
        raise ValueError('job resources differ from its components or CPU-per-GPU budget')
    budget = spec['resources']
    now = dt.datetime.now(dt.timezone.utc)
    extra_until = dt.datetime.fromisoformat(budget['extra_gpu_until_utc'])
    deadline = dt.datetime.fromisoformat(budget['deadline_utc'])
    worst_end = now + dt.timedelta(minutes=minutes * ((len(indices) + parallel - 1) // parallel))
    if worst_end >= deadline:
        raise ValueError('job wall limit can run beyond the declared deadline')
    gpu_limit = int(budget['gpus_after_extra_window'])
    if worst_end <= extra_until:
        gpu_limit = int(budget['gpus_before_extra_window'])
    hard_stop = extra_until if gpus * parallel > int(budget['gpus_after_extra_window']) else deadline
    queued = sp.check_output(['squeue', '-h', '-r', '-u', os.environ['USER'], '-o', '%i'], text=True).split()
    owned = []
    for job_id in queued:
        line = sp.check_output(['scontrol', 'show', 'job', '-o', job_id], text=True)
        fields = dict(piece.split('=', 1) for piece in line.split() if '=' in piece)
        if not job_belongs_to_run(fields, root):
            continue
        match = re.search(r'(?:^|\s)TRES=([^\s]+)', line)
        if not match:
            raise ValueError('cannot establish resources of an existing task job')
        tres = dict(piece.split('=', 1) for piece in match.group(1).split(',') if '=' in piece)
        owned.append({'job_id': job_id, 'gpus': int(tres.get('gres/gpu', 0)),
                      'cpus': int(tres.get('cpu', 0)), 'state': fields.get('JobState')})
    if sum(row['gpus'] for row in owned) + gpus * parallel > gpu_limit:
        raise ValueError(f'GPU resource budget is occupied: {owned}')
    if sum(row['cpus'] for row in owned) + cpus * parallel > 6 * gpu_limit:
        raise ValueError('CPU resource budget is occupied')
    (root / 'logs').mkdir(exist_ok=True)
    allocated_gpus = gpus * parallel if single_allocation else gpus
    allocated_cpus = cpus * parallel if single_allocation else cpus
    allocation_minutes = minutes * ((len(indices) + parallel - 1) // parallel) if single_allocation else minutes
    frozen = dict(spec, job_runtime={'hard_stop_utc': hard_stop.isoformat(), 'cpus_per_task': allocated_cpus,
                                    'worker_cpus': cpus, 'gpus_per_component': gpus,
                                    'job': args.job, 'fingerprint': fingerprint})
    encoded = (json.dumps(frozen, sort_keys=True, indent=2) + '\n').encode()
    manifest_digest = hashlib.sha256(encoded).hexdigest()
    snapshot = records / (args.job + '.' + manifest_digest[:16] + '.manifest.json')
    if snapshot.exists():
        if snapshot.read_bytes() != encoded:
            raise ValueError('immutable manifest snapshot differs')
    else:
        with snapshot.open('xb') as handle:
            handle.write(encoded)
    script = records / (args.job + '.sbatch')
    remaining_code = ('import datetime as d,sys; end=d.datetime.fromisoformat(sys.argv[1]);'
                      'print(int((end-d.datetime.now(d.timezone.utc)).total_seconds())-40)')
    content = ('#!/usr/bin/env bash\nset -Eeuo pipefail\nseconds="$(' + shlex.quote(spec['python'])
               + ' -c ' + shlex.quote(remaining_code) + ' ' + shlex.quote(hard_stop.isoformat()) + ')"\n'
               + 'if [ "$seconds" -le 0 ]; then exit 75; fi\n'
               + 'exec timeout --signal=TERM --kill-after=30s "${seconds}s" ' + shlex.quote(spec['python'])
               + ' ' + shlex.quote(str(source / 'operations/r03_c3fd_main_20260907/run_component.py'))
               + ' --config ' + shlex.quote(str(snapshot)) + ' --config-sha256 ' + manifest_digest
               + (' --component-indices ' + ' '.join(map(str, indices))
                  + ' --parallel-components ' + str(parallel) + '\n' if single_allocation else
                  ' --component-index "${SLURM_ARRAY_TASK_ID:-' + str(indices[0]) + '}"\n'))
    script.write_text(content, encoding='utf-8')
    command = ['sbatch', '--parsable', '--no-requeue', '--partition=' + job.get('partition', 'gpu'),
               '--job-name=' + args.job, '--nodes=1', '--ntasks=1', f'--cpus-per-task={allocated_cpus}',
               *([f'--gres=gpu:NVIDIAA800-SXM4-80GB:{allocated_gpus}'] if allocated_gpus else []),
               '--mem=' + str(job.get('memory', '96G')),
               f'--time={allocation_minutes}', '--chdir=' + str(source),
               '--output=' + str(root / 'logs' / (args.job + '_%A_%a.out')),
               '--error=' + str(root / 'logs' / (args.job + '_%A_%a.err'))]
    if len(indices) > 1 and not single_allocation:
        command.append('--array=' + ','.join(map(str, indices)) + '%' + str(parallel))
    command.append(str(script))
    with lock.open('x') as handle:
        json.dump({'command': command, 'fingerprint': fingerprint, 'manifest_sha256': manifest_digest,
                   'requested_utc': now.isoformat(), 'gpus_reserved': gpus * parallel}, handle)
    result = sp.run(command, capture_output=True, text=True)
    if result.returncode:
        # A nonzero sbatch result is retained for explicit review; never infer
        # that a dropped transport or wrapper error means no job was created.
        print(json.dumps({'submitted': False, 'returncode': result.returncode,
                          'stdout': result.stdout, 'stderr': result.stderr}))
        raise SystemExit(result.returncode)
    job_id = result.stdout.strip().split(';')[0]
    if not job_id.isdigit():
        raise ValueError('submission output does not establish a numeric job identity')
    record = {'job_id': job_id, 'job': args.job, 'command': command, 'manifest': str(snapshot),
              'manifest_sha256': manifest_digest, 'fingerprint': fingerprint,
              'submitted_utc': now.isoformat(), 'max_parallel_tasks': parallel,
              'allocation_mode': 'single' if single_allocation else 'array',
              'gpus_per_allocation': allocated_gpus, 'cpus_per_allocation': allocated_cpus,
              'allocation_wall_minutes': allocation_minutes,
              'gpus_per_task': gpus, 'cpus_per_task': cpus, 'queue_before': owned,
              'gpu_budget_applied': gpu_limit, 'worst_end_utc': worst_end.isoformat(),
              'absolute_stop_utc': hard_stop.isoformat(), 'runtime_cutoff_enforced': True}
    with receipt.open('x') as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--afterok", help="Queue pilot_GP after its recorded pilot_RI job succeeds")
    parser.add_argument("--stage", choices=["canary", "canary_v2", "physics", "canary_repair", "canary_geometry", "pilot_RI", "pilot_GP", "canary_hull", "pilot_hull", "canary_evaluate", "pilot_evaluate", "formal", "formal_evaluate"], required=True)
    parser.add_argument("--producer-source", type=Path)
    parser.add_argument("--wall-minutes", type=int)
    args = parser.parse_args()
    if args.afterok:
        parent = json.loads((args.run_root / 'PILOT_RI_SUBMISSION.json').read_text())
        if args.stage != 'pilot_GP' or args.afterok != parent['job_id'] or str(args.source) != parent['source']:
            raise ValueError('--afterok must name this source version\'s recorded pilot_RI job')
    receipt = args.run_root / (args.stage.upper() + "_SUBMISSION.json")
    if receipt.exists():
        print(receipt.read_text())
        return
    assert (args.source / "_CODE_READY").is_file()
    assert (args.run_root / "pointer_40395" / "_SUCCESS").is_file()
    stage_specs = {"canary": (2, 80, "200G", "canary.sbatch"),
                   "canary_v2": (2, 80, "200G", "canary.sbatch"),
                   "physics": (1, 150, "120G", "train_physics.sbatch"),
                   "canary_repair": (2, 80, "200G", "trial.sbatch"),
                   "canary_geometry": (2, 80, "200G", "trial.sbatch"),
                   "pilot_RI": (2, 270, "200G", "trial.sbatch"),
                   "pilot_GP": (2, 270, "200G", "trial.sbatch"),
                   "canary_hull": (0, 70, "24G", "hull.sbatch"),
                   "pilot_hull": (0, 70, "24G", "hull.sbatch"),
                   "canary_evaluate": (0, 45, "32G", "evaluate.sbatch"),
                   "pilot_evaluate": (0, 80, "32G", "evaluate.sbatch")}
    stage_specs['formal'] = (2, 270, '200G', 'trial.sbatch')
    stage_specs['formal_evaluate'] = (0, 20, '32G', 'evaluate.sbatch')
    gpus, minutes, memory, script = stage_specs[args.stage]
    if args.stage == 'formal':
        if not args.producer_source or not args.wall_minutes or not 1 <= args.wall_minutes <= 270:
            parser.error('formal requires its immutable producer source and a wall limit within the remaining budget')
        minutes = args.wall_minutes
        registration = json.loads((args.run_root / 'FORMAL_REGISTRATION.json').read_text())
        frozen = json.loads((args.run_root / 'METHOD_FREEZE.json').read_text())
        assert registration['producer_source'] == str(args.producer_source)
        assert registration['wall_minutes'] == minutes
        assert registration['selected_role'] == frozen['selected_role']
        assert args.producer_source.name == frozen['producer_commit']
        assert registration['requests_per_seed'] in (256, 500)
        assert len(set(registration['planner_seeds'])) == 2
        assert (args.run_root / 'evaluation_pilot256/_SUCCESS').is_file()
        assert (args.producer_source / '_CODE_READY').is_file()
        require_geometry_canary_acceptance(args.run_root, args.producer_source)
    elif args.stage == 'formal_evaluate':
        if args.producer_source or (args.wall_minutes is not None and not 1 <= args.wall_minutes <= 80):
            parser.error('formal evaluation accepts only a CPU wall limit from 1 to 80 minutes')
        if args.wall_minutes:
            minutes = args.wall_minutes
        registration = json.loads((args.run_root / 'FORMAL_REGISTRATION.json').read_text())
        assert (args.run_root / f"formal_{2 * registration['requests_per_seed']}" / '_SUCCESS').is_file()
        assert (args.run_root / 'hull_formal/official_mp_cache/completion_SUCCESS').is_file()
    elif args.producer_source or args.wall_minutes:
        parser.error('producer and wall overrides apply only to formal sampling')
    cpus = 6 if args.stage.endswith(('_hull', '_evaluate')) else gpus * 4
    if args.stage == "canary_repair":
        assert (args.run_root / "canary_40400/_SUCCESS").is_file(), 'original/interface canary incomplete'
    if args.stage == "canary_geometry":
        assert (args.run_root / "canary_repair_40403/_SUCCESS").is_file(), 'same-condition repair canary incomplete'
    if args.stage in ("pilot_RI", "pilot_GP"):
        require_geometry_canary_acceptance(args.run_root, args.source)
    if args.stage == "pilot_GP":
        assert (args.run_root / "pilot_256/shared_I/_SUCCESS").is_file(), 'shared trial Plans are not ready'
        assert (args.run_root / "physics_40399/train/_SUCCESS").is_file(), 'physical adapter is not complete'
    if args.stage == 'canary_hull':
        assert (args.run_root / 'canary_40400/_SUCCESS').is_file()
    if args.stage == 'pilot_hull':
        assert (args.run_root / 'pilot_256/R/planner/_SUCCESS').is_file()
        assert (args.run_root / 'pilot_256/shared_I/_SUCCESS').is_file()
    if args.stage == 'canary_evaluate':
        assert (args.run_root / 'GEOMETRY_CANARY_COMPLETE.json').is_file()
        assert (args.run_root / 'hull_canary/official_mp_cache/completion_SUCCESS').is_file()
    if args.stage == 'pilot_evaluate':
        assert all((args.run_root / 'pilot_256' / role / '_SUCCESS').is_file() for role in ('R', 'I', 'G', 'P'))
        assert (args.run_root / 'hull_pilot256/official_mp_cache/completion_SUCCESS').is_file()
    now = dt.datetime.now(dt.timezone.utc)
    assert now + dt.timedelta(minutes=minutes + 5) < dt.datetime(2026, 9, 7, 15, 35, 26, tzinfo=dt.timezone.utc)
    jobs = sp.check_output(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"], text=True).split()
    resources = []
    for job in jobs:
        line = sp.check_output(["scontrol", "show", "job", "-o", job], text=True)
        # Old Slurm duplicates TresPerNode (%b); the total TRES is authoritative.
        match = re.search(r"(?:^|\s)TRES=([^\s]+)", line)
        assert match, line
        fields = dict(piece.split("=", 1) for piece in match.group(1).split(",") if "=" in piece)
        job_fields = dict(piece.split('=', 1) for piece in line.split() if '=' in piece)
        owned_by_run = job_belongs_to_run(job_fields, args.run_root)
        resources.append({"job_id": job, "gpus": int(fields.get("gres/gpu", "0")), "cpus": int(fields['cpu']),
                          "tres": fields, "owned_by_this_run": owned_by_run,
                          "job_name": job_fields.get('JobName'), "work_dir": job_fields.get('WorkDir')})
    # Latest user instruction: this task uses two A800s. This shared account
    # also carries an unrelated experiment, identified by its
    # actual work/command/output paths and retained in the queue receipt.
    owned = [row for row in resources if row['owned_by_this_run']]
    assert len(owned) < 2, owned
    overlapping = [row for row in owned if row['job_id'] != args.afterok]
    assert sum(row["gpus"] for row in overlapping) + gpus <= 2, overlapping
    assert sum(row['cpus'] for row in overlapping) + cpus <= 8, overlapping
    env = dict(os.environ, R03_SOURCE_ROOT=str(args.source), R03_RUN_ROOT=str(args.run_root), R03_STAGE=args.stage)
    if args.stage == 'formal':
        env.update(R03_PRODUCER_SOURCE=str(args.producer_source),
                   R03_REQUESTS_PER_SEED=str(registration['requests_per_seed']),
                   R03_FORMAL_SEED_0=str(registration['planner_seeds'][0]),
                   R03_FORMAL_SEED_1=str(registration['planner_seeds'][1]))
    if not gpus:
        for name in tuple(env):
            if name.startswith(('SBATCH_GRES', 'SBATCH_GPUS', 'SBATCH_TRES', 'SBATCH_CPUS_PER_GPU')):
                env.pop(name)
    command = ["sbatch", "--parsable", "--job-name=r03" + args.stage + "033526", "--partition=" + ('gpu' if gpus else 'normal'),
               "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(cpus),
               *(["--gres=gpu:NVIDIAA800-SXM4-80GB:" + str(gpus)] if gpus else []), "--mem=" + memory,
               "--time=" + str(minutes), "--no-requeue", "--chdir=" + str(args.source),
               "--output=" + str(args.run_root / "logs" / (args.stage + "_%j.out")),
               "--error=" + str(args.run_root / "logs" / (args.stage + "_%j.err")),
               str(args.source / "operations/r03_c3fd_main_20260907" / script)]
    if args.afterok:
        command.insert(1, '--dependency=afterok:' + args.afterok)
    result = sp.run(command, env=env, check=False, capture_output=True, text=True)
    if result.returncode:
        print(json.dumps({'stage': args.stage, 'submitted': False, 'returncode': result.returncode,
                          'stdout': result.stdout, 'stderr': result.stderr, 'command': command}))
        raise SystemExit(result.returncode)
    job = result.stdout.strip().split(";")[0]
    assert job.isdigit(), result.stdout
    record = {"job_id": job, "source": str(args.source), "stage": args.stage, "gpus": gpus,
              "cpus": cpus, "submitted_utc": now.isoformat(), "queue_before": resources,
              "resource_scope": "this_registered_run", "task_resource_ceiling": {"A800": 2, "CPUs": 8, "Slurm_jobs": 2},
              "afterok": args.afterok,
              "formal_registration": registration if args.stage == 'formal' else None,
              "command": command, "physics_updates_fixed_before_evaluation": 128 if args.stage == "physics" else None}
    with receipt.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    import sys
    if '--config' in sys.argv:
        configured_dispatch()
    else:
        main()
