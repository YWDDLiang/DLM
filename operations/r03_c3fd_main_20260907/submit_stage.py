"""Idempotent stage dispatch inside the existing A800 pane, after archive deployment."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess as sp
from trial_preflight import require_geometry_canary_acceptance


def job_belongs_to_run(job_fields, run_root):
    run_root = Path(run_root).resolve()
    paths = [Path(job_fields[key]) for key in ('Command', 'WorkDir', 'StdOut') if job_fields.get(key)]
    return any(path == run_root or run_root in path.parents for path in paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage", choices=["canary", "canary_v2", "physics", "canary_repair", "canary_geometry", "pilot_RI", "pilot_GP", "canary_hull", "pilot_hull"], required=True)
    args = parser.parse_args()
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
                   "pilot_hull": (0, 70, "24G", "hull.sbatch")}
    gpus, minutes, memory, script = stage_specs[args.stage]
    cpus = 6 if args.stage.endswith('_hull') else gpus * 4
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
    assert sum(row["gpus"] for row in owned) + gpus <= 2, owned
    assert sum(row['cpus'] for row in owned) + cpus <= 8, owned
    env = dict(os.environ, R03_SOURCE_ROOT=str(args.source), R03_RUN_ROOT=str(args.run_root), R03_STAGE=args.stage)
    command = ["sbatch", "--parsable", "--job-name=r03" + args.stage + "033526", "--partition=gpu",
               "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(cpus),
               ("--gres=gpu:NVIDIAA800-SXM4-80GB:" + str(gpus)) if gpus else '--gres=none', "--mem=" + memory,
               "--time=" + str(minutes), "--no-requeue", "--chdir=" + str(args.source),
               "--output=" + str(args.run_root / "logs" / (args.stage + "_%j.out")),
               "--error=" + str(args.run_root / "logs" / (args.stage + "_%j.err")),
               str(args.source / "operations/r03_c3fd_main_20260907" / script)]
    result = sp.run(command, env=env, check=True, capture_output=True, text=True)
    job = result.stdout.strip().split(";")[0]
    assert job.isdigit(), result.stdout
    record = {"job_id": job, "source": str(args.source), "stage": args.stage, "gpus": gpus,
              "cpus": cpus, "submitted_utc": now.isoformat(), "queue_before": resources,
              "resource_scope": "this_registered_run", "task_resource_ceiling": {"A800": 2, "CPUs": 8, "Slurm_jobs": 2},
              "command": command, "physics_updates_fixed_before_evaluation": 128 if args.stage == "physics" else None}
    with receipt.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
