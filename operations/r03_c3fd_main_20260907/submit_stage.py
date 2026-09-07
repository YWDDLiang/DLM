"""Idempotent stage dispatch inside the existing A800 pane, after archive deployment."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage", choices=["canary", "physics"], required=True)
    args = parser.parse_args()
    receipt = args.run_root / (args.stage.upper() + "_SUBMISSION.json")
    if receipt.exists():
        print(receipt.read_text())
        return
    assert (args.source / "_CODE_READY").is_file()
    assert (args.run_root / "pointer_40395" / "_SUCCESS").is_file()
    gpus, minutes, memory, script = ((2, 80, "200G", "canary.sbatch") if args.stage == "canary"
                                   else (1, 150, "120G", "train_physics.sbatch"))
    now = dt.datetime.now(dt.timezone.utc)
    assert now + dt.timedelta(minutes=minutes + 5) < dt.datetime(2026, 9, 7, 15, 35, 26, tzinfo=dt.timezone.utc)
    jobs = sp.check_output(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"], text=True).split()
    assert len(jobs) < 2, jobs
    resources = []
    for job in jobs:
        line = sp.check_output(["scontrol", "show", "job", "-o", job], text=True)
        # Old Slurm duplicates TresPerNode (%b); the total TRES is authoritative.
        match = re.search(r"(?:^|\s)TRES=([^\s]+)", line)
        assert match, line
        fields = dict(piece.split("=", 1) for piece in match.group(1).split(",") if "=" in piece)
        resources.append({"job_id": job, "gpus": int(fields.get("gres/gpu", "0")), "tres": fields})
    assert sum(row["gpus"] for row in resources) + gpus <= 4, resources
    env = dict(os.environ, R03_SOURCE_ROOT=str(args.source), R03_RUN_ROOT=str(args.run_root))
    command = ["sbatch", "--parsable", "--job-name=r03" + args.stage + "033526", "--partition=gpu",
               "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(gpus * 4),
               "--gres=gpu:NVIDIAA800-SXM4-80GB:" + str(gpus), "--mem=" + memory,
               "--time=" + str(minutes), "--no-requeue", "--chdir=" + str(args.source),
               "--output=" + str(args.run_root / "logs" / (args.stage + "_%j.out")),
               "--error=" + str(args.run_root / "logs" / (args.stage + "_%j.err")),
               str(args.source / "operations/r03_c3fd_main_20260907" / script)]
    result = sp.run(command, env=env, check=True, capture_output=True, text=True)
    job = result.stdout.strip().split(";")[0]
    assert job.isdigit(), result.stdout
    record = {"job_id": job, "source": str(args.source), "stage": args.stage, "gpus": gpus,
              "cpus": gpus * 4, "submitted_utc": now.isoformat(), "queue_before": resources,
              "command": command, "physics_updates_fixed_before_evaluation": 128 if args.stage == "physics" else None}
    with receipt.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
