#!/usr/bin/env python3
"""Fail-closed prerequisite barrier and one-time submission of the queued V2 run.

Empty Slurm output alone is insufficient: every registered predecessor phase
must have complete outcome accounting, including phases not submitted yet.
This tool does not train, inspect model losses, or manage unrelated jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess


def beneath(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("queue path escaped the declared experiment root")
    return path


def value_at(value, dotted):
    for key in dotted.split("."):
        value = value[key]
    return value


def project_jobs(text, *, allowed_job_id=None):
    jobs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        job_id, name, state = line.split("|", 2)
        if (name.startswith(("spad-", "periodic-"))
                and job_id != str(allowed_job_id)):
            jobs.append({"job_id": job_id, "name": name, "state": state})
    return jobs


def assess_queue(root, manifest, *, active_jobs=()):
    root = Path(root)
    reasons, phases = [], {}
    if manifest.get("schema") != "periodic_dlm_v2_launch_manifest_v1" or manifest.get("ready") is not True:
        reasons.append("implementation_manifest_not_ready")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("implementation_commit", ""))):
        reasons.append("implementation_commit_not_frozen")
    if manifest.get("resources") != {"gpus": 6, "cpus": 24, "simultaneous_project_jobs_max": 2}:
        reasons.append("resource_contract_changed")
    required = {"raw_v1_train", "raw_v1_native", "raw_v1_tau800", "old_k8_train",
                "old_k8_native", "old_k8_tau800", "old_k8_main_plans", "old_k8_main_evaluation"}
    prerequisites = manifest.get("prerequisites", [])
    if {entry.get("id") for entry in prerequisites} != required or len(prerequisites) != len(required):
        reasons.append("predecessor_phase_set_changed")
    for entry in prerequisites:
        identity, problems = entry["id"], []
        try:
            if "job_pointer" in entry:
                pointer = beneath(root, entry["job_pointer"])
                job_id = pointer.read_text(encoding="utf-8").strip()
                if not re.fullmatch(r"[0-9]+", job_id):
                    raise ValueError("invalid predecessor job id")
                run = beneath(root, entry["run_pattern"].format(job_id=job_id))
            else:
                run = beneath(root, entry["run_dir"])
            if (run / "_FAILED").exists():
                problems.append("failed_run")
            for marker in entry["success_markers"]:
                if not beneath(run, marker).is_file():
                    problems.append("missing:" + marker)
            for check in entry.get("reports", []):
                report = json.loads(beneath(run, check["path"]).read_text(encoding="utf-8"))
                for field, expected in check["equals"].items():
                    if value_at(report, field) != expected:
                        problems.append("report_mismatch:" + check["path"] + ":" + field)
            phases[identity] = {"run_dir": str(run), "ready": not problems, "problems": problems}
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(str(error))
            phases[identity] = {"ready": False, "problems": problems}
        if problems:
            reasons.append("predecessor_incomplete:" + identity)
    if active_jobs:
        reasons.append("project_quota_not_fully_released")
    return {"schema": "periodic_v2_queue_readiness_v1", "ready": not reasons,
            "reasons": reasons, "phases": phases, "active_project_jobs": list(active_jobs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--allow-job-id", help="Only the already allocated V2 job may pass its own start check")
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    if args.submit and args.allow_job_id:
        raise ValueError("submission cannot ignore a live project job")
    if args.allow_job_id and args.allow_job_id != os.environ.get("SLURM_JOB_ID"):
        raise ValueError("only this allocation may be excluded from its start barrier")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip()
    queue = subprocess.check_output(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i|%j|%T"], text=True)
    report = assess_queue(args.root, manifest, active_jobs=project_jobs(queue, allowed_job_id=args.allow_job_id))
    if head != manifest.get("implementation_commit"):
        report["ready"] = False
        report["reasons"].append("source_checkout_differs_from_frozen_implementation")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["ready"]:
        raise SystemExit(2)
    if not args.submit:
        return
    args.experiment.mkdir(parents=True, exist_ok=True)
    claim = args.experiment / "V2_TRAIN_SUBMISSION.json"
    state = {"schema": "periodic_v2_once_only_submission_v1", "status": "submitting",
             "created_at": datetime.now(timezone.utc).isoformat(), "manifest": str(args.manifest),
             "implementation_commit": head, "prerequisites": report["phases"]}
    with claim.open("x", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    env = os.environ.copy()
    env.update(H1A2_CANDIDATE_ROOT=str(args.root), PERIODIC_REPAIR_SOURCE=str(args.source),
               PERIODIC_V2_ROOT=str(args.experiment), PERIODIC_V2_LAUNCH_MANIFEST=str(args.manifest))
    try:
        completed = subprocess.run(["sbatch", "--parsable", str(args.source / "slurm/249_train_periodic_dlm_v2.sbatch")],
                                   env=env, text=True, capture_output=True, check=True)
        job = completed.stdout.strip().split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+", job):
            raise RuntimeError("submission response is uncertain; inspect Slurm before any retry")
        state.update(status="submitted", job_id=job)
        (args.experiment / "V2_TRAIN_JOB").write_text(job + "\n", encoding="utf-8")
    except Exception as error:
        state.update(status="submission_needs_inspection", error=str(error))
        raise
    finally:
        claim.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
