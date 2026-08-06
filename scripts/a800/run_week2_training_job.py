#!/usr/bin/env python3
"""Execute one immutable training job from a frozen Week-2 plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    location = path.resolve()
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _training_evidence_path(resolved: Mapping[str, Any], output_dir: Path) -> Path:
    """Return the terminal evidence emitted by the frozen training phase."""

    phase = str(resolved["phase"])
    if phase == "shared-60000":
        return output_dir / "shared_training_complete.json"
    if phase == "screen-60000-to-85000":
        return output_dir / "partial_training_complete.json"
    raise ValueError(f"unsupported Week-2 training phase: {phase}")


def resolve_job(
    *,
    plan_path: str | Path,
    job_id: str,
    project_root: str | Path,
) -> dict[str, Any]:
    plan_location = Path(plan_path).resolve()
    plan = json.loads(plan_location.read_text(encoding="utf-8"))
    if plan.get("schema") != "wqcodiff_week2_training_plan_v1":
        raise ValueError("unsupported Week-2 training-plan schema")
    jobs = {str(job["job_id"]): job for job in plan["jobs"]}
    if len(jobs) != len(plan["jobs"]):
        raise ValueError("Week-2 training plan has duplicate job IDs")
    if job_id not in jobs:
        raise ValueError(f"unknown Week-2 job ID: {job_id}")
    job = dict(jobs[job_id])
    argv = [str(value) for value in job["argv"]]
    if argv[:4] != ["python", "-m", "crystal_dlm.wqcodiff", "--protocol"]:
        raise ValueError("Week-2 job is not the registered Python module command")
    if "train" not in argv or argv.index("train") <= 4:
        raise ValueError("Week-2 job is not a training command")
    argv[0] = sys.executable

    root = Path(project_root).resolve()
    output_dir = (root / str(job["output_dir"])).resolve()
    try:
        output_dir.relative_to(root / "runs")
    except ValueError as exc:
        raise ValueError("Week-2 output escapes the project runs directory") from exc
    if output_dir.exists():
        raise FileExistsError(f"immutable Week-2 output already exists: {output_dir}")

    dependency_artifacts: dict[str, dict[str, Any]] = {}
    for dependency in job.get("depends_on", ()):
        if dependency not in jobs:
            raise ValueError(f"Week-2 job has unknown dependency: {dependency}")
        checkpoint = (root / str(jobs[dependency]["continuation_checkpoint"])).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Week-2 dependency checkpoint is incomplete: {checkpoint}"
            )
        dependency_artifacts[str(dependency)] = file_identity(checkpoint)

    return {
        "schema": "wqcodiff_week2_training_job_v1",
        "materialized_plan": file_identity(plan_location),
        "registry_sha256": plan["registry_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "source_bundle_sha256": plan["source_bundle_sha256"],
        "job_id": job_id,
        "phase": job["phase"],
        "variant": job["variant"],
        "target_update": int(job["target_update"]),
        "output_dir": str(output_dir),
        "continuation_checkpoint": str(
            (root / str(job["continuation_checkpoint"])).resolve()
        ),
        "validation_ema": (
            None
            if job.get("validation_ema") is None
            else str((root / str(job["validation_ema"])).resolve())
        ),
        "dependencies": dependency_artifacts,
        "argv": argv,
    }


def execute_job(
    resolved: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    events_path: str | Path,
    complete_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    events = Path(events_path).resolve()
    complete = Path(complete_path).resolve()
    if manifest.exists() or events.exists() or complete.exists():
        raise FileExistsError("immutable Week-2 job evidence already exists")
    output_dir = Path(str(resolved["output_dir"]))
    if output_dir.exists():
        raise FileExistsError(f"immutable Week-2 output already exists: {output_dir}")
    _write_exclusive(manifest, resolved)
    _append(
        events,
        {
            "schema": "wqcodiff_week2_training_event_v1",
            "event": "started",
            "job_id": resolved["job_id"],
        },
    )
    started = time.monotonic()
    result = subprocess.run(
        list(resolved["argv"]),
        cwd=Path(project_root).resolve(),
        check=False,
    )
    training_evidence = _training_evidence_path(resolved, output_dir)
    continuation = Path(str(resolved["continuation_checkpoint"]))
    validation_ema = (
        None
        if resolved.get("validation_ema") is None
        else Path(str(resolved["validation_ema"]))
    )
    terminal = {
        "schema": "wqcodiff_week2_training_event_v1",
        "event": "terminal",
        "job_id": resolved["job_id"],
        "returncode": int(result.returncode),
        "elapsed_s": time.monotonic() - started,
        "training_evidence": (
            file_identity(training_evidence) if training_evidence.is_file() else None
        ),
        "continuation_checkpoint": (
            file_identity(continuation) if continuation.is_file() else None
        ),
        "validation_ema": (
            file_identity(validation_ema)
            if validation_ema is not None and validation_ema.is_file()
            else None
        ),
    }
    _append(events, terminal)
    expected_ema = resolved.get("validation_ema") is not None
    ok = bool(
        result.returncode == 0
        and terminal["training_evidence"] is not None
        and terminal["continuation_checkpoint"] is not None
        and (not expected_ema or terminal["validation_ema"] is not None)
    )
    if not ok:
        raise RuntimeError(f"Week-2 training job failed evidence gate: {resolved['job_id']}")
    payload = {
        "schema": "wqcodiff_week2_training_complete_v1",
        "ok": True,
        "job_id": resolved["job_id"],
        "target_update": resolved["target_update"],
        "manifest_sha256": sha256_file(manifest),
        "events_sha256": sha256_file(events),
        "training_evidence": terminal["training_evidence"],
        "continuation_checkpoint": terminal["continuation_checkpoint"],
        "validation_ema": terminal["validation_ema"],
    }
    _write_exclusive(complete, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--complete", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    resolved = resolve_job(
        plan_path=args.plan,
        job_id=args.job_id,
        project_root=args.project_root,
    )
    if not args.execute:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    if args.manifest is None or args.events is None or args.complete is None:
        raise SystemExit("--execute requires --manifest, --events, and --complete")
    result = execute_job(
        resolved,
        manifest_path=args.manifest,
        events_path=args.events,
        complete_path=args.complete,
        project_root=args.project_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
