#!/usr/bin/env python3
"""Execute one immutable modulo-partitioned Week-2 sampling lane."""

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


PHASES = ("preflight", "development")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    location = Path(path).resolve()
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }


def verify_identity(payload: Mapping[str, Any]) -> Path:
    path = Path(str(payload["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(payload["bytes"]):
        raise ValueError(f"artifact size differs from frozen identity: {path}")
    if sha256_file(path) != str(payload["sha256"]):
        raise ValueError(f"artifact SHA256 differs from frozen identity: {path}")
    return path


def _rooted(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def resolve_lane(
    *,
    plan_path: str | Path,
    phase: str,
    lane_index: int,
    lane_count: int,
    project_root: str | Path,
) -> dict[str, Any]:
    plan_location = Path(plan_path).resolve()
    plan = json.loads(plan_location.read_text(encoding="utf-8"))
    if plan.get("schema") != "wqcodiff_week2_sampling_plan_v1":
        raise ValueError("unsupported Week-2 sampling plan schema")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if lane_count != int(plan.get("maximum_concurrent_lanes", -1)):
        raise ValueError("lane count differs from the frozen sampling plan")
    if not 0 <= lane_index < lane_count:
        raise ValueError("lane index is outside the lane count")

    root = Path(project_root).resolve()
    revision_lock = verify_identity(plan["revision_threshold_lock"])
    phase_jobs = [job for job in plan["jobs"] if job.get("phase") == phase]
    expected = int(plan["phase_summary"][phase]["jobs"])
    if len(phase_jobs) != expected:
        raise ValueError("materialized phase job count disagrees with its summary")
    chosen = [
        (phase_ordinal, job)
        for phase_ordinal, job in enumerate(phase_jobs)
        if phase_ordinal % lane_count == lane_index
    ]
    if not chosen:
        raise ValueError("sampling lane has no registered cells")

    resolved_jobs: list[dict[str, Any]] = []
    for phase_ordinal, job in chosen:
        checkpoint = verify_identity(job["checkpoint"])
        argv = [str(value) for value in job["argv"]]
        if argv[:4] != ["python", "-m", "crystal_dlm.wqcodiff", "--protocol"]:
            raise ValueError("registered sampling cell is not a Python module command")
        if "sample" not in argv:
            raise ValueError("registered Week-2 cell is not a sample command")
        argv[0] = sys.executable
        replacements = {
            "--checkpoint": checkpoint,
            "--revision-lock": revision_lock,
            "--output": _rooted(root, str(job["output"])),
            "--ledger": _rooted(root, str(job["ledger"])),
        }
        for flag, path in replacements.items():
            position = argv.index(flag) + 1
            argv[position] = str(path)
        if Path(argv[argv.index("--checkpoint") + 1]) != checkpoint:
            raise ValueError("sampling argv checkpoint differs from its frozen identity")
        if Path(argv[argv.index("--revision-lock") + 1]) != revision_lock:
            raise ValueError("sampling argv threshold lock differs from its frozen identity")
        output = replacements["--output"]
        ledger = replacements["--ledger"]
        for artifact in (output, ledger):
            try:
                artifact.relative_to(root / "runs" / str(plan["run_id"]))
            except ValueError as exc:
                raise ValueError("Week-2 sampling output escapes its run directory") from exc
        resolved_jobs.append(
            {
                "phase_ordinal": phase_ordinal,
                "cell_id": job["cell_id"],
                "configuration_id": job["configuration_id"],
                "variant": job["variant"],
                "attempts": int(job["attempts"]),
                "backbone_calls_per_attempt": int(
                    job["backbone_calls_per_attempt"]
                ),
                "checkpoint": file_identity(checkpoint),
                "argv": argv,
                "output": str(output),
                "ledger": str(ledger),
            }
        )

    return {
        "schema": "wqcodiff_week2_sampling_lane_plan_v1",
        "materialized_plan": file_identity(plan_location),
        "registry_sha256": plan["registry_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "source_bundle_sha256": plan["source_bundle_sha256"],
        "run_id": plan["run_id"],
        "phase": phase,
        "lane_index": lane_index,
        "lane_count": lane_count,
        "revision_threshold_lock": file_identity(revision_lock),
        "cells": len(resolved_jobs),
        "attempts": sum(job["attempts"] for job in resolved_jobs),
        "backbone_calls": sum(
            job["attempts"] * job["backbone_calls_per_attempt"]
            for job in resolved_jobs
        ),
        "jobs": resolved_jobs,
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


def execute_lane(
    lane: Mapping[str, Any],
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
        raise FileExistsError("immutable Week-2 sampling evidence already exists")
    for job in lane["jobs"]:
        output = Path(str(job["output"]))
        ledger = Path(str(job["ledger"]))
        if output.exists() or output.with_suffix(".summary.json").exists() or ledger.exists():
            raise FileExistsError(
                f"registered sampling cell already has artifacts: {job['cell_id']}"
            )
    _write_exclusive(manifest, lane)

    started_all = time.monotonic()
    completed_cells = 0
    for job in lane["jobs"]:
        started = time.monotonic()
        _append(
            events,
            {
                "schema": "wqcodiff_week2_sampling_event_v1",
                "event": "started",
                "cell_id": job["cell_id"],
                "phase_ordinal": job["phase_ordinal"],
            },
        )
        result = subprocess.run(
            list(job["argv"]),
            cwd=Path(project_root).resolve(),
            check=False,
        )
        output = Path(str(job["output"]))
        summary = output.with_suffix(".summary.json")
        ledger = Path(str(job["ledger"]))
        terminal = {
            "schema": "wqcodiff_week2_sampling_event_v1",
            "event": "terminal",
            "cell_id": job["cell_id"],
            "phase_ordinal": job["phase_ordinal"],
            "returncode": int(result.returncode),
            "elapsed_s": time.monotonic() - started,
            "output_sha256": sha256_file(output) if output.is_file() else None,
            "summary_sha256": sha256_file(summary) if summary.is_file() else None,
            "ledger_sha256": sha256_file(ledger) if ledger.is_file() else None,
        }
        _append(events, terminal)
        if result.returncode != 0:
            raise RuntimeError(
                f"Week-2 sampling lane stopped after cell failure: {job['cell_id']}"
            )
        if any(terminal[key] is None for key in ("output_sha256", "summary_sha256", "ledger_sha256")):
            raise RuntimeError(f"sampling cell lacks immutable artifacts: {job['cell_id']}")
        completed_cells += 1

    payload = {
        "schema": "wqcodiff_week2_sampling_complete_v1",
        "ok": completed_cells == int(lane["cells"]),
        "phase": lane["phase"],
        "lane_index": lane["lane_index"],
        "cells": lane["cells"],
        "completed_cells": completed_cells,
        "attempts": lane["attempts"],
        "backbone_calls": lane["backbone_calls"],
        "elapsed_s": time.monotonic() - started_all,
        "manifest_sha256": sha256_file(manifest),
        "events_sha256": sha256_file(events),
    }
    _write_exclusive(complete, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--lane-index", type=int, choices=range(4), required=True)
    parser.add_argument("--lane-count", type=int, default=4)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--complete", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    lane = resolve_lane(
        plan_path=args.plan,
        phase=args.phase,
        lane_index=args.lane_index,
        lane_count=args.lane_count,
        project_root=args.project_root,
    )
    if not args.execute:
        print(json.dumps(lane, indent=2, sort_keys=True))
        return
    if args.manifest is None or args.events is None or args.complete is None:
        raise SystemExit("--execute requires --manifest, --events, and --complete")
    result = execute_lane(
        lane,
        manifest_path=args.manifest,
        events_path=args.events,
        complete_path=args.complete,
        project_root=args.project_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
