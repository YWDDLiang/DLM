#!/usr/bin/env python3
"""Execute one immutable modulo-partitioned lane of a materialized Day-7 plan."""

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


PHASES = ("threshold-calibration", "day7-primary", "day7-intervention")
REVISION_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


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


def parse_checkpoint_map(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if not separator or not method or not raw_path or method in result:
            raise ValueError("--checkpoint must be unique METHOD=PATH entries")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[method] = path
    return result


def _checkpoint_placeholder(method: str) -> str:
    return "${CHECKPOINT_" + method.replace("-", "_") + "}"


def resolve_lane(
    *,
    plan_path: str | Path,
    phase: str,
    lane_index: int,
    lane_count: int,
    dataset_path: str | Path,
    checkpoints: Mapping[str, Path],
    revision_threshold: float | None,
) -> dict[str, Any]:
    plan_location = Path(plan_path).resolve()
    plan = json.loads(plan_location.read_text(encoding="utf-8"))
    if plan.get("schema") != "wqcodiff_materialized_job_plan_v1":
        raise ValueError("unsupported materialized Day-7 plan schema")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if lane_count != int(plan.get("maximum_concurrent_lanes", -1)):
        raise ValueError("lane count differs from the frozen plan")
    if not 0 <= lane_index < lane_count:
        raise ValueError("lane index is outside the lane count")
    if phase != "threshold-calibration" and revision_threshold not in REVISION_THRESHOLDS:
        raise ValueError("non-calibration lanes require a frozen revision threshold")

    dataset = Path(dataset_path).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
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
        raise ValueError("lane has no registered cells")

    resolved_jobs: list[dict[str, Any]] = []
    for phase_ordinal, job in chosen:
        method = str(job["method"])
        if method not in checkpoints:
            raise ValueError(f"missing checkpoint for {method}")
        replacements = {
            "${DAY7_VAL_WQ}": str(dataset),
            _checkpoint_placeholder(method): str(checkpoints[method]),
        }
        if revision_threshold is not None:
            replacements["${REVISION_THRESHOLD}"] = str(revision_threshold)
        argv = [replacements.get(str(value), str(value)) for value in job["argv"]]
        if argv[0] != "python":
            raise ValueError("registered cell does not start with the Python interpreter")
        argv[0] = sys.executable
        if any("${" in value for value in argv):
            raise ValueError(f"unresolved placeholder in cell {job['cell_id']}")
        output = Path(argv[argv.index("--output") + 1]).resolve()
        ledger = Path(argv[argv.index("--ledger") + 1]).resolve()
        resolved_jobs.append(
            {
                "phase_ordinal": phase_ordinal,
                "cell_id": job["cell_id"],
                "experiment_id": job["experiment_id"],
                "method": method,
                "attempts": int(job["attempts"]),
                "backbone_calls_per_attempt": int(job["backbone_calls_per_attempt"]),
                "argv": argv,
                "output": str(output),
                "ledger": str(ledger),
            }
        )

    required_methods = sorted({str(job["method"]) for _, job in chosen})
    return {
        "schema": "wqcodiff_day7_lane_plan_v1",
        "materialized_plan": file_identity(plan_location),
        "registry_sha256": plan["registry_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "source_bundle_sha256": plan["source_bundle_sha256"],
        "run_id": plan["run_id"],
        "phase": phase,
        "lane_index": lane_index,
        "lane_count": lane_count,
        "revision_threshold": revision_threshold,
        "dataset": file_identity(dataset),
        "checkpoints": {
            method: file_identity(checkpoints[method]) for method in required_methods
        },
        "cells": len(resolved_jobs),
        "attempts": sum(item["attempts"] for item in resolved_jobs),
        "backbone_calls": sum(
            item["attempts"] * item["backbone_calls_per_attempt"]
            for item in resolved_jobs
        ),
        "jobs": resolved_jobs,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
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
    if events.exists() or complete.exists():
        raise FileExistsError("immutable lane events/complete output already exists")
    for job in lane["jobs"]:
        output = Path(job["output"])
        if (
            output.exists()
            or output.with_suffix(".summary.json").exists()
            or Path(job["ledger"]).exists()
        ):
            raise FileExistsError(
                f"registered cell already has attempt artifacts: {job['cell_id']}"
            )
    _write_json_exclusive(manifest, lane)

    started_all = time.monotonic()
    completed_cells = 0
    for job in lane["jobs"]:
        started = time.monotonic()
        _append_event(
            events,
            {
                "schema": "wqcodiff_day7_lane_event_v1",
                "event": "started",
                "cell_id": job["cell_id"],
                "phase_ordinal": job["phase_ordinal"],
            },
        )
        result = subprocess.run(job["argv"], cwd=Path(project_root).resolve(), check=False)
        output = Path(job["output"])
        summary = output.with_suffix(".summary.json")
        event = {
            "schema": "wqcodiff_day7_lane_event_v1",
            "event": "terminal",
            "cell_id": job["cell_id"],
            "phase_ordinal": job["phase_ordinal"],
            "returncode": int(result.returncode),
            "elapsed_s": time.monotonic() - started,
            "output_sha256": sha256_file(output) if output.is_file() else None,
            "summary_sha256": sha256_file(summary) if summary.is_file() else None,
            "ledger_sha256": (
                sha256_file(Path(job["ledger"]))
                if Path(job["ledger"]).is_file()
                else None
            ),
        }
        _append_event(events, event)
        if result.returncode != 0:
            raise RuntimeError(
                f"Day-7 lane stopped after terminal cell failure: {job['cell_id']}"
            )
        if event["output_sha256"] is None or event["summary_sha256"] is None:
            raise RuntimeError(f"cell lacks immutable output/summary: {job['cell_id']}")
        completed_cells += 1

    payload = {
        "schema": "wqcodiff_day7_lane_complete_v1",
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
    _write_json_exclusive(complete, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--lane-index", type=int, choices=range(4), required=True)
    parser.add_argument("--lane-count", type=int, default=4)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--revision-threshold", type=float)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--complete", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    lane = resolve_lane(
        plan_path=args.plan,
        phase=args.phase,
        lane_index=args.lane_index,
        lane_count=args.lane_count,
        dataset_path=args.dataset,
        checkpoints=parse_checkpoint_map(args.checkpoint),
        revision_threshold=args.revision_threshold,
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
