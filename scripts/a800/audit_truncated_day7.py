#!/usr/bin/env python3
"""Audit the immutable completed subset of an intentionally stopped Day-7 run.

This is deliberately not a replacement for the preregistered full-grid auditor.
It verifies that every cell which reached a successful terminal lane event is
internally complete and immutable, while recording all active and never-started
cells as missing from the full scientific gate.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from crystal_dlm.wqcodiff.contracts import AttemptLedger, write_json_exclusive


NONTERMINAL_STATUSES = {"submitted", "running"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    location = path.resolve()
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": _sha256(location),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSON object required: {path}:{line_number}")
            rows.append(payload)
    return rows


def _summary_path(output: Path) -> Path:
    return output.with_suffix(".summary.json")


def _status_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(collections.Counter(str(row.get("status")) for row in rows).items())
    )


def _sacct(job_id: str) -> dict[str, Any]:
    command = [
        "sacct",
        "-n",
        "-P",
        "-j",
        job_id,
        "--format=JobID,JobIDRaw,State,Elapsed,TotalCPU,MaxRSS,ExitCode,AllocTRES",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-root", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--expected-cells", type=int, default=180)
    parser.add_argument("--expected-attempts", type=int, default=184_320)
    parser.add_argument("--stop-reason", required=True)
    args = parser.parse_args()

    lane_root = args.lane_root.resolve()
    manifest_paths = sorted(lane_root.glob("lane*/lane_manifest.json"))
    event_paths = sorted(lane_root.glob("lane*/lane_events.jsonl"))
    if len(manifest_paths) != 4 or len(event_paths) != 4:
        raise ValueError("exactly four lane manifests and event streams are required")

    manifests = [_read_json(path) for path in manifest_paths]
    jobs: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for job in manifest.get("jobs", []):
            cell_id = str(job.get("cell_id", ""))
            if not cell_id or cell_id in jobs:
                raise ValueError(f"invalid or duplicate cell ID: {cell_id!r}")
            jobs[cell_id] = job

    events = [event for path in event_paths for event in _read_jsonl(path)]
    started = collections.Counter(
        str(event.get("cell_id", ""))
        for event in events
        if event.get("event") == "started"
    )
    terminal_events: dict[str, dict[str, Any]] = {}
    terminal_failures: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") != "terminal":
            continue
        cell_id = str(event.get("cell_id", ""))
        target = terminal_events if int(event.get("returncode", 1)) == 0 else terminal_failures
        if cell_id in target:
            raise ValueError(f"duplicate terminal lane event: {cell_id}")
        target[cell_id] = event

    completed_ids = set(terminal_events)
    active_ids = set(started) - set(terminal_events) - set(terminal_failures)
    never_started_ids = set(jobs) - set(started)
    unknown_event_ids = (set(started) | set(terminal_events) | set(terminal_failures)) - set(jobs)

    cell_audits: list[dict[str, Any]] = []
    for cell_id in sorted(completed_ids, key=lambda value: int(jobs[value]["phase_ordinal"])):
        job = jobs[cell_id]
        event = terminal_events[cell_id]
        output_path = Path(str(job["output"])).resolve()
        ledger_path = Path(str(job["ledger"])).resolve()
        summary_path = _summary_path(output_path)
        rows = _read_jsonl(output_path)
        attempt_ids = [str(row.get("attempt_id", "")) for row in rows]
        summary = _read_json(summary_path)
        ledger_audit = AttemptLedger(ledger_path).audit(
            terminal_stage="recovery",
            expected_attempt_ids=attempt_ids,
        )
        output_identity = _identity(output_path)
        ledger_identity = _identity(ledger_path)
        summary_identity = _identity(summary_path)
        checks = {
            "single_started_event": started[cell_id] == 1,
            "manifest_attempt_count": int(job.get("attempts", -1)) == 1024,
            "output_attempt_count": len(rows) == int(job.get("attempts", -1)),
            "output_schema": all(
                row.get("schema") == "wqcodiff_recovery_attempt_v1" for row in rows
            ),
            "output_unique_attempt_ids": len(attempt_ids) == len(set(attempt_ids)),
            "output_all_terminal": all(
                str(row.get("status")) not in NONTERMINAL_STATUSES for row in rows
            ),
            "ledger_audit": ledger_audit.ok,
            "ledger_record_count": ledger_audit.records == 2 * len(rows),
            "ledger_attempt_count": ledger_audit.attempts == len(rows),
            "summary_terminal": summary.get("all_attempts_terminal") is True,
            "summary_attempt_count": int(summary.get("structures", -1)) == len(rows),
            "event_output_hash": event.get("output_sha256") == output_identity["sha256"],
            "event_ledger_hash": event.get("ledger_sha256") == ledger_identity["sha256"],
            "event_summary_hash": event.get("summary_sha256") == summary_identity["sha256"],
        }
        first = rows[0] if rows else {}
        cell_audits.append(
            {
                "cell_id": cell_id,
                "phase_ordinal": int(job["phase_ordinal"]),
                "method": job.get("method"),
                "level": first.get("corruption_level"),
                "operator": first.get("operator"),
                "corruption_seed": first.get("corruption_seed"),
                "ok": all(checks.values()),
                "checks": checks,
                "status_counts": _status_counts(rows),
                "lane_event": event,
                "artifacts": {
                    "output": output_identity,
                    "ledger": ledger_identity,
                    "summary": summary_identity,
                },
                "attempt_audit": dataclasses.asdict(ledger_audit),
            }
        )

    aggregate_path = args.aggregate.resolve()
    aggregate = _read_json(aggregate_path)
    completed_attempts = sum(int(jobs[cell_id]["attempts"]) for cell_id in completed_ids)
    by_method = dict(
        sorted(
            collections.Counter(str(jobs[cell_id]["method"]) for cell_id in completed_ids).items()
        )
    )
    completed_subset_valid = (
        not unknown_event_ids
        and not terminal_failures
        and len(completed_ids) == len(cell_audits)
        and all(cell["ok"] for cell in cell_audits)
        and aggregate.get("records") == completed_attempts
    )
    full_gate_eligible = (
        completed_subset_valid
        and len(completed_ids) == args.expected_cells
        and completed_attempts == args.expected_attempts
        and not active_ids
        and not never_started_ids
    )
    result = {
        "schema": "wqcodiff_day7_intentionally_truncated_audit_v1",
        "ok": completed_subset_valid,
        "full_preregistered_gate_eligible": full_gate_eligible,
        "scientific_use": "engine_selection_diagnostic_only",
        "stop": {
            "requested": True,
            "reason": args.stop_reason,
            "slurm_job_id": args.slurm_job_id,
            "accounting": _sacct(args.slurm_job_id),
        },
        "expected": {
            "cells": args.expected_cells,
            "attempts": args.expected_attempts,
        },
        "observed": {
            "completed_cells": len(completed_ids),
            "completed_attempts": completed_attempts,
            "by_method": by_method,
            "active_at_cancel": sorted(active_ids),
            "never_started": sorted(never_started_ids),
            "terminal_failures": sorted(terminal_failures),
            "unknown_event_ids": sorted(unknown_event_ids),
        },
        "aggregate": {
            "artifact": _identity(aggregate_path),
            "records": aggregate.get("records"),
            "dlm_promoted": aggregate.get("dlm_promoted"),
            "required_claim_action": aggregate.get("required_claim_action"),
        },
        "lane_inputs": {
            "manifests": [_identity(path) for path in manifest_paths],
            "events": [_identity(path) for path in event_paths],
        },
        "cells": cell_audits,
    }
    write_json_exclusive(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "ok": completed_subset_valid,
                "full_preregistered_gate_eligible": full_gate_eligible,
                "completed_cells": len(completed_ids),
                "completed_attempts": completed_attempts,
                "by_method": by_method,
                "active_at_cancel": sorted(active_ids),
                "never_started_count": len(never_started_ids),
                "dlm_promoted": aggregate.get("dlm_promoted"),
                "required_claim_action": aggregate.get("required_claim_action"),
                "output": _identity(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if completed_subset_valid else 47


if __name__ == "__main__":
    raise SystemExit(main())
