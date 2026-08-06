#!/usr/bin/env python3
"""Assemble the terminal report for the bounded B1/B2 DDP smoke DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def parse_scheduler(path: Path, array_job_id: str) -> dict[str, Any]:
    observed: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4:
            continue
        job_id, state, exit_code, elapsed = fields[:4]
        observed[job_id] = {
            "state": state,
            "exit_code": exit_code,
            "elapsed": elapsed,
        }
    return {
        arm: {
            "job_id": f"{array_job_id}_{index}",
            **observed.get(f"{array_job_id}_{index}", {}),
        }
        for index, arm in enumerate(("B1", "B2"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--scheduler-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scheduler = parse_scheduler(args.scheduler_record, args.array_job_id)
    reasons: list[str] = []
    arms: dict[str, Any] = {}
    for arm in ("B1", "B2"):
        scheduler_entry = scheduler[arm]
        if (
            scheduler_entry.get("state") != "COMPLETED"
            or scheduler_entry.get("exit_code") != "0:0"
        ):
            reasons.append(f"{arm} scheduler state is not COMPLETED 0:0")
        report_path = args.run_root / "arms" / arm / "engineering_report.json"
        if not report_path.exists():
            reasons.append(f"{arm} engineering report is missing")
            continue
        report = read_json(report_path)
        if report.get("engineering_gate_passed") is not True:
            reasons.append(f"{arm} engineering gate did not pass")
        arms[arm] = {
            "scheduler": scheduler_entry,
            "report": report,
            "report_sha256": sha256_file(report_path),
        }

    gate_passed = not reasons and set(arms) == {"B1", "B2"}
    payload = {
        "schema": "h1a2_dlm_b1_b2_2xa800_smoke32_terminal_v1",
        "status": "complete" if gate_passed else "failed",
        "engineering_gate_passed": gate_passed,
        "array_job_id": args.array_job_id,
        "arms": arms,
        "failure_reasons": reasons,
        "next_gate": (
            "freeze_B1_B2_learning_rate_and_scientific_training_manifest"
            if gate_passed
            else "stop_and_preserve_engineering_failure"
        ),
        "scientific_result": False,
        "eligible_for_checkpoint_selection": False,
        "eligible_for_later_initialization": False,
        "automatic_downstream": False,
        "scientific_training_authorized": False,
        "crystal_generation": False,
        "sun_evaluation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
