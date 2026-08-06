#!/usr/bin/env python3
"""Reassemble the B1/B2 smoke terminal using array-aware Slurm identities."""

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


def parse_scheduler(path: Path, array_job_id: str) -> dict[str, dict[str, str]]:
    """Parse JobID|JobIDRaw|State|ExitCode|Elapsed from sacct.

    Slurm may assign an array element a distinct allocation JobIDRaw. The
    display JobID retains the stable ``array_job_id_array_task_id`` identity
    and is therefore the only correct key for arm matching.
    """

    observed: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.strip().split("|")
        if len(fields) < 5:
            continue
        job_id, job_id_raw, state, exit_code, elapsed = fields[:5]
        if not job_id.startswith(f"{array_job_id}_"):
            continue
        if job_id in observed:
            raise RuntimeError(
                f"duplicate scheduler JobID {job_id!r} on line {line_number}"
            )
        observed[job_id] = {
            "job_id": job_id,
            "job_id_raw": job_id_raw,
            "state": state,
            "exit_code": exit_code,
            "elapsed": elapsed,
        }
    return {
        arm: observed.get(
            f"{array_job_id}_{index}",
            {"job_id": f"{array_job_id}_{index}"},
        )
        for index, arm in enumerate(("B1", "B2"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-run-root", type=Path, required=True)
    parser.add_argument("--repair-run-root", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--scheduler-record", type=Path, required=True)
    parser.add_argument("--failed-terminal", type=Path, required=True)
    parser.add_argument("--expected-failed-terminal-sha256", required=True)
    parser.add_argument("--expected-b1-report-sha256", required=True)
    parser.add_argument("--expected-b2-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reasons: list[str] = []
    failed_terminal_sha = sha256_file(args.failed_terminal)
    if failed_terminal_sha != args.expected_failed_terminal_sha256:
        reasons.append("preserved failed terminal identity mismatch")

    scheduler = parse_scheduler(args.scheduler_record, args.array_job_id)
    expected_report_shas = {
        "B1": args.expected_b1_report_sha256,
        "B2": args.expected_b2_report_sha256,
    }
    arms: dict[str, Any] = {}
    for arm in ("B1", "B2"):
        scheduler_entry = scheduler[arm]
        if (
            scheduler_entry.get("state") != "COMPLETED"
            or scheduler_entry.get("exit_code") != "0:0"
        ):
            reasons.append(f"{arm} scheduler state is not COMPLETED 0:0")

        report_path = (
            args.evidence_run_root / "arms" / arm / "engineering_report.json"
        )
        if not report_path.exists():
            reasons.append(f"{arm} engineering report is missing")
            continue
        report_sha = sha256_file(report_path)
        if report_sha != expected_report_shas[arm]:
            reasons.append(f"{arm} engineering report identity mismatch")
        report = read_json(report_path)
        if report.get("engineering_gate_passed") is not True:
            reasons.append(f"{arm} engineering gate did not pass")
        arms[arm] = {
            "scheduler": scheduler_entry,
            "report": report,
            "report_sha256": report_sha,
        }

    gate_passed = not reasons and set(arms) == {"B1", "B2"}
    payload = {
        "schema": "h1a2_dlm_b1_b2_2xa800_smoke32_terminal_repair_v2",
        "status": "complete" if gate_passed else "failed",
        "engineering_gate_passed": gate_passed,
        "repair_kind": "array_aware_sacct_jobid_parser",
        "original_array_job_id": args.array_job_id,
        "original_failed_assembly_job_id": "29338",
        "original_failed_terminal_sha256": failed_terminal_sha,
        "scheduler_record_sha256": sha256_file(args.scheduler_record),
        "arms": arms,
        "failure_reasons": reasons,
        "next_gate": (
            "freeze_resource_envelope_and_review_scientific_learning_rate"
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
    args.repair_run_root.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
