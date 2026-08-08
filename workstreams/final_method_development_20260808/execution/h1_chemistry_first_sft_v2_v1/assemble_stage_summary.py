#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ALLOWED = {"sft_v2", "sft_v2_c"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(64, 256), required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [value for value in args.candidates.split(",") if value]
    if not candidates or len(candidates) != len(set(candidates)) or not set(candidates) <= ALLOWED:
        raise SystemExit("candidate list is empty, duplicated, or invalid")
    terminals = {}
    passing = []
    scientific_stops = []
    engineering_failures = []
    for candidate in candidates:
        path = args.stage_root / "terminal" / f"{candidate}_terminal_report.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        status = value.get("status")
        if value.get("candidate_id") != candidate or int(value.get("stage", -1)) != args.stage:
            raise SystemExit(f"terminal identity mismatch for {candidate}")
        terminals[candidate] = {
            "status": status,
            "decision": value.get("decision"),
            "gate_passed": value.get("gate_passed"),
            "sha256": sha256_file(path),
        }
        if status == "planner_gate_pass" and value.get("gate_passed") is True:
            passing.append(candidate)
        elif status == "scientific_stop" and value.get("gate_passed") is False:
            scientific_stops.append(candidate)
        elif status == "engineering_failure":
            engineering_failures.append(candidate)
        else:
            raise SystemExit(f"unexpected terminal status for {candidate}: {status!r}")
    status = (
        "engineering_failure"
        if engineering_failures
        else "planner_gate_pass"
        if passing
        else "scientific_stop"
    )
    payload = {
        "schema": "h1_chemistry_first_stage_summary_v1",
        "identity": "h1_chemistry_first_sft_v2_v1",
        "stage": args.stage,
        "status": status,
        "candidates": candidates,
        "passing_candidates": passing,
        "scientific_stop_candidates": scientific_stops,
        "engineering_failure_candidates": engineering_failures,
        "terminals": terminals,
        "automatic_downstream": False,
        "automatic_rl": False,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
