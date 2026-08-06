#!/usr/bin/env python3
"""Compare a fresh zero-change P0 replay with the immutable H1 output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mismatch_ordinals(
    reference: list[dict[str, Any]],
    replay: list[dict[str, Any]],
) -> list[int]:
    limit = min(len(reference), len(replay))
    mismatches = [
        index for index in range(limit) if reference[index] != replay[index]
    ]
    mismatches.extend(range(limit, max(len(reference), len(replay))))
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    names = ("body_attempts.jsonl", "generation.jsonl")
    paths = {
        name: (
            args.reference_dir / name,
            args.replay_dir / name,
        )
        for name in names
    }
    for reference_path, replay_path in paths.values():
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        if not replay_path.is_file():
            raise FileNotFoundError(replay_path)

    checks: dict[str, Any] = {}
    exact = True
    for name, (reference_path, replay_path) in paths.items():
        reference_rows = read_jsonl(reference_path)
        replay_rows = read_jsonl(replay_path)
        mismatches = mismatch_ordinals(reference_rows, replay_rows)
        reference_sha = sha256_file(reference_path)
        replay_sha = sha256_file(replay_path)
        file_exact = reference_sha == replay_sha
        row_exact = not mismatches
        exact = exact and file_exact and row_exact and len(reference_rows) == 256
        checks[name] = {
            "reference_sha256": reference_sha,
            "replay_sha256": replay_sha,
            "reference_rows": len(reference_rows),
            "replay_rows": len(replay_rows),
            "file_bytes_exact": file_exact,
            "json_rows_exact": row_exact,
            "mismatch_count": len(mismatches),
            "first_mismatch_ordinals": mismatches[:16],
        }

    report = {
        "schema": "h1_exact_replay_report_v1",
        "status": "complete" if exact else "parity_failed",
        "gate_passed": exact,
        "arm": "P0",
        "attempts": 256,
        "zero_scientific_change": True,
        "body_generation_rerun": True,
        "refinement_rerun": True,
        "diffusion_reverse_steps": 800,
        "direct_metrics_run": False,
        "sun_run": False,
        "automatic_downstream": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
