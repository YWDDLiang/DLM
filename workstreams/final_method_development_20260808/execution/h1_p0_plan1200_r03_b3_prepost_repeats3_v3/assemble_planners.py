#!/usr/bin/env python3
"""Fail-closed assembly of three independently sampled P0 Plan1200 cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (17029, 27183, 31415)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    repeats: list[dict[str, Any]] = []
    failures: list[str] = []
    cohort_hashes: list[str] = []
    for repeat, seed in enumerate(EXPECTED_SEEDS):
        success = run_root / "status" / f"planner_repeat_{repeat}_SUCCESS"
        failed = run_root / "status" / f"planner_repeat_{repeat}_FAILED"
        exit_code = run_root / "status" / f"planner_repeat_{repeat}_exit_code.txt"
        manifest_path = run_root / "repeats" / str(repeat) / "cohort" / "cohort_manifest.json"
        if not success.is_file() or failed.exists() or not exit_code.is_file():
            failures.append(f"repeat_{repeat}:slurm_or_marker_failure")
            continue
        if exit_code.read_text(encoding="ascii").strip() != "0":
            failures.append(f"repeat_{repeat}:nonzero_exit")
            continue
        try:
            manifest = read_json(manifest_path)
            cohort = manifest.get("artifacts", {}).get("cohort1000", {})
            if (
                manifest.get("schema") != "h1_p0_plan1200_frozen_cohort1000_v1"
                or manifest.get("status") != "complete"
                or int(manifest.get("repeat", -1)) != repeat
                or int(manifest.get("planner_seed", -1)) != seed
                or int(manifest.get("raw_attempts", -1)) != 1200
                or int(manifest.get("selected_attempts", -1)) != 1000
                or manifest.get("shared_between_R03_and_B3") is not True
                or manifest.get("arm_outcome_dependent_replacement") is not False
                or manifest.get("raw_rich_seven_line_forwarded") is not False
                or manifest.get("canonical_charge_bucket_visible") is not True
            ):
                raise ValueError("cohort manifest contract changed")
            cohort_path = Path(str(cohort["path"]))
            cohort_sha = str(cohort["sha256"])
            if not cohort_path.is_file() or sha256_file(cohort_path) != cohort_sha:
                raise ValueError("cohort identity changed")
            cohort_hashes.append(cohort_sha)
            repeats.append(
                {
                    "repeat": repeat,
                    "planner_seed": seed,
                    "raw_attempts": int(manifest["raw_attempts"]),
                    "parse_successes": int(manifest["parse_successes"]),
                    "parse_failures": int(manifest["parse_failures"]),
                    "reserve_parse_success_count": int(
                        manifest["reserve_parse_success_count"]
                    ),
                    "cohort_manifest": {
                        "path": str(manifest_path),
                        "bytes": manifest_path.stat().st_size,
                        "sha256": sha256_file(manifest_path),
                    },
                    "cohort1000": cohort,
                    "planner_artifacts": {
                        key: value
                        for key, value in manifest["artifacts"].items()
                        if key != "cohort1000"
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - terminal evidence retains error.
            failures.append(f"repeat_{repeat}:{type(exc).__name__}:{exc}")

    if len(cohort_hashes) == 3 and len(set(cohort_hashes)) != 3:
        failures.append("cross_repeat_cohort_sha_collision")
    report = {
        "schema": "h1_p0_plan1200_repeats3_terminal_v1",
        "status": "failed" if failures else "complete",
        "run_id": run_root.name,
        "repeat_count": 3,
        "planner_seeds": list(EXPECTED_SEEDS),
        "three_independent_plan_batches": not failures,
        "raw_attempts_per_repeat": 1200,
        "frozen_cohort_attempts_per_repeat": 1000,
        "cohort_selection": "first_1000_parse_successes_by_planner_ordinal",
        "shared_between_R03_and_B3_within_repeat": True,
        "cross_repeat_reuse": False,
        "repeats": repeats,
        "failures": failures,
        "planner_source_manifest_sha256": args.source_manifest_sha256,
        "automatic_body_submission": False,
        "automatic_training": False,
        "automatic_rl": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if not failures:
        with (run_root / "status" / "planner_assembly_SUCCESS").open(
            "x", encoding="ascii"
        ) as handle:
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))
    return 3 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
