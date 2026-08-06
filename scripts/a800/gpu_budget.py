#!/usr/bin/env python3
"""Audit consumed A800 GPU-hours before each four-job submission wave."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from collections.abc import Sequence
from typing import Any


CUMULATIVE_LIMITS = {1: 180.0, 2: 750.0, 3: 1250.0, 4: 2050.0}
TOTAL_LIMIT = 2050.0
WEEK4_RESERVE = 800.0


def audit_budget(
    runs_root: str | Path | Sequence[str | Path],
    *,
    current_week: int,
    proposed_gpu_hours: float = 0.0,
) -> dict[str, Any]:
    if current_week not in CUMULATIVE_LIMITS:
        raise ValueError("current_week must be in {1,2,3,4}")
    if proposed_gpu_hours < 0.0:
        raise ValueError("proposed_gpu_hours must be nonnegative")
    raw_roots = (
        (runs_root,)
        if isinstance(runs_root, (str, Path))
        else tuple(runs_root)
    )
    if not raw_roots:
        raise ValueError("at least one runs root is required")
    roots = tuple(Path(value).resolve() for value in raw_roots)
    if len(roots) != len(set(roots)):
        raise ValueError("duplicate runs roots are forbidden")
    by_week: dict[int, float] = defaultdict(float)
    jobs: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for root in roots:
        for path in sorted(root.glob("**/*.job_usage.json")) if root.exists() else ():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("schema") != "wqcodiff_slurm_usage_v1":
                    raise ValueError("schema")
                job_id = str(payload["slurm_job_id"])
                week = int(payload["week"])
                gpu_hours = float(payload["gpu_hours"])
                if week not in CUMULATIVE_LIMITS or gpu_hours < 0.0:
                    raise ValueError("week/gpu_hours")
                if job_id in jobs:
                    raise ValueError(
                        f"duplicate job id also seen in {jobs[job_id]['path']}"
                    )
                jobs[job_id] = {
                    "path": str(path),
                    "week": week,
                    "gpu_hours": gpu_hours,
                    "run_id": payload.get("run_id"),
                    "job_name": payload.get("slurm_job_name"),
                }
                by_week[week] += gpu_hours
            except Exception as exc:
                errors.append(f"{path}:{type(exc).__name__}:{exc}")
    consumed_total = sum(by_week.values())
    consumed_through_week = sum(
        value for week, value in by_week.items() if week <= current_week
    )
    projected_total = consumed_total + proposed_gpu_hours
    projected_cumulative = consumed_through_week + proposed_gpu_hours
    checks = {
        "usage_records_valid": not errors,
        "current_week_cumulative_within_limit": projected_cumulative
        <= CUMULATIVE_LIMITS[current_week],
        "total_within_2050": projected_total <= TOTAL_LIMIT,
        "week4_reserve_preserved": current_week == 4
        or TOTAL_LIMIT - projected_total >= WEEK4_RESERVE,
    }
    return {
        "schema": "wqcodiff_gpu_budget_audit_v1",
        "runs_roots": [str(root) for root in roots],
        "ok": all(checks.values()),
        "current_week": current_week,
        "proposed_gpu_hours": proposed_gpu_hours,
        "limits": {
            "cumulative": CUMULATIVE_LIMITS,
            "total": TOTAL_LIMIT,
            "week4_reserve": WEEK4_RESERVE,
        },
        "consumed_by_week": {
            str(week): by_week.get(week, 0.0) for week in sorted(CUMULATIVE_LIMITS)
        },
        "consumed_total": consumed_total,
        "projected_current_week_cumulative": projected_cumulative,
        "projected_total": projected_total,
        "remaining_total_after_proposal": TOTAL_LIMIT - projected_total,
        "checks": checks,
        "usage_records": len(jobs),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--current-week", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--proposed-gpu-hours", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_budget(
        args.runs_root,
        current_week=args.current_week,
        proposed_gpu_hours=args.proposed_gpu_hours,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 3)


if __name__ == "__main__":
    main()
