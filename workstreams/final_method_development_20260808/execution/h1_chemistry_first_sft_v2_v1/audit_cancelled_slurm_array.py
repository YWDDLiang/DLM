#!/usr/bin/env python3
"""Verify cancelled Slurm array tasks, including compressed sacct rows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


COMPRESSED_ARRAY = re.compile(
    r"^(?P<job>\d+)_\[(?P<start>\d+)(?:-(?P<end>\d+))?(?:%\d+)?\]$"
)
EXPLICIT_TASK = re.compile(r"^(?P<job>\d+)_(?P<task>\d+)$")


def expanded_task_ids(job_id: str) -> set[str]:
    explicit = EXPLICIT_TASK.fullmatch(job_id)
    if explicit:
        return {job_id}
    compressed = COMPRESSED_ARRAY.fullmatch(job_id)
    if not compressed:
        return set()
    start = int(compressed.group("start"))
    end = int(compressed.group("end") or start)
    if end < start:
        raise ValueError(f"descending compressed array range: {job_id}")
    return {f"{compressed.group('job')}_{task}" for task in range(start, end + 1)}


def cancelled_task_ids(sacct_text: str) -> set[str]:
    cancelled: set[str] = set()
    for line in sacct_text.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[1].startswith("CANCELLED"):
            cancelled.update(expanded_task_ids(fields[0]))
    return cancelled


def verify_cancelled_tasks(sacct_text: str, expected: set[str]) -> None:
    observed = cancelled_task_ids(sacct_text)
    missing = expected - observed
    if missing:
        raise RuntimeError(
            f"cancelled Slurm array evidence is incomplete: missing={sorted(missing)}, "
            f"observed={sorted(observed)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sacct", type=Path, required=True)
    parser.add_argument("--expected", nargs="+", required=True)
    args = parser.parse_args()
    verify_cancelled_tasks(
        args.sacct.read_text(encoding="utf-8"), set(args.expected)
    )


if __name__ == "__main__":
    main()
