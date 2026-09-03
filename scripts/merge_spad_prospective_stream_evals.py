#!/usr/bin/env python3
"""Create one immutable view over the two independently scheduled SPAD streams."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_report(run: Path, expected_stream: int) -> dict[str, Any]:
    if not (run / "_OFFLINE_SUCCESS").is_file():
        raise FileNotFoundError(f"stream {expected_stream} evaluation is incomplete")
    report = json.loads((run / "OFFLINE_FINAL.json").read_text(encoding="utf-8"))
    if (
        report.get("schema") != "spad_prospective_offline_stream_v1"
        or int(report.get("stream", -1)) != expected_stream
        or report.get("official") is not False
        or len(report.get("cells") or ()) != 6
    ):
        raise ValueError(f"stream {expected_stream} report is malformed")
    if {int(row["stream"]) for row in report["cells"]} != {expected_stream}:
        raise ValueError("cell stream labels differ from run stream")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream17-run", type=Path, required=True)
    parser.add_argument("--stream18-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    runs = {17: args.stream17_run.resolve(), 18: args.stream18_run.resolve()}
    reports = {stream: read_report(run, stream) for stream, run in runs.items()}
    cohorts = {str(report["cohort"]) for report in reports.values()}
    if len(cohorts) != 1:
        raise ValueError("stream evaluations use different frozen cohorts")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    cells = []
    for arm in ("B0", "BC", "BS"):
        (args.output_dir / arm).mkdir()
        for stream, run in runs.items():
            source = run / arm / f"stream{stream}"
            if not source.is_dir():
                raise FileNotFoundError(source)
            os.symlink(
                source,
                args.output_dir / arm / f"stream{stream}",
                target_is_directory=True,
            )
            cells.extend(
                row for row in reports[stream]["cells"] if row["arm"] == arm
            )
    if len(cells) != 12:
        raise ValueError("merged prospective evaluation does not have 12 cells")
    report = {
        "schema": "spad_prospective_offline_union_v1",
        "cohort": next(iter(cohorts)),
        "source_runs": {str(stream): str(run) for stream, run in runs.items()},
        "cells": cells,
        "official": False,
        "scientific_recomputation": False,
    }
    (args.output_dir / "OFFLINE_FINAL.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_OFFLINE_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
