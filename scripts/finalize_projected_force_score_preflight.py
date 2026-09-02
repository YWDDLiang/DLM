#!/usr/bin/env python3
"""Recompute projected-force preflight accounting from immutable row outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_projected_force_score_preflight import build_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows_path = args.rows_jsonl.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = [
        json.loads(line)
        for line in rows_path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 512:
        raise ValueError("projected force finalizer requires exactly 512 rows")
    report = build_report(rows)
    report["source_rows_jsonl"] = str(rows_path)
    output.mkdir(parents=True)
    (output / "PROJECTED_FORCE_SCORE_FINAL.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
