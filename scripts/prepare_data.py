#!/usr/bin/env python3
"""Report preparation status for the frozen MP-20 release split."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mp20"
EXPECTED = {"train.csv": 27136, "val.csv": 9047, "test.csv": 9046}


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.reader(handle))
    return max(0, rows - 1)


def main() -> None:
    report: dict[str, object] = {"data_root": str(DATA), "files": {}}
    for name, expected in EXPECTED.items():
        path = DATA / name
        observed = row_count(path) if path.is_file() else None
        report["files"][name] = {
            "present": path.is_file(),
            "expected_rows": expected,
            "observed_rows": observed,
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

