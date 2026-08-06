#!/usr/bin/env python3
"""Verify all frozen H1 bundles and source manifests without writing files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_integrity import verify_h1_fallback  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include every file entry instead of manifest-level counts only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_h1_fallback(project_root=args.project_root)
    if not args.details:
        report = {
            **{key: value for key, value in report.items() if key != "manifests"},
            "manifests": [
                {key: value for key, value in manifest.items() if key != "entries"}
                for manifest in report["manifests"]
            ],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
