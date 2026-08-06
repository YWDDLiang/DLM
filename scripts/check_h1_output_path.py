#!/usr/bin/env python3
"""Check proposed experiment output paths against the frozen H1 fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_readonly_guard import (  # noqa: E402
    H1ReadOnlyViolation,
    assert_writable_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Proposed output paths")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Checkout root used for relative paths",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: list[dict[str, object]] = []
    blocked = False
    for raw_path in args.paths:
        try:
            normalized = assert_writable_output_path(
                raw_path,
                project_root=args.project_root,
            )
        except H1ReadOnlyViolation as exc:
            blocked = True
            results.append(
                {
                    "path": raw_path,
                    "allowed": False,
                    "reason": str(exc),
                }
            )
        else:
            results.append(
                {
                    "path": raw_path,
                    "allowed": True,
                    "normalized": str(normalized),
                }
            )
    print(
        json.dumps(
            {"allowed": not blocked, "results": results}, indent=2, sort_keys=True
        )
    )
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
