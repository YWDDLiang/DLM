#!/usr/bin/env python3
"""Retain the selected best checkpoint and the latest two checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.checkpoint_retention import (  # noqa: E402
    apply_retention_plan,
    build_retention_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        type=Path,
        help="Direct parent containing checkpoint-N or step-N directories.",
    )
    parser.add_argument(
        "--best-checkpoint",
        required=True,
        type=Path,
        help="Selected best checkpoint; must be a direct child of checkpoint-root.",
    )
    parser.add_argument("--keep-latest", type=int, default=2)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the plan. Without this flag the command is a dry run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_retention_plan(
        args.checkpoint_root,
        args.best_checkpoint,
        keep_latest=args.keep_latest,
    )
    result = plan.to_dict()
    result["applied"] = bool(args.apply)
    if args.apply:
        result["deleted_paths"] = list(apply_retention_plan(plan))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

