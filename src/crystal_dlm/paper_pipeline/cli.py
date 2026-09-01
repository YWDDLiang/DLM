"""Read-only CLI for validating and discovering paper pipeline contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .manifest import MAINLINE_STAGE_ORDER, command_for_stage, load_and_validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crystal-paper-pipeline")
    parser.add_argument("--config", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("validate", help="validate all frozen method contracts")
    subparsers.add_parser("show", help="show the coupled method and stage order")
    stage = subparsers.add_parser("stage", help="show one audited stage command")
    stage.add_argument("stage_id", choices=MAINLINE_STAGE_ORDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, report = load_and_validate(args.config)
    if args.action == "validate":
        payload = report
    elif args.action == "show":
        payload = {
            "schema": manifest["schema"],
            "method_name": manifest["method_name"],
            "factorization": manifest["factorization"],
            "stage_order": report["stage_order"],
            "profiles": manifest["profiles"],
        }
    else:
        payload = command_for_stage(manifest, args.stage_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
