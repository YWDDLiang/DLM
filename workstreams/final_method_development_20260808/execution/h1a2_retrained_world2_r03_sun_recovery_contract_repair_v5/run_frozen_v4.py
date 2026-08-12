#!/usr/bin/env python3
"""Run one byte-verified frozen V4 module with this protocol preloaded."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import protocol
from protocol import require_file


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--script-sha256", required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        raise ValueError("frozen module arguments are required")
    script = require_file(args.script, args.script_sha256, "frozen V4 module")
    sys.modules["protocol"] = protocol
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

