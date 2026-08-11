#!/usr/bin/env python3
"""Run the byte-frozen current S.U.N. adapter for environment-repair V2."""

from __future__ import annotations

import argparse
import importlib
import os
import runpy
import sys
from pathlib import Path

from protocol import require_file, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--expected-mp-cache-sha256", required=True)
    parser.add_argument("--expected-relax-cache-sha256", required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if not forwarded:
        raise ValueError("S.U.N. runner arguments are required")
    if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
        raise RuntimeError("MP credentials are forbidden in Slurm evaluation")

    runner = require_file(args.runner, args.runner_sha256, "current S.U.N. runner")
    module = importlib.import_module("crystal_dlm.wqcodiff.crysllmgen.a100_sun")
    module_path = Path(module.__file__).resolve()
    if sha256_file(module_path) != args.module_sha256:
        raise ValueError("current S.U.N. module identity changed")
    module.MP_HULL_CACHE_SHA256 = args.expected_mp_cache_sha256
    module.CHGNET_RELAX_CACHE_SHA256 = args.expected_relax_cache_sha256
    sys.argv = [str(runner), *forwarded]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
