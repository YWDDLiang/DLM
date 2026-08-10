#!/usr/bin/env python3
"""Run the byte-frozen R03E S.U.N. adapter with one completed-cache SHA."""

from __future__ import annotations

import argparse
import importlib
import os
import runpy
import sys
from pathlib import Path

from protocol import require_file, require_hex_sha, sha256_file


BASE_CACHE_SHA256 = (
    "93d6532cd93c1cfebcbc969d0299852359d6a2950b66b259c028e971f8f7e4ff"
)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--a100-sun-module-sha256", required=True)
    parser.add_argument("--expected-mp-hull-cache-sha256", required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        raise ValueError("R03E S.U.N. arguments are required")
    if any(
        os.environ.get(name)
        for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")
    ):
        raise RuntimeError("MP credentials must be absent in Slurm evaluation")

    expected_cache_sha = require_hex_sha(
        args.expected_mp_hull_cache_sha256,
        "completed MP hull cache SHA256",
    )
    positions = [
        index
        for index, value in enumerate(forwarded)
        if value == "--mp-hull-cache"
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(forwarded):
        raise ValueError("exactly one --mp-hull-cache argument is required")
    cache = Path(forwarded[positions[0] + 1]).resolve()
    if sha256_file(cache) != expected_cache_sha:
        raise ValueError("completed MP hull cache identity changed")

    runner = require_file(
        args.runner, args.runner_sha256, "byte-frozen R03E S.U.N. runner"
    )
    sun_module = importlib.import_module(
        "crystal_dlm.wqcodiff.crysllmgen.a100_sun"
    )
    module_path = Path(sun_module.__file__).resolve()
    if sha256_file(module_path) != require_hex_sha(
        args.a100_sun_module_sha256, "R03E a100_sun module SHA256"
    ):
        raise ValueError("byte-frozen R03E a100_sun module changed")
    if sun_module.MP_HULL_CACHE_SHA256 != BASE_CACHE_SHA256:
        raise ValueError("R03E base-cache identity constant changed")

    # The scientific implementation remains byte-for-byte R03E. Only its
    # immutable asset identity is rebound to the pre-Slurm completed cache.
    sun_module.MP_HULL_CACHE_SHA256 = expected_cache_sha
    sys.argv = [str(runner), *forwarded]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
