#!/usr/bin/env python3
"""Byte-verify and invoke one proven official-S evaluator module."""

from __future__ import annotations

import argparse
import hashlib
import runpy
import sys
from pathlib import Path


FROZEN_SOURCE = Path(
    "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/"
    "workstreams/final_method_development_20260808/execution/"
    "h1_a2_r03_prepost_sun256_official_recovery_v1_terminal_contract_repair_v3"
)
FROZEN_MANIFEST_SHA256 = (
    "7c470d346ca374b8fd42d3c14e130e1c79257246847afa821248a4c5482f20c2"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_source() -> None:
    manifest = FROZEN_SOURCE / "SOURCE_SHA256.txt"
    if sha256_file(manifest) != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen official evaluator manifest changed")
    listed: set[str] = set()
    for line_number, raw in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        digest, separator, relative = raw.partition("  ")
        path = Path(relative)
        if (
            not separator
            or len(digest) != 64
            or path.is_absolute()
            or ".." in path.parts
            or sha256_file(FROZEN_SOURCE / path) != digest
        ):
            raise ValueError(f"frozen official evaluator changed at line {line_number}")
        listed.add(path.as_posix())
    observed = {
        path.relative_to(FROZEN_SOURCE).as_posix()
        for path in FROZEN_SOURCE.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_SHA256.txt"
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if listed != observed:
        raise ValueError(
            f"frozen official evaluator file set changed: "
            f"missing={sorted(listed-observed)} extra={sorted(observed-listed)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--script-sha256", required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    verify_frozen_source()
    relative = Path(args.script)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe frozen evaluator script path")
    script = (FROZEN_SOURCE / relative).resolve()
    if not script.is_relative_to(FROZEN_SOURCE.resolve()):
        raise ValueError("frozen evaluator script escapes source")
    if sha256_file(script) != args.script_sha256:
        raise ValueError("requested frozen evaluator script changed")
    sys.path.insert(0, str(FROZEN_SOURCE))
    sys.modules.pop("protocol", None)
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
