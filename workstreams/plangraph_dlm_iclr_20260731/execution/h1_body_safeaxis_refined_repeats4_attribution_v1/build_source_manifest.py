#!/usr/bin/env python3
"""Create the exact immutable source inventory for R03H."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignored(path: Path, source: Path) -> bool:
    relative = path.relative_to(source)
    return (
        path.name == "SOURCE_SHA256.txt"
        or "__pycache__" in relative.parts
        or ".pytest_cache" in relative.parts
        or path.suffix in {".pyc", ".pyo"}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = source / "SOURCE_SHA256.txt"
    if output.exists():
        raise FileExistsError(output)
    files = sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file() and not ignored(path, source)
        ),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    if not files:
        raise ValueError("source inventory would be empty")
    with output.open("x", encoding="utf-8") as handle:
        for path in files:
            relative = path.relative_to(source).as_posix()
            handle.write(f"{sha256_file(path)}  {relative}\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(sha256_file(output))


if __name__ == "__main__":
    main()
