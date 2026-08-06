#!/usr/bin/env python3
"""Build a deterministic source-only tar with an embedded file/hash manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
from pathlib import Path
from typing import Iterable


FORBIDDEN_PARTS = {
    ".git",
    ".secrets",
    "archive",
    "reference",
    "data",
    "runs",
    "reports",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidates(root: Path) -> Iterable[Path]:
    fixed = (
        root / "README.md",
        root / "crystal_dlm/__init__.py",
        root / "crystal_dlm/crysllmgen_text.py",
        root / "scripts/__init__.py",
        root / "scripts/run_mattergen_sun_eval.py",
        root / "requirements/wqcodiff-py310.txt",
        root / "requirements/wqcodiff-constraints.txt",
        root / "scripts/a800/README.md",
    )
    yield from fixed
    yield from sorted(
        path
        for path in (root / "crystal_dlm/wqcodiff").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (
            path.suffix in {".py", ".sh", ".json", ".md", ".txt", ".yaml", ".yml"}
            or path.name == "LICENSE"
        )
    )
    yield from sorted(
        path
        for path in (root / "configs/experiments/wyckoff_codiffusion").glob("*")
        if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}
    )
    yield from sorted((root / "docs/experiment_program").glob("*.md"))
    yield from sorted((root / "scripts/a800").glob("*.py"))
    yield from sorted((root / "scripts/a800").glob("*.sh"))
    yield from sorted((root / "scripts/a800").glob("*.yaml"))
    yield from sorted((root / "scripts/a800").glob("*.json"))
    yield from sorted((root / "tests").glob("test_wqcodiff_*.py"))
    yield from sorted((root / "tests").glob("test_crysllmgen_*.py"))


def _files(root: Path) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in _candidates(root):
        if not path.is_file():
            raise FileNotFoundError(f"registered source file is missing: {path}")
        if path.is_symlink():
            raise ValueError(f"source bundle refuses symlink: {path}")
        relative = path.relative_to(root).as_posix()
        parts = set(Path(relative).parts)
        if parts & FORBIDDEN_PARTS:
            raise ValueError(f"forbidden source path selected: {relative}")
        if relative not in seen:
            seen.add(relative)
            result.append(path)
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="wqcodiff_source")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _files(root)
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
            "mode": oct(path.stat().st_mode & 0o777),
        }
        for path in files
    ]
    embedded = {
        "schema": "wqcodiff_source_manifest_v1",
        "source_root": ".",
        "files": entries,
        "exclusions": sorted(FORBIDDEN_PARTS),
    }
    embedded_bytes = (json.dumps(embedded, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    bundle = output_dir / f"{args.name}.tar.gz"
    if bundle.exists():
        raise FileExistsError(bundle)
    with bundle.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path, entry in zip(files, entries):
                    info = archive.gettarinfo(str(path), arcname=entry["path"])
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
                info = tarfile.TarInfo("SOURCE_MANIFEST.json")
                info.size = len(embedded_bytes)
                info.mode = 0o644
                info.uid = info.gid = info.mtime = 0
                archive.addfile(info, __import__("io").BytesIO(embedded_bytes))
    bundle_sha = _sha(bundle)
    external = {
        **embedded,
        "source_root": str(root),
        "bundle": str(bundle),
        "bundle_bytes": bundle.stat().st_size,
        "bundle_sha256": bundle_sha,
    }
    manifest = output_dir / f"{args.name}.manifest.json"
    sha_file = output_dir / f"{args.name}.sha256"
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(external, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with sha_file.open("x", encoding="utf-8") as handle:
        handle.write(f"{bundle_sha}  {bundle.name}\n")
    print(
        json.dumps(
            {
                "schema": "wqcodiff_source_bundle_v1",
                "bundle": str(bundle),
                "bundle_sha256": bundle_sha,
                "bundle_bytes": bundle.stat().st_size,
                "manifest": str(manifest),
                "sha256_file": str(sha_file),
                "files": len(entries),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
