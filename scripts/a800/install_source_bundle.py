#!/usr/bin/env python3
"""Safely verify and overlay one source-only bundle onto remote white-list paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


ALLOWED_TOP_LEVEL = {"crystal_dlm", "configs", "requirements", "scripts", "tests", "docs"}
ALLOWED_ROOT_FILES = {"README.md"}
PRESERVED_TOP_LEVEL = {"data", "reference", "runs", "reports", "archive", ".secrets", ".git"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"unsafe tar path: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"links/devices are forbidden in source tar: {member.name}")
        allowed_source = (
            path.parts[0] in ALLOWED_TOP_LEVEL
            or (len(path.parts) == 1 and member.name in ALLOWED_ROOT_FILES)
        )
        if member.name != "SOURCE_MANIFEST.json" and not allowed_source:
            raise ValueError(f"tar member is outside source whitelist: {member.name}")
        if set(path.parts) & PRESERVED_TOP_LEVEL:
            raise ValueError(f"tar member touches preserved path: {member.name}")
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    target = args.target.resolve()
    digest = _sha(bundle)
    if digest != args.expected_sha256:
        raise ValueError(f"uploaded bundle SHA256 mismatch: {digest}")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        members = _safe_members(archive)
        manifest_member = archive.getmember("SOURCE_MANIFEST.json")
        source = archive.extractfile(manifest_member)
        if source is None:
            raise ValueError("embedded source manifest is unreadable")
        manifest = json.loads(source.read())
        if manifest.get("schema") != "wqcodiff_source_manifest_v1":
            raise ValueError("embedded source manifest schema mismatch")
        expected_names = {entry["path"] for entry in manifest["files"]} | {
            "SOURCE_MANIFEST.json"
        }
        actual_names = {member.name for member in members if member.isfile()}
        if actual_names != expected_names:
            raise ValueError("tar members differ from embedded manifest")
        with tempfile.TemporaryDirectory(prefix="wqcodiff-source-", dir=target) as directory:
            staging = Path(directory)
            archive.extractall(staging, members=members)
            for entry in manifest["files"]:
                source_path = staging / entry["path"]
                if source_path.stat().st_size != int(entry["bytes"]):
                    raise ValueError(f"source size mismatch: {entry['path']}")
                if _sha(source_path) != entry["sha256"]:
                    raise ValueError(f"source hash mismatch: {entry['path']}")
            for entry in manifest["files"]:
                source_path = staging / entry["path"]
                destination = target / entry["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".source-sync-tmp")
                shutil.copy2(source_path, temporary)
                os.replace(temporary, destination)
            record_dir = target / ".artifacts" / "source_sync"
            record_dir.mkdir(parents=True, exist_ok=True)
            record = record_dir / f"{digest}.json"
            with record.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        **manifest,
                        "source_root": str(target),
                        "uploaded_bundle": str(bundle),
                        "bundle_sha256": digest,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
    print(
        json.dumps(
            {
                "ok": True,
                "bundle": str(bundle),
                "bundle_sha256": digest,
                "target": str(target),
                "files_installed": len(manifest["files"]),
                "preserved_paths": sorted(PRESERVED_TOP_LEVEL),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
