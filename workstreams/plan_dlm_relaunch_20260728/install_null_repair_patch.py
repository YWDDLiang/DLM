#!/usr/bin/env python3
"""Verify and install the Plan+DLM null-repair source overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_replace(source: Path, target: Path, *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.null-repair.",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def safe_target(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe target path: {relative}")
    target = root / candidate
    cursor = root
    for part in candidate.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"symlinked target parent is forbidden: {cursor}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve(strict=True)
    if manifest_path.is_symlink():
        raise ValueError("manifest symlinks are forbidden")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError("execution manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "plan-dlm-null-repair-source-patch@1"
        or manifest.get("identity") != "plan_dlm_null_repair_v1"
        or manifest.get("training") is not False
        or manifest.get("sun_or_mlip") is not False
    ):
        raise ValueError("unsupported execution manifest")

    root = args.project_root.resolve(strict=True)
    if not root.is_dir() or args.project_root.is_symlink():
        raise ValueError("unsafe project root")
    bundle_root = manifest_path.parent
    payload_root = (bundle_root / "payload" / "llm_plan_diff").resolve(strict=True)
    if bundle_root not in payload_root.parents:
        raise ValueError("payload escaped the extracted bundle")

    verified: list[tuple[dict[str, Any], Path, Path, str | None]] = []
    for entry in manifest["files"]:
        source = (payload_root / entry["target"]).resolve(strict=True)
        if payload_root not in source.parents:
            raise ValueError(f"payload file escaped its root: {entry['target']}")
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"unsafe payload file: {entry['target']}")
        if source.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"payload size mismatch: {entry['target']}")
        if sha256_file(source) != entry["sha256"]:
            raise ValueError(f"payload SHA256 mismatch: {entry['target']}")

        target = safe_target(root, entry["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ValueError(f"target symlink is forbidden: {entry['target']}")
        before = sha256_file(target) if target.exists() else None
        allowed = set(entry["allowed_preinstall_sha256"])
        before_token = before if before is not None else "ABSENT"
        if before_token not in allowed and before != entry["sha256"]:
            raise ValueError(f"unexpected preinstall source: {entry['target']}")
        verified.append((entry, source, target, before))

    changes: list[dict[str, Any]] = []
    for entry, source, target, before in verified:
        if before != entry["sha256"]:
            atomic_replace(source, target, mode=int(entry["mode"], 8))
        after = sha256_file(target)
        if after != entry["sha256"]:
            raise RuntimeError(f"postinstall SHA256 mismatch: {entry['target']}")
        changes.append(
            {
                "target": entry["target"],
                "before_sha256": before,
                "after_sha256": after,
                "changed": before != after,
            }
        )

    record_dir = root / ".artifacts" / "source_sync"
    record_dir.mkdir(parents=True, exist_ok=True)
    record = record_dir / f"authorized_patch_{manifest_sha256}.json"
    payload = {
        "schema_version": "plan-dlm-null-repair-install-record@1",
        "status": "complete",
        "identity": manifest["identity"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "project_root": str(root),
        "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": changes,
        "training": False,
        "sun_or_mlip": False,
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    try:
        descriptor = os.open(record, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError:
        existing = json.loads(record.read_text(encoding="utf-8"))
        if (
            existing.get("status") != "complete"
            or existing.get("manifest_sha256") != manifest_sha256
        ):
            raise
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
