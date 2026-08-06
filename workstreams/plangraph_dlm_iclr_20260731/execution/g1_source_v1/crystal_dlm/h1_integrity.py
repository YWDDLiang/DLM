"""Read-only SHA-256 verification for the frozen H1 fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Dict, Iterable

from crystal_dlm.h1_readonly_guard import default_project_root


H1_CHECKSUM_MANIFESTS: tuple[str, ...] = (
    "workstreams/r5c_reactivation_20260728/r5c_frozen_baseline_20260728.tar.gz.sha256",
    "workstreams/r5c_reactivation_20260728/r5c_reactivation_bundle_20260728.tar.gz.sha256",
    "workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/SOURCE_SHA256.txt",
    "workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/H1A2_EPOCH2_CODE_SHA256.txt",
    (
        "workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/"
        "sun_exploratory_p0_p1_v1/SOURCE_SHA256.txt"
    ),
    (
        "workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/"
        "sun_mp_completion_p0_p1_v1/SOURCE_SHA256.txt"
    ),
)

_SHA256_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")


class H1IntegrityError(ValueError):
    """Raised for a malformed checksum manifest."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_manifest(
    manifest_path: str | Path,
    *,
    allowed_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Verify one standard ``sha256sum`` manifest without changing files."""

    manifest = Path(manifest_path).expanduser().resolve()
    boundary = Path(allowed_root or manifest.parent).expanduser().resolve()
    entries: list[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SHA256_LINE.fullmatch(line)
        if match is None:
            raise H1IntegrityError(f"{manifest}:{line_number}: malformed SHA-256 line")
        expected, raw_relative = match.groups()
        relative = Path(raw_relative)
        if relative.is_absolute():
            raise H1IntegrityError(
                f"{manifest}:{line_number}: unsafe manifest path {raw_relative!r}"
            )
        target = (manifest.parent / relative).resolve(strict=False)
        if target != boundary and boundary not in target.parents:
            raise H1IntegrityError(
                f"{manifest}:{line_number}: path escapes allowed root {boundary}"
            )
        exists = target.is_file()
        actual = sha256_file(target) if exists else None
        entries.append(
            {
                "path": raw_relative,
                "expected_sha256": expected.lower(),
                "actual_sha256": actual,
                "exists": exists,
                "ok": exists and actual == expected.lower(),
            }
        )
    if not entries:
        raise H1IntegrityError(f"{manifest}: no checksum entries")
    return {
        "manifest": str(manifest),
        "entry_count": len(entries),
        "ok_count": sum(bool(entry["ok"]) for entry in entries),
        "ok": all(bool(entry["ok"]) for entry in entries),
        "entries": entries,
    }


def verify_h1_fallback(
    *,
    project_root: str | Path | None = None,
    manifests: Iterable[str] = H1_CHECKSUM_MANIFESTS,
) -> Dict[str, Any]:
    root = Path(project_root or default_project_root()).expanduser().resolve()
    reports: list[Dict[str, Any]] = []
    for relative_manifest in manifests:
        path = root / relative_manifest
        if not path.is_file():
            reports.append(
                {
                    "manifest": str(path),
                    "entry_count": 0,
                    "ok_count": 0,
                    "ok": False,
                    "error": "manifest_missing",
                    "entries": [],
                }
            )
            continue
        try:
            reports.append(verify_sha256_manifest(path, allowed_root=root))
        except (H1IntegrityError, OSError) as exc:
            reports.append(
                {
                    "manifest": str(path),
                    "entry_count": 0,
                    "ok_count": 0,
                    "ok": False,
                    "error": str(exc),
                    "entries": [],
                }
            )
    return {
        "project_root": str(root),
        "manifest_count": len(reports),
        "entry_count": sum(int(report["entry_count"]) for report in reports),
        "ok_count": sum(int(report["ok_count"]) for report in reports),
        "ok": bool(reports) and all(bool(report["ok"]) for report in reports),
        "manifests": reports,
    }


__all__ = [
    "H1_CHECKSUM_MANIFESTS",
    "H1IntegrityError",
    "sha256_file",
    "verify_h1_fallback",
    "verify_sha256_manifest",
]
