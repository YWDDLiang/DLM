"""Fail-closed helpers for the post-stop H1-A2 V3 S.U.N. diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


ARM_ORDER = ("M00", "M10", "M01", "M11")
ARM_COMPONENTS = {
    "M00": ("P0", "B0"),
    "M10": ("Pstar", "B0"),
    "M01": ("P0", "B2"),
    "M11": ("Pstar", "B2"),
}
HEX_SHA = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected one JSON object")
            rows.append(value)
    return rows


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(
    path: str | Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def require_hex_sha(value: str, label: str) -> str:
    observed = str(value).strip().lower()
    if HEX_SHA.fullmatch(observed) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return observed


def require_sha(path: str | Path, expected: str, label: str) -> Path:
    location = Path(path).resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    expected_sha = require_hex_sha(expected, label)
    observed = sha256_file(location)
    if observed != expected_sha:
        raise ValueError(
            f"{label} changed: expected={expected_sha} observed={observed}"
        )
    return location


def _manifest_entries(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise ValueError(f"{path}:{line_number}: malformed manifest entry")
        expected, relative = pieces
        require_hex_sha(expected, f"{path}:{line_number}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{path}:{line_number}: unsafe relative path")
        entries.append((expected, relative))
    if not entries:
        raise ValueError(f"{path}: empty manifest")
    return entries


def _is_python_bytecode_cache(path: Path, source: Path) -> bool:
    relative = path.relative_to(source)
    return "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}


def require_source_manifest(
    source_dir: str | Path, expected_manifest_sha256: str
) -> Path:
    source = Path(source_dir).resolve()
    manifest = require_sha(
        source / "SOURCE_SHA256.txt",
        expected_manifest_sha256,
        "execution source manifest",
    )
    entries = _manifest_entries(manifest)
    listed = {relative for _, relative in entries}
    observed = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if (
            path.is_file()
            and path.name != "SOURCE_SHA256.txt"
            and not _is_python_bytecode_cache(path, source)
        )
    }
    if listed != observed:
        raise ValueError(
            "execution source file set changed: "
            f"missing={sorted(listed-observed)}, extra={sorted(observed-listed)}"
        )
    for expected, relative in entries:
        require_sha(source / relative, expected, f"source file {relative}")
    return manifest


def require_runtime_manifest(
    project_root: str | Path, source_dir: str | Path
) -> Path:
    project = Path(project_root).resolve()
    if not project.is_dir():
        raise FileNotFoundError(project)
    source = Path(source_dir).resolve()
    runtime = source / "runtime"
    manifest = source / "RUNTIME_REQUIRED_SHA256.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not runtime.is_dir():
        raise FileNotFoundError(runtime)
    for expected, relative in _manifest_entries(manifest):
        require_sha(runtime / relative, expected, f"runtime file {relative}")
    return manifest


def validate_arm(value: str) -> str:
    arm = str(value)
    if arm not in ARM_ORDER:
        raise ValueError(f"arm must be one of {ARM_ORDER}")
    return arm


def attempt_id(arm: str, sample_idx: int) -> str:
    return f"h1a2-v3-poststop-sun256:{int(sample_idx):04d}:{validate_arm(arm)}"


def rows_by_attempt(path: str | Path, schema: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("schema") != schema:
            raise ValueError(f"{path}: unexpected schema")
        key = str(row.get("attempt_id") or "")
        if not key or key in result:
            raise ValueError(f"{path}: missing or duplicate attempt_id")
        result[key] = row
    return result
