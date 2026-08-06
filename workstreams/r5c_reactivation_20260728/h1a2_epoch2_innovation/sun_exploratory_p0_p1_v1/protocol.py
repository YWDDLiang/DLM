"""Small fail-closed helpers for the exploratory paired S.U.N. DAG."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


ARM_ORDER = ("P0", "P1")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
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


def write_jsonl_exclusive(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def resolve_project_path(project_root: str | Path, value: str | Path) -> Path:
    location = Path(value)
    if not location.is_absolute():
        location = Path(project_root) / location
    return location.resolve()


def require_sha(path: str | Path, expected: str, label: str) -> Path:
    location = Path(path).resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    observed = sha256_file(location)
    if observed != expected:
        raise ValueError(
            f"{label} changed: expected={expected}, observed={observed}, path={location}"
        )
    return location


def require_hex_sha(value: str, label: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def require_source_manifest(
    source_dir: str | Path, expected_manifest_sha256: str
) -> Path:
    require_hex_sha(expected_manifest_sha256, "execution source manifest")
    manifest = Path(source_dir).resolve() / "SOURCE_SHA256.txt"
    require_sha(manifest, expected_manifest_sha256, "execution source manifest")
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise ValueError(f"{manifest}:{line_number}: invalid SHA256 entry")
        expected, relative = pieces
        require_hex_sha(expected, f"{manifest}:{line_number}")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"{manifest}:{line_number}: unsafe relative path")
        entries.append((expected, relative))
    if not entries:
        raise ValueError("execution source manifest is empty")
    listed = {relative for _, relative in entries}
    observed = {
        path.relative_to(Path(source_dir).resolve()).as_posix()
        for path in Path(source_dir).resolve().rglob("*")
        if path.is_file() and path.name != "SOURCE_SHA256.txt"
    }
    if observed != listed:
        raise ValueError(
            "execution source file set changed: "
            f"missing={sorted(listed - observed)}, extra={sorted(observed - listed)}"
        )
    for expected, relative in entries:
        require_sha(Path(source_dir) / relative, expected, f"source file {relative}")
    return manifest


def require_runtime_manifest(project_root: str | Path, source_dir: str | Path) -> Path:
    manifest = Path(source_dir).resolve() / "RUNTIME_REQUIRED_SHA256.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    entries = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise ValueError(f"{manifest}:{line_number}: invalid runtime SHA256 entry")
        expected, relative = pieces
        require_hex_sha(expected, f"{manifest}:{line_number}")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"{manifest}:{line_number}: unsafe runtime path")
        entries.append((expected, relative))
    if not entries:
        raise ValueError("runtime manifest is empty")
    for expected, relative in entries:
        require_sha(
            Path(project_root).resolve() / relative,
            expected,
            f"runtime file {relative}",
        )
    return manifest


def validate_arm(arm: str) -> str:
    value = str(arm)
    if value not in ARM_ORDER:
        raise ValueError(f"arm must be one of {ARM_ORDER}")
    return value


def plan_body_eligible(row: Mapping[str, Any]) -> tuple[bool, str]:
    plan = row.get("plan_state")
    if not bool(row.get("parsed")) or not isinstance(plan, Mapping):
        return False, "planner_parse_failed"
    try:
        num_atoms = int(plan["N"])
        elements = [str(value) for value in plan["elements"]]
        counts = [int(value) for value in plan["counts"]]
    except Exception:
        return False, "planner_plan_state_incomplete"
    if not 1 <= num_atoms <= 20:
        return False, "planner_N_out_of_range"
    if (
        not elements
        or len(elements) != len(counts)
        or any(count <= 0 for count in counts)
        or sum(counts) != num_atoms
    ):
        return False, "planner_composition_shape_invalid"
    return True, ""


def rows_by_attempt(path: str | Path, *, schema: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("schema") != schema:
            raise ValueError(f"{path}: unexpected schema {row.get('schema')!r}")
        attempt_id = str(row.get("attempt_id", ""))
        if not attempt_id or attempt_id in result:
            raise ValueError(f"{path}: missing or duplicate attempt_id")
        result[attempt_id] = row
    return result


__all__ = [
    "ARM_ORDER",
    "canonical_sha256",
    "plan_body_eligible",
    "read_json",
    "read_jsonl",
    "require_hex_sha",
    "require_sha",
    "require_runtime_manifest",
    "require_source_manifest",
    "resolve_project_path",
    "rows_by_attempt",
    "sha256_file",
    "validate_arm",
    "write_json_exclusive",
    "write_jsonl_exclusive",
]
