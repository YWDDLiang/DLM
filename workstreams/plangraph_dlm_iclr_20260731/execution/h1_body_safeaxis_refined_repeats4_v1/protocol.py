"""Fail-closed helpers for the H1 R03E repeated-refiner diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


ARMS = ("control", "candidate")
REPEATS = (0, 1, 2, 3)
DENOMINATOR = 256
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


def require_identity(specification: Mapping[str, Any], label: str) -> Path:
    path = Path(str(specification.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = require_hex_sha(str(specification.get("sha256", "")), label)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} changed: expected={expected} observed={observed}"
        )
    if "bytes" in specification and path.stat().st_size != int(
        specification["bytes"]
    ):
        raise ValueError(f"{label} byte size changed")
    return path


def validate_arm(value: str) -> str:
    arm = str(value)
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    return arm


def validate_repeat(value: int | str) -> int:
    repeat = int(value)
    if repeat not in REPEATS:
        raise ValueError(f"repeat must be one of {REPEATS}")
    return repeat


def ordered_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    ordinal_field: str,
    expected_count: int = DENOMINATOR,
) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row[ordinal_field]))
    if (
        len(ordered) != int(expected_count)
        or [int(row.get(ordinal_field, -1)) for row in ordered]
        != list(range(int(expected_count)))
    ):
        raise ValueError(f"{ordinal_field} coverage changed")
    return ordered


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
        entries.append((expected, relative_path.as_posix()))
    if not entries:
        raise ValueError(f"{path}: empty manifest")
    return entries


def _ignored_source_file(path: Path, source: Path) -> bool:
    relative = path.relative_to(source)
    return (
        "__pycache__" in relative.parts
        or ".pytest_cache" in relative.parts
        or path.suffix in {".pyc", ".pyo"}
    )


def require_source_manifest(
    source_dir: str | Path, expected_manifest_sha256: str
) -> Path:
    source = Path(source_dir).resolve()
    manifest = source / "SOURCE_SHA256.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    expected_manifest = require_hex_sha(
        expected_manifest_sha256, "execution source manifest"
    )
    observed_manifest = sha256_file(manifest)
    if observed_manifest != expected_manifest:
        raise ValueError(
            "execution source manifest changed: "
            f"expected={expected_manifest} observed={observed_manifest}"
        )
    entries = _manifest_entries(manifest)
    listed = {relative for _, relative in entries}
    observed = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if (
            path.is_file()
            and path.name != "SOURCE_SHA256.txt"
            and not _ignored_source_file(path, source)
        )
    }
    if listed != observed:
        raise ValueError(
            "execution source file set changed: "
            f"missing={sorted(listed-observed)}, extra={sorted(observed-listed)}"
        )
    for expected, relative in entries:
        path = source / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"execution source file changed: {relative}")
    return manifest


def validate_config(config: Mapping[str, Any]) -> None:
    protocol = config.get("protocol") or {}
    firewall = config.get("decision_firewall") or {}
    if (
        config.get("schema")
        != "h1_body_safeaxis_refined_repeats4_config_v1"
        or config.get("status") != "user_authorized_single_factor_diagnostic"
        or protocol.get("repeats") != 4
        or protocol.get("repeat_ids") != [0, 1, 2, 3]
        or protocol.get("new_scientific_seed_per_repeat") is not False
        or protocol.get("same_frozen_refiner_seed_ledger_each_repeat") is not True
        or int(protocol.get("raw_attempts_per_arm_per_repeat", -1)) != DENOMINATOR
        or protocol.get("denominator") != "raw_all_attempt"
        or (config.get("sun") or {}).get("mp_api_enabled") is not False
        or firewall.get("formal_g3") is not False
        or firewall.get("automatic_promotion") is not False
        or firewall.get("automatic_training") is not False
        or firewall.get("automatic_downstream") is not False
        or firewall.get("checkpoint_reselection") is not False
    ):
        raise ValueError("R03E frozen protocol or decision firewall changed")


def attempt_id(repeat: int, arm: str, ordinal: int) -> str:
    return f"h1-r03e-r{validate_repeat(repeat)}-{validate_arm(arm)}-{int(ordinal):04d}"
