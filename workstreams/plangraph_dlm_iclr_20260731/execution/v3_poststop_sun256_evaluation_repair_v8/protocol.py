"""Fail-closed helpers for the evaluation-only H1-A2 V3 S.U.N. repair."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


ARM_ORDER = ("M00", "M10", "M01", "M11")
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


def validate_arm(value: str) -> str:
    arm = str(value)
    if arm not in ARM_ORDER:
        raise ValueError(f"arm must be one of {ARM_ORDER}")
    return arm


def rows_by_attempt(
    rows: Iterable[Mapping[str, Any]], schema: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != schema:
            raise ValueError("unexpected attempt schema")
        key = str(row.get("attempt_id") or "")
        if not key or key in result:
            raise ValueError("missing or duplicate attempt_id")
        result[key] = dict(row)
    return result


def verify_frozen_arm(
    input_manifest_path: str | Path, arm_value: str
) -> dict[str, Any]:
    """Verify one immutable v7 generation/refine/direct evidence chain."""

    arm = validate_arm(arm_value)
    manifest = read_json(input_manifest_path)
    if (
        manifest.get("schema")
        != "h1a2_v3_poststop_sun256_evaluation_repair_input_v1"
        or manifest.get("status") != "frozen"
        or manifest.get("denominator") != "raw_all_attempt"
        or int(manifest.get("attempts_per_arm", -1)) != DENOMINATOR
        or manifest.get("retry") is not False
        or manifest.get("replacement") is not False
        or manifest.get("repair") is not False
        or manifest.get("filter") is not False
        or manifest.get("rerank") is not False
    ):
        raise ValueError("frozen input manifest contract changed")
    specification = manifest["arms"][arm]
    method = str(specification["method"])
    generation_path = require_identity(
        specification["generation_jsonl"], f"{arm} generation.jsonl"
    )
    generation_report_path = require_identity(
        specification["generation_report"], f"{arm} generation report"
    )
    generation_success_path = require_identity(
        specification["generation_success"], f"{arm} generation _SUCCESS"
    )
    direct_attempts_path = require_identity(
        specification["direct_attempt_metrics"], f"{arm} direct attempts"
    )
    direct_report_path = require_identity(
        specification["direct_report"], f"{arm} direct report"
    )

    if generation_success_path.stat().st_size != 0:
        raise ValueError(f"{arm} generation _SUCCESS is not the frozen marker")
    generation = read_jsonl(generation_path)
    generation_report = read_json(generation_report_path)
    attempt_ids = [str(row.get("attempt_id") or "") for row in generation]
    expected_successes = int(specification["expected"]["generation_succeeded"])
    if (
        len(generation) != DENOMINATOR
        or [int(row.get("ordinal", -1)) for row in generation]
        != list(range(DENOMINATOR))
        or any(not attempt_id for attempt_id in attempt_ids)
        or len(set(attempt_ids)) != DENOMINATOR
        or {str(row.get("method")) for row in generation} != {method}
        or any(row.get("retry_or_replacement_used") is not False for row in generation)
        or sum(row.get("status") == "succeeded" for row in generation)
        != expected_successes
        or any(
            row.get("status") == "succeeded"
            and (
                row.get("diffusion_refinement_applied") is not True
                or int(row.get("diffusion_refinement_steps", -1)) != 800
            )
            for row in generation
        )
    ):
        raise ValueError(f"{arm} generation/refine800 evidence changed")
    if (
        generation_report.get("ok") is not True
        or generation_report.get("all_successes_diffusion_refined") is not True
        or int(generation_report.get("diffusion_steps", -1)) != 800
    ):
        raise ValueError(f"{arm} generation report changed")

    direct_attempts = read_jsonl(direct_attempts_path)
    direct_report = read_json(direct_report_path)
    direct_by_attempt = rows_by_attempt(
        direct_attempts, "crysllmgen_metric_attempt_v1"
    )
    if (
        list(direct_by_attempt) != attempt_ids
        or any(row.get("method") != method for row in direct_attempts)
        or direct_report.get("ok") is not True
        or int(direct_report.get("attempts", -1)) != DENOMINATOR
        or direct_report.get("denominator") != "all_generation_attempts"
        or direct_report.get("method") != method
    ):
        raise ValueError(f"{arm} direct attempt mapping changed")
    observed_counts = {
        "composition_valid": sum(
            bool(row.get("comp_valid")) for row in direct_attempts
        ),
        "structure_valid": sum(
            bool(row.get("struct_valid")) for row in direct_attempts
        ),
        "joint_valid": sum(bool(row.get("valid")) for row in direct_attempts),
    }
    expected_counts = {
        key: int(specification["expected"][key]) for key in observed_counts
    }
    if (
        observed_counts != expected_counts
        or int(direct_report.get("comp_valid_count", -1))
        != expected_counts["composition_valid"]
        or int(direct_report.get("struct_valid_count", -1))
        != expected_counts["structure_valid"]
        or int(direct_report.get("valid_count", -1))
        != expected_counts["joint_valid"]
    ):
        raise ValueError(f"{arm} direct metric counts changed")
    return {
        "arm": arm,
        "method": method,
        "manifest": manifest,
        "specification": specification,
        "generation": generation,
        "generation_report": generation_report,
        "attempt_ids": attempt_ids,
        "direct_attempts": direct_attempts,
        "direct_by_attempt": direct_by_attempt,
        "counts": {
            "generation_succeeded": expected_successes,
            **observed_counts,
        },
    }
