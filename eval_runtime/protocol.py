#!/usr/bin/env python3
"""Fail-closed helpers for the early-H1-A2 exact-plan reproduction."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


class ContractError(RuntimeError):
    """Raised when a frozen scientific contract is violated."""


HEX_SHA = re.compile(r"^[0-9a-f]{64}$")
PLANNER_ATTEMPTS = 1200
HISTORICAL_ATTEMPTS = 1000
SCREEN_ATTEMPTS = 256
REMAINDER_ATTEMPTS = 139
REPEATS = (0,)
ARMS = ("historical", "control", "candidate")
TRAINING_SEED = 17
SAMPLING_SEED = 17029
PROTOCOL_NAME = "h1a2c_p0_p1_sun256_exploratory_v1"
PAIRING_EXPERIMENT = "h1a2c-p0-p1-sun256-exploratory-v1"
BODY_STAGE = "r5c_exact_body_suffix_noise"
REFINER_STAGE = "crysllmgen_parent_reverse_noise_max20"


def _active_denominator() -> int:
    raw = os.environ.get("H1_ACTIVE_DENOMINATOR")
    if raw is None:
        raise RuntimeError("H1_ACTIVE_DENOMINATOR is required before protocol import")
    value = int(raw)
    if value not in (
        REMAINDER_ATTEMPTS,
        SCREEN_ATTEMPTS,
        HISTORICAL_ATTEMPTS,
        PLANNER_ATTEMPTS,
    ):
        raise ValueError("active denominator must be 139, 256, 1000, or 1200")
    return value


RAW_DENOMINATOR = _active_denominator()
DENOMINATOR = RAW_DENOMINATOR


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(canonical_json(value))


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} is not finite")
    return result


def validate_arm(value: str) -> str:
    arm = str(value)
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return arm


def validate_repeat(value: int | str) -> int:
    repeat = int(value)
    if repeat not in REPEATS:
        raise ValueError(f"unknown repeat: {repeat}")
    return repeat


def seed_derivation_ordinal(repeat: int, raw_ordinal: int) -> int:
    validate_repeat(repeat)
    ordinal = int(raw_ordinal)
    if ordinal not in range(PLANNER_ATTEMPTS):
        raise ValueError("raw ordinal outside 0..1199")
    return ordinal


def paired_seed(repeat: int, raw_ordinal: int, channel: str) -> int:
    if channel == "body":
        stage = BODY_STAGE
    elif channel == "refiner":
        stage = REFINER_STAGE
    else:
        raise ValueError("unknown seed channel")
    payload = {
        "protocol": PROTOCOL_NAME,
        "pairing_experiment": PAIRING_EXPERIMENT,
        "training_seed": TRAINING_SEED,
        "sampling_seed": SAMPLING_SEED,
        "ordinal": seed_derivation_ordinal(repeat, raw_ordinal),
        "stage": stage,
    }
    return int.from_bytes(
        hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()[:8],
        "big",
    ) & ((1 << 63) - 1)


def attempt_id(repeat: int, arm: str, ordinal: int) -> str:
    validate_repeat(repeat)
    validate_arm(arm)
    index = int(ordinal)
    if index not in range(RAW_DENOMINATOR):
        raise ValueError("attempt ordinal outside active denominator")
    return f"h1a2-exactplan-r{repeat}-{arm}-{index:04d}"


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ContractError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def identity(path: str | Path) -> dict[str, Any]:
    location = Path(path).resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    return {"path": str(location), "bytes": location.stat().st_size, "sha256": sha256_file(location)}


def require_hex_sha(value: str, label: str) -> str:
    observed = str(value).strip().lower()
    if HEX_SHA.fullmatch(observed) is None:
        raise ContractError(f"{label} must be one lowercase SHA-256")
    return observed


def require_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    location = Path(path).resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    if sha256_file(location) != require_hex_sha(expected_sha256, label):
        raise ContractError(f"{label} identity changed")
    return location


def write_source_manifest(
    source_dir: str | Path,
    relative_files: Iterable[str | Path],
) -> Path:
    """Write the complete, relative-path manifest consumed by query jobs."""

    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    normalized: list[Path] = []
    seen: set[str] = set()
    for raw in relative_files:
        relative = Path(raw)
        text = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in str(raw)
            or text == "SOURCE_SHA256.txt"
        ):
            raise ContractError(f"unsafe source-manifest path: {raw}")
        if text in seen:
            raise ContractError(f"duplicate source-manifest path: {text}")
        location = (source / relative).resolve()
        try:
            location.relative_to(source)
        except ValueError as exc:
            raise ContractError(f"source-manifest path escapes source: {raw}") from exc
        if not location.is_file():
            raise FileNotFoundError(location)
        seen.add(text)
        normalized.append(relative)
    if not normalized:
        raise ContractError("source manifest cannot be empty")
    manifest = source / "SOURCE_SHA256.txt"
    with manifest.open("x", encoding="utf-8", newline="\n") as handle:
        for relative in sorted(normalized, key=lambda value: value.as_posix()):
            handle.write(f"{sha256_file(source / relative)}  {relative.as_posix()}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def ordered_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    ordinal_field: str,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    count = RAW_DENOMINATOR if expected_count is None else int(expected_count)
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row[ordinal_field]))
    if len(ordered) != count or [int(row.get(ordinal_field, -1)) for row in ordered] != list(range(count)):
        raise ContractError(f"{ordinal_field} coverage changed")
    return ordered


def require_source_manifest(source_dir: str | Path, expected_manifest_sha256: str) -> Path:
    source = Path(source_dir).resolve()
    manifest = require_file(source / "SOURCE_SHA256.txt", expected_manifest_sha256, "execution source manifest")
    expected: dict[str, str] = {}
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        digest, separator, relative = raw.partition("  ")
        if not separator:
            raise ContractError(f"manifest line {line_number} is malformed")
        require_hex_sha(digest, f"manifest line {line_number}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in relative:
            raise ContractError(f"manifest line {line_number} is unsafe")
        expected[relative_path.as_posix()] = digest
    observed = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
        and path.resolve() != manifest.resolve()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if observed != set(expected):
        raise ContractError("source file set changed")
    for relative, digest in expected.items():
        if sha256_file(source / relative) != digest:
            raise ContractError(f"source file changed: {relative}")
    return manifest


def validate_config(config: Mapping[str, Any]) -> None:
    planner = config.get("planner") or {}
    body = config.get("body") or {}
    refiner = config.get("refiner") or {}
    evaluation = config.get("evaluation") or {}
    resources = config.get("resources") or {}
    if (
        config.get("schema") != "h1a2_epoch2_exactplan1200_h1a2_r03_fullsun_config_v1"
        or int(config.get("planner_attempts", -1)) != PLANNER_ATTEMPTS
        or int(config.get("historical_frozen_attempts", -1)) != HISTORICAL_ATTEMPTS
        or int(planner.get("world_size", -1)) != 2
        or int(planner.get("batch_size", -1)) != 4
        or int(planner.get("seed", -1)) != 17
        or planner.get("sampler_sha256") != "d38743f2f647d798800724b09537fbe492706805c00d7ee34c5ca8d74e39adc8"
        or body.get("checkpoint_role") != "historical_B0_R5C_exact_length"
        or body.get("paired_arms") != ["control_D1", "candidate_D2_SAFE_AXIS"]
        or int(body.get("max_batch_size", -1)) != 8
        or int(refiner.get("diffusion_steps", -1)) != 800
        or int(refiner.get("effective_batch_size", -1)) != 1
        or evaluation.get("stability_scope") != "all_reconstructed_before_NU_intersection"
        or int(resources.get("a800_total", -1)) != 2
        or int(resources.get("cpus_total", -1)) != 32
        or any(config.get(key) is not False for key in ("retry", "replacement", "repair", "filter", "rerank", "training", "rl"))
    ):
        raise ContractError("experiment contract changed")
