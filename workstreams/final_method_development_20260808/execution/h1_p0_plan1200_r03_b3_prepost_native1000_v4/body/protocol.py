"""Fail-closed helpers for the immutable V4 Plan1200 R03/B3 panel."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


ARMS = ("R03", "B3")
REPEATS = (0, 1, 2)
DENOMINATOR = 1000
PAIRED_SEED_NAMESPACE = "20260810_h1_p0_plan1200_r03_b3_prepost_repeats3_v1"
HEX_SHA = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def paired_seed(repeat: int, ordinal: int, channel: str) -> int:
    repeat = validate_repeat(repeat)
    if ordinal not in range(DENOMINATOR):
        raise ValueError("ordinal outside frozen cohort")
    if channel not in {"body", "refiner"}:
        raise ValueError("unknown seed channel")
    material = (
        f"{PAIRED_SEED_NAMESPACE}|"
        f"repeat={repeat}|ordinal={ordinal}|channel={channel}"
    )
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big") % (
        2**63 - 1
    )


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
                raise ValueError(f"{path}:{line_number}: expected one object")
            rows.append(value)
    return rows


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
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


def require_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    location = Path(path).resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    expected = require_hex_sha(expected_sha256, label)
    observed = sha256_file(location)
    if observed != expected:
        raise ValueError(f"{label} changed: expected={expected} observed={observed}")
    return location


def ordered_rows(
    rows: Iterable[Mapping[str, Any]], *, ordinal_field: str
) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row[ordinal_field]))
    if (
        len(ordered) != DENOMINATOR
        or [int(row.get(ordinal_field, -1)) for row in ordered]
        != list(range(DENOMINATOR))
    ):
        raise ValueError(f"{ordinal_field} coverage changed")
    return ordered


def validate_frozen_cohort_row(
    row: Mapping[str, Any], *, repeat: int, ordinal: int
) -> dict[str, Any]:
    """Validate the producer's parse-success evidence without inventing fields.

    V3's cohort producer selected rows from ``parsed_plans.jsonl`` and persisted
    both ``parsed_plan`` and ``plan_state``.  It intentionally did not copy the
    raw-attempt ledger's top-level ``parsed`` boolean.  This validator encodes
    the actual frozen producer schema and is shared by preflight and runtime so
    those two gates cannot drift again.
    """

    repeat = validate_repeat(repeat)
    if ordinal not in range(DENOMINATOR):
        raise ValueError("cohort ordinal outside frozen denominator")
    parsed_plan = row.get("parsed_plan")
    plan_state = row.get("plan_state")
    prompt = row.get("body_prompt")
    prompt_sha = row.get("body_prompt_sha256")
    expected_attempt_id = f"p0-plan1200-r{repeat}-{ordinal:04d}"
    if (
        int(row.get("repeat", -1)) != repeat
        or int(row.get("cohort_ordinal", -1)) != ordinal
        or str(row.get("attempt_id", "")) != expected_attempt_id
        or int(row.get("planner_candidate_ordinal", -1)) < 0
        or not isinstance(parsed_plan, Mapping)
        or not isinstance(plan_state, Mapping)
        or not isinstance(prompt, str)
        or not prompt
        or prompt_sha != sha256_text(prompt)
        or row.get("raw_rich_seven_line_forwarded") is not False
        or row.get("canonical_charge_bucket_visible") is not True
        or row.get("body_prompt_contract")
        != "historical_r5c_plan_state_json_exact_length"
        or "plan_state:" not in prompt
        or '"charge_bucket"' not in prompt
        or not prompt.endswith("dynamic_crystal_body:")
    ):
        raise ValueError(f"frozen cohort contract changed at ordinal {ordinal}")
    return dict(plan_state)


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


def attempt_id(arm: str, repeat: int, ordinal: int, stage: str) -> str:
    arm = validate_arm(arm)
    repeat = validate_repeat(repeat)
    if stage not in {"pre_model494", "post_model494"}:
        raise ValueError("invalid evaluation stage")
    return f"h1-plan1200-{arm.lower()}-r{repeat}-{stage}-{int(ordinal):04d}"


def require_source_manifest(source_dir: str | Path, expected_manifest_sha256: str) -> Path:
    source = Path(source_dir).resolve()
    manifest = require_file(
        source / "SOURCE_SHA256.txt", expected_manifest_sha256, "execution source manifest"
    )
    listed: set[str] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise ValueError(f"manifest line {line_number} is malformed")
        expected, relative = pieces
        require_hex_sha(expected, f"manifest line {line_number}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"manifest line {line_number} has unsafe path")
        listed.add(relative_path.as_posix())
        if sha256_file(source / relative_path) != expected:
            raise ValueError(f"source file changed: {relative}")
    observed = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_SHA256.txt"
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if listed != observed:
        raise ValueError(
            f"source file set changed: missing={sorted(listed-observed)}, "
            f"extra={sorted(observed-listed)}"
        )
    return manifest


def validate_config(config: Mapping[str, Any]) -> None:
    body = config.get("body") or {}
    refiner = config.get("refiner") or {}
    evaluation = config.get("evaluation") or {}
    statistics = config.get("statistics") or {}
    if (
        config.get("schema")
        != "h1_p0_plan1200_r03_b3_prepost_native1000_body_config_v4"
        or config.get("paired_seed_namespace") != PAIRED_SEED_NAMESPACE
        or int(config.get("denominator", -1)) != DENOMINATOR
        or int(config.get("repeat_count", -1)) != 3
        or set((body.get("models") or {}).keys()) != set(ARMS)
        or body.get("prompt_contract") != "historical_r5c_plan_state_json_exact_length"
        or body.get("schedule") != "d2_safe_axis"
        or body.get("exact_length_generation") is not True
        or float(body.get("temperature", -1)) != 0.7
        or float(body.get("cfg_scale", -1)) != 0.0
        or int(body.get("max_batch_size", -1)) != 8
        or int(refiner.get("timesteps", -1)) != 1000
        or int(refiner.get("diffusion_steps", -1)) != 800
        or int(refiner.get("effective_batch_size", -1)) != 1
        or evaluation.get("stages") != ["pre_model494", "post_model494"]
        or evaluation.get("headline_sun_denominator")
        != "reconstructed_structures_exact_legacy"
        or evaluation.get("secondary_sun_denominator") != "all_1000_attempts"
        or int(statistics.get("hierarchical_paired_bootstrap_draws", -1)) != 50000
        or any(config.get(key) is not False for key in (
            "retry", "replacement", "repair", "filter", "rerank",
            "automatic_training", "automatic_promotion", "automatic_rl",
        ))
    ):
        raise ValueError("P0 Plan1200 R03/B3 pre/post contract changed")
