"""Fail-closed helpers for the immutable V4 CrysLLMGen-native supplement."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


ARMS = ("R03", "B3")
REPEATS = (0, 1, 2)
RAW_ATTEMPTS = 1200
PREFIX_COUNT = 1000
NATIVE_DENOMINATOR = 1000
SEED_NAMESPACE = "20260810_h1_p0_plan1200_r03_b3_prepost_repeats3_v1"


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


def candidate_seed(repeat: int, candidate_rank: int, channel: str) -> int:
    repeat = validate_repeat(repeat)
    if candidate_rank not in range(RAW_ATTEMPTS):
        raise ValueError("candidate rank outside frozen raw-1200 draw")
    if channel not in {"body", "refiner"}:
        raise ValueError("unknown seed channel")
    material = (
        f"{SEED_NAMESPACE}|repeat={repeat}|ordinal={candidate_rank}|channel={channel}"
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
    with location.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(
    path: str | Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8", newline="\n") as handle:
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


def ordered_candidate_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows), key=lambda row: int(row["candidate_rank"])
    )
    if (
        not PREFIX_COUNT <= len(ordered) <= RAW_ATTEMPTS
        or [int(row.get("candidate_rank", -1)) for row in ordered]
        != list(range(len(ordered)))
    ):
        raise ValueError("candidate-rank coverage changed")
    return ordered


def validate_frozen_candidate_row(
    row: Mapping[str, Any], *, repeat: int, candidate_rank: int
) -> dict[str, Any]:
    """Validate one parse-success candidate using the producer's real schema.

    ``plans_for_dlm.jsonl`` persists ``parsed_plan`` and ``plan_state`` but not
    the raw-attempt ledger's top-level ``parsed`` boolean.  Candidate freezing
    preserved that schema.  Both preflight and reserve generation use this
    validator so a consumer cannot silently reintroduce the V3 mismatch.
    """

    repeat = validate_repeat(repeat)
    if candidate_rank not in range(RAW_ATTEMPTS):
        raise ValueError("candidate rank outside frozen raw-1200 draw")
    parsed_plan = row.get("parsed_plan")
    plan_state = row.get("plan_state")
    prompt = row.get("body_prompt")
    expected_partition = (
        "v3_prefix" if candidate_rank < PREFIX_COUNT else "frozen_reserve"
    )
    if (
        int(row.get("repeat", -1)) != repeat
        or int(row.get("candidate_rank", -1)) != candidate_rank
        or str(row.get("candidate_id", ""))
        != f"p0-native-r{repeat}-{candidate_rank:04d}"
        or int(row.get("planner_candidate_ordinal", -1)) not in range(RAW_ATTEMPTS)
        or not isinstance(parsed_plan, Mapping)
        or not isinstance(plan_state, Mapping)
        or row.get("plan_state_sha256") != canonical_sha256(plan_state)
        or not isinstance(prompt, str)
        or not prompt
        or row.get("body_prompt_sha256") != sha256_text(prompt)
        or row.get("body_prompt_contract")
        != "historical_r5c_plan_state_json_exact_length"
        or row.get("raw_rich_seven_line_forwarded") is not False
        or row.get("canonical_charge_bucket_visible") is not True
        or int(row.get("body_noise_seed", -1))
        != candidate_seed(repeat, candidate_rank, "body")
        or int(row.get("refiner_noise_seed", -1))
        != candidate_seed(repeat, candidate_rank, "refiner")
        or row.get("candidate_partition") != expected_partition
        or "plan_state:" not in prompt
        or '"charge_bucket"' not in prompt
        or not prompt.endswith("dynamic_crystal_body:")
    ):
        raise ValueError(
            f"frozen candidate contract changed at repeat {repeat} "
            f"rank {candidate_rank}"
        )
    return dict(plan_state)


def first_success_ranks(
    attempts: Mapping[int, Mapping[str, Any]],
    candidate_count: int,
    required: int = NATIVE_DENOMINATOR,
) -> list[int]:
    """Return the upstream-style prefix of successful candidates in frozen order."""

    if sorted(attempts) != list(range(candidate_count)):
        raise ValueError("candidate attempt ledger is not continuous")
    successes = [
        rank
        for rank in range(candidate_count)
        if attempts[rank].get("status") == "succeeded"
    ]
    if len(successes) < required:
        raise ValueError(
            f"only {len(successes)} body successes are available; {required} required"
        )
    return successes[:required]


def identity(path: str | Path) -> dict[str, Any]:
    location = Path(path).resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": sha256_file(location),
    }
