"""Deterministically pool immutable per-seed generation artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import AttemptStatus, write_json_exclusive


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationPoolConfig:
    inputs: tuple[str, ...]
    output_jsonl: str
    manifest_json: str
    expected_method: str | None = None
    expected_total: int | None = None
    expected_training_seed_counts: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("generation pooling requires at least one input")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("generation pooling inputs must be unique")
        if self.expected_total is not None and self.expected_total <= 0:
            raise ValueError("expected_total must be positive")
        seeds = [int(seed) for seed, _ in self.expected_training_seed_counts]
        if len(seeds) != len(set(seeds)):
            raise ValueError("expected training-seed counts must be unique")
        if any(int(count) <= 0 for _, count in self.expected_training_seed_counts):
            raise ValueError("expected training-seed counts must be positive")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    location = path.resolve()
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": _sha256_file(location),
    }


def _canonical_line(row: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def _parse_record(
    payload: Any,
    *,
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}:{line_number}: generation record is not an object")
    row = dict(payload)
    if row.get("schema") != "wqcodiff_generation_attempt_v1":
        raise ValueError(f"{path}:{line_number}: invalid generation schema")
    for field in (
        "attempt_id",
        "pair_id",
        "method",
        "training_seed",
        "sampling_seed",
        "ordinal",
        "status",
    ):
        if field not in row or row[field] in (None, ""):
            raise ValueError(f"{path}:{line_number}: missing generation field {field}")
    try:
        status = AttemptStatus(str(row["status"]))
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number}: unknown generation status {row['status']}"
        ) from exc
    if not status.terminal:
        raise ValueError(f"{path}:{line_number}: pooled generation is not terminal")
    row["training_seed"] = int(row["training_seed"])
    row["sampling_seed"] = int(row["sampling_seed"])
    row["ordinal"] = int(row["ordinal"])
    if row["ordinal"] < 0:
        raise ValueError(f"{path}:{line_number}: ordinal must be non-negative")
    return row


def pool_generation_artifacts(config: GenerationPoolConfig) -> dict[str, Any]:
    """Pool complete per-seed JSONL files without changing attempt content.

    The pooled order is independent of the order in which input paths are
    supplied.  Any duplicate attempt or method-independent pair identity is a
    fatal error because it would alter uniqueness and paired denominators.
    """

    output = Path(config.output_jsonl).resolve()
    manifest = Path(config.manifest_json).resolve()
    if output == manifest:
        raise ValueError("pooled output and manifest must be different files")
    if output.exists() or manifest.exists():
        raise FileExistsError("pooled generation output/manifest is immutable")

    input_paths = [Path(value).resolve() for value in config.inputs]
    if output in input_paths or manifest in input_paths:
        raise ValueError("pooled output/manifest cannot also be an input")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    attempts: set[str] = set()
    pairs: set[str] = set()
    source_artifacts: list[dict[str, Any]] = []
    for path in sorted(input_paths, key=lambda value: str(value)):
        source_artifacts.append(_identity(path))
        input_records = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                row = _parse_record(payload, path=path, line_number=line_number)
                attempt_id = str(row["attempt_id"])
                pair_id = str(row["pair_id"])
                if attempt_id in attempts:
                    raise ValueError(f"duplicate pooled attempt_id: {attempt_id}")
                if pair_id in pairs:
                    raise ValueError(f"duplicate pooled pair_id: {pair_id}")
                attempts.add(attempt_id)
                pairs.add(pair_id)
                rows.append(row)
                input_records += 1
        if input_records == 0:
            raise ValueError(f"generation pooling input is empty: {path}")
        source_artifacts[-1]["records"] = input_records

    methods = sorted({str(row["method"]) for row in rows})
    if len(methods) != 1:
        raise ValueError(f"generation pool mixes methods: {methods}")
    if config.expected_method is not None and methods != [config.expected_method]:
        raise ValueError(
            f"generation pool method mismatch: expected {config.expected_method}, found {methods}"
        )
    if config.expected_total is not None and len(rows) != config.expected_total:
        raise ValueError(
            f"generation pool count mismatch: expected {config.expected_total}, found {len(rows)}"
        )

    training_counts = Counter(int(row["training_seed"]) for row in rows)
    expected_counts = {
        int(seed): int(count) for seed, count in config.expected_training_seed_counts
    }
    if expected_counts and dict(sorted(training_counts.items())) != dict(
        sorted(expected_counts.items())
    ):
        raise ValueError(
            "generation pool training-seed counts mismatch: "
            f"expected {dict(sorted(expected_counts.items()))}, "
            f"found {dict(sorted(training_counts.items()))}"
        )

    rows.sort(
        key=lambda row: (
            int(row["training_seed"]),
            int(row["sampling_seed"]),
            int(row["ordinal"]),
            str(row["attempt_id"]),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_line(row))
        handle.flush()
        os.fsync(handle.fileno())

    status_counts = Counter(str(row["status"]) for row in rows)
    sampling_counts = Counter(
        (int(row["training_seed"]), int(row["sampling_seed"])) for row in rows
    )
    checkpoint_hashes = sorted(
        {
            str(row["checkpoint_sha256"])
            for row in rows
            if row.get("checkpoint_sha256")
        }
    )
    source_hashes = sorted(
        {
            str(row["source_bundle_sha256"])
            for row in rows
            if row.get("source_bundle_sha256")
        }
    )
    revision_locks = sorted(
        {
            str(row["revision_lock_sha256"])
            for row in rows
            if row.get("revision_lock_sha256")
        }
    )
    result = {
        "schema": "wqcodiff_generation_pool_v1",
        "method": methods[0],
        "records": len(rows),
        "terminal_records": len(rows),
        "order": ["training_seed", "sampling_seed", "ordinal", "attempt_id"],
        "training_seed_counts": {
            str(seed): count for seed, count in sorted(training_counts.items())
        },
        "training_sampling_seed_counts": {
            f"{training_seed}/{sampling_seed}": count
            for (training_seed, sampling_seed), count in sorted(sampling_counts.items())
        },
        "status_counts": dict(sorted(status_counts.items())),
        "attempt_id_set_sha256": hashlib.sha256(
            "\n".join(sorted(attempts)).encode("utf-8")
        ).hexdigest(),
        "pair_id_set_sha256": hashlib.sha256(
            "\n".join(sorted(pairs)).encode("utf-8")
        ).hexdigest(),
        "checkpoint_sha256s": checkpoint_hashes,
        "source_bundle_sha256s": source_hashes,
        "revision_lock_sha256s": revision_locks,
        "source_artifacts": source_artifacts,
        "pooled_artifact": _identity(output),
    }
    write_json_exclusive(manifest, result)
    return result


def parse_seed_count(value: str) -> tuple[int, int]:
    seed, separator, count = value.partition("=")
    if not separator:
        raise ValueError("training-seed counts must use SEED=COUNT")
    try:
        result = int(seed), int(count)
    except ValueError as exc:
        raise ValueError("training-seed counts must use integer SEED=COUNT") from exc
    if result[1] <= 0:
        raise ValueError("training-seed counts must be positive")
    return result
