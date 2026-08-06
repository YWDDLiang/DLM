"""Attempt identity, seed derivation, and append-only evidence contracts."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # Linux/Slurm path; kept optional so CPU model tests also run on Windows.
    import fcntl
except ImportError:  # pragma: no cover - Windows-only development fallback
    fcntl = None  # type: ignore[assignment]


_LOCAL_LEDGER_LOCK = threading.RLock()


@contextmanager
def _exclusive_file_lock(handle: Any) -> Iterable[None]:
    if fcntl is None:
        # The registered execution environment is Linux and uses flock.  This
        # fallback is deliberately process-local and exists only for Windows
        # unit/smoke tests; paper runs reject non-Linux hosts in env_doctor.
        with _LOCAL_LEDGER_LOCK:
            yield
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json_exclusive(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> None:
    """Create one immutable JSON artifact and reject non-finite numbers."""

    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class AttemptStatus(str, enum.Enum):
    """Allowed lifecycle states.

    Failure subtypes are terminal so that no caller can silently replace or
    retry them while preserving an apparently successful denominator.
    """

    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NONCONVERGED = "nonconverged"
    UNSUPPORTED_ELEMENT = "unsupported_element"
    MISSING_HULL = "missing_hull"
    INVALID_TOPOLOGY = "invalid_topology"
    BRIDGE_FAILURE = "bridge_failure"
    PROJECTION_FAILURE = "projection_failure"
    CACHE_MISMATCH = "cache_mismatch"
    SEED_MISMATCH = "seed_mismatch"

    @property
    def terminal(self) -> bool:
        return self not in {AttemptStatus.SUBMITTED, AttemptStatus.RUNNING}

    @property
    def success(self) -> bool:
        return self is AttemptStatus.SUCCEEDED


@dataclasses.dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One immutable stage record for a submitted generation attempt."""

    attempt_id: str
    method: str
    training_seed: int
    sampling_seed: int
    stage: str
    status: AttemptStatus
    reason: str = ""
    artifact_hash: str = ""
    seed: int | None = None
    calls: Mapping[str, int] = dataclasses.field(default_factory=dict)
    flops: float = 0.0
    walltime_s: float = 0.0
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.attempt_id or any(ch.isspace() for ch in self.attempt_id):
            raise ValueError("attempt_id must be non-empty and contain no whitespace")
        if not self.method or not self.stage:
            raise ValueError("method and stage are required")
        if self.flops < 0 or self.walltime_s < 0:
            raise ValueError("flops and walltime_s must be non-negative")
        if any(int(value) < 0 for value in self.calls.values()):
            raise ValueError("component call counts must be non-negative")
        if self.status.terminal and not self.status.success and not self.reason:
            raise ValueError("terminal failures require a reason")

    @property
    def key(self) -> tuple[str, str]:
        return self.attempt_id, self.stage

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["status"] = self.status.value
        data["calls"] = dict(sorted(self.calls.items()))
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttemptRecord":
        payload = dict(data)
        payload["status"] = AttemptStatus(payload["status"])
        return cls(**payload)


class SeedDeriver:
    """Derive attempt and stage seeds without rank/world-size dependence."""

    def __init__(self, protocol_name: str, experiment_id: str) -> None:
        if not protocol_name or not experiment_id:
            raise ValueError("protocol_name and experiment_id are required")
        self.protocol_name = protocol_name
        self.experiment_id = experiment_id

    def attempt_id(
        self,
        *,
        training_seed: int,
        sampling_seed: int,
        ordinal: int,
        method: str,
    ) -> str:
        if ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        payload = {
            "protocol": self.protocol_name,
            "experiment": self.experiment_id,
            "training_seed": int(training_seed),
            "sampling_seed": int(sampling_seed),
            "ordinal": int(ordinal),
            "method": method,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return f"a-{digest[:24]}"

    def derive(
        self,
        *,
        training_seed: int,
        sampling_seed: int,
        attempt_id: str,
        stage: str,
    ) -> int:
        payload = {
            "protocol": self.protocol_name,
            "experiment": self.experiment_id,
            "training_seed": int(training_seed),
            "sampling_seed": int(sampling_seed),
            "attempt_id": attempt_id,
            "stage": stage,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).digest()
        # Keep the result valid for Python, NumPy, and torch generators.
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    def pair_id(
        self,
        *,
        training_seed: int,
        sampling_seed: int,
        ordinal: int,
    ) -> str:
        """Method-independent identity for matched-noise comparisons."""

        if ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        payload = {
            "protocol": self.protocol_name,
            "pairing_experiment": self.experiment_id,
            "training_seed": int(training_seed),
            "sampling_seed": int(sampling_seed),
            "ordinal": int(ordinal),
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return f"p-{digest[:24]}"

    def paired_derive(
        self,
        *,
        training_seed: int,
        sampling_seed: int,
        ordinal: int,
        stage: str,
    ) -> int:
        payload = {
            "protocol": self.protocol_name,
            "pairing_experiment": self.experiment_id,
            "training_seed": int(training_seed),
            "sampling_seed": int(sampling_seed),
            "ordinal": int(ordinal),
            "stage": stage,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclasses.dataclass(frozen=True, slots=True)
class AttemptAudit:
    records: int
    attempts: int
    duplicate_stage_records: tuple[tuple[str, str], ...]
    conflicting_terminal_records: tuple[tuple[str, str], ...]
    missing_terminal_attempts: tuple[str, ...]
    seed_mismatches: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        return not (
            self.duplicate_stage_records
            or self.conflicting_terminal_records
            or self.missing_terminal_attempts
            or self.seed_mismatches
        )


class AttemptLedger:
    """Process-safe append-only JSONL ledger.

    A lock covers validation plus append.  Existing bytes are never rewritten;
    a duplicate ``(attempt_id, stage, status)`` is a fatal contract violation.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_identity: tuple[int, int] | None = None
        self._cache_offset = 0
        self._cache_lines = 0
        self._cache_by_key: dict[tuple[str, str], list[AttemptRecord]] = {}

    def _reset_append_cache(self, identity: tuple[int, int]) -> None:
        self._cache_identity = identity
        self._cache_offset = 0
        self._cache_lines = 0
        self._cache_by_key = {}

    def _refresh_append_cache(self, handle: Any) -> None:
        """Incrementally validate bytes added since this instance last appended.

        The file lock is held by the caller. Cooperative writers only append
        complete newline-terminated records, so every existing byte is parsed
        exactly once per ledger instance instead of once per append.
        """

        stat = os.fstat(handle.fileno())
        identity = (int(stat.st_dev), int(stat.st_ino))
        if self._cache_identity != identity or stat.st_size < self._cache_offset:
            self._reset_append_cache(identity)
        if stat.st_size == self._cache_offset:
            return
        handle.seek(self._cache_offset)
        added = handle.read(stat.st_size - self._cache_offset)
        if len(added) != stat.st_size - self._cache_offset:
            raise ValueError("attempt ledger changed while its append lock was held")
        if added and not added.endswith(b"\n"):
            raise ValueError("attempt ledger has a partial trailing JSONL record")

        parsed: list[AttemptRecord] = []
        for relative_line, raw_line in enumerate(added.splitlines(), start=1):
            if not raw_line.strip():
                continue
            line_number = self._cache_lines + relative_line
            try:
                parsed.append(
                    AttemptRecord.from_dict(json.loads(raw_line.decode("utf-8")))
                )
            except Exception as exc:
                raise ValueError(
                    f"invalid ledger record at line {line_number}: {exc}"
                ) from exc
        for item in parsed:
            self._cache_by_key.setdefault(item.key, []).append(item)
        self._cache_offset = int(stat.st_size)
        self._cache_lines += added.count(b"\n")

    def records(self) -> list[AttemptRecord]:
        if not self.path.exists():
            return []
        result: list[AttemptRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    result.append(AttemptRecord.from_dict(json.loads(line)))
                except Exception as exc:  # pragma: no cover - message is the contract
                    raise ValueError(f"invalid ledger record at line {line_number}: {exc}") from exc
        return result

    def append(self, record: AttemptRecord) -> None:
        line = (_canonical_json(record.to_dict()) + "\n").encode("utf-8")
        with self.path.open("a+b") as handle:
            with _exclusive_file_lock(handle):
                self._refresh_append_cache(handle)
                same_stage = self._cache_by_key.get(record.key, [])
                if any(item.status == record.status for item in same_stage):
                    raise ValueError(
                        f"duplicate attempt stage/status: {record.key} {record.status.value}"
                    )
                if record.status.terminal and any(item.status.terminal for item in same_stage):
                    raise ValueError(f"attempt stage already has a terminal record: {record.key}")
                handle.seek(0, os.SEEK_END)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                self._cache_by_key.setdefault(record.key, []).append(record)
                self._cache_offset += len(line)
                self._cache_lines += 1

    def audit(
        self,
        *,
        seed_deriver: SeedDeriver | None = None,
        terminal_stage: str = "sun",
        expected_attempt_ids: Iterable[str] | None = None,
    ) -> AttemptAudit:
        records = self.records()
        by_key: dict[tuple[str, str], list[AttemptRecord]] = {}
        attempts: set[str] = set()
        for record in records:
            attempts.add(record.attempt_id)
            by_key.setdefault(record.key, []).append(record)

        duplicate = tuple(sorted(key for key, values in by_key.items() if len(values) != len({v.status for v in values})))
        conflicting = tuple(
            sorted(key for key, values in by_key.items() if sum(v.status.terminal for v in values) > 1)
        )
        required = set(expected_attempt_ids) if expected_attempt_ids is not None else attempts
        terminal_ids = {
            record.attempt_id
            for record in records
            if record.stage == terminal_stage and record.status.terminal
        }
        missing = tuple(sorted(required - terminal_ids))

        mismatches: list[tuple[str, str]] = []
        if seed_deriver is not None:
            for record in records:
                expected = seed_deriver.derive(
                    training_seed=record.training_seed,
                    sampling_seed=record.sampling_seed,
                    attempt_id=record.attempt_id,
                    stage=record.stage,
                )
                if record.seed != expected:
                    mismatches.append(record.key)
        return AttemptAudit(
            records=len(records),
            attempts=len(attempts),
            duplicate_stage_records=duplicate,
            conflicting_terminal_records=conflicting,
            missing_terminal_attempts=missing,
            seed_mismatches=tuple(sorted(mismatches)),
        )


class ArtifactLedger:
    """Append-only JSONL artifacts with a unique immutable record key."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        key_fields: tuple[str, ...] = ("attempt_id",),
    ) -> None:
        if not key_fields:
            raise ValueError("artifact key_fields cannot be empty")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_fields = key_fields
        self._cache_identity: tuple[int, int] | None = None
        self._cache_offset = 0
        self._cache_lines = 0
        self._cache_keys: set[tuple[str, ...]] = set()

    def _reset_append_cache(self, identity: tuple[int, int]) -> None:
        self._cache_identity = identity
        self._cache_offset = 0
        self._cache_lines = 0
        self._cache_keys = set()

    def _refresh_append_cache(self, handle: Any) -> None:
        stat = os.fstat(handle.fileno())
        identity = (int(stat.st_dev), int(stat.st_ino))
        if self._cache_identity != identity or stat.st_size < self._cache_offset:
            self._reset_append_cache(identity)
        if stat.st_size == self._cache_offset:
            return
        handle.seek(self._cache_offset)
        added = handle.read(stat.st_size - self._cache_offset)
        if len(added) != stat.st_size - self._cache_offset:
            raise ValueError("artifact ledger changed while its append lock was held")
        if added and not added.endswith(b"\n"):
            raise ValueError("artifact ledger has a partial trailing JSONL record")

        parsed_keys: list[tuple[str, ...]] = []
        for relative_line, raw_line in enumerate(added.splitlines(), start=1):
            if not raw_line.strip():
                continue
            line_number = self._cache_lines + relative_line
            try:
                payload = json.loads(raw_line.decode("utf-8"))
            except Exception as exc:
                raise ValueError(
                    f"invalid artifact JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"artifact line {line_number} is not an object")
            parsed_keys.append(self._key(payload))
        self._cache_keys.update(parsed_keys)
        self._cache_offset = int(stat.st_size)
        self._cache_lines += added.count(b"\n")

    def _key(self, payload: Mapping[str, Any]) -> tuple[str, ...]:
        try:
            key = tuple(str(payload[field]) for field in self.key_fields)
        except KeyError as exc:
            raise ValueError(f"artifact is missing key field {exc.args[0]}") from exc
        if any(not value for value in key):
            raise ValueError("artifact key values cannot be empty")
        return key

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        result: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except Exception as exc:
                    raise ValueError(
                        f"invalid artifact JSON at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"artifact line {line_number} is not an object")
                result.append(payload)
        return result

    def append(self, payload: Mapping[str, Any]) -> str:
        record = dict(payload)
        key = self._key(record)
        line = (_canonical_json(record) + "\n").encode("utf-8")
        digest = hashlib.sha256(line).hexdigest()
        with self.path.open("a+b") as handle:
            with _exclusive_file_lock(handle):
                self._refresh_append_cache(handle)
                if key in self._cache_keys:
                    raise ValueError(f"duplicate immutable artifact key {key}")
                handle.seek(0, os.SEEK_END)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                self._cache_keys.add(key)
                self._cache_offset += len(line)
                self._cache_lines += 1
        return digest
