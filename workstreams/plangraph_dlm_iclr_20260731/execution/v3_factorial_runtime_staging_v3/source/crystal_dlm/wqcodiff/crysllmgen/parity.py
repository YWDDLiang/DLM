"""Dependency-light contracts and comparison utilities for Gate-A parity."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar


REQUIRED_CHECKS = (
    "source_snapshot",
    "atom_parser",
    "beta_sigma_tables",
    "checkpoint_mapping",
    "one_step_csp_tensors",
    "deterministic_sampler",
    "attempt_accounting",
)

T = TypeVar("T")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class ParityContract:
    schema: str
    upstream_commit: str
    upstream_relative_manifest_sha256: str
    proposal_count: int
    selection_salt: str
    absolute_tolerance: float
    relative_tolerance: float
    required_checks: tuple[str, ...]
    scheduler_timesteps: int
    parent_run_type: str
    one_step_device: str
    source_path: Path
    sha256: str

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ParityContract":
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != "crysllmgen_disabled_extension_parity_v1":
            raise ValueError("unsupported CrysLLMGen parity contract schema")
        checks = tuple(str(value) for value in payload.get("required_checks", ()))
        if checks != REQUIRED_CHECKS:
            raise ValueError("parity checks or their frozen order changed")
        proposal_count = int(payload["proposal_count"])
        if proposal_count != 256:
            raise ValueError("Gate A requires exactly 256 hash-fixed proposals")
        atol = float(payload["numeric_tolerance"]["absolute"])
        rtol = float(payload["numeric_tolerance"]["relative"])
        if not 0.0 <= atol <= 1.0e-6 or not 0.0 <= rtol <= 1.0e-6:
            raise ValueError("parity tolerances may not exceed 1e-6")
        scheduler_timesteps = int(payload["tensor_contract"]["scheduler_timesteps"])
        if scheduler_timesteps != 1000:
            raise ValueError("released MP20 checkpoint requires a 1000-step scheduler")
        parent_run_type = str(payload["tensor_contract"]["parent_run_type"])
        if parent_run_type != "train":
            raise ValueError("official CrysLLMGen sampling uses parent run_type=train")
        one_step_device = str(payload["tensor_contract"]["one_step_device"])
        if one_step_device != "cpu":
            raise ValueError("one-step transparency must avoid nondeterministic CUDA scatter")
        commit = str(payload["upstream_commit"])
        manifest_sha = str(payload["upstream_relative_manifest_sha256"])
        for label, value in (("upstream_commit", commit), ("manifest", manifest_sha)):
            expected_length = 40 if label == "upstream_commit" else 64
            if len(value) != expected_length:
                raise ValueError(f"invalid {label} digest")
            if any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"invalid {label} digest")
        return cls(
            schema=str(payload["schema"]),
            upstream_commit=commit,
            upstream_relative_manifest_sha256=manifest_sha,
            proposal_count=proposal_count,
            selection_salt=str(payload["selection_salt"]),
            absolute_tolerance=atol,
            relative_tolerance=rtol,
            required_checks=checks,
            scheduler_timesteps=scheduler_timesteps,
            parent_run_type=parent_run_type,
            one_step_device=one_step_device,
            source_path=source,
            sha256=sha256_file(source),
        )


def hash_fixed_select(
    records: Iterable[T],
    *,
    identity: Callable[[T], str],
    count: int,
    salt: str,
) -> list[T]:
    """Select records independently of input order, rank, and world size."""

    if count <= 0 or not salt:
        raise ValueError("positive count and non-empty salt are required")
    keyed: list[tuple[str, str, T]] = []
    identities: set[str] = set()
    for record in records:
        record_id = str(identity(record))
        if not record_id or record_id in identities:
            raise ValueError("hash-fixed selection requires unique non-empty identities")
        identities.add(record_id)
        digest = hashlib.sha256(f"{salt}\0{record_id}".encode("utf-8")).hexdigest()
        keyed.append((digest, record_id, record))
    if len(keyed) < count:
        raise ValueError(f"requested {count} records from a pool of {len(keyed)}")
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in keyed[:count]]


def _to_builtin(value: Any) -> Any:
    """Recursively normalize tensor/array containers for comparison and hashing."""

    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "tolist"):
        current = current.tolist()
    if dataclasses.is_dataclass(current):
        current = dataclasses.asdict(current)
    if isinstance(current, Mapping):
        return {key: _to_builtin(item) for key, item in current.items()}
    if isinstance(current, (list, tuple)):
        return [_to_builtin(item) for item in current]
    return current


@dataclasses.dataclass(frozen=True, slots=True)
class ValueComparison:
    passed: bool
    numeric_values: int
    exact_values: int
    max_absolute_error: float
    max_relative_error: float
    first_mismatch: str | None
    upstream_sha256: str
    derived_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compare_values(
    upstream: Any,
    derived: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> ValueComparison:
    """Recursively compare tensors/arrays/containers with one tight contract."""

    upstream_builtin = _to_builtin(upstream)
    derived_builtin = _to_builtin(derived)
    numeric_values = 0
    exact_values = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    first_mismatch: str | None = None

    def mismatch(path: str, reason: str) -> None:
        nonlocal first_mismatch
        if first_mismatch is None:
            first_mismatch = f"{path}: {reason}"

    def visit(left: Any, right: Any, path: str) -> None:
        nonlocal numeric_values, exact_values, maximum_absolute, maximum_relative
        left = _to_builtin(left)
        right = _to_builtin(right)
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                mismatch(path, "mapping/non-mapping type mismatch")
                return
            left_keys = tuple(sorted(str(key) for key in left))
            right_keys = tuple(sorted(str(key) for key in right))
            if left_keys != right_keys:
                mismatch(path, f"mapping keys differ: {left_keys!r} != {right_keys!r}")
                return
            for key in left_keys:
                visit(left[key], right[key], f"{path}.{key}")
            return
        sequence_types = (list, tuple)
        if isinstance(left, sequence_types) or isinstance(right, sequence_types):
            if not isinstance(left, sequence_types) or not isinstance(right, sequence_types):
                mismatch(path, "sequence/non-sequence type mismatch")
                return
            if len(left) != len(right):
                mismatch(path, f"sequence lengths differ: {len(left)} != {len(right)}")
                return
            for index, (left_value, right_value) in enumerate(zip(left, right)):
                visit(left_value, right_value, f"{path}[{index}]")
            return
        numeric = (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        )
        if numeric:
            numeric_values += 1
            left_float = float(left)
            right_float = float(right)
            if not math.isfinite(left_float) or not math.isfinite(right_float):
                mismatch(path, "non-finite numeric value")
                return
            absolute = abs(left_float - right_float)
            relative = absolute / max(abs(left_float), abs(right_float), 1.0e-30)
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            allowed = absolute_tolerance + relative_tolerance * abs(left_float)
            if absolute > allowed:
                mismatch(path, f"numeric error {absolute:.9g} exceeds {allowed:.9g}")
            return
        exact_values += 1
        if type(left) is not type(right) or left != right:
            mismatch(path, f"exact values differ: {left!r} != {right!r}")

    visit(upstream_builtin, derived_builtin, "root")
    return ValueComparison(
        passed=first_mismatch is None,
        numeric_values=numeric_values,
        exact_values=exact_values,
        max_absolute_error=maximum_absolute,
        max_relative_error=maximum_relative,
        first_mismatch=first_mismatch,
        upstream_sha256=sha256_json(upstream_builtin),
        derived_sha256=sha256_json(derived_builtin),
    )


def audit_parity_report(
    payload: Mapping[str, Any], contract: ParityContract
) -> dict[str, Any]:
    """Reject partial, retried, or numerically loose Gate-A reports."""

    errors: list[str] = []
    if payload.get("schema") != "crysllmgen_disabled_extension_parity_report_v1":
        errors.append("report_schema_mismatch")
    if payload.get("contract_sha256") != contract.sha256:
        errors.append("contract_hash_mismatch")
    if int(payload.get("proposal_count", -1)) != contract.proposal_count:
        errors.append("proposal_count_mismatch")
    if bool(payload.get("retry_or_replacement_used", True)):
        errors.append("retry_or_replacement_used")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        errors.append("checks_missing")
        checks = {}
    if tuple(checks) != contract.required_checks:
        errors.append("check_set_or_order_mismatch")
    for name in contract.required_checks:
        cell = checks.get(name)
        if not isinstance(cell, Mapping) or not bool(cell.get("passed")):
            errors.append(f"check_failed:{name}")
        if isinstance(cell, Mapping):
            maximum = float(cell.get("max_absolute_error", 0.0))
            if maximum > contract.absolute_tolerance:
                errors.append(f"absolute_tolerance_exceeded:{name}")
    terminal_attempts = int(payload.get("terminal_attempts", -1))
    if terminal_attempts != contract.proposal_count:
        errors.append("terminal_attempt_denominator_mismatch")
    return {
        "schema": "crysllmgen_disabled_extension_parity_audit_v1",
        "ok": not errors,
        "errors": errors,
        "contract_sha256": contract.sha256,
        "report_sha256": sha256_json(payload),
    }


def write_json_exclusive(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
