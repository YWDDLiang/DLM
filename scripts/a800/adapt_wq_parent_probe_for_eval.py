#!/usr/bin/env python3
"""Create an immutable evaluation view of parent-CSP probe attempts.

The parent probe deliberately records both the expanded proposal structure and
the final parent-CSP structure.  The unchanged CrysLLMGen and A100 S.U.N.
evaluators consume the common ``wqcodiff_generation_attempt_v1`` schema and a
field named ``structure``.  This adapter maps ``final_structure`` to that field
without changing attempt IDs, structures, statuses, or the all-attempt
denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


SOURCE_SCHEMA = "wq_parent_csp_probe_attempt_v1"
TARGET_SCHEMA = "wqcodiff_generation_attempt_v1"
METHOD = "DIAG-WQ-PROPOSAL-PARENT-CSP32"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not a mapping")
            rows.append(row)
    return rows


def adapt_parent_probe_for_eval(
    *,
    source_jsonl: str | Path,
    output_jsonl: str | Path,
    manifest_json: str | Path,
    expected_attempts: int,
    expected_start_ordinal: int,
    adapter_training_execution_patch_sha256: str,
    evaluation_execution_patch_sha256: str,
) -> dict[str, Any]:
    source = Path(source_jsonl).resolve()
    output = Path(output_jsonl).resolve()
    manifest_path = Path(manifest_json).resolve()
    if expected_attempts <= 0 or expected_start_ordinal < 0:
        raise ValueError("invalid expected attempt range")
    for label, value in (
        ("adapter training patch", adapter_training_execution_patch_sha256),
        ("evaluation patch", evaluation_execution_patch_sha256),
    ):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{label} must be one lowercase SHA256")
    if not source.is_file():
        raise FileNotFoundError(source)
    for path in (output, manifest_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(source)
    expected_ordinals = list(
        range(expected_start_ordinal, expected_start_ordinal + expected_attempts)
    )
    if len(rows) != expected_attempts:
        raise ValueError("parent probe denominator changed")
    if [int(row.get("ordinal", -1)) for row in rows] != expected_ordinals:
        raise ValueError("parent probe ordinal contract changed")
    attempt_ids = [str(row.get("attempt_id", "")) for row in rows]
    if any(not value for value in attempt_ids) or len(set(attempt_ids)) != len(rows):
        raise ValueError("parent probe attempt IDs are missing or duplicated")

    adapted: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    mapped_structure_hashes: list[str] = []
    for row in rows:
        if row.get("schema") != SOURCE_SCHEMA:
            raise ValueError("unexpected parent probe schema")
        if row.get("method") != METHOD:
            raise ValueError("unexpected parent probe method")
        if row.get("retry_or_replacement_used") is not False:
            raise ValueError("retry/replacement evidence cannot enter evaluation")
        if (
            row.get("adapter_training_execution_patch_sha256")
            != adapter_training_execution_patch_sha256
            or row.get("diagnostic_execution_patch_sha256")
            != evaluation_execution_patch_sha256
        ):
            raise ValueError("parent probe execution identity changed")
        status = str(row.get("status", ""))
        if status not in {"succeeded", "failed"}:
            raise ValueError("parent probe attempt is not terminal")

        target = dict(row)
        target["source_schema"] = SOURCE_SCHEMA
        target["schema"] = TARGET_SCHEMA
        target["structure_stage"] = "released_parent_cspdiffusion_32_step_final"
        target["evaluation_execution_patch_sha256"] = (
            evaluation_execution_patch_sha256
        )
        target["source_attempt_sha256"] = _sha256_bytes(
            _canonical(row).encode("utf-8")
        )
        if status == "succeeded":
            structure = row.get("final_structure")
            if not isinstance(structure, Mapping):
                raise ValueError("successful parent attempt has no final structure")
            volume = float(row.get("final_volume", float("nan")))
            atom_count = int(row.get("atom_count", 0))
            if not math.isfinite(volume) or volume <= 0.0 or atom_count <= 0:
                raise ValueError("successful parent attempt has invalid geometry")
            target["structure"] = dict(structure)
            structure_sha = _sha256_bytes(
                _canonical(structure).encode("utf-8")
            )
            target["structure_sha256"] = structure_sha
            mapped_structure_hashes.append(structure_sha)
            succeeded += 1
        else:
            target["structure"] = None
            failed += 1
        adapted.append(target)

    with output.open("x", encoding="utf-8") as handle:
        for row in adapted:
            handle.write(_canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    manifest = {
        "schema": "wq_parent_csp_evaluation_adapter_manifest_v1",
        "ok": True,
        "source_schema": SOURCE_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "method": METHOD,
        "attempts": expected_attempts,
        "start_ordinal": expected_start_ordinal,
        "end_ordinal_inclusive": expected_ordinals[-1],
        "succeeded": succeeded,
        "failed": failed,
        "structure_mapping": "final_structure_to_structure_byte_semantic_identity",
        "mapped_structure_hashes": mapped_structure_hashes,
        "source_jsonl": str(source),
        "source_jsonl_sha256": _sha256_file(source),
        "output_jsonl": str(output),
        "output_jsonl_sha256": _sha256_file(output),
        "adapter_training_execution_patch_sha256": (
            adapter_training_execution_patch_sha256
        ),
        "evaluation_execution_patch_sha256": evaluation_execution_patch_sha256,
        "retry_or_replacement_used": False,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--expected-attempts", type=int, required=True)
    parser.add_argument("--expected-start-ordinal", type=int, required=True)
    parser.add_argument(
        "--adapter-training-execution-patch-sha256",
        required=True,
    )
    parser.add_argument("--evaluation-execution-patch-sha256", required=True)
    args = parser.parse_args()
    result = adapt_parent_probe_for_eval(
        source_jsonl=args.source_jsonl,
        output_jsonl=args.output_jsonl,
        manifest_json=args.manifest_json,
        expected_attempts=args.expected_attempts,
        expected_start_ordinal=args.expected_start_ordinal,
        adapter_training_execution_patch_sha256=(
            args.adapter_training_execution_patch_sha256
        ),
        evaluation_execution_patch_sha256=(
            args.evaluation_execution_patch_sha256
        ),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
