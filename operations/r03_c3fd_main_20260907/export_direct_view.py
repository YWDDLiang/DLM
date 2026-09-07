"""Expose existing R03 endpoint metadata to the frozen Direct evaluator.

Only status/ordinal/method field names are adapted. The actual structure, all
request identities and failure rows are preserved. No chemistry or geometry is
evaluated, repaired, reconstructed or selected here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


INPUT_SCHEMA = "r03_common_evaluation_input_v1"
SCHEMA = "r03_direct_metadata_view_v1"
FROZEN_EVAL_UTILS_SHA256 = "68e6d0a9703f412cfd3215e6d0ae687e5b153e941d16f3fa4f2fffeedb505cb6"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_digest(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    text = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def direct_view(rows: Sequence[Mapping[str, Any]], *, expected_denominator: int):
    if isinstance(expected_denominator, bool) or expected_denominator <= 0 or len(rows) != expected_denominator:
        raise ValueError("Direct view requires its unchanged positive all-request denominator")
    if [row.get("evaluation_ordinal") for row in rows] != list(range(expected_denominator)):
        raise ValueError("Direct view cannot drop, reorder, or renumber source evaluation ordinals")
    attempts = [row.get("attempt_id") for row in rows]
    sample_ids = [row.get("sample_idx") for row in rows]
    trajectories = [row.get("trajectory_id") for row in rows]
    for values, name in ((attempts, "attempt_id"), (trajectories, "trajectory_id")):
        if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"original {name} values must be present and unique")
    if any(type(value) is not int or value < 0 for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("original global sample_idx values must be nonnegative integers and unique")
    endpoints = {row.get("endpoint") for row in rows}
    methods = {row.get("method_id") for row in rows}
    if len(endpoints) != 1 or not endpoints <= {"native", "tau800"}:
        raise ValueError("native and refined endpoints cannot share one Direct view")
    if len(methods) != 1 or any(not isinstance(method, str) or not method for method in methods):
        raise ValueError("different methods cannot share one Direct view")
    result = []
    for ordinal, row in enumerate(rows):
        if (row.get("schema") != INPUT_SCHEMA or row.get("purpose") != "evaluation"
                or row.get("source_split") != "evaluation"):
            raise ValueError("Direct view accepts only explicit R03 evaluation endpoint artifacts")
        if type(row.get("success")) is not bool or type(row.get("parseable")) is not bool:
            raise ValueError("endpoint success and parseable flags must be explicit booleans")
        structure = row.get("structure")
        succeeded = row["success"]
        if succeeded and (not row["parseable"] or not isinstance(structure, dict) or not structure):
            raise ValueError("successful endpoint lacks its real parseable structure")
        if succeeded and row.get("source_attempt_status") == "controller_failure":
            raise ValueError("a failed controller cannot become a successful Direct endpoint")
        status = "succeeded" if succeeded else "failed"
        reason = None if succeeded else str(
            row.get("artifact_error") or row.get("source_reason") or row.get("source_attempt_status")
            or "endpoint_failure"
        )
        result.append({
            "schema": SCHEMA, "ordinal": ordinal, "evaluation_ordinal": row["evaluation_ordinal"],
            "attempt_id": row["attempt_id"], "original_attempt_id": row.get("original_attempt_id", row["attempt_id"]),
            "trajectory_id": row["trajectory_id"], "sample_idx": row["sample_idx"],
            "method": row["method_id"], "method_id": row["method_id"], "endpoint": row["endpoint"],
            "status": status, "reason": reason, "structure": structure,
            "structure_json_sha256": structure_digest(structure),
            "source_success": succeeded, "source_parseable": row["parseable"],
            "source_attempt_status": row.get("source_attempt_status"),
            "planner_seed": row.get("planner_seed"), "component_id": row.get("component_id"),
            "purpose": "evaluation", "source_split": "evaluation", "training_use_allowed": False,
            "geometry_changed": False, "retry_or_replacement_used": False,
        })
    report = {
        "schema": SCHEMA, "expected_denominator": expected_denominator, "input_rows": len(rows),
        "output_rows": len(result), "generation_succeeded": sum(row["status"] == "succeeded" for row in result),
        "failed_requests_retained": sum(row["status"] == "failed" for row in result),
        "method_id": next(iter(methods)), "endpoint": next(iter(endpoints)),
        "structure_payloads_unchanged": True, "original_attempt_ids_preserved": True,
        "global_sample_ids_preserved": True, "new_model_calls": 0, "new_physics_calls": 0,
        "new_validity_checks": 0, "field_mapping": {"ordinal": "evaluation_ordinal", "status": "success", "method": "method_id"},
    }
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-jsonl", type=Path, required=True)
    parser.add_argument("--expected-denominator", type=int, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    evaluator = args.snapshot_root / "eval_utils.py"
    evaluator_sha = file_sha256(evaluator)
    if evaluator_sha != FROZEN_EVAL_UTILS_SHA256:
        raise ValueError("Direct eval_utils.py differs from the frozen R03 snapshot")
    rows = [json.loads(line) for line in args.paths_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("every endpoint row must be a JSON object")
    output, report = direct_view(rows, expected_denominator=args.expected_denominator)
    report.update(source_paths_jsonl=str(args.paths_jsonl.resolve()), source_paths_sha256=file_sha256(args.paths_jsonl),
                  snapshot_root=str(args.snapshot_root.resolve()), eval_utils_sha256=evaluator_sha,
                  field_adapter_sha256=file_sha256(Path(__file__)))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    generated = args.output_dir / "generation.jsonl"
    with generated.open("x", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report["generation_sha256"] = file_sha256(generated)
    (args.output_dir / "DIRECT_VIEW_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
