#!/usr/bin/env python3
"""Assemble the paired H1 D1 versus D2-safe-axis schedule screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_sha(path: Path, expected: str, label: str) -> Path:
    location = path.resolve()
    if not location.is_file():
        raise FileNotFoundError(f"{label} is missing: {location}")
    observed = sha256_file(location)
    if observed != str(expected):
        raise ValueError(f"{label} SHA changed: {observed}")
    return location


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def failure_class(record: Mapping[str, Any]) -> str | None:
    if record.get("status") == "succeeded":
        return None
    reason = str(record.get("reason") or "unknown")
    return ":".join(reason.split(":")[:2])


def duplicate_coordinate_failure(record: Mapping[str, Any]) -> bool:
    if record.get("status") == "succeeded":
        return False
    reason = str(record.get("reason") or "").lower()
    return "duplicate" in reason and "coord" in reason


def summarize(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    classes: dict[str, int] = {}
    for record in records:
        name = failure_class(record)
        if name is not None:
            classes[name] = classes.get(name, 0) + 1
    return {
        "attempts": len(records),
        "succeeded": sum(row.get("status") == "succeeded" for row in records),
        "failed": sum(row.get("status") != "succeeded" for row in records),
        "duplicate_coordinate_failures": sum(
            duplicate_coordinate_failure(row) for row in records
        ),
        "failure_classes": classes,
    }


def paired_identity_mismatches(
    control: list[Mapping[str, Any]],
    candidate: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "attempt_id",
        "pair_id",
        "body_noise_seed",
        "plan_state_sha256",
    )
    mismatches: list[dict[str, Any]] = []
    for ordinal, (left, right) in enumerate(zip(control, candidate, strict=True)):
        if int(left.get("ordinal", -1)) != ordinal:
            mismatches.append(
                {"ordinal": ordinal, "field": "control_ordinal", "value": left.get("ordinal")}
            )
        if int(right.get("ordinal", -1)) != ordinal:
            mismatches.append(
                {
                    "ordinal": ordinal,
                    "field": "candidate_ordinal",
                    "value": right.get("ordinal"),
                }
            )
        for field in fields:
            if left.get(field) != right.get(field):
                mismatches.append(
                    {
                        "ordinal": ordinal,
                        "field": field,
                        "control": left.get(field),
                        "candidate": right.get(field),
                    }
                )
    return mismatches


def output_agreement(
    historical: list[Mapping[str, Any]],
    control: list[Mapping[str, Any]],
) -> dict[str, int]:
    status_matches = 0
    text_matches = 0
    both_succeeded = 0
    for left, right in zip(historical, control, strict=True):
        if left.get("status") == right.get("status"):
            status_matches += 1
        if left.get("status") == "succeeded" and right.get("status") == "succeeded":
            both_succeeded += 1
            if left.get("text") == right.get("text"):
                text_matches += 1
    return {
        "status_matches": status_matches,
        "both_succeeded": both_succeeded,
        "exact_text_matches_among_both_succeeded": text_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    config = read_json(args.config.resolve())
    denominator = int(config["denominator"])
    if (
        config.get("schema") != "h1_body_safeaxis32_config_v1"
        or denominator != 32
        or config.get("automatic_downstream") is not False
    ):
        raise ValueError("schedule-screen configuration changed")

    frozen = config["frozen_h1"]
    control_path = require_sha(
        Path(frozen["control_body_attempts"]),
        frozen["control_body_attempts_sha256"],
        "frozen H1 control body attempts",
    )
    screen_dir = args.candidate_dir.resolve()
    control_path_current = screen_dir / "control_body_attempts.jsonl"
    candidate_path = screen_dir / "candidate_body_attempts.jsonl"
    generation_report_path = screen_dir / "generation_report.json"
    partition_path = screen_dir / "shared_batch_partition.json"
    invariant_path = screen_dir / "schedule_invariants.json"
    if not all(
        path.is_file()
        for path in (
            control_path_current,
            candidate_path,
            generation_report_path,
            partition_path,
            invariant_path,
        )
    ):
        raise FileNotFoundError("paired schedule output is incomplete")

    full_control = read_jsonl(control_path)
    historical = full_control[:denominator]
    control = read_jsonl(control_path_current)
    candidate = read_jsonl(candidate_path)
    if (
        len(full_control) != 256
        or len(historical) != denominator
        or len(control) != denominator
        or len(candidate) != denominator
    ):
        raise ValueError("historical, control, or candidate denominator changed")
    generation_report = read_json(generation_report_path)
    partition_report = read_json(partition_path)
    invariant_report = read_json(invariant_path)
    partition = partition_report.get("partition")
    if (
        generation_report.get("status") != "complete"
        or int(generation_report.get("attempts_per_arm", -1)) != denominator
        or generation_report.get("shared_batch_partition_applied_identically")
        is not True
        or generation_report.get("refinement_run") is not False
        or generation_report.get("direct_metrics_run") is not False
        or generation_report.get("sun_run") is not False
        or generation_report.get("automatic_downstream") is not False
        or generation_report.get("body_adapter_sha256_recorded")
        != config["body"]["adapter_sha256"]
        or generation_report.get("generation_policies")
        != ["d1", "d2_safe_axis"]
        or generation_report.get("all_candidate_invariants_passed") is not True
        or generation_report.get("schedule_invariants_sha256")
        != sha256_file(invariant_path)
        or invariant_report.get("status") != "complete"
        or int(invariant_report.get("attempts", -1)) != denominator
        or invariant_report.get("all_candidate_invariants_passed") is not True
        or int(invariant_report.get("required_z_before_xy_count", -1)) != 0
        or not isinstance(partition, list)
        or sorted(item for batch in partition for item in batch)
        != list(range(denominator))
        or canonical_sha256(partition) != partition_report.get("sha256")
        or partition_report.get("sha256")
        != generation_report.get("shared_batch_partition_sha256")
    ):
        raise ValueError("paired generation report changed")
    for label, schedule_arm, generation_policy, attempts_path in (
        ("control", "D1", "d1", control_path_current),
        ("candidate", "D2_SAFE_AXIS", "d2_safe_axis", candidate_path),
    ):
        arm_report = generation_report.get("arms", {}).get(label, {})
        if (
            arm_report.get("schedule_arm") != schedule_arm
            or arm_report.get("generation_policy") != generation_policy
            or int(arm_report.get("attempts", -1)) != denominator
            or arm_report.get("body_attempts_sha256")
            != sha256_file(attempts_path)
        ):
            raise ValueError(f"{label} generation report identity changed")

    identity_mismatches = paired_identity_mismatches(control, candidate)
    historical_identity_mismatches = paired_identity_mismatches(historical, control)
    historical_summary = summarize(historical)
    control_summary = summarize(control)
    candidate_summary = summarize(candidate)
    historical_classes = set(historical_summary["failure_classes"])
    control_classes = set(control_summary["failure_classes"])
    candidate_classes = set(candidate_summary["failure_classes"])
    control_new_classes = sorted(control_classes - historical_classes)
    new_classes = sorted(candidate_classes - control_classes)
    control_completion_drop = (
        int(historical_summary["succeeded"]) - int(control_summary["succeeded"])
    )
    completion_drop = (
        int(control_summary["succeeded"]) - int(candidate_summary["succeeded"])
    )
    duplicate_excess = (
        int(candidate_summary["duplicate_coordinate_failures"])
        - int(control_summary["duplicate_coordinate_failures"])
    )
    treatment_applied = int(
        generation_report["arms"]["candidate"]["schedule_treatment_applied"]
    )
    control_policy_exact = all(
        row.get("generation_policy") == "d1"
        and row.get("schedule_arm") == "D1"
        and row.get("body_checkpoint_arm") == "B0"
        and row.get("planner_arm") == "P0"
        for row in control
    )
    candidate_policy_exact = all(
        row.get("generation_policy") == "d2_safe_axis"
        and row.get("schedule_arm") == "D2_SAFE_AXIS"
        and row.get("body_checkpoint_arm") == "B0"
        and row.get("planner_arm") == "P0"
        for row in candidate
    )
    schedule_invariant_exact = all(
        row.get("schedule_z_before_xy_count") == 0
        and row.get("schedule_all_xy_precede_all_z") is True
        and isinstance(row.get("schedule_invariant_sha256"), str)
        and len(row["schedule_invariant_sha256"]) == 64
        for row in [*control, *candidate]
    )
    no_forbidden_attempt_actions = all(
        all(
            row.get(field) is False
            for field in (
                "retry_used",
                "replacement_used",
                "repair_used",
                "filter_used",
                "rerank_used",
            )
        )
        for row in [*control, *candidate]
    )

    gate = config["gate"]
    gate_passed = (
        not identity_mismatches
        and not historical_identity_mismatches
        and control_completion_drop
        <= int(gate["control_completion_drop_vs_historical_max_count"])
        and len(control_new_classes)
        <= int(gate["control_new_failure_classes_allowed"])
        and completion_drop <= int(gate["candidate_completion_drop_max_count"])
        and duplicate_excess
        <= int(gate["candidate_excess_duplicate_coordinate_failures_max_count"])
        and len(new_classes)
        <= int(gate["new_candidate_failure_classes_allowed"])
        and (
            treatment_applied == denominator
            if gate["require_schedule_treatment_applied"]
            else True
        )
        and control_policy_exact
        and candidate_policy_exact
        and schedule_invariant_exact
        and no_forbidden_attempt_actions
    )
    report = {
        "schema": "h1_body_safeaxis32_terminal_report_v1",
        "status": "complete",
        "gate_passed": gate_passed,
        "decision": (
            "safe_axis32_safe_to_expand"
            if gate_passed
            else "scientific_stop_retain_h1_exact_plan"
        ),
        "denominator_per_arm": denominator,
        "historical_h1_calibration": {
            "identity": "frozen_H1_P0_B0_exact_plan",
            "body_attempts_sha256": sha256_file(control_path),
            **historical_summary,
            **output_agreement(historical, control),
            "input_identity_mismatch_count": len(historical_identity_mismatches),
            "control_completion_drop_count": control_completion_drop,
            "new_control_failure_classes": control_new_classes,
        },
        "control": {
            "identity": "H1_P0_B0_exact_plan",
            "body_attempts_sha256": sha256_file(control_path_current),
            **control_summary,
        },
        "candidate": {
            "identity": "H1_P0_B0_plangraph_safe_axis_schedule",
            "body_attempts_sha256": sha256_file(candidate_path),
            **candidate_summary,
        },
        "paired": {
            "input_identity_mismatch_count": len(identity_mismatches),
            "input_identity_mismatches": identity_mismatches[:16],
            "control_policy_exact": control_policy_exact,
            "candidate_completion_drop_count": completion_drop,
            "candidate_excess_duplicate_coordinate_failures": duplicate_excess,
            "new_candidate_failure_classes": new_classes,
            "schedule_treatment_applied": treatment_applied,
            "candidate_policy_exact": candidate_policy_exact,
            "schedule_invariant_exact": schedule_invariant_exact,
            "schedule_invariants_sha256": sha256_file(invariant_path),
            "shared_batch_partition_applied_identically": True,
        },
        "historical_control_reused_read_only_for_calibration": True,
        "body_checkpoint_changed": False,
        "planner_changed": False,
        "training_run": False,
        "refinement_run": False,
        "direct_metrics_run": False,
        "sun_run": False,
        "retry_replacement_repair_filter_rerank": False,
        "formal_g3": False,
        "automatic_promotion": False,
        "automatic_downstream": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
