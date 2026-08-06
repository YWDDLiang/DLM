#!/usr/bin/env python3
"""Compile frozen real-ledger Plans without model inference or generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
sys.path.insert(0, str(RUNTIME))

from crystal_dlm.h1a2_factorial_contract import (  # noqa: E402
    MODEL_SAMPLED_PLAN_PROVENANCE,
    assert_planner_input_identity,
    build_factorial_arm_input,
    build_factorial_ordinal_record,
    persist_parser_accepted_model_sampled_plan,
)
from crystal_dlm.ordinal_rng import sha256_text  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> Path:
    location = path.resolve()
    if not location.is_file():
        raise FileNotFoundError(f"{label} is missing: {location}")
    observed = sha256_file(location)
    if observed != str(expected):
        raise ValueError(f"{label} SHA changed: {observed}")
    return location


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_exclusive(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_row_sha256(source: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        source,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(encoded)


def convert_parsed_row(
    *,
    planner_arm: str,
    factorial_arm: str,
    source: Mapping[str, Any],
    sample_idx: int,
    planner_base_seed: int,
    input_contract: Mapping[str, Any],
) -> dict[str, Any]:
    ordinal = build_factorial_ordinal_record(
        planner_base_seed,
        sample_idx=sample_idx,
    )
    expected_seed = int(ordinal["planner_sampling_seed"])
    if int(source.get("planner_sampling_seed", -1)) != expected_seed:
        raise ValueError("frozen Planner seed changed")
    if source.get("parsed") is not True:
        raise ValueError("convert_parsed_row requires parsed=true")

    raw_text = str(source.get("raw_plan_text") or "")
    canonical_text = str(source.get("plan_text") or "")
    persisted = persist_parser_accepted_model_sampled_plan(
        raw_text,
        canonical_text,
        planner_arm=planner_arm,
        sample_idx=sample_idx,
        planner_sampling_seed=expected_seed,
        planner_input_contract=input_contract,
        max_atoms=20,
    )
    body = build_factorial_arm_input(
        persisted,
        factorial_arm=factorial_arm,
        ordinal_record=ordinal,
    )
    if body["raw_plan_text_sha256"] != sha256_text(raw_text):
        raise ValueError("raw sampled Plan SHA mismatch")
    if body["plan_text_sha256"] != sha256_text(canonical_text):
        raise ValueError("canonical sampled Plan SHA mismatch")
    if body["body_prompt_sha256"] != persisted["body_prompt_sha256"]:
        raise ValueError("compiled body prompt SHA mismatch")

    persisted.update(
        {
            "attempt_status": "complete",
            "body_compilation_reached": True,
            "factorial_arm_checked": factorial_arm,
            "source_planner_row_sha256": source_row_sha256(source),
        }
    )
    return persisted


def failed_planner_row(
    *,
    planner_arm: str,
    source: Mapping[str, Any],
    sample_idx: int,
    planner_base_seed: int,
) -> dict[str, Any]:
    ordinal = build_factorial_ordinal_record(
        planner_base_seed,
        sample_idx=sample_idx,
    )
    expected_seed = int(ordinal["planner_sampling_seed"])
    if int(source.get("planner_sampling_seed", -1)) != expected_seed:
        raise ValueError("frozen Planner seed changed")
    raw_text = str(source.get("raw_plan_text") or "")
    return {
        "schema": "h1_real_ledger_plan_preflight_attempt_v1",
        "sample_idx": sample_idx,
        "planner_arm": planner_arm,
        "attempt_status": "failed",
        "earliest_failure_stage": "planner",
        "failure_reason": str(source.get("reason") or "planner_parse_failed"),
        "failure_message": str(source.get("message") or ""),
        "raw_model_sampled_plan_text": raw_text,
        "raw_plan_text_sha256": sha256_text(raw_text),
        "planner_sampling_seed": expected_seed,
        "plan_provenance": MODEL_SAMPLED_PLAN_PROVENANCE,
        "model_proposed_plan": True,
        "body_compilation_reached": False,
        "retry_used": False,
        "replacement_used": False,
        "repair_used": False,
        "filter_used": False,
        "rerank_used": False,
        "source_planner_row_sha256": source_row_sha256(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config.resolve())
    if (
        config.get("schema") != "h1_real_ledger_plan_preflight_config_v1"
        or int(config.get("denominator", -1)) != 256
        or config.get("automatic_downstream") is not False
    ):
        raise ValueError("preflight config or decision firewall changed")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    contracts: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for arm, arm_config in config["arms"].items():
        contract_path = require_sha(
            Path(arm_config["planner_input_contract"]),
            arm_config["planner_input_contract_sha256"],
            f"{arm} Planner input contract",
        )
        raw_path = require_sha(
            Path(arm_config["raw_generations"]),
            arm_config["raw_generations_sha256"],
            f"{arm} raw Planner generations",
        )
        contracts[arm] = read_json(contract_path)
        sources[arm] = read_jsonl(raw_path)
        if (
            contracts[arm].get("planner_arm") != arm
            or contracts[arm].get("include_sample_id") is not False
            or len(sources[arm]) != 512
            or [int(row.get("sample_idx", -1)) for row in sources[arm]]
            != list(range(512))
        ):
            raise ValueError(f"{arm} frozen Planner identity changed")
    assert_planner_input_identity(contracts["P0"], contracts["Pstar"])

    arm_reports: dict[str, dict[str, Any]] = {}
    gate_passed = True
    for arm, arm_config in config["arms"].items():
        rows: list[dict[str, Any]] = []
        conversion_errors: list[dict[str, Any]] = []
        for sample_idx, source in enumerate(sources[arm][:256]):
            if source.get("parsed") is not True:
                rows.append(
                    failed_planner_row(
                        planner_arm=arm,
                        source=source,
                        sample_idx=sample_idx,
                        planner_base_seed=int(config["planner_base_seed"]),
                    )
                )
                continue
            try:
                rows.append(
                    convert_parsed_row(
                        planner_arm=arm,
                        factorial_arm=arm_config["factorial_arm"],
                        source=source,
                        sample_idx=sample_idx,
                        planner_base_seed=int(config["planner_base_seed"]),
                        input_contract=contracts[arm],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                conversion_errors.append(
                    {
                        "sample_idx": sample_idx,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                rows.append(
                    {
                        "schema": "h1_real_ledger_plan_preflight_attempt_v1",
                        "sample_idx": sample_idx,
                        "planner_arm": arm,
                        "attempt_status": "conversion_failed",
                        "body_compilation_reached": False,
                        "raw_model_sampled_plan_text": str(
                            source.get("raw_plan_text") or ""
                        ),
                        "source_planner_row_sha256": source_row_sha256(source),
                    }
                )

        ledger_path = output / f"{arm}_converted_attempts.jsonl"
        write_jsonl_exclusive(ledger_path, rows)
        parsed = sum(source.get("parsed") is True for source in sources[arm][:256])
        compiled = sum(row.get("body_compilation_reached") is True for row in rows)
        warnings = sum(
            row.get("raw_plan_contract_conforming") is False for row in rows
        )
        canonicalized = sum(row.get("canonicalization_used") is True for row in rows)
        arm_passed = (
            len(rows) == 256
            and parsed == int(arm_config["expected_parsed"])
            and compiled == parsed
            and not conversion_errors
        )
        gate_passed = gate_passed and arm_passed
        arm_reports[arm] = {
            "attempts": len(rows),
            "parsed_true": parsed,
            "body_compilation_reached": compiled,
            "planner_failures_preserved": 256 - parsed,
            "raw_format_advisory_warnings": warnings,
            "canonicalized_raw_outputs": canonicalized,
            "conversion_error_count": len(conversion_errors),
            "conversion_errors": conversion_errors[:16],
            "converted_attempts_sha256": sha256_file(ledger_path),
            "gate_passed": arm_passed,
        }

    report = {
        "schema": "h1_real_ledger_plan_preflight_report_v1",
        "status": "complete" if gate_passed else "failed",
        "gate_passed": gate_passed,
        "denominator_per_arm": 256,
        "planner_base_seed": int(config["planner_base_seed"]),
        "arms": arm_reports,
        "model_inference_run": False,
        "body_generation_run": False,
        "refinement_run": False,
        "sun_run": False,
        "retry_replacement_repair_filter_rerank": False,
        "automatic_downstream": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_exclusive(output / "preflight_report.json", report)
    if gate_passed:
        (output / "_SUCCESS").touch(exist_ok=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
