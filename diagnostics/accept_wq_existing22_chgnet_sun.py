#!/usr/bin/env python3
"""Validate and seal the existing-22 CHGNet R5-C S.U.N. result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from diagnostics.prepare_wq_existing22_chgnet_sun import (
    CONTRACT_SCHEMA,
    Existing22SunInputError,
    load_json,
    load_jsonl,
    sha256_file,
    write_json_exclusive,
)


TERMINAL_SCHEMA = "wqcodiff_existing22_chgnet_sun_terminal_acceptance_v1"


def scientific_decision(
    *,
    strict_count: int,
    meta_count: int,
    unknown_count: int,
    minimum_strict: int,
    minimum_meta: int,
) -> str:
    if strict_count >= minimum_strict and meta_count >= minimum_meta:
        return "PASS"
    if (
        strict_count + unknown_count < minimum_strict
        or meta_count + unknown_count < minimum_meta
    ):
        return "FAIL"
    return "INCONCLUSIVE_MP_COVERAGE"


def validate_result(
    *,
    contract: Mapping[str, Any],
    adapter_manifest: Mapping[str, Any],
    sun_summary: Mapping[str, Any],
    sun_attempts: list[dict[str, Any]],
    sun_run_contract: Mapping[str, Any],
    execution_patch_sha256: str,
) -> dict[str, Any]:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise Existing22SunInputError("terminal contract schema changed")
    if (
        adapter_manifest.get("schema")
        != "wqcodiff_existing22_chgnet_sun_adapter_manifest_v1"
        or int(adapter_manifest.get("attempts", -1)) != 22
        or int(adapter_manifest.get("reconstructed_structures", -1)) != 17
        or int(adapter_manifest.get("failed_placeholders", -1)) != 5
        or adapter_manifest.get("new_generation") is not False
        or adapter_manifest.get("geometry_repair_or_rescue") is not False
        or adapter_manifest.get("retry_or_replacement_used") is not False
    ):
        raise Existing22SunInputError("adapter manifest integrity failed")
    failed_expected = sorted(
        int(value)
        for value in contract["denominator"][
            "frozen_structural_failure_ordinals"
        ]
    )
    if adapter_manifest.get("frozen_structural_failure_ordinals") != failed_expected:
        raise Existing22SunInputError("frozen failure ordinals changed")
    if (
        sun_summary.get("schema") != "crysllmgen_r5c_a100_sun_summary_v1"
        or sun_summary.get("ok") is not True
        or sun_summary.get("denominator") != "all_generation_attempts"
        or sun_summary.get("retry_or_replacement_used") is not False
        or sun_summary.get("execution_patch_sha256") != execution_patch_sha256
    ):
        raise Existing22SunInputError("exact S.U.N. summary integrity failed")
    counts = sun_summary.get("counts")
    if not isinstance(counts, Mapping):
        raise Existing22SunInputError("S.U.N. summary counts are missing")
    if (
        int(counts.get("total_attempts", -1)) != 22
        or int(counts.get("reconstructed", -1)) != 17
        or len(sun_attempts) != 22
        or len({row.get("attempt_id") for row in sun_attempts}) != 22
        or any(
            row.get("execution_patch_sha256") != execution_patch_sha256
            or row.get("retry_or_replacement_used") is not False
            for row in sun_attempts
        )
    ):
        raise Existing22SunInputError("all-22 S.U.N. attempt mapping changed")
    generated_failures = sorted(
        int(adapter_manifest["attempt_records"][int(row["generation_ordinal"])]["ordinal"])
        for row in sun_attempts
        if row.get("generation_status") == "failed"
    )
    if generated_failures != failed_expected:
        raise Existing22SunInputError("known structural failures were not preserved")
    thresholds = sun_summary.get("thresholds_ev_per_atom")
    if (
        not isinstance(thresholds, Mapping)
        or float(thresholds.get("strict", 1.0)) != 0.0
        or float(thresholds.get("meta_like", 1.0)) != 0.1
    ):
        raise Existing22SunInputError("S.U.N. thresholds changed")
    if (
        sun_run_contract.get("environment") != "diff_meets_diff"
        or int(sun_run_contract.get("threads", -1)) != 8
        or int(sun_run_contract.get("expected_attempts", -1)) != 22
        or "A800" not in str(sun_run_contract.get("cuda_device", ""))
        or sun_run_contract.get("offline") is not True
        or sun_run_contract.get("retry_or_replacement_used") is not False
    ):
        raise Existing22SunInputError("exact evaluator runtime contract changed")

    strict = int(counts["strict_full_sun"])
    meta = int(counts["meta_full_sun"])
    unknown = int(counts["relaxation_or_hull_unknown"])
    if strict < 0 or meta < strict or unknown < 0 or meta + unknown > 17:
        raise Existing22SunInputError("S.U.N. count relationships are invalid")
    minimum_strict = int(
        contract["decision_rule"]["minimum_strict_full_sun_count"]
    )
    minimum_meta = int(
        contract["decision_rule"]["minimum_meta_full_sun_count"]
    )
    decision = scientific_decision(
        strict_count=strict,
        meta_count=meta,
        unknown_count=unknown,
        minimum_strict=minimum_strict,
        minimum_meta=minimum_meta,
    )
    consequence_key = {
        "PASS": "pass_consequence",
        "FAIL": "fail_consequence",
        "INCONCLUSIVE_MP_COVERAGE": "inconclusive_consequence",
    }[decision]
    return {
        "scientific_decision": decision,
        "decision_consequence": contract["decision_rule"][consequence_key],
        "minimum_counts": {
            "strict_full_sun": minimum_strict,
            "meta_full_sun": minimum_meta,
        },
        "observed_counts": {
            "strict_full_sun": strict,
            "meta_full_sun": meta,
            "relaxation_or_hull_unknown": unknown,
        },
        "optimistic_upper_counts": {
            "strict_full_sun": strict + unknown,
            "meta_full_sun": meta + unknown,
        },
    }


def execute(
    *,
    contract_path: Path,
    output_directory: Path,
    execution_patch_sha256: str,
    gpu_csv: Path,
) -> dict[str, Any]:
    if len(execution_patch_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in execution_patch_sha256
    ):
        raise Existing22SunInputError("execution patch is not a lowercase SHA256")
    contract_path = contract_path.resolve()
    output = output_directory.resolve()
    gpu_csv = gpu_csv.resolve()
    contract = load_json(contract_path)
    adapter_manifest_path = output / str(
        contract["output"]["adapter_manifest"]
    )
    sun_directory = output / str(contract["output"]["sun_directory"])
    sun_summary_path = sun_directory / "attempt_summary.json"
    sun_attempts_path = sun_directory / "attempt_results.jsonl"
    sun_run_contract_path = sun_directory / "run_contract.json"
    terminal_path = output / str(contract["output"]["terminal_acceptance"])
    if terminal_path.exists():
        raise FileExistsError(terminal_path)
    for path in (
        adapter_manifest_path,
        sun_summary_path,
        sun_attempts_path,
        sun_run_contract_path,
        gpu_csv,
    ):
        if not path.is_file():
            raise Existing22SunInputError(f"terminal input is missing: {path}")
    adapter_manifest = load_json(adapter_manifest_path)
    sun_summary = load_json(sun_summary_path)
    sun_attempts = load_jsonl(sun_attempts_path)
    sun_run_contract = load_json(sun_run_contract_path)
    decision = validate_result(
        contract=contract,
        adapter_manifest=adapter_manifest,
        sun_summary=sun_summary,
        sun_attempts=sun_attempts,
        sun_run_contract=sun_run_contract,
        execution_patch_sha256=execution_patch_sha256,
    )
    payload = {
        "schema": TERMINAL_SCHEMA,
        "ok": True,
        "evaluation_integrity": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "run_id": contract["run_id"],
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "authorization_record_sha256": contract["authorization"]["sha256"],
        "execution_patch_sha256": execution_patch_sha256,
        "historical_formal_survival_gate": {
            "result": contract["frozen_history"]["formal_survival_result"],
            "formal_gate_rewritten": False,
            "observed_survival": contract["frozen_history"]["observed_survival"],
        },
        "continuation_identity": "user_accepted_exploratory_gate",
        "denominator": "all_existing_projected_states",
        "attempts": 22,
        "reconstructed_structures": 17,
        "frozen_structural_failures": 5,
        "known_structural_failures_relaxed_or_rescued": False,
        "adapter_manifest_sha256": sha256_file(adapter_manifest_path),
        "adapter_generation_sha256": adapter_manifest[
            "adapter_generation_sha256"
        ],
        "sun_summary_sha256": sha256_file(sun_summary_path),
        "sun_attempt_results_sha256": sha256_file(sun_attempts_path),
        "sun_run_contract_sha256": sha256_file(sun_run_contract_path),
        "gpu_csv_sha256": sha256_file(gpu_csv),
        "scientific_decision": decision["scientific_decision"],
        "decision_consequence": decision["decision_consequence"],
        "minimum_counts": decision["minimum_counts"],
        "observed_counts": decision["observed_counts"],
        "optimistic_upper_counts": decision["optimistic_upper_counts"],
        "rates": sun_summary["rates"],
        "historical_directional_reference": contract[
            "historical_directional_reference"
        ],
        "mp_api_calls": 0,
        "new_generation": False,
        "training": False,
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(terminal_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = execute(
        contract_path=args.contract,
        output_directory=args.output_dir,
        execution_patch_sha256=args.execution_patch_sha256,
        gpu_csv=args.gpu_csv,
    )
    print("WQ_EXISTING22_CHGNET_SUN_EVALUATION_INTEGRITY=PASS")
    print(f"WQ_EXISTING22_CHGNET_SUN_DECISION={payload['scientific_decision']}")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
