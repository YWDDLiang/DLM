#!/usr/bin/env python3
"""Adapt the retrained seed17 cohort to the frozen H1-A2 B0/D1 body contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from crystal_dlm.h1_llm_planner import (
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    teacher_formula_answer,
)
from crystal_dlm.h1a2_factorial_contract import (
    MODEL_SAMPLED_PLAN_PROVENANCE,
    build_planner_input_contract,
    persist_model_sampled_plan,
)
from crystal_dlm.ordinal_rng import sha256_text as frozen_sha256_text

from protocol import (
    DENOMINATOR,
    PLANNER_RAW_ATTEMPTS,
    canonical_sha256,
    ordered_rows,
    paired_seed,
    read_jsonl,
    sha256_file,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def _failed(
    *, row: Mapping[str, Any], raw: Mapping[str, Any], ordinal: int
) -> dict[str, Any]:
    return {
        "schema": "h1a2_factorial_contract_v1",
        "sample_idx": ordinal,
        "planner_arm": "P0",
        "attempt_status": "failed",
        "earliest_failure_stage": "planner",
        "failure_reason": str(row.get("ineligible_reason") or "planner_parse_failed"),
        "failure_message": str(raw.get("message") or ""),
        "raw_model_sampled_plan_text": str(raw.get("raw_plan_text") or ""),
        "planner_sampling_seed": int(row["planner_effective_rank_seed"]),
        "plan_provenance": MODEL_SAMPLED_PLAN_PROVENANCE,
        "model_proposed_plan": True,
        "registered_body_sampling_seed": paired_seed(0, ordinal, "body"),
        "registered_refiner_sampling_seed": paired_seed(0, ordinal, "refiner"),
        "retry_used": False,
        "replacement_used": False,
        "repair_used": False,
        "filter_used": False,
        "rerank_used": False,
        "source_cohort_row_sha256": canonical_sha256(row),
        "source_raw_row_sha256": canonical_sha256(raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--planner-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort = ordered_rows(read_jsonl(args.cohort.resolve()), ordinal_field="cohort_ordinal")
    raw_rows = read_jsonl(args.raw_generations.resolve())
    if len(raw_rows) != PLANNER_RAW_ATTEMPTS:
        raise ValueError("raw planner denominator changed")
    raw_by_sample: dict[int, dict[str, Any]] = {}
    for raw in raw_rows:
        sample_idx = int(raw.get("sample_idx", -1))
        if sample_idx in raw_by_sample:
            raise ValueError("duplicate raw planner sample_idx")
        raw_by_sample[sample_idx] = raw
    if set(raw_by_sample) != set(range(PLANNER_RAW_ATTEMPTS)):
        raise ValueError("raw planner sample_idx coverage changed")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model.resolve(), trust_remote_code=True, local_files_only=True
    )
    input_contract = build_planner_input_contract(
        tokenizer,
        planner_arm="P0",
        checkpoint_sha256=args.planner_checkpoint_sha256,
    )
    attempts: list[dict[str, Any]] = []
    raw_warnings = 0
    for ordinal, row in enumerate(cohort):
        sample_idx = int(row["planner_sample_idx"])
        raw = raw_by_sample[sample_idx]
        if canonical_sha256(raw) != row.get("raw_record_sha256"):
            raise ValueError(f"raw planner identity changed at ordinal {ordinal}")
        if row.get("body_eligible") is not True:
            attempts.append(_failed(row=row, raw=raw, ordinal=ordinal))
            continue
        plan = row.get("plan_state")
        if not isinstance(plan, Mapping):
            raise ValueError(f"eligible cohort row lacks plan_state at {ordinal}")
        canonical_text = teacher_formula_answer(
            plan, prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN
        )
        rank_seed = int(row["planner_effective_rank_seed"])
        record = persist_model_sampled_plan(
            canonical_text,
            planner_arm="P0",
            sample_idx=ordinal,
            planner_sampling_seed=rank_seed,
            planner_input_contract=input_contract,
            max_atoms=20,
        )
        if canonical_sha256(record["plan_state"]) != row.get("plan_state_sha256"):
            raise ValueError(f"H1-A2 canonical plan_state changed at {ordinal}")
        raw_text = str(raw.get("raw_plan_text") or "")
        warning: str | None = None
        try:
            persist_model_sampled_plan(
                raw_text,
                planner_arm="P0",
                sample_idx=ordinal,
                planner_sampling_seed=rank_seed,
                planner_input_contract=input_contract,
                max_atoms=20,
            )
        except ValueError as exc:
            warning = str(exc)
            raw_warnings += 1
        record.update(
            {
                "attempt_status": "complete",
                "earliest_failure_stage": None,
                "failure_reason": None,
                "failure_message": None,
                "raw_model_sampled_plan_text": raw_text,
                "raw_plan_text_sha256": frozen_sha256_text(raw_text),
                "frozen_canonical_plan_text": canonical_text,
                "canonical_plan_text_source": "frozen_retrained_plan_state",
                "raw_plan_format_gate": "advisory_nonblocking",
                "raw_plan_contract_conforming": warning is None,
                "raw_plan_contract_warning": warning,
                "canonicalization_used": raw_text != canonical_text,
                "registered_body_sampling_seed": paired_seed(0, ordinal, "body"),
                "registered_refiner_sampling_seed": paired_seed(0, ordinal, "refiner"),
                "planner_rng_provenance": "legacy_world2_stateful_rank_seed",
                "planner_rank": int(row["planner_rank"]),
                "planner_effective_rank_seed": rank_seed,
                "source_cohort_row_sha256": canonical_sha256(row),
                "source_raw_row_sha256": canonical_sha256(raw),
            }
        )
        attempts.append(record)

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    attempts_path = output / "planner_attempts.jsonl"
    write_jsonl_exclusive(attempts_path, attempts)
    write_json_exclusive(output / "planner_input_contract.json", input_contract)
    report = {
        "schema": "h1a2_retrained_seed17_to_historical_b0_input_adapter_v1",
        "status": "complete",
        "attempts": DENOMINATOR,
        "source_raw_attempts": PLANNER_RAW_ATTEMPTS,
        "source_selection": "first_256_raw_records_in_merged_file_order_with_failures_preserved",
        "complete": sum(row["attempt_status"] == "complete" for row in attempts),
        "failed": sum(row["attempt_status"] == "failed" for row in attempts),
        "raw_plan_contract_warnings": raw_warnings,
        "planner_attempts_sha256": sha256_file(attempts_path),
        "planner_input_contract_sha256": sha256_file(
            output / "planner_input_contract.json"
        ),
        "source_cohort_sha256": sha256_file(args.cohort.resolve()),
        "source_raw_generations_sha256": sha256_file(args.raw_generations.resolve()),
        "body_policy": "historical_H1A2_B0_d1_exact_plan_schedule",
        "body_and_refiner_seed_source": "frozen_20260731_attempt_ledger",
        "same_seed17_world2_plans_as_topology_R03": True,
        "retry_replacement_repair_filter_rerank": False,
    }
    report["report_payload_sha256"] = canonical_sha256(report)
    write_json_exclusive(output / "input_adapter_report.json", report)
    (output / "inputs_SUCCESS").touch(exist_ok=False)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
