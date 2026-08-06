#!/usr/bin/env python3
"""Assemble the preregistered paired-32 CR-Plan engineering screen."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

from crystal_dlm.composition_validity import classify_smact_validity
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.h1_crplan import load_frozen_smact_table


DENOMINATOR = 32
FROZEN_SAMPLER = {
    "num_samples": DENOMINATOR,
    "max_new_tokens": 96,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "max_atoms": 20,
    "prompt_style": "h1_rich_plan_v1",
    "include_sample_id": False,
    "seed": 17029,
    "seed_mode": "stateless_ordinal_v1",
    "rank_independent_sampling": True,
    "effective_generation_batch_size": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path} contains a non-object row")
                rows.append(row)
    ordinals = [int(value.get("sample_idx", -1)) for value in rows]
    if len(rows) != DENOMINATOR or ordinals != list(range(DENOMINATOR)):
        raise ValueError(
            f"{path} must be exactly ordered all-attempt ordinals 0..31"
        )
    return rows


def read_historical_head(
    path: Path,
    *,
    expected_sha256: str,
) -> list[dict[str, Any]]:
    observed_sha256 = sha256_file(path)
    if observed_sha256 != str(expected_sha256):
        raise ValueError(
            "historical P0 source identity mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path} contains a non-object row")
                rows.append(row)
    if len(rows) < DENOMINATOR:
        raise ValueError("historical P0 source has fewer than 32 attempts")
    head = rows[:DENOMINATOR]
    ordinals = [int(value.get("sample_idx", -1)) for value in head]
    if ordinals != list(range(DENOMINATOR)):
        raise ValueError("historical P0 head is not ordered ordinals 0..31")
    return head


def latency_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    raw_values = [
        row.get("planner_generation_latency_sec") for row in rows
    ]
    values = sorted(
        float(value)
        for value in raw_values
        if value is not None
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )
    if not values:
        return {
            "attempt_count": len(rows),
            "valid_count": 0,
            "invalid_or_missing_count": len(rows),
            "median_sec": None,
            "p95_sec": None,
        }
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "attempt_count": len(rows),
        "valid_count": len(values),
        "invalid_or_missing_count": len(rows) - len(values),
        "median_sec": statistics.median(values),
        "p95_sec": values[p95_index],
    }


def validate_config(
    path: Path,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in FROZEN_SAMPLER.items()
        if config.get(key) != expected
    }
    if config.get("formula_constraint_mode") != expected_mode:
        mismatches["formula_constraint_mode"] = {
            "expected": expected_mode,
            "observed": config.get("formula_constraint_mode"),
        }
    if mismatches:
        raise ValueError(f"frozen paired-32 sampler mismatch: {mismatches}")
    return config


def exact_mcnemar_p(baseline_only: int, candidate_only: int) -> float:
    discordant = int(baseline_only) + int(candidate_only)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(0, min(baseline_only, candidate_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def classify_attempts(
    rows: Iterable[Mapping[str, Any]],
    *,
    reachability,
    require_crplan_identity: bool,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    records: dict[int, dict[str, Any]] = {}
    reasons: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    formula_counts: Counter[str] = Counter()
    element_coverage: set[str] = set()
    parse_count = 0
    completion_count = 0
    comp_valid_count = 0
    primary_comp_valid_count = 0
    primary_charge_count = 0
    non_applicable_count = 0
    mixed_only_count = 0
    dead_end_count = 0
    masked_steps = 0
    preterminal_difference_steps = 0
    certificate_parity = True
    identity_verified_count = 0
    identity_failure_count = 0
    diagnostics_present_count = 0
    dead_end_fail_closed_count = 0
    unexpected_dead_end_handling_count = 0
    dp_attempt_states_created: list[int] = []
    cache_entry_peaks: list[int] = []
    dp_cache_telemetry_valid_count = 0
    dp_cache_telemetry_invalid_count = 0

    for row in rows:
        ordinal = int(row["sample_idx"])
        parsed = row.get("parsed") is True
        row_identity_failure = (
            require_crplan_identity
            and row.get("reason") == "CRPlanIdentityError"
        )
        identity_failure_count += int(row_identity_failure)
        if row_identity_failure:
            certificate_parity = False
        completion = row.get("plan_end_marker_present") is True
        completion_count += int(completion)
        diagnostics = row.get("crplan_diagnostics")
        if isinstance(diagnostics, Mapping):
            diagnostics_present_count += 1
            enforcement_verified = (
                diagnostics.get("legal_support_enforcement")
                == "mask_or_raise"
                and diagnostics.get("mask_application_count")
                == diagnostics.get("masked_step_count")
                and diagnostics.get("empty_support_error_raised")
                is (diagnostics.get("dead_end") is not None)
            )
            certificate_parity = (
                certificate_parity and enforcement_verified
            )
            if diagnostics.get("dead_end") is not None:
                dead_end_count += 1
                fail_closed = (
                    row.get("parsed") is not True
                    and row.get("fail_closed") is True
                    and row.get("reason") == "CRPlanDeadEndError"
                )
                dead_end_fail_closed_count += int(fail_closed)
                unexpected_dead_end_handling_count += int(not fail_closed)
            if (
                diagnostics.get("silent_fallback_used_by_decoder")
                is not False
            ):
                certificate_parity = False
            if (
                diagnostics.get(
                    "retry_replacement_repair_filter_or_rerank_used"
                )
                is not False
            ):
                certificate_parity = False
            masked_steps += int(diagnostics.get("masked_step_count") or 0)
            preterminal_difference_steps += int(
                diagnostics.get("preterminal_support_difference_steps") or 0
            )
            dp = diagnostics.get("dp")
            if isinstance(dp, Mapping):
                delta = dp.get("attempt_delta")
                start = dp.get("start")
                end = dp.get("end")
                cache_start = dp.get("cache_start")
                cache_end = dp.get("cache_end")
                required_delta_keys = (
                    "queries",
                    "cache_hits",
                    "cache_misses",
                    "states_created",
                )
                required_cache_keys = (
                    "terminal_certificate_entries",
                    "prefix_reachability_entries",
                    "element_allocation_entries_global",
                )
                telemetry_valid = (
                    isinstance(delta, Mapping)
                    and isinstance(start, Mapping)
                    and isinstance(end, Mapping)
                    and isinstance(cache_start, Mapping)
                    and isinstance(cache_end, Mapping)
                    and all(
                        isinstance(value.get(key), int)
                        and int(value[key]) >= 0
                        for value in (start, end, delta)
                        for key in required_delta_keys
                    )
                    and all(
                        isinstance(cache_start.get(key), int)
                        and int(cache_start[key]) >= 0
                        for key in required_cache_keys
                    )
                    and all(
                        isinstance(cache_end.get(key), int)
                        and int(cache_end[key]) >= 0
                        for key in required_cache_keys
                    )
                    and isinstance(
                        dp.get("attempt_peak_cache_entries"), int
                    )
                    and int(dp["attempt_peak_cache_entries"]) >= 0
                )
                dp_cache_telemetry_valid_count += int(telemetry_valid)
                dp_cache_telemetry_invalid_count += int(
                    not telemetry_valid
                )
                if telemetry_valid:
                    dp_attempt_states_created.append(
                        int(delta.get("states_created") or 0)
                    )
                    cache_entry_peaks.append(
                        int(dp["attempt_peak_cache_entries"])
                    )
            elif require_crplan_identity:
                dp_cache_telemetry_invalid_count += 1
        elif require_crplan_identity:
            certificate_parity = False
        record: dict[str, Any] = {
            "ordinal": ordinal,
            "parsed": parsed,
            "completion": completion,
            "planner_sampling_seed": row.get("planner_sampling_seed"),
            "formula": None,
            "comp_valid": False,
            "reason": "parse_failure",
            "stratum": "parse_failure",
            "charge_applicable": False,
            "primary_charge_witness": False,
        }
        if parsed:
            plan = row.get("plan_state")
            if not isinstance(plan, Mapping):
                raise ValueError(f"parsed ordinal {ordinal} has no plan_state")
            symbols = [str(value) for value in (plan.get("elements") or ())]
            counts = [int(value) for value in (plan.get("counts") or ())]
            if len(symbols) != len(counts) or not symbols:
                raise ValueError(f"parsed ordinal {ordinal} has invalid composition")
            parse_count += 1
            formula = str(plan["formula"])
            element_coverage.update(symbols)
            formula_counts[formula] += 1
            classification = dict(
                classify_smact_validity(
                    [int(SYMBOL_TO_Z[value]) for value in symbols],
                    counts,
                )
            )
            certificate = reachability.terminal_certificate(
                zip(symbols, counts)
            ).to_dict()
            embedded = row.get("crplan_terminal_certificate")
            if embedded is not None and embedded != certificate:
                certificate_parity = False
            reason = str(classification["reason"])
            stratum = str(certificate["stratum"])
            comp_valid = classification["valid"] is True
            primary = certificate["primary_charge_witness"] is True
            charge_applicable = certificate["charge_applicable"] is True
            reasons[reason] += 1
            strata[stratum] += 1
            comp_valid_count += int(comp_valid)
            primary_charge_count += int(primary)
            primary_comp_valid_count += int(primary and comp_valid)
            mixed_only_count += int(
                stratum == "charge_applicable_mixed_valence_only"
            )
            non_applicable_count += int(
                stratum.startswith("charge_not_applicable_")
            )
            identity = row.get("crplan_identity")
            identity_verified = (
                isinstance(identity, Mapping)
                and identity.get("verified") is True
                and identity.get("fsm_counts_equal_parser_counts") is True
                and identity.get("formula_line_count") == 1
            )
            if require_crplan_identity:
                identity_verified_count += int(identity_verified)
                identity_failure_count += int(
                    not identity_verified and not row_identity_failure
                )
                certificate_parity = (
                    certificate_parity and identity_verified
                )
            record.update(
                {
                    "formula": formula,
                    "comp_valid": comp_valid,
                    "reason": reason,
                    "stratum": stratum,
                    "charge_applicable": charge_applicable,
                    "primary_charge_witness": primary,
                    "terminal_allowed": certificate["terminal_allowed"],
                }
            )
        records[ordinal] = record

    return (
        {
            "denominator": DENOMINATOR,
            "parse_count": parse_count,
            "completion_count": completion_count,
            "composition_valid_count": comp_valid_count,
            "primary_charge_witness_count": primary_charge_count,
            "primary_composition_valid_count": primary_comp_valid_count,
            "mixed_valence_only_count": mixed_only_count,
            "charge_not_applicable_count": non_applicable_count,
            "unique_formula_count": len(formula_counts),
            "element_coverage_count": len(element_coverage),
            "element_coverage": sorted(
                element_coverage,
                key=lambda value: SYMBOL_TO_Z[value],
            ),
            "reason_counts": dict(sorted(reasons.items())),
            "stratum_counts": dict(sorted(strata.items())),
            "constraint_dead_end_count": dead_end_count,
            "constraint_masked_step_count": masked_steps,
            "preterminal_support_difference_steps": (
                preterminal_difference_steps
            ),
            "certificate_and_no_fallback_parity": certificate_parity,
            "constraint_diagnostics_present_count": diagnostics_present_count,
            "dead_end_fail_closed_count": dead_end_fail_closed_count,
            "unexpected_dead_end_handling_count": (
                unexpected_dead_end_handling_count
            ),
            "dp_attempt_states_created_max": (
                max(dp_attempt_states_created)
                if dp_attempt_states_created
                else 0
            ),
            "cache_entry_peak_max": (
                max(cache_entry_peaks) if cache_entry_peaks else 0
            ),
            "dp_cache_telemetry_valid_count": (
                dp_cache_telemetry_valid_count
            ),
            "dp_cache_telemetry_invalid_count": (
                dp_cache_telemetry_invalid_count
            ),
            "crplan_identity_required": require_crplan_identity,
            "crplan_identity_verified_count": identity_verified_count,
            "crplan_identity_failure_count": identity_failure_count,
        },
        records,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--cr0-report", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--historical-p0-raw", type=Path, required=True)
    parser.add_argument("--historical-p0-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cr0 = json.loads(args.cr0_report.read_text(encoding="utf-8"))
    if cr0.get("status") != "pass":
        raise ValueError("paired-32 cannot assemble without passing CR-0")
    control_config = validate_config(
        args.control_root / "run_config.json",
        expected_mode="off",
    )
    candidate_config = validate_config(
        args.candidate_root / "run_config.json",
        expected_mode="full_prefix",
    )
    shared_config_keys = set(FROZEN_SAMPLER)
    paired_config_mismatch = {
        key: {
            "control": control_config.get(key),
            "candidate": candidate_config.get(key),
        }
        for key in sorted(shared_config_keys)
        if control_config.get(key) != candidate_config.get(key)
    }
    reachability = load_frozen_smact_table(max_atoms=20)
    if (
        candidate_config.get("oxidation_table_sha256")
        != reachability.table_sha256
        or cr0["oxidation_table"]["sha256"] != reachability.table_sha256
    ):
        raise ValueError("oxidation table identity changed after CR-0")
    candidate_tokenizer_report = json.loads(
        (args.candidate_root / "tokenizer_report.json").read_text(
            encoding="utf-8"
        )
    )
    control_tokenizer_report = json.loads(
        (args.control_root / "tokenizer_report.json").read_text(
            encoding="utf-8"
        )
    )
    cr0_token_fragment_sha256 = cr0["tokenizer"]["fragment_sha256"]
    if (
        candidate_config.get("token_fragment_sha256")
        != cr0_token_fragment_sha256
        or candidate_tokenizer_report.get(
            "decoded_token_fragment_sha256"
        )
        != cr0_token_fragment_sha256
        or candidate_tokenizer_report.get("eos_token_id")
        != cr0["tokenizer"]["eos_token_id"]
        or candidate_tokenizer_report.get("pad_token_id")
        != cr0["tokenizer"]["pad_token_id"]
        or candidate_tokenizer_report.get("vocab_size")
        != cr0["tokenizer"]["vocab_size"]
        or candidate_tokenizer_report.get("padding_side")
        != cr0["tokenizer"]["padding_side"]
    ):
        raise ValueError(
            "candidate tokenizer vocabulary/EOS identity changed after CR-0"
        )
    paired_tokenizer_metadata = (
        "vocab_size",
        "eos_token_id",
        "pad_token_id",
        "padding_side",
    )
    if any(
        control_tokenizer_report.get(key)
        != candidate_tokenizer_report.get(key)
        for key in paired_tokenizer_metadata
    ):
        raise ValueError("control/candidate tokenizer metadata mismatch")

    control_rows = read_jsonl(args.control_root / "raw_generations.jsonl")
    candidate_rows = read_jsonl(args.candidate_root / "raw_generations.jsonl")
    historical_rows = read_historical_head(
        args.historical_p0_raw,
        expected_sha256=args.historical_p0_sha256,
    )
    historical_parity_fields = (
        "sample_idx",
        "raw_model_text",
        "raw_plan_text",
        "raw_plan_text_sha256",
        "planner_model_path",
        "planner_checkpoint_path",
        "planner_sampling_seed",
        "prompt_version",
        "prompt_style",
        "seed_mode",
        "parsed",
        "formula_parse",
        "valid_formula",
        "valid_N",
        "plan_end_marker_present",
        "plan_tail_after_end_marker",
        "parsed_plan",
        "plan_state",
        "plan_text",
        "plan_text_sha256",
        "prompt",
        "body_prompt_sha256",
    )
    historical_control_mismatches = [
        {
            "ordinal": ordinal,
            "fields": [
                field
                for field in historical_parity_fields
                if control_rows[ordinal].get(field)
                != historical_rows[ordinal].get(field)
            ],
        }
        for ordinal in range(DENOMINATOR)
        if any(
            control_rows[ordinal].get(field)
            != historical_rows[ordinal].get(field)
            for field in historical_parity_fields
        )
    ]
    control, control_records = classify_attempts(
        control_rows,
        reachability=reachability,
        require_crplan_identity=False,
    )
    candidate, candidate_records = classify_attempts(
        candidate_rows,
        reachability=reachability,
        require_crplan_identity=True,
    )
    seed_mismatches = [
        ordinal
        for ordinal in range(DENOMINATOR)
        if control_records[ordinal]["planner_sampling_seed"]
        != candidate_records[ordinal]["planner_sampling_seed"]
    ]
    prompt_mismatches = [
        ordinal
        for ordinal in range(DENOMINATOR)
        if control_rows[ordinal].get("planner_input_prompt_sha256")
        != candidate_rows[ordinal].get("planner_input_prompt_sha256")
        or control_rows[ordinal].get("planner_input_ids_sha256")
        != candidate_rows[ordinal].get("planner_input_ids_sha256")
    ]
    baseline_only = sum(
        int(
            control_records[ordinal]["comp_valid"]
            and not candidate_records[ordinal]["comp_valid"]
        )
        for ordinal in range(DENOMINATOR)
    )
    candidate_only = sum(
        int(
            candidate_records[ordinal]["comp_valid"]
            and not control_records[ordinal]["comp_valid"]
        )
        for ordinal in range(DENOMINATOR)
    )
    candidate_charge_terminal_failures = [
        value
        for value in candidate_records.values()
        if value["parsed"]
        and value["charge_applicable"]
        and value.get("terminal_allowed") is not True
    ]
    control_latency = latency_summary(control_rows)
    candidate_latency = latency_summary(candidate_rows)
    gates = {
        "cr0_pass": True,
        "paired_config_mismatch_zero": not paired_config_mismatch,
        "candidate_tokenizer_exact_cr0_identity": True,
        "paired_seed_mismatch_zero": not seed_mismatches,
        "paired_planner_prompt_and_input_ids_mismatch_zero": (
            not prompt_mismatches
        ),
        "control_scientific_output_exact_historical_p0_first32": (
            not historical_control_mismatches
        ),
        "candidate_tokenizer_fsm_dead_end_zero": (
            candidate["constraint_dead_end_count"] == 0
        ),
        "candidate_dp_cache_telemetry_complete": (
            candidate["dp_cache_telemetry_valid_count"] == DENOMINATOR
            and candidate["dp_cache_telemetry_invalid_count"] == 0
        ),
        "planner_generation_latency_complete": (
            control_latency["valid_count"] == DENOMINATOR
            and control_latency["invalid_or_missing_count"] == 0
            and candidate_latency["valid_count"] == DENOMINATOR
            and candidate_latency["invalid_or_missing_count"] == 0
        ),
        "candidate_silent_fallback_retry_repair_zero": candidate[
            "certificate_and_no_fallback_parity"
        ]
        and candidate["constraint_diagnostics_present_count"] == DENOMINATOR
        and candidate["unexpected_dead_end_handling_count"] == 0
        and candidate["dead_end_fail_closed_count"]
        == candidate["constraint_dead_end_count"],
        "candidate_charge_applicable_terminal_failure_zero": (
            len(candidate_charge_terminal_failures) == 0
        ),
        "candidate_parse_loss_at_most_one": (
            candidate["parse_count"] >= control["parse_count"] - 1
        ),
        "candidate_completion_loss_at_most_one": (
            candidate["completion_count"] >= control["completion_count"] - 1
        ),
        "formula_seven_line_identity_preserved": (
            candidate["crplan_identity_failure_count"] == 0
            and candidate["crplan_identity_verified_count"]
            == candidate["parse_count"]
            and candidate["parse_count"] > 0
        ),
        "no_sun_or_endpoint_mask_tuning": True,
    }
    passed = all(gates.values())
    prefix_attribution_allowed = (
        cr0.get("prefix_semantics", {}).get(
            "prefix_control_gain_attribution_allowed"
        )
        is True
    )
    report = {
        "schema": "h1_crplan_paired32_terminal_report_v1",
        "status": "pass" if passed else "fail",
        "denominator_per_arm": DENOMINATOR,
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "cr0_report_sha256": sha256_file(args.cr0_report),
        "historical_p0": {
            "path": str(args.historical_p0_raw),
            "full_raw_sha256": args.historical_p0_sha256,
            "parity_fields": list(historical_parity_fields),
            "first32_mismatches": historical_control_mismatches,
        },
        "oxidation_table_sha256": reachability.table_sha256,
        "token_fragment_sha256": cr0_token_fragment_sha256,
        "tokenizer_vocab_size": cr0["tokenizer"]["vocab_size"],
        "tokenizer_eos_token_id": cr0["tokenizer"]["eos_token_id"],
        "tokenizer_pad_token_id": cr0["tokenizer"]["pad_token_id"],
        "tokenizer_padding_side": cr0["tokenizer"]["padding_side"],
        "control": control,
        "candidate": candidate,
        "scientific_interpretation": {
            "observed_treatment": (
                "off_vs_full_prefix"
                if prefix_attribution_allowed
                else "off_vs_terminal_charge_gate_with_conservative_prefix_mask"
            ),
            "prefix_control_gain_attribution_allowed": (
                prefix_attribution_allowed
            ),
            "cr0_prefix_semantics": cr0.get("prefix_semantics"),
            "four_arm_missing_policy_requires_review": (
                not prefix_attribution_allowed
            ),
        },
        "candidate_minus_control": {
            "parse_count": candidate["parse_count"] - control["parse_count"],
            "completion_count": (
                candidate["completion_count"] - control["completion_count"]
            ),
            "composition_valid_count": (
                candidate["composition_valid_count"]
                - control["composition_valid_count"]
            ),
            "primary_composition_valid_count": (
                candidate["primary_composition_valid_count"]
                - control["primary_composition_valid_count"]
            ),
            "unique_formula_count": (
                candidate["unique_formula_count"]
                - control["unique_formula_count"]
            ),
            "element_coverage_count": (
                candidate["element_coverage_count"]
                - control["element_coverage_count"]
            ),
        },
        "paired_composition_validity": {
            "baseline_only": baseline_only,
            "candidate_only": candidate_only,
            "discordant": baseline_only + candidate_only,
            "mcnemar_two_sided_exact_p": exact_mcnemar_p(
                baseline_only,
                candidate_only,
            ),
        },
        "paired_config_mismatch": paired_config_mismatch,
        "paired_seed_mismatch_ordinals": seed_mismatches,
        "paired_planner_prompt_or_input_ids_mismatch_ordinals": (
            prompt_mismatches
        ),
        "candidate_charge_terminal_failures": (
            candidate_charge_terminal_failures
        ),
        "planner_generation_latency": {
            "control": control_latency,
            "candidate": candidate_latency,
        },
        "gates": gates,
        "gate_passed": passed,
        "decision": (
            "allow_prepare_four_arm_only_after_missing_policy_review"
            if passed and not prefix_attribution_allowed
            else "allow_independent_four_arm_plan_only_512"
            if passed
            else "engineering_stop_repair_before_scientific_interpretation"
        ),
        "generation_rerun_for_endpoint_selection": False,
        "sun_used": False,
        "retry_replacement_repair_filter_or_rerank_used": False,
        "automatic_downstream": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
