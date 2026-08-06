#!/usr/bin/env python3
"""Assemble the preregistered H1 CR-Plan four-arm 512 terminal."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from crystal_dlm.composition_validity import classify_smact_validity
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.h1_crplan import load_frozen_smact_table


IDENTITY = "h1_crplan_fourarm512_route_amendment_v1"
DENOMINATOR = 512
MODES = ("off", "grammar_only", "terminal_only", "full_prefix")
FORBIDDEN_TRUE_FIELDS = (
    "retry",
    "replacement",
    "repair",
    "filter",
    "rerank",
    "fallback",
    "body_rerun",
    "refiner_rerun",
    "direct_rerun",
    "sun_rerun",
    "network",
    "training",
    "checkpoint_reselection",
    "automatic_downstream",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object row")
                rows.append(value)
    ordinals = [int(row.get("sample_idx", -1)) for row in rows]
    if len(rows) != DENOMINATOR or ordinals != list(range(DENOMINATOR)):
        raise ValueError(
            f"{path} must contain ordered all-attempt ordinals 0..511"
        )
    return rows


def exact_mcnemar_p(baseline_only: int, candidate_only: int) -> float:
    discordant = int(baseline_only) + int(candidate_only)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(
            0,
            min(int(baseline_only), int(candidate_only)) + 1,
        )
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def quantile_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if (
        len(ordered) != DENOMINATOR
        or any(not math.isfinite(value) or value <= 0.0 for value in ordered)
    ):
        raise ValueError("latency vector is incomplete or non-finite")
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "median_sec": statistics.median(ordered),
        "p95_sec": ordered[p95_index],
        "min_sec": ordered[0],
        "max_sec": ordered[-1],
        "mean_sec": statistics.fmean(ordered),
        "raw_values_retained_in_attempt_records": True,
    }


def aggregate_step_telemetry(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rejection_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    masked_steps = 0
    preterminal_difference_steps = 0
    blocked_newlines = 0
    removed_mass: list[float] = []
    reachability_removed_mass: list[float] = []
    prefix_only_removed_mass: list[float] = []
    state_work: list[int] = []
    cache_peaks: list[int] = []
    affected_attempts = 0
    diagnostics_count = 0
    telemetry_valid_count = 0

    for row in rows:
        diagnostics = row.get("crplan_diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        diagnostics_count += 1
        masked_steps += int(diagnostics.get("masked_step_count") or 0)
        attempt_difference_steps = int(
            diagnostics.get("preterminal_support_difference_steps") or 0
        )
        preterminal_difference_steps += attempt_difference_steps
        affected_attempts += int(attempt_difference_steps > 0)
        blocked_newlines += int(
            diagnostics.get("blocked_newline_token_count") or 0
        )
        steps = diagnostics.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                phase_counts[str(step.get("phase"))] += 1
                for reason, count in (
                    step.get("rejection_counts") or {}
                ).items():
                    rejection_counts[str(reason)] += int(count)
                for destination, key in (
                    (removed_mass, "removed_probability_mass"),
                    (
                        reachability_removed_mass,
                        "reachability_removed_probability_mass",
                    ),
                    (
                        prefix_only_removed_mass,
                        "prefix_only_removed_probability_mass",
                    ),
                ):
                    value = float(step.get(key) or 0.0)
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(
                            f"invalid {key} in ordinal {row.get('sample_idx')}"
                        )
                    destination.append(value)
        dp = diagnostics.get("dp")
        delta = dp.get("attempt_delta") if isinstance(dp, Mapping) else None
        peak = (
            dp.get("attempt_peak_cache_entries")
            if isinstance(dp, Mapping)
            else None
        )
        telemetry_valid = (
            isinstance(delta, Mapping)
            and isinstance(delta.get("states_created"), int)
            and int(delta["states_created"]) >= 0
            and isinstance(peak, int)
            and int(peak) >= 0
        )
        telemetry_valid_count += int(telemetry_valid)
        if telemetry_valid:
            state_work.append(int(delta["states_created"]))
            cache_peaks.append(int(peak))

    return {
        "diagnostics_count": diagnostics_count,
        "telemetry_valid_count": telemetry_valid_count,
        "masked_step_count": masked_steps,
        "preterminal_support_difference_step_count": (
            preterminal_difference_steps
        ),
        "preterminal_affected_attempt_count": affected_attempts,
        "blocked_newline_token_count": blocked_newlines,
        "phase_counts": dict(sorted(phase_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "removed_probability_mass": {
            "step_count": len(removed_mass),
            "sum": sum(removed_mass),
            "max": max(removed_mass, default=0.0),
        },
        "reachability_removed_probability_mass": {
            "step_count": len(reachability_removed_mass),
            "sum": sum(reachability_removed_mass),
            "max": max(reachability_removed_mass, default=0.0),
        },
        "prefix_only_removed_probability_mass": {
            "step_count": len(prefix_only_removed_mass),
            "sum": sum(prefix_only_removed_mass),
            "max": max(prefix_only_removed_mass, default=0.0),
        },
        "semantic_state_work_report_only": {
            "attempt_count": len(state_work),
            "max_states_created": max(state_work, default=0),
            "median_states_created": (
                statistics.median(state_work) if state_work else 0
            ),
            "sum_states_created": sum(state_work),
            "max_cache_entries": max(cache_peaks, default=0),
            "v4_100000_state_gate_reused": False,
        },
    }


def classify_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    reachability: Any,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    parsed_count = 0
    completion_count = 0
    comp_valid_count = 0
    nonshortcut_comp_valid_count = 0
    primary_comp_valid_count = 0
    shortcut_valid_count = 0
    charge_applicable_count = 0
    charge_terminal_failures: list[int] = []
    identity_verified_count = 0
    identity_failure_count = 0
    engineering_failure_count = 0
    direct_reasons: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    formulas: Counter[str] = Counter()
    element_coverage: set[str] = set()
    atom_counts: list[int] = []
    records: dict[int, dict[str, Any]] = {}

    for row in rows:
        ordinal = int(row["sample_idx"])
        if (
            row.get("experiment_identity") != IDENTITY
            or row.get("mode") != mode
            or row.get("execution_variant")
            != "optimized_scientific_primary"
            or int(row.get("ledger_ordinal", -1)) != ordinal
            or row.get("ledger_role") != "shared"
        ):
            engineering_failure_count += 1
        if row.get("generation_error") is True:
            engineering_failure_count += 1
        for field in FORBIDDEN_TRUE_FIELDS:
            if row.get(field) is not False:
                engineering_failure_count += 1

        completion = row.get("plan_end_marker_present") is True
        parsed = row.get("parsed") is True
        completion_count += int(completion)
        record = {
            "ordinal": ordinal,
            "parsed": parsed,
            "completion": completion,
            "formula": None,
            "composition_valid": False,
            "nonshortcut_composition_valid": False,
            "shortcut_valid": False,
            "charge_applicable": False,
            "terminal_allowed": None,
            "planner_sampling_seed": row.get("planner_sampling_seed"),
            "prompt_sha256": row.get("planner_input_prompt_sha256"),
            "input_ids_sha256": row.get("planner_input_ids_sha256"),
        }

        diagnostics = row.get("crplan_diagnostics")
        if mode == "off":
            if diagnostics is not None:
                engineering_failure_count += 1
        else:
            diagnostics_ok = (
                isinstance(diagnostics, Mapping)
                and diagnostics.get("mode") == mode
                and diagnostics.get("legal_support_enforcement")
                == "mask_or_raise"
                and diagnostics.get("dead_end") is None
                and diagnostics.get("empty_support_error_raised") is False
                and diagnostics.get(
                    "silent_fallback_used_by_decoder"
                )
                is False
                and diagnostics.get(
                    "retry_replacement_repair_filter_or_rerank_used"
                )
                is False
                and diagnostics.get("mask_application_count")
                == diagnostics.get("masked_step_count")
            )
            if not diagnostics_ok:
                engineering_failure_count += 1

        if not parsed:
            records[ordinal] = record
            continue

        plan = row.get("plan_state")
        if not isinstance(plan, Mapping):
            raise ValueError(f"parsed ordinal {ordinal} has no plan_state")
        symbols = [str(value) for value in (plan.get("elements") or ())]
        counts = [int(value) for value in (plan.get("counts") or ())]
        if len(symbols) != len(counts) or not symbols or any(
            value <= 0 for value in counts
        ):
            raise ValueError(f"parsed ordinal {ordinal} composition invalid")
        if any(symbol not in SYMBOL_TO_Z for symbol in symbols):
            raise ValueError(f"parsed ordinal {ordinal} has unknown element")

        parsed_count += 1
        formula = str(plan["formula"])
        formulas[formula] += 1
        element_coverage.update(symbols)
        atom_counts.append(sum(counts))
        classification = dict(
            classify_smact_validity(
                [int(SYMBOL_TO_Z[symbol]) for symbol in symbols],
                counts,
            )
        )
        certificate = reachability.terminal_certificate(
            zip(symbols, counts)
        ).to_dict()
        reason = str(classification["reason"])
        stratum = str(certificate["stratum"])
        comp_valid = classification.get("valid") is True
        shortcut = reason in (
            "single_element_shortcut",
            "all_metal_shortcut",
        )
        charge_applicable = certificate["charge_applicable"] is True
        nonshortcut_comp_valid = (
            comp_valid and charge_applicable and not shortcut
        )
        primary_comp_valid = (
            comp_valid
            and certificate["primary_charge_witness"] is True
            and not shortcut
        )
        direct_reasons[reason] += 1
        strata[stratum] += 1
        comp_valid_count += int(comp_valid)
        shortcut_valid_count += int(comp_valid and shortcut)
        nonshortcut_comp_valid_count += int(nonshortcut_comp_valid)
        primary_comp_valid_count += int(primary_comp_valid)
        charge_applicable_count += int(charge_applicable)
        if charge_applicable and certificate["terminal_allowed"] is not True:
            charge_terminal_failures.append(ordinal)

        if mode != "off":
            embedded = row.get("crplan_terminal_certificate")
            diagnostic_certificate = (
                diagnostics.get("terminal_certificate")
                if isinstance(diagnostics, Mapping)
                else None
            )
            identity = row.get("crplan_identity")
            identity_ok = (
                embedded == certificate
                and diagnostic_certificate == certificate
                and isinstance(identity, Mapping)
                and identity.get("verified") is True
                and identity.get("fsm_counts_equal_parser_counts") is True
                and identity.get("formula_line_count") == 1
            )
            identity_verified_count += int(identity_ok)
            identity_failure_count += int(not identity_ok)
            engineering_failure_count += int(not identity_ok)

        record.update(
            {
                "formula": formula,
                "composition_valid": comp_valid,
                "nonshortcut_composition_valid": nonshortcut_comp_valid,
                "primary_composition_valid": primary_comp_valid,
                "shortcut_valid": comp_valid and shortcut,
                "direct_reason": reason,
                "stratum": stratum,
                "charge_applicable": charge_applicable,
                "terminal_allowed": certificate["terminal_allowed"],
            }
        )
        records[ordinal] = record

    unique_formula_count = len(formulas)
    coverage_denominator = len(SYMBOL_TO_Z)
    telemetry = (
        {
            "diagnostics_count": 0,
            "telemetry_valid_count": 0,
            "semantic_state_work_report_only": {
                "attempt_count": 0,
                "max_states_created": 0,
                "median_states_created": 0,
                "sum_states_created": 0,
                "max_cache_entries": 0,
                "v4_100000_state_gate_reused": False,
            },
        }
        if mode == "off"
        else aggregate_step_telemetry(rows)
    )
    return (
        {
            "denominator": DENOMINATOR,
            "parse_count": parsed_count,
            "completion_count": completion_count,
            "composition_valid_count": comp_valid_count,
            "nonshortcut_composition_valid_count": (
                nonshortcut_comp_valid_count
            ),
            "primary_composition_valid_count": primary_comp_valid_count,
            "shortcut_valid_count": shortcut_valid_count,
            "charge_applicable_count": charge_applicable_count,
            "charge_applicable_terminal_failure_count": len(
                charge_terminal_failures
            ),
            "charge_applicable_terminal_failure_ordinals": (
                charge_terminal_failures
            ),
            "unique_formula_count": unique_formula_count,
            "unique_formula_rate": unique_formula_count / DENOMINATOR,
            "top1_formula_count": max(formulas.values(), default=0),
            "top1_formula_rate": (
                max(formulas.values(), default=0) / DENOMINATOR
            ),
            "element_coverage_count": len(element_coverage),
            "element_coverage_denominator": coverage_denominator,
            "element_coverage_rate": (
                len(element_coverage) / coverage_denominator
            ),
            "element_coverage": sorted(
                element_coverage,
                key=lambda value: SYMBOL_TO_Z[value],
            ),
            "mean_atoms_when_parsed": (
                statistics.fmean(atom_counts) if atom_counts else None
            ),
            "direct_reason_counts": dict(sorted(direct_reasons.items())),
            "endpoint_stratum_counts": dict(sorted(strata.items())),
            "crplan_identity_verified_count": identity_verified_count,
            "crplan_identity_failure_count": identity_failure_count,
            "engineering_failure_count": engineering_failure_count,
            "planner_generation_latency": quantile_summary(
                [
                    float(row["planner_generation_latency_sec"])
                    for row in rows
                ]
            ),
            "mechanism_telemetry": telemetry,
        },
        records,
    )


def paired_binary_report(
    baseline: Mapping[int, Mapping[str, Any]],
    candidate: Mapping[int, Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    baseline_only = sum(
        int(
            baseline[ordinal].get(field) is True
            and candidate[ordinal].get(field) is not True
        )
        for ordinal in range(DENOMINATOR)
    )
    candidate_only = sum(
        int(
            candidate[ordinal].get(field) is True
            and baseline[ordinal].get(field) is not True
        )
        for ordinal in range(DENOMINATOR)
    )
    return {
        "field": field,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "discordant": baseline_only + candidate_only,
        "candidate_minus_baseline": candidate_only - baseline_only,
        "mcnemar_two_sided_exact_p": exact_mcnemar_p(
            baseline_only,
            candidate_only,
        ),
    }


def scientific_gates_from_metrics(
    *,
    terminal: Mapping[str, Any],
    full: Mapping[str, Any],
    minimum_gain: int = 11,
    minimum_affected_rate: float = 0.05,
) -> tuple[dict[str, bool], dict[str, Any]]:
    full_telemetry = full["mechanism_telemetry"]
    applicable = int(full["charge_applicable_count"])
    affected = int(full_telemetry["preterminal_affected_attempt_count"])
    affected_rate = 0.0 if applicable == 0 else affected / applicable
    deltas = {
        "composition_valid_count": (
            int(full["composition_valid_count"])
            - int(terminal["composition_valid_count"])
        ),
        "nonshortcut_composition_valid_count": (
            int(full["nonshortcut_composition_valid_count"])
            - int(terminal["nonshortcut_composition_valid_count"])
        ),
        "primary_composition_valid_count": (
            int(full["primary_composition_valid_count"])
            - int(terminal["primary_composition_valid_count"])
        ),
        "parse_count": (
            int(full["parse_count"]) - int(terminal["parse_count"])
        ),
        "completion_count": (
            int(full["completion_count"])
            - int(terminal["completion_count"])
        ),
        "shortcut_valid_count": (
            int(full["shortcut_valid_count"])
            - int(terminal["shortcut_valid_count"])
        ),
        "unique_formula_rate": (
            float(full["unique_formula_rate"])
            - float(terminal["unique_formula_rate"])
        ),
        "element_coverage_rate": (
            float(full["element_coverage_rate"])
            - float(terminal["element_coverage_rate"])
        ),
        "charge_applicable_preterminal_affected_rate": affected_rate,
    }
    gates = {
        "nonshortcut_composition_valid_gain_ge_11": (
            deltas["nonshortcut_composition_valid_count"]
            >= int(minimum_gain)
        ),
        "charge_applicable_preterminal_affected_rate_ge_5pct": (
            applicable > 0
            and affected_rate >= float(minimum_affected_rate)
        ),
        "full_charge_applicable_terminal_failure_zero": (
            int(full["charge_applicable_terminal_failure_count"]) == 0
        ),
        "full_parse_not_below_terminal": deltas["parse_count"] >= 0,
        "full_completion_not_below_terminal": (
            deltas["completion_count"] >= 0
        ),
        "full_unique_formula_rate_loss_at_most_2pp": (
            deltas["unique_formula_rate"] >= -0.02 - 1e-12
        ),
        "full_element_coverage_rate_loss_at_most_2pp": (
            deltas["element_coverage_rate"] >= -0.02 - 1e-12
        ),
        "full_shortcut_valid_count_not_increased": (
            deltas["shortcut_valid_count"] <= 0
        ),
    }
    return gates, deltas


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    run_root = args.run_root.resolve()
    config = read_json(args.config)
    ledger = read_json(args.ledger)
    source_manifest_sha256 = sha256_file(args.source_manifest)
    ledger_sha256 = sha256_file(args.ledger)
    if config.get("identity") != IDENTITY:
        raise ValueError("config identity mismatch")
    if config.get("science_ledger_sha256") != ledger_sha256:
        raise ValueError("config ledger SHA mismatch")
    if ledger.get("identity") != IDENTITY:
        raise ValueError("ledger identity mismatch")
    ledger_rows = ledger.get("rows")
    if not isinstance(ledger_rows, list) or len(ledger_rows) != DENOMINATOR:
        raise ValueError("ledger is incomplete")
    ledger_seeds = {
        int(row["ordinal"]): int(row["planner_sampling_seed"])
        for row in ledger_rows
        if isinstance(row, Mapping)
    }
    if set(ledger_seeds) != set(range(DENOMINATOR)):
        raise ValueError("ledger ordinals are not exactly 0..511")

    predecessor_checks = {
        "v4_terminal_sha256": sha256_file(args.v4_terminal)
        == config["frozen_predecessors"]["v4_terminal_sha256"],
        "e1_terminal_sha256": sha256_file(args.e1_terminal)
        == config["frozen_predecessors"]["e1_terminal_sha256"],
    }
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    setup_by_mode: dict[str, dict[str, Any]] = {}
    arm_exit_codes: dict[str, int | None] = {}
    for mode in MODES:
        arm_root = run_root / "arms" / mode
        rows_by_mode[mode] = read_jsonl(
            arm_root / "raw_generations.jsonl"
        )
        setup_by_mode[mode] = read_json(arm_root / "setup.json")
        runner = read_json(arm_root / "runner_report.json")
        if (
            runner.get("status") != "runner_complete"
            or runner.get("mode") != mode
            or int(runner.get("attempt_count", -1)) != DENOMINATOR
        ):
            raise ValueError(f"{mode} runner report invalid")
        exit_path = run_root / "status" / f"{mode}_exit_code.txt"
        arm_exit_codes[mode] = (
            int(exit_path.read_text(encoding="utf-8").strip())
            if exit_path.exists()
            else None
        )

    reachability = load_frozen_smact_table(
        max_atoms=int(config["max_atoms"]),
        missing_state_policy=str(config["missing_state_policy"]),
    )
    metrics: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[int, dict[str, Any]]] = {}
    for mode in MODES:
        metrics[mode], records[mode] = classify_arm(
            rows_by_mode[mode],
            mode=mode,
            reachability=reachability,
        )

    identity_mismatch_ordinals: list[int] = []
    ledger_mismatch_ordinals: list[int] = []
    for ordinal in range(DENOMINATOR):
        identity_values = {
            (
                records[mode][ordinal]["planner_sampling_seed"],
                records[mode][ordinal]["prompt_sha256"],
                records[mode][ordinal]["input_ids_sha256"],
            )
            for mode in MODES
        }
        if len(identity_values) != 1:
            identity_mismatch_ordinals.append(ordinal)
        if any(
            int(records[mode][ordinal]["planner_sampling_seed"])
            != ledger_seeds[ordinal]
            for mode in MODES
        ):
            ledger_mismatch_ordinals.append(ordinal)

    tokenizer_metadata_fields = (
        "vocab_size",
        "eos_token_id",
        "pad_token_id",
        "padding_side",
    )
    tokenizer_identity = all(
        len(
            {
                setup_by_mode[mode].get(field)
                for mode in MODES
            }
        )
        == 1
        for field in tokenizer_metadata_fields
    )
    constrained = ("grammar_only", "terminal_only", "full_prefix")
    constrained_contract_identity = all(
        len(
            {
                setup_by_mode[mode].get(field)
                for mode in constrained
            }
        )
        == 1
        and None
        not in {
            setup_by_mode[mode].get(field)
            for mode in constrained
        }
        for field in (
            "token_fragment_sha256",
            "oxidation_table_sha256",
            "constraint_contract_sha256",
        )
    )
    source_and_ledger_identity = all(
        setup_by_mode[mode].get("source_manifest_sha256")
        == source_manifest_sha256
        and setup_by_mode[mode].get("science_ledger_sha256")
        == ledger_sha256
        for mode in MODES
    )
    adapter_identity = all(
        setup_by_mode[mode].get("adapter_model_sha256")
        == config["adapter_model_sha256"]
        for mode in MODES
    )
    arm_exit_zero = all(value == 0 for value in arm_exit_codes.values())
    no_row_engineering_failures = all(
        int(metrics[mode]["engineering_failure_count"]) == 0
        for mode in MODES
    )
    constrained_telemetry_complete = all(
        metrics[mode]["mechanism_telemetry"]["diagnostics_count"]
        == DENOMINATOR
        and metrics[mode]["mechanism_telemetry"][
            "telemetry_valid_count"
        ]
        == DENOMINATOR
        for mode in constrained
    )
    constrained_identity_complete = all(
        metrics[mode]["crplan_identity_failure_count"] == 0
        and metrics[mode]["crplan_identity_verified_count"]
        == metrics[mode]["parse_count"]
        for mode in constrained
    )
    engineering_gates = {
        "all_arm_process_exit_codes_zero": arm_exit_zero,
        "all_arms_exact_512_ordinals": True,
        "common_science_ledger_seed_identity_100pct": (
            not ledger_mismatch_ordinals
        ),
        "common_prompt_input_seed_identity_100pct": (
            not identity_mismatch_ordinals
        ),
        "tokenizer_metadata_identity_100pct": tokenizer_identity,
        "constrained_policy_contract_identity_100pct": (
            constrained_contract_identity
        ),
        "source_and_ledger_sha_identity_100pct": (
            source_and_ledger_identity
        ),
        "adapter_identity_100pct": adapter_identity,
        "constrained_diagnostics_and_state_telemetry_complete": (
            constrained_telemetry_complete
        ),
        "constrained_parser_fsm_certificate_identity_100pct": (
            constrained_identity_complete
        ),
        "no_generation_or_forbidden_operation_failure": (
            no_row_engineering_failures
        ),
        "v4_terminal_unchanged": predecessor_checks[
            "v4_terminal_sha256"
        ],
        "e1_terminal_unchanged": predecessor_checks[
            "e1_terminal_sha256"
        ],
    }

    scientific_gates, deltas = scientific_gates_from_metrics(
        terminal=metrics["terminal_only"],
        full=metrics["full_prefix"],
        minimum_gain=int(
            config["scientific_gates"][
                "nonshortcut_composition_valid_count_gain_min"
            ]
        ),
        minimum_affected_rate=float(
            config["scientific_gates"][
                "charge_applicable_preterminal_affected_rate_min"
            ]
        ),
    )
    engineering_passed = all(engineering_gates.values())
    scientific_passed = all(scientific_gates.values())
    gate_passed = engineering_passed and scientific_passed
    if not engineering_passed:
        status = "engineering_failure"
        decision = "stop_crplan_retain_frozen_h1"
        process_exit_code = 2
    elif scientific_passed:
        status = "mechanism_gate_pass"
        decision = "eligible_for_separately_authorized_paired64_only"
        process_exit_code = 0
    else:
        status = "scientific_stop"
        decision = "stop_crplan_retain_frozen_h1"
        process_exit_code = 0

    terminal_report = {
        "schema": "h1_crplan_fourarm512_route_amendment_terminal_v1",
        "identity": IDENTITY,
        "status": status,
        "decision": decision,
        "gate_passed": gate_passed,
        "engineering_passed": engineering_passed,
        "scientific_passed": scientific_passed,
        "denominator_per_arm": DENOMINATOR,
        "source_manifest_sha256": source_manifest_sha256,
        "science_ledger_sha256": ledger_sha256,
        "science_base_seed": ledger["base_seed"],
        "adapter_model_sha256": config["adapter_model_sha256"],
        "array_job_id": args.array_job_id,
        "arm_exit_codes": arm_exit_codes,
        "frozen_predecessor_checks": predecessor_checks,
        "arms": metrics,
        "primary_contrast": {
            "name": "full_prefix_minus_terminal_only",
            "shortcut_excluded_from_primary_gain": True,
            "deltas": deltas,
        },
        "paired_full_vs_terminal": {
            "raw_composition_validity": paired_binary_report(
                records["terminal_only"],
                records["full_prefix"],
                field="composition_valid",
            ),
            "nonshortcut_composition_validity": paired_binary_report(
                records["terminal_only"],
                records["full_prefix"],
                field="nonshortcut_composition_valid",
            ),
            "primary_uniform_composition_validity": paired_binary_report(
                records["terminal_only"],
                records["full_prefix"],
                field="primary_composition_valid",
            ),
        },
        "secondary_pairwise_raw_composition_validity": {
            "grammar_minus_off": paired_binary_report(
                records["off"],
                records["grammar_only"],
                field="composition_valid",
            ),
            "terminal_minus_grammar": paired_binary_report(
                records["grammar_only"],
                records["terminal_only"],
                field="composition_valid",
            ),
        },
        "identity_mismatch_ordinals": identity_mismatch_ordinals,
        "ledger_mismatch_ordinals": ledger_mismatch_ordinals,
        "engineering_gates": engineering_gates,
        "scientific_gates": scientific_gates,
        "v4_100000_state_gate_reused": False,
        "semantic_states_report_only": True,
        "new_cross_job_latency_ratio_gate": False,
        "raw_attempt_latency_retained": True,
        "body_rerun": False,
        "refiner_rerun": False,
        "direct_rerun": False,
        "sun_rerun": False,
        "network": False,
        "training": False,
        "checkpoint_reselection": False,
        "promotion": False,
        "formal_g3": False,
        "automatic_downstream": False,
        "paired64_submitted": False,
    }
    return terminal_report, process_exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--v4-terminal", type=Path, required=True)
    parser.add_argument("--e1-terminal", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        terminal, exit_code = evaluate(args)
    except Exception as exc:  # noqa: BLE001
        terminal = {
            "schema": "h1_crplan_fourarm512_route_amendment_terminal_v1",
            "identity": IDENTITY,
            "status": "engineering_failure",
            "decision": "stop_crplan_retain_frozen_h1",
            "gate_passed": False,
            "engineering_passed": False,
            "scientific_passed": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "v4_100000_state_gate_reused": False,
            "semantic_states_report_only": True,
            "body_rerun": False,
            "refiner_rerun": False,
            "direct_rerun": False,
            "sun_rerun": False,
            "network": False,
            "training": False,
            "checkpoint_reselection": False,
            "promotion": False,
            "formal_g3": False,
            "automatic_downstream": False,
            "paired64_submitted": False,
        }
        exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(terminal, sort_keys=True), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
