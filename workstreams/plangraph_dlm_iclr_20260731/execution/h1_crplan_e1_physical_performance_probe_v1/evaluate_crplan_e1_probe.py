#!/usr/bin/env python3
"""Evaluate the immutable H1 CR-Plan E1 physical-performance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


ATTEMPTS = 18
MODES = ("off", "terminal_only", "full_prefix")
REFERENCE_ORDINALS = (2, 11)
MEDIAN_RATIO_MAX = 1.5
P95_RATIO_MAX = 2.0
MINIMUM_AFFECTED_RATE = 0.05
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
                    raise ValueError(f"{path} has a non-object row")
                rows.append(value)
    return rows


def latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(
        float(row["planner_generation_latency_sec"]) for row in rows
    )
    if (
        len(values) != ATTEMPTS
        or any(not math.isfinite(value) or value <= 0 for value in values)
    ):
        raise ValueError("latency vector is incomplete or non-finite")
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "count": len(values),
        "median_sec": statistics.median(values),
        "p95_sec": values[p95_index],
        "min_sec": values[0],
        "max_sec": values[-1],
        "values_sec": values,
    }


def decision_from_components(
    *,
    p0_latency: Mapping[str, Any],
    full_latency: Mapping[str, Any],
    identity_ok: bool,
    trace_parity_ok: bool,
    scalar_parity_ok: bool,
    no_engineering_failures: bool,
    applicable_count: int,
    affected_count: int,
) -> dict[str, Any]:
    p0_median = float(p0_latency["median_sec"])
    p0_p95 = float(p0_latency["p95_sec"])
    full_median = float(full_latency["median_sec"])
    full_p95 = float(full_latency["p95_sec"])
    median_ratio = full_median / p0_median
    p95_ratio = full_p95 / p0_p95
    affected_rate = (
        0.0
        if int(applicable_count) <= 0
        else int(affected_count) / int(applicable_count)
    )
    gates = {
        "full_median_ratio_le_1p5": median_ratio <= MEDIAN_RATIO_MAX,
        "full_p95_ratio_le_2p0": p95_ratio <= P95_RATIO_MAX,
        "input_prompt_seed_identity_100pct": bool(identity_ok),
        "actual_trace_support_parity_100pct": bool(trace_parity_ok),
        "scalar_token_rerun_parity_100pct": bool(scalar_parity_ok),
        "no_engineering_failure_or_forbidden_operation": bool(
            no_engineering_failures
        ),
        "charge_applicable_attempt_exists": int(applicable_count) > 0,
        "preterminal_affected_rate_ge_5pct": (
            int(applicable_count) > 0
            and affected_rate >= MINIMUM_AFFECTED_RATE
        ),
    }
    return {
        "gates": gates,
        "gate_passed": all(gates.values()),
        "full_over_p0_median_ratio": median_ratio,
        "full_over_p0_p95_ratio": p95_ratio,
        "charge_applicable_attempt_count": int(applicable_count),
        "preterminal_affected_attempt_count": int(affected_count),
        "preterminal_affected_rate": affected_rate,
    }


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    run_root = args.run_root.resolve()
    probe_root = run_root / "probe"
    failures: list[str] = []
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    schedule = read_json(probe_root / "schedule.json")
    schedule_rows = schedule.get("schedule")
    if not isinstance(schedule_rows, list) or len(schedule_rows) != 54:
        failures.append("balanced_schedule_invalid")

    for mode in MODES:
        path = (
            probe_root / "primary" / mode / "raw_generations.jsonl"
        )
        try:
            rows = read_jsonl(path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{mode}_read_error:{type(exc).__name__}:{exc}")
            rows = []
        rows_by_mode[mode] = rows
        ordinals = [int(row.get("sample_idx", -1)) for row in rows]
        if len(rows) != ATTEMPTS or ordinals != list(range(ATTEMPTS)):
            failures.append(f"{mode}_ordinal_ledger_invalid")
        if any(row.get("mode") != mode for row in rows):
            failures.append(f"{mode}_mode_identity_invalid")
        if any(
            row.get("execution_variant") != "optimized_primary"
            for row in rows
        ):
            failures.append(f"{mode}_execution_variant_invalid")

    identity_ok = True
    for ordinal in range(ATTEMPTS):
        values = []
        for mode in MODES:
            by_ordinal = {
                int(row.get("sample_idx", -1)): row
                for row in rows_by_mode[mode]
            }
            row = by_ordinal.get(ordinal)
            if row is None:
                identity_ok = False
                continue
            values.append(
                (
                    row.get("planner_sampling_seed"),
                    row.get("planner_input_prompt_sha256"),
                    row.get("planner_input_ids_sha256"),
                )
            )
        if len(values) != len(MODES) or len(set(values)) != 1:
            identity_ok = False
    if not identity_ok:
        failures.append("input_prompt_or_seed_identity_failure")

    no_engineering_failures = True
    forbidden_hits: list[str] = []
    max_states_by_mode: dict[str, int] = {}
    for mode, rows in rows_by_mode.items():
        states: list[int] = []
        for row in rows:
            if row.get("generation_error") is True:
                no_engineering_failures = False
            if row.get("reason") in (
                "CRPlanIdentityError",
                "CRPlanDeadEndError",
            ):
                no_engineering_failures = False
            for field in FORBIDDEN_TRUE_FIELDS:
                if row.get(field) is not False:
                    no_engineering_failures = False
                    forbidden_hits.append(
                        f"{mode}:{row.get('sample_idx')}:{field}"
                    )
            if mode != "off":
                diagnostics = row.get("crplan_diagnostics")
                if not isinstance(diagnostics, Mapping):
                    no_engineering_failures = False
                    continue
                if diagnostics.get("dead_end") is not None:
                    no_engineering_failures = False
                if (
                    diagnostics.get("silent_fallback_used_by_decoder")
                    is not False
                    or diagnostics.get(
                        "retry_replacement_repair_filter_or_rerank_used"
                    )
                    is not False
                ):
                    no_engineering_failures = False
                dp = diagnostics.get("dp")
                delta = (
                    dp.get("attempt_delta")
                    if isinstance(dp, Mapping)
                    else None
                )
                if isinstance(delta, Mapping):
                    states.append(int(delta.get("states_created") or 0))
        max_states_by_mode[mode] = max(states, default=0)
    if not no_engineering_failures:
        failures.append("engineering_failure_or_forbidden_operation")

    try:
        latency = {
            mode: latency_summary(rows_by_mode[mode]) for mode in MODES
        }
    except Exception as exc:  # noqa: BLE001
        failures.append(f"latency_invalid:{type(exc).__name__}:{exc}")
        latency = {
            mode: {
                "count": 0,
                "median_sec": float("inf"),
                "p95_sec": float("inf"),
                "min_sec": None,
                "max_sec": None,
                "values_sec": [],
            }
            for mode in MODES
        }

    trace_audit = read_json(probe_root / "trace_support_audit.json")
    reference_parity = read_json(probe_root / "reference_parity.json")
    runner_report = read_json(probe_root / "runner_report.json")
    trace_parity_ok = trace_audit.get("all_equal") is True
    scalar_parity_ok = (
        reference_parity.get("all_equal") is True
        and reference_parity.get("reference_ordinals")
        == list(REFERENCE_ORDINALS)
    )
    if not trace_parity_ok:
        failures.append("actual_trace_support_parity_failure")
    if not scalar_parity_ok:
        failures.append("scalar_token_rerun_parity_failure")
    if int(args.runner_exit_code) != 0:
        failures.append(f"runner_exit_code_{int(args.runner_exit_code)}")
        no_engineering_failures = False

    applicable_count = 0
    affected_count = 0
    for row in rows_by_mode["full_prefix"]:
        diagnostics = row.get("crplan_diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        certificate = diagnostics.get("terminal_certificate")
        stratum = (
            certificate.get("stratum")
            if isinstance(certificate, Mapping)
            else None
        )
        if isinstance(stratum, str) and stratum.startswith(
            "charge_applicable_"
        ):
            applicable_count += 1
            if int(
                diagnostics.get(
                    "preterminal_support_difference_steps"
                )
                or 0
            ) > 0:
                affected_count += 1

    decision = decision_from_components(
        p0_latency=latency["off"],
        full_latency=latency["full_prefix"],
        identity_ok=identity_ok,
        trace_parity_ok=trace_parity_ok,
        scalar_parity_ok=scalar_parity_ok,
        no_engineering_failures=no_engineering_failures,
        applicable_count=applicable_count,
        affected_count=affected_count,
    )
    for gate, passed in decision["gates"].items():
        if not passed:
            failures.append(f"gate_failed:{gate}")
    gate_passed = decision["gate_passed"] and not failures
    terminal = {
        "schema": "h1_crplan_e1_physical_performance_terminal_v1",
        "status": (
            "exploratory_physical_feasibility_pass_not_v4_repair"
            if gate_passed
            else "exploratory_physical_feasibility_fail"
        ),
        "gate_passed": gate_passed,
        "decision": (
            "eligible_for_new_preregistered_scientific_route_amendment_only"
            if gate_passed
            else "stop_crplan_retain_frozen_h1"
        ),
        "failures": sorted(set(failures)),
        "runner_exit_code": int(args.runner_exit_code),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "attempts_per_mode": ATTEMPTS,
        "primary_attempt_count": sum(
            len(rows) for rows in rows_by_mode.values()
        ),
        "latency": latency,
        "physical_gate": decision,
        "input_prompt_seed_identity": identity_ok,
        "trace_support_audit": trace_audit,
        "scalar_reference_parity": reference_parity,
        "max_cumulative_semantic_states_by_mode": max_states_by_mode,
        "logical_state_gate_reused_from_v4": False,
        "logical_state_definition_changed": False,
        "forbidden_operation_hits": forbidden_hits,
        "runner_report": runner_report,
        "v4_terminal_report_sha256": (
            "55df7801e24f3bfd013e2e41f0cb96babe2a20ff3c5fb8a5b8e0b23073a5e627"
        ),
        "v4_terminal_modified": False,
        "four_arm_512_submitted": False,
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
    }
    return terminal, 0 if gate_passed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--runner-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        terminal, exit_code = evaluate(args)
    except Exception as exc:  # noqa: BLE001
        terminal = {
            "schema": "h1_crplan_e1_physical_performance_terminal_v1",
            "status": "exploratory_physical_feasibility_evaluator_failure",
            "gate_passed": False,
            "decision": "stop_crplan_retain_frozen_h1",
            "failures": [f"{type(exc).__name__}: {exc}"],
            "runner_exit_code": int(args.runner_exit_code),
            "v4_terminal_modified": False,
            "four_arm_512_submitted": False,
            "automatic_downstream": False,
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
