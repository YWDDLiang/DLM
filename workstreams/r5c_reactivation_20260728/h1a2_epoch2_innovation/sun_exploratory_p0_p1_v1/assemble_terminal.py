#!/usr/bin/env python3
"""Assemble the paired P0/P1 exploratory result without authorizing follow-up."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT_FALLBACK = HERE.parents[3]
for location in (PROJECT_ROOT_FALLBACK, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from protocol import (  # noqa: E402
    ARM_ORDER,
    read_json,
    read_jsonl,
    require_hex_sha,
    require_runtime_manifest,
    require_sha,
    require_source_manifest,
    rows_by_attempt,
    sha256_file,
    write_json_exclusive,
)


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260731


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - fraction)
        + float(sorted_values[upper]) * fraction
    )


def _exact_mcnemar(
    candidate: Sequence[bool], baseline: Sequence[bool]
) -> dict[str, Any]:
    candidate_only = sum(
        bool(left) and not bool(right) for left, right in zip(candidate, baseline)
    )
    baseline_only = sum(
        not bool(left) and bool(right) for left, right in zip(candidate, baseline)
    )
    discordant = candidate_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(candidate_only, baseline_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "candidate_only": candidate_only,
        "baseline_only": baseline_only,
        "discordant": discordant,
        "two_sided_exact_p_value": p_value,
    }


def _paired_effect(
    candidate: Sequence[bool],
    baseline: Sequence[bool],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    if len(candidate) != 256 or len(baseline) != 256:
        raise ValueError("paired effects require the full 256-pair denominator")
    differences = [
        float(bool(left)) - float(bool(right))
        for left, right in zip(candidate, baseline)
    ]
    rng = random.Random(BOOTSTRAP_SEED + int(seed_offset))
    samples = [
        100.0 * sum(differences[rng.randrange(256)] for _ in range(256)) / 256
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    samples.sort()
    candidate_count = sum(bool(value) for value in candidate)
    baseline_count = sum(bool(value) for value in baseline)
    return {
        "attempts": 256,
        "candidate_arm": "P1",
        "baseline_arm": "P0",
        "candidate_count": candidate_count,
        "baseline_count": baseline_count,
        "candidate_rate": candidate_count / 256,
        "baseline_rate": baseline_count / 256,
        "difference_percentage_points": 100.0
        * (candidate_count - baseline_count)
        / 256,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED + int(seed_offset),
            "ci95_lower_percentage_points": _quantile(samples, 0.025),
            "ci95_upper_percentage_points": _quantile(samples, 0.975),
        },
        "exact_mcnemar": _exact_mcnemar(candidate, baseline),
    }


def _all_metal(plan: Any) -> bool:
    import smact

    if not isinstance(plan, dict):
        return False
    elements = [str(value) for value in plan.get("elements") or []]
    return bool(elements) and all(value in smact.metals for value in elements)


def _unique_formula_summary(plans: list[Any]) -> dict[str, Any]:
    formulas = [
        str(plan.get("reduced_formula") or plan.get("formula"))
        for plan in plans
        if isinstance(plan, dict)
        and str(plan.get("reduced_formula") or plan.get("formula") or "")
    ]
    return {
        "observed_formula_rows": len(formulas),
        "unique_formulas": len(set(formulas)),
        "unique_formula_rate_all_attempts": len(set(formulas)) / 256,
    }


def _arm_evidence(
    *,
    arm: str,
    arm_root: Path,
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    generation_dir = arm_root / "generation"
    evaluation_dir = arm_root / "evaluation"
    if not (generation_dir / "_SUCCESS").is_file():
        raise FileNotFoundError(generation_dir / "_SUCCESS")
    if not (evaluation_dir / "_SUCCESS").is_file():
        raise FileNotFoundError(evaluation_dir / "_SUCCESS")
    generation_report = read_json(generation_dir / "generation_report.json")
    evaluation_report = read_json(evaluation_dir / "evaluation_report.json")
    generation = read_jsonl(generation_dir / "generation.jsonl")
    direct = rows_by_attempt(
        evaluation_dir / "crysllmgen_metrics" / "attempt_metrics.jsonl",
        schema="crysllmgen_metric_attempt_v1",
    )
    sun = rows_by_attempt(
        evaluation_dir / "r5c_a100_sun" / "attempt_results.jsonl",
        schema="crysllmgen_r5c_a100_sun_attempt_v1",
    )
    expected_ids = [str(cell["arms"][arm]["attempt_id"]) for cell in ledger]
    if (
        generation_report.get("ok") is not True
        or evaluation_report.get("ok") is not True
        or len(generation) != 256
        or [str(row.get("attempt_id")) for row in generation] != expected_ids
        or list(direct) != expected_ids
        or list(sun) != expected_ids
        or any(row.get("retry_or_replacement_used") is not False for row in generation)
    ):
        raise ValueError(f"{arm} terminal attempt mapping changed")
    plans = [cell["arms"][arm].get("plan_state") for cell in ledger]
    vectors = {
        "body_graph_yield": [
            bool(row.get("status") == "succeeded")
            for row in read_jsonl(generation_dir / "body_attempts.jsonl")
        ],
        "generation_succeeded": [
            bool(row.get("status") == "succeeded") for row in generation
        ],
        "direct_comp_valid": [
            bool(direct[value]["comp_valid"]) for value in expected_ids
        ],
        "direct_struct_valid": [
            bool(direct[value]["struct_valid"]) for value in expected_ids
        ],
        "direct_joint_valid": [bool(direct[value]["valid"]) for value in expected_ids],
        "sun_novel": [bool(sun[value]["metrics"]["novel"]) for value in expected_ids],
        "sun_unique": [
            bool(sun[value]["metrics"]["unique_representative"])
            for value in expected_ids
        ],
        "sun_novel_unique": [
            bool(sun[value]["metrics"]["novel_unique"]) for value in expected_ids
        ],
        "sun_strict": [
            bool(sun[value]["metrics"]["strict_full_sun"]) for value in expected_ids
        ],
        "sun_meta": [
            bool(sun[value]["metrics"]["meta_full_sun"]) for value in expected_ids
        ],
        "plan_all_metal": [_all_metal(plan) for plan in plans],
    }
    if any(len(values) != 256 for values in vectors.values()):
        raise ValueError(f"{arm} vectors do not retain all 256 attempts")
    return {
        "vectors": vectors,
        "summary": {
            "attempts": 256,
            "counts": {name: sum(values) for name, values in vectors.items()},
            "rates": {name: sum(values) / 256 for name, values in vectors.items()},
            "formula_diversity": _unique_formula_summary(plans),
            "generation_report": _identity(generation_dir / "generation_report.json"),
            "evaluation_report": _identity(evaluation_dir / "evaluation_report.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    args = parser.parse_args()

    execution_sha = require_hex_sha(
        args.execution_manifest_sha256, "execution source manifest"
    )
    require_source_manifest(args.source_dir.resolve(), execution_sha)
    require_runtime_manifest(args.project_root.resolve(), args.source_dir.resolve())
    config = read_json(args.config.resolve())
    if config.get("status") != "user_authorized_exploratory_execution":
        raise ValueError("exploratory authorization changed")
    data = args.data_dir.resolve()
    if not (data / "_SUCCESS").is_file():
        raise FileNotFoundError(data / "_SUCCESS")
    ledger_manifest = read_json(data / "ledger_manifest.json")
    if (
        ledger_manifest.get("ok") is not True
        or ledger_manifest.get("execution_manifest_sha256") != execution_sha
    ):
        raise ValueError("paired ledger terminal identity changed")
    require_sha(
        data / "attempt_ledger.jsonl",
        ledger_manifest["attempt_ledger"]["sha256"],
        "paired attempt ledger",
    )
    ledger = read_jsonl(data / "attempt_ledger.jsonl")
    if len(ledger) != 256:
        raise ValueError("paired ledger denominator changed")

    arms_root = args.arms_root.resolve()
    arm_evidence = {
        arm: _arm_evidence(
            arm=arm,
            arm_root=arms_root / arm,
            ledger=ledger,
        )
        for arm in ARM_ORDER
    }
    metric_order = (
        "body_graph_yield",
        "generation_succeeded",
        "direct_comp_valid",
        "direct_struct_valid",
        "direct_joint_valid",
        "sun_novel",
        "sun_unique",
        "sun_novel_unique",
        "sun_strict",
        "sun_meta",
        "plan_all_metal",
    )
    effects = {
        metric: _paired_effect(
            arm_evidence["P1"]["vectors"][metric],
            arm_evidence["P0"]["vectors"][metric],
            seed_offset=index,
        )
        for index, metric in enumerate(metric_order)
    }
    thresholds = config["screening_gates"]
    gate_values = {
        "direct_composition_gain": {
            "observed_pp": effects["direct_comp_valid"]["difference_percentage_points"],
            "operator": ">=",
            "threshold_pp": float(thresholds["direct_composition_gain_pp_min"]),
        },
        "joint_valid_gain": {
            "observed_pp": effects["direct_joint_valid"][
                "difference_percentage_points"
            ],
            "operator": ">=",
            "threshold_pp": float(thresholds["joint_valid_gain_pp_min"]),
        },
        "structure_noninferiority": {
            "observed_pp": effects["direct_struct_valid"][
                "difference_percentage_points"
            ],
            "operator": ">=",
            "threshold_pp": float(thresholds["structure_noninferiority_pp_min"]),
        },
        "graph_yield_noninferiority": {
            "observed_pp": effects["body_graph_yield"]["difference_percentage_points"],
            "operator": ">=",
            "threshold_pp": float(thresholds["graph_yield_noninferiority_pp_min"]),
        },
        "unique_noninferiority": {
            "observed_pp": effects["sun_unique"]["difference_percentage_points"],
            "operator": ">=",
            "threshold_pp": float(thresholds["unique_noninferiority_pp_min"]),
        },
        "novel_noninferiority": {
            "observed_pp": effects["sun_novel"]["difference_percentage_points"],
            "operator": ">=",
            "threshold_pp": float(thresholds["novel_noninferiority_pp_min"]),
        },
        "strict_sun_noninferiority": {
            "observed_pp": effects["sun_strict"]["difference_percentage_points"],
            "operator": ">=",
            "threshold_pp": float(thresholds["strict_sun_noninferiority_pp_min"]),
        },
        "meta_sun_gain": {
            "observed_pp": effects["sun_meta"]["difference_percentage_points"],
            "operator": ">=",
            "threshold_pp": float(thresholds["meta_sun_gain_pp_min"]),
        },
        "all_metal_inflation": {
            "observed_pp": effects["plan_all_metal"]["difference_percentage_points"],
            "operator": "<=",
            "threshold_pp": float(thresholds["all_metal_inflation_pp_max"]),
        },
    }
    for gate in gate_values.values():
        if gate["operator"] == ">=":
            gate["passed"] = gate["observed_pp"] >= gate["threshold_pp"]
        else:
            gate["passed"] = gate["observed_pp"] <= gate["threshold_pp"]
    screening_passed = all(bool(value["passed"]) for value in gate_values.values())
    decision = (
        "exploratory_support_only_formal_promotion_ineligible"
        if screening_passed
        else "stop_exploratory_screen"
    )
    failed_gates = [
        name for name, value in gate_values.items() if not bool(value["passed"])
    ]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    terminal = {
        "schema": "h1a2c_p0_p1_sun256_exploratory_terminal_v1",
        "ok": True,
        "identity": config["identity"],
        "run_id": config["run_id"],
        "decision": decision,
        "screening_passed": screening_passed,
        "failed_screening_gates": failed_gates,
        "formal_promotion_eligible": False,
        "formal_ineligibility_reasons": [
            "corrected JointChem formal selection is null",
            "matched-vs-shuffled-plan gate was not run in this two-arm exploratory screen",
        ],
        "attempts_per_arm": 256,
        "failure_denominator": "all_registered_attempts",
        "arms": {arm: arm_evidence[arm]["summary"] for arm in ARM_ORDER},
        "paired_effects_P1_minus_P0": effects,
        "screening_gates": gate_values,
        "statistics": {
            "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed_base": BOOTSTRAP_SEED,
            "exact_test": "two-sided exact McNemar/binomial",
        },
        "execution_manifest_sha256": execution_sha,
        "ledger_manifest": _identity(data / "ledger_manifest.json"),
        "attempt_ledger": _identity(data / "attempt_ledger.jsonl"),
        "retry_or_replacement_used": False,
        "manual_crystal_evaluation_authorized": True,
        "manual_authorization_includes_afterok_sun_evaluation": True,
        "automatic_crystal_evaluation_authorized": False,
        "automatic_promotion_authorized": False,
        "automatic_1000_attempt_confirmation_authorized": False,
        "automatic_training_authorized": False,
        "automatic_downstream_authorized": False,
        "recommended_next_step": (
            "Request explicit review before designing matched-vs-shuffled confirmation."
            if screening_passed
            else "Stop; inspect paired failure modes before any new crystal run."
        ),
    }
    terminal_path = output / "terminal_report.json"
    write_json_exclusive(terminal_path, terminal)
    decision_record = {
        "schema": "h1a2c_p0_p1_sun256_exploratory_decision_v1",
        "decision": decision,
        "screening_passed": screening_passed,
        "failed_screening_gates": failed_gates,
        "formal_promotion_eligible": False,
        "terminal_report_sha256": sha256_file(terminal_path),
        "automatic_crystal_evaluation_authorized": False,
        "automatic_downstream_authorized": False,
    }
    write_json_exclusive(output / "decision.json", decision_record)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(decision_record, sort_keys=True))


if __name__ == "__main__":
    main()
