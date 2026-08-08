#!/usr/bin/env python3
"""Assemble preregistered raw64/raw256 gates for one SFT-v2 candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.h1_chemistry_first_sft import H1_CHEMISTRY_FIRST_SFT_SCHEMA  # noqa: E402
from scripts.evaluate_h1_nocharge_sft_planner_gate import (  # noqa: E402
    classify_arm,
    load_ledger,
    load_smact4_audit,
    metric_deltas,
    paired_binary_report,
    read_json,
    read_raw,
    sha256_file,
)


SCHEMA = "h1_chemistry_first_planner_gate_v1"
ALLOWED_CANDIDATES = {"sft_v2", "sft_v2_c"}
COARSE_FIELDS = (
    "anion_framework",
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)


def distribution_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    histograms = {field: Counter() for field in COARSE_FIELDS}
    arity = Counter()
    parsed_count = 0
    for row in rows:
        if row.get("parsed") is not True:
            continue
        plan = row.get("plan_state")
        if not isinstance(plan, Mapping):
            raise ValueError("parsed Planner row is missing plan_state")
        parsed_count += 1
        for field in COARSE_FIELDS:
            value = str(plan.get(field) or "")
            if not value:
                raise ValueError(f"parsed Planner row omitted {field}")
            histograms[field][value] += 1
        elements = plan.get("elements")
        if not isinstance(elements, list) or not elements:
            raise ValueError("parsed Planner row omitted elements")
        arity[str(len(elements))] += 1
    complete = all(sum(hist.values()) == parsed_count for hist in histograms.values())
    complete = complete and sum(arity.values()) == parsed_count
    return {
        "parsed_count": parsed_count,
        "complete": complete,
        "arity_counts": dict(sorted(arity.items(), key=lambda item: int(item[0]))),
        "coarse_field_counts": {
            field: dict(sorted(histogram.items()))
            for field, histogram in histograms.items()
        },
    }


def total_variation(
    left: Mapping[str, int],
    right: Mapping[str, int],
    *,
    left_denominator: int,
    right_denominator: int,
) -> float | None:
    if int(left_denominator) <= 0 or int(right_denominator) <= 0:
        return None
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(
            int(left.get(key, 0)) / int(left_denominator)
            - int(right.get(key, 0)) / int(right_denominator)
        )
        for key in keys
    )


def distribution_deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "arity_total_variation": total_variation(
            baseline["arity_counts"],
            candidate["arity_counts"],
            left_denominator=int(baseline["parsed_count"]),
            right_denominator=int(candidate["parsed_count"]),
        ),
        "coarse_field_total_variation": {},
    }
    for field in COARSE_FIELDS:
        result["coarse_field_total_variation"][field] = total_variation(
            baseline["coarse_field_counts"][field],
            candidate["coarse_field_counts"][field],
            left_denominator=int(baseline["parsed_count"]),
            right_denominator=int(candidate["parsed_count"]),
        )
    return result


def load_training_terminal(path: Path, candidate: str) -> dict[str, Any]:
    terminal = read_json(path)
    gate = terminal.get("conditional_structural_anchor_nll_gate")
    if (
        terminal.get("schema") != H1_CHEMISTRY_FIRST_SFT_SCHEMA
        or terminal.get("candidate") != candidate
        or terminal.get("status") != "pass"
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or terminal.get("generated_metric_checkpoint_selection") is not False
    ):
        raise ValueError("candidate training terminal or anchor-NLL identity mismatch")
    return terminal


def candidate_scientific_gates(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    stage: int,
    anchor_nll_passed: bool,
    distribution_complete: bool,
) -> tuple[dict[str, bool], dict[str, Any]]:
    if int(stage) not in {64, 256}:
        raise ValueError("Planner stage must be 64 or 256")
    deltas = metric_deltas(candidate, baseline)
    new_failure_classes = sorted(
        set(candidate["failure_class_counts"])
        - set(baseline["failure_class_counts"])
    )
    shortcut_delta = (
        deltas["single_element_shortcut_count"]
        + deltas["all_metal_shortcut_count"]
    )
    gates = {
        "legacy_comp_valid_literal_gain_ge_1": deltas["legacy_comp_valid_count"] >= 1,
        "legacy_nonshortcut_primary_literal_gain_ge_1": deltas[
            "legacy_primary_nonshortcut_count"
        ]
        >= 1,
        "parse_loss_at_most_1_attempt": deltas["parse_count"] >= -1,
        "completion_loss_at_most_1_attempt": deltas["completion_count"] >= -1,
        "shortcut_count_not_increased": shortcut_delta <= 0,
        "unique_formula_rate_loss_at_most_2pp": deltas["unique_formula_rate"]
        >= -0.02 - 1e-12,
        "element_coverage_rate_loss_at_most_2pp": deltas["element_coverage_rate"]
        >= -0.02 - 1e-12,
        "absolute_mean_N_drift_at_most_0p5": deltas["mean_N_when_parsed"]
        is not None
        and abs(float(deltas["mean_N_when_parsed"])) <= 0.5 + 1e-12,
        "conditional_anchor_NLL_degradation_at_most_1pct": bool(anchor_nll_passed),
        "no_new_generation_failure_class": not new_failure_classes,
    }
    if int(stage) == 256:
        gates.update(
            {
                "top1_formula_rate_increase_at_most_2pp": deltas[
                    "top1_formula_rate"
                ]
                <= 0.02 + 1e-12,
                # Arity and coarse marginals were preregistered as complete
                # audits, not as post-hoc numeric superiority thresholds.
                "arity_and_coarse_field_marginal_audit_complete": bool(
                    distribution_complete
                ),
                "smact4_validity_audit_complete": sum(
                    int(value)
                    for value in candidate["smact4_stratum_counts"].values()
                )
                == int(candidate["denominator"])
                and int(candidate["smact4_valid_count"])
                <= int(candidate["parse_count"]),
            }
        )
    return gates, {
        "candidate_minus_p0": deltas,
        "shortcut_count_delta": shortcut_delta,
        "new_failure_classes": new_failure_classes,
    }


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    candidate_id = str(args.candidate_id)
    if candidate_id not in ALLOWED_CANDIDATES:
        raise ValueError(f"unsupported candidate {candidate_id!r}")
    denominator = int(args.stage)
    ledger_seeds = load_ledger(args.ledger, denominator)
    raw_paths = {"p0": args.p0_raw, candidate_id: args.candidate_raw}
    smact_paths = {"p0": args.p0_smact4, candidate_id: args.candidate_smact4}
    raw_rows = {
        arm: read_raw(path, denominator) for arm, path in raw_paths.items()
    }
    smact_rows: dict[str, dict[int, dict[str, Any]]] = {}
    smact_contracts: dict[str, dict[str, Any]] = {}
    for arm in ("p0", candidate_id):
        smact_rows[arm], smact_contracts[arm] = load_smact4_audit(
            smact_paths[arm],
            raw_path=raw_paths[arm],
            arm=arm,
            denominator=denominator,
        )
    summaries: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ("p0", candidate_id):
        summaries[arm], records[arm] = classify_arm(
            raw_rows[arm],
            smact_rows[arm],
            arm=arm,
            denominator=denominator,
        )
    training_terminal = load_training_terminal(
        args.training_terminal,
        candidate_id,
    )

    seed_mismatches = []
    for ordinal in range(denominator):
        observed = {
            records["p0"][ordinal]["sampling_seed"],
            records[candidate_id][ordinal]["sampling_seed"],
        }
        if observed != {ledger_seeds[ordinal]}:
            seed_mismatches.append(ordinal)
    contract_shas = {
        contract.get("contract_sha256") for contract in smact_contracts.values()
    }
    distribution = {
        "p0": distribution_summary(raw_rows["p0"]),
        candidate_id: distribution_summary(raw_rows[candidate_id]),
    }
    distribution_delta = distribution_deltas(
        distribution["p0"],
        distribution[candidate_id],
    )
    engineering_gates = {
        "both_arms_exact_all_attempt_denominator": True,
        "common_registered_ordinal_seeds_100pct": not seed_mismatches,
        "exact_common_smact4_contract": len(contract_shas) == 1
        and None not in contract_shas,
        "all_identity_checks_zero": all(
            all(int(value) == 0 for value in summaries[arm]["identity_failures"].values())
            for arm in ("p0", candidate_id)
        ),
        "candidate_generated_charge_field_zero": summaries[candidate_id][
            "generated_charge_field_count"
        ]
        == 0,
        "latency_complete": all(
            summaries[arm]["planner_generation_latency"]["count"] == denominator
            for arm in ("p0", candidate_id)
        ),
        "training_terminal_and_anchor_identity": True,
        "distribution_accounting_complete": all(
            value["complete"] for value in distribution.values()
        ),
    }
    scientific_gates, deltas = candidate_scientific_gates(
        summaries["p0"],
        summaries[candidate_id],
        stage=denominator,
        anchor_nll_passed=training_terminal[
            "conditional_structural_anchor_nll_gate"
        ]["passed"]
        is True,
        distribution_complete=engineering_gates["distribution_accounting_complete"],
    )
    engineering_passed = all(engineering_gates.values())
    scientific_passed = all(scientific_gates.values())
    if not engineering_passed:
        status = "engineering_failure"
        decision = f"stop_{candidate_id}_engineering"
        exit_code = 2
    elif scientific_passed:
        status = "planner_gate_pass"
        decision = (
            "eligible_for_raw256" if denominator == 64 else "eligible_for_planner_shortlist"
        )
        exit_code = 0
    else:
        status = "scientific_stop"
        decision = f"stop_{candidate_id}_no_rl"
        exit_code = 0
    paired = {
        field: paired_binary_report(
            records["p0"],
            records[candidate_id],
            field=field,
            denominator=denominator,
        )
        for field in (
            "legacy_comp_valid",
            "legacy_primary_nonshortcut",
            "smact4_valid",
            "smact4_uniform_primary",
            "parsed",
            "completion",
        )
    }
    result = {
        "schema": SCHEMA,
        "stage": denominator,
        "candidate_id": candidate_id,
        "status": status,
        "decision": decision,
        "engineering_passed": engineering_passed,
        "scientific_passed": scientific_passed,
        "gate_passed": engineering_passed and scientific_passed,
        "paper_comparable_primary_evaluator": "frozen_legacy_crysllmgen_smact",
        "secondary_evaluator": "exact_SMACT_4.0.0_ICSD24_frozen_contract",
        "arms": summaries,
        "deltas": deltas,
        "distribution_audit": distribution,
        "distribution_deltas": distribution_delta,
        "paired_candidate_vs_p0": paired,
        "engineering_gates": engineering_gates,
        "scientific_gates": scientific_gates,
        "seed_mismatch_ordinals": seed_mismatches,
        "training_terminal_sha256": sha256_file(args.training_terminal),
        "science_ledger_sha256": sha256_file(args.ledger),
        "raw_sha256": {
            arm: sha256_file(path) for arm, path in raw_paths.items()
        },
        "smact4_audit_sha256": {
            arm: sha256_file(path) for arm, path in smact_paths.items()
        },
        "automatic_downstream": False,
        "automatic_rl": False,
    }
    return result, exit_code


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    candidate = str(result["candidate_id"])
    lines = [
        f"# H1 chemistry-first {candidate} raw{result['stage']} gate",
        "",
        f"- status: `{result['status']}`",
        f"- decision: `{result['decision']}`",
        f"- engineering_passed: `{result['engineering_passed']}`",
        f"- scientific_passed: `{result['scientific_passed']}`",
        "",
        "| arm | parse | completion | legacy comp | legacy primary | SMACT4 valid | SMACT4 uniform | unique | mean N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("p0", candidate):
        row = result["arms"][arm]
        denominator = row["denominator"]
        lines.append(
            f"| {arm} | {row['parse_count']}/{denominator} | {row['completion_count']}/{denominator} | "
            f"{row['legacy_comp_valid_count']}/{denominator} | {row['legacy_primary_nonshortcut_count']}/{denominator} | "
            f"{row['smact4_valid_count']}/{denominator} | {row['smact4_uniform_primary_count']}/{denominator} | "
            f"{row['unique_formula_count']}/{denominator} | {row['mean_N_when_parsed']} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in {
            **result["engineering_gates"],
            **result["scientific_gates"],
        }.items()
    )
    lines.extend(
        [
            "",
            "## Paired candidate vs P0",
            "",
            "```json",
            json.dumps(result["paired_candidate_vs_p0"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(64, 256), required=True)
    parser.add_argument("--candidate-id", choices=tuple(sorted(ALLOWED_CANDIDATES)), required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--training-terminal", type=Path, required=True)
    parser.add_argument("--p0-raw", type=Path, required=True)
    parser.add_argument("--p0-smact4", type=Path, required=True)
    parser.add_argument("--candidate-raw", type=Path, required=True)
    parser.add_argument("--candidate-smact4", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result, exit_code = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(result, args.output_md)
    print(
        json.dumps(
            {"status": result["status"], "gate_passed": result["gate_passed"]}
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
