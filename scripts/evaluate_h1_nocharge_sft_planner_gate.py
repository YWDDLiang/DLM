#!/usr/bin/env python3
"""Assemble the preregistered H1 no-charge ion-auxiliary Planner gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.composition_validity import classify_smact_validity  # noqa: E402
from crystal_dlm.fixed_slot import SYMBOL_TO_Z  # noqa: E402
from crystal_dlm.h1_llm_planner import (  # noqa: E402
    H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
)
from crystal_dlm.h1_nocharge_ion_aux import (  # noqa: E402
    SMACT4_VERSION,
)
from crystal_dlm.h1_local_smact4_ledger import (  # noqa: E402
    EXPECTED_SMACT4_CONTRACT_SHA256,
)


SCHEMA = "h1_nocharge_ion_aux_planner_gate_v1"
SMACT4_AUDIT_SCHEMA = "h1_nocharge_planner_smact4_audit_v1"
ARMS = ("p0", "c0", "c1")
FORBIDDEN_TRUE_FIELDS = (
    "retry",
    "replacement",
    "repair",
    "filter",
    "rerank",
    "fallback",
    "network",
    "training_during_generation",
    "checkpoint_reselection",
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


def read_raw(path: Path, denominator: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object row")
                rows.append(value)
    ordinals = [int(row.get("sample_idx", -1)) for row in rows]
    if len(rows) != denominator or ordinals != list(range(denominator)):
        raise ValueError(f"{path} does not contain exact ordered ordinals 0..{denominator - 1}")
    return rows


def load_ledger(path: Path, denominator: int) -> dict[int, int]:
    ledger = read_json(path)
    if int(ledger.get("denominator", -1)) != denominator:
        raise ValueError("science ledger denominator mismatch")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != denominator:
        raise ValueError("science ledger rows are incomplete")
    result = {
        int(row["ordinal"]): int(row["planner_sampling_seed"])
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(result) != set(range(denominator)):
        raise ValueError("science ledger ordinals are incomplete")
    return result


def load_smact4_audit(
    path: Path,
    *,
    raw_path: Path,
    arm: str,
    denominator: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    report = read_json(path)
    if (
        report.get("schema") != SMACT4_AUDIT_SCHEMA
        or report.get("status") != "pass"
        or report.get("arm") != arm
        or int(report.get("denominator", -1)) != denominator
        or report.get("raw_generations_sha256") != sha256_file(raw_path)
    ):
        raise ValueError(f"{arm} SMACT4 audit identity mismatch")
    contract = report.get("smact4_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("smact_version") != SMACT4_VERSION
        or contract.get("contract_sha256") != EXPECTED_SMACT4_CONTRACT_SHA256
    ):
        raise ValueError(f"{arm} did not use exact SMACT {SMACT4_VERSION}")
    attempts = report.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != denominator:
        raise ValueError(f"{arm} SMACT4 audit attempts are incomplete")
    by_ordinal = {
        int(row["ordinal"]): dict(row)
        for row in attempts
        if isinstance(row, Mapping)
    }
    if set(by_ordinal) != set(range(denominator)):
        raise ValueError(f"{arm} SMACT4 audit ordinals are incomplete")
    return by_ordinal, dict(contract)


def exact_mcnemar_p(baseline_only: int, candidate_only: int) -> float:
    discordant = int(baseline_only) + int(candidate_only)
    if discordant == 0:
        return 1.0
    k = min(int(baseline_only), int(candidate_only))
    tail = sum(math.comb(discordant, value) for value in range(k + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty vector")
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    differences: Sequence[int],
    *,
    draws: int = 10_000,
    seed: int = 26080619,
) -> dict[str, Any]:
    values = [int(value) for value in differences]
    if not values or any(value not in {-1, 0, 1} for value in values):
        raise ValueError("paired binary differences must be non-empty and in {-1,0,1}")
    rng = random.Random(int(seed))
    n = len(values)
    samples = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(int(draws))
    ]
    return {
        "draws": int(draws),
        "seed": int(seed),
        "observed_delta_rate": sum(values) / n,
        "lower_95": percentile(samples, 0.025),
        "upper_95": percentile(samples, 0.975),
    }


def paired_binary_report(
    baseline: Mapping[int, Mapping[str, Any]],
    candidate: Mapping[int, Mapping[str, Any]],
    *,
    field: str,
    denominator: int,
) -> dict[str, Any]:
    differences = [
        int(candidate[ordinal].get(field) is True)
        - int(baseline[ordinal].get(field) is True)
        for ordinal in range(denominator)
    ]
    baseline_only = sum(value == -1 for value in differences)
    candidate_only = sum(value == 1 for value in differences)
    return {
        "field": field,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "discordant": baseline_only + candidate_only,
        "candidate_minus_baseline_count": candidate_only - baseline_only,
        "candidate_minus_baseline_rate": (candidate_only - baseline_only) / denominator,
        "mcnemar_two_sided_exact_p": exact_mcnemar_p(baseline_only, candidate_only),
        "paired_bootstrap_95": paired_bootstrap_ci(differences),
    }


def latency_summary(values: Sequence[float], denominator: int) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if len(ordered) != denominator or any(not math.isfinite(value) or value <= 0 for value in ordered):
        raise ValueError("planner latency vector is incomplete/non-finite")
    return {
        "count": denominator,
        "median_sec": statistics.median(ordered),
        "p95_sec": ordered[max(0, math.ceil(0.95 * denominator) - 1)],
        "mean_sec": statistics.fmean(ordered),
        "min_sec": ordered[0],
        "max_sec": ordered[-1],
    }


def classify_arm(
    rows: Sequence[Mapping[str, Any]],
    smact4_rows: Mapping[int, Mapping[str, Any]],
    *,
    arm: str,
    denominator: int,
    legacy_classifier: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if legacy_classifier is None:
        legacy_classifier = classify_smact_validity
    expected_style = H1_PLANNER_PROMPT_STYLE_RICH_PLAN if arm == "p0" else H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE
    formulas: Counter[str] = Counter()
    elements: set[str] = set()
    atom_counts: list[int] = []
    arity_counts: Counter[str] = Counter()
    anion_counts: Counter[str] = Counter()
    legacy_reasons: Counter[str] = Counter()
    smact4_strata: Counter[str] = Counter()
    failure_classes: Counter[str] = Counter()
    latencies: list[float] = []
    records: dict[int, dict[str, Any]] = {}
    counters: Counter[str] = Counter()

    for ordinal, row in enumerate(rows):
        prompt_style = str(row.get("prompt_style") or "")
        if prompt_style != expected_style:
            counters["prompt_style_identity_failure"] += 1
        if row.get("seed_mode") != "stateless_ordinal_v1":
            counters["seed_mode_identity_failure"] += 1
        if row.get("formula_constraint_mode") != "off":
            counters["unexpected_formula_constraint"] += 1
        if row.get("crplan_diagnostics") is not None:
            counters["unexpected_crplan_diagnostics"] += 1
        for field in FORBIDDEN_TRUE_FIELDS:
            if row.get(field) is True:
                counters["forbidden_operation_failure"] += 1
        latency = float(row.get("planner_generation_latency_sec") or 0.0)
        latencies.append(latency)
        completion = row.get("plan_end_marker_present") is True
        parsed = row.get("parsed") is True
        counters["completion"] += int(completion)
        smact4 = smact4_rows[ordinal]
        record: dict[str, Any] = {
            "ordinal": ordinal,
            "parsed": parsed,
            "completion": completion,
            "formula": None,
            "legacy_comp_valid": False,
            "legacy_primary_nonshortcut": False,
            "single_element_shortcut": False,
            "all_metal_shortcut": False,
            "smact4_valid": False,
            "smact4_uniform_primary": False,
            "sampling_seed": row.get("planner_sampling_seed"),
            "prompt_sha256": row.get("planner_input_prompt_sha256"),
            "input_ids_sha256": row.get("planner_input_ids_sha256"),
        }
        if not parsed:
            failure_class = str(row.get("reason") or "planner_parse_failure")
            failure_classes[failure_class] += 1
            if smact4.get("parsed") is not False:
                counters["smact4_parse_identity_failure"] += 1
            legacy_reasons["planner_parse_failure"] += 1
            smact4_strata["planner_parse_failure"] += 1
            records[ordinal] = record
            continue

        plan = row.get("plan_state")
        if not isinstance(plan, Mapping):
            raise ValueError(f"{arm} parsed ordinal {ordinal} is missing plan_state")
        formula = str(plan.get("formula") or "")
        symbols = [str(value) for value in (plan.get("elements") or ())]
        counts = [int(value) for value in (plan.get("counts") or ())]
        if not formula or not symbols or len(symbols) != len(counts) or any(value <= 0 for value in counts):
            raise ValueError(f"{arm} parsed ordinal {ordinal} has invalid formula arrays")
        if any(symbol not in SYMBOL_TO_Z for symbol in symbols):
            raise ValueError(f"{arm} parsed ordinal {ordinal} has an unsupported element")
        classification = dict(
            legacy_classifier([SYMBOL_TO_Z[symbol] for symbol in symbols], counts)
        )
        valid = classification.get("valid") is True
        reason = str(classification.get("reason") or "unknown")
        embedded = plan.get("validator")
        if not isinstance(embedded, Mapping) or embedded.get("valid") != classification.get("valid") or embedded.get("reason") != classification.get("reason"):
            counters["legacy_embedded_identity_failure"] += 1
        generated_fields = plan.get("generated_rich_fields")
        raw_text = str(row.get("raw_plan_text") or "")
        generated_charge = (
            isinstance(generated_fields, Mapping) and "charge" in generated_fields
        ) or any(line.strip().lower().startswith("charge:") for line in raw_text.splitlines())
        counters["generated_charge_field"] += int(generated_charge)
        if arm != "p0" and generated_charge:
            counters["nocharge_schema_failure"] += 1

        if smact4.get("parsed") is not True or smact4.get("formula") != formula:
            counters["smact4_formula_identity_failure"] += 1
        if smact4.get("official_witness_parity") is not True:
            counters["smact4_official_witness_parity_failure"] += 1
        smact_valid = smact4.get("valid") is True
        smact_stratum = str(smact4.get("stratum") or "unknown")
        legacy_primary = valid and reason == "charge_neutral_pauling_valid"
        single = valid and reason == "single_element_shortcut"
        all_metal = valid and reason == "all_metal_shortcut"
        smact_uniform = smact_valid and smact_stratum == "uniform_primary"
        counters["parse"] += 1
        counters["legacy_comp_valid"] += int(valid)
        counters["legacy_primary_nonshortcut"] += int(legacy_primary)
        counters["single_element_shortcut"] += int(single)
        counters["all_metal_shortcut"] += int(all_metal)
        counters["smact4_valid"] += int(smact_valid)
        counters["smact4_uniform_primary"] += int(smact_uniform)
        legacy_reasons[reason] += 1
        smact4_strata[smact_stratum] += 1
        formulas[formula] += 1
        elements.update(symbols)
        n_atoms = sum(counts)
        atom_counts.append(n_atoms)
        arity_counts[str(len(symbols))] += 1
        anion_counts[str(plan.get("anion_framework") or "unknown")] += 1
        record.update(
            {
                "formula": formula,
                "legacy_comp_valid": valid,
                "legacy_primary_nonshortcut": legacy_primary,
                "single_element_shortcut": single,
                "all_metal_shortcut": all_metal,
                "legacy_reason": reason,
                "smact4_valid": smact_valid,
                "smact4_uniform_primary": smact_uniform,
                "smact4_stratum": smact_stratum,
            }
        )
        records[ordinal] = record

    top1 = max(formulas.values(), default=0)
    summary = {
        "denominator": denominator,
        "parse_count": counters["parse"],
        "parse_rate": counters["parse"] / denominator,
        "completion_count": counters["completion"],
        "completion_rate": counters["completion"] / denominator,
        "legacy_comp_valid_count": counters["legacy_comp_valid"],
        "legacy_comp_valid_rate": counters["legacy_comp_valid"] / denominator,
        "legacy_primary_nonshortcut_count": counters["legacy_primary_nonshortcut"],
        "legacy_primary_nonshortcut_rate": counters["legacy_primary_nonshortcut"] / denominator,
        "single_element_shortcut_count": counters["single_element_shortcut"],
        "all_metal_shortcut_count": counters["all_metal_shortcut"],
        "smact4_valid_count": counters["smact4_valid"],
        "smact4_valid_rate": counters["smact4_valid"] / denominator,
        "smact4_uniform_primary_count": counters["smact4_uniform_primary"],
        "smact4_uniform_primary_rate": counters["smact4_uniform_primary"] / denominator,
        "legacy_reason_counts": dict(sorted(legacy_reasons.items())),
        "smact4_stratum_counts": dict(sorted(smact4_strata.items())),
        "failure_class_counts": dict(sorted(failure_classes.items())),
        "unique_formula_count": len(formulas),
        "unique_formula_rate": len(formulas) / denominator,
        "top1_formula_count": top1,
        "top1_formula_rate": top1 / denominator,
        "element_coverage_count": len(elements),
        "element_coverage_denominator": len(SYMBOL_TO_Z),
        "element_coverage_rate": len(elements) / len(SYMBOL_TO_Z),
        "elements": sorted(elements, key=lambda symbol: SYMBOL_TO_Z[symbol]),
        "mean_N_when_parsed": statistics.fmean(atom_counts) if atom_counts else None,
        "arity_counts": dict(sorted(arity_counts.items(), key=lambda item: int(item[0]))),
        "anion_family_counts": dict(sorted(anion_counts.items())),
        "planner_generation_latency": latency_summary(latencies, denominator),
        "identity_failures": {
            key: counters[key]
            for key in (
                "prompt_style_identity_failure",
                "seed_mode_identity_failure",
                "unexpected_formula_constraint",
                "unexpected_crplan_diagnostics",
                "forbidden_operation_failure",
                "legacy_embedded_identity_failure",
                "nocharge_schema_failure",
                "smact4_parse_identity_failure",
                "smact4_formula_identity_failure",
                "smact4_official_witness_parity_failure",
            )
        },
        "generated_charge_field_count": counters["generated_charge_field"],
    }
    return summary, records


def metric_deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "parse_count",
        "completion_count",
        "legacy_comp_valid_count",
        "legacy_primary_nonshortcut_count",
        "single_element_shortcut_count",
        "all_metal_shortcut_count",
        "smact4_valid_count",
        "smact4_uniform_primary_count",
        "unique_formula_rate",
        "top1_formula_rate",
        "element_coverage_rate",
    )
    deltas = {key: float(candidate[key]) - float(baseline[key]) for key in keys}
    candidate_mean = candidate.get("mean_N_when_parsed")
    baseline_mean = baseline.get("mean_N_when_parsed")
    deltas["mean_N_when_parsed"] = (
        None
        if candidate_mean is None or baseline_mean is None
        else float(candidate_mean) - float(baseline_mean)
    )
    return deltas


def scientific_gates(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    stage: int,
) -> tuple[dict[str, bool], dict[str, Any]]:
    c0 = summaries["c0"]
    c1 = summaries["c1"]
    p0 = summaries["p0"]
    c1_c0 = metric_deltas(c1, c0)
    c1_p0 = metric_deltas(c1, p0)
    count_gain = 3 if stage == 64 else 8
    parse_loss = 1 if stage == 64 else 2
    c0_failure_classes = set(c0["failure_class_counts"])
    c1_new_failure_classes = sorted(set(c1["failure_class_counts"]) - c0_failure_classes)
    gates = {
        f"c1_minus_c0_legacy_comp_valid_gain_ge_{count_gain}": c1_c0["legacy_comp_valid_count"] >= count_gain,
        f"c1_minus_c0_legacy_primary_gain_ge_{count_gain}": c1_c0["legacy_primary_nonshortcut_count"] >= count_gain,
        "c1_parse_loss_within_limit": c1_c0["parse_count"] >= -parse_loss,
        "c1_completion_loss_within_limit": c1_c0["completion_count"] >= -parse_loss,
        "single_element_shortcut_not_increased": c1_c0["single_element_shortcut_count"] <= 0,
        "all_metal_shortcut_not_increased": c1_c0["all_metal_shortcut_count"] <= 0,
        "unique_formula_rate_loss_at_most_2pp": c1_c0["unique_formula_rate"] >= -0.02 - 1e-12,
        "element_coverage_rate_loss_at_most_2pp": c1_c0["element_coverage_rate"] >= -0.02 - 1e-12,
        "absolute_mean_N_drift_at_most_0p5": c1_c0["mean_N_when_parsed"] is not None and abs(c1_c0["mean_N_when_parsed"]) <= 0.5 + 1e-12,
        "smact4_valid_not_decreased": c1_c0["smact4_valid_count"] >= 0,
        "smact4_uniform_primary_not_decreased": c1_c0["smact4_uniform_primary_count"] >= 0,
        "no_new_generation_failure_class": not c1_new_failure_classes,
    }
    if stage == 256:
        gates.update(
            {
                "c1_legacy_comp_valid_not_below_p0": c1_p0["legacy_comp_valid_count"] >= 0,
                "c1_legacy_primary_not_below_p0": c1_p0["legacy_primary_nonshortcut_count"] >= 0,
                "top1_formula_rate_increase_at_most_2pp": c1_c0["top1_formula_rate"] <= 0.02 + 1e-12,
            }
        )
    return gates, {
        "c1_minus_c0": c1_c0,
        "c1_minus_p0": c1_p0,
        "c1_new_failure_classes": c1_new_failure_classes,
    }


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    denominator = int(args.stage)
    ledger_seeds = load_ledger(args.ledger, denominator)
    raw_paths = {arm: getattr(args, f"{arm}_raw") for arm in ARMS}
    smact_paths = {arm: getattr(args, f"{arm}_smact4") for arm in ARMS}
    raw_rows = {arm: read_raw(raw_paths[arm], denominator) for arm in ARMS}
    smact_rows: dict[str, dict[int, dict[str, Any]]] = {}
    smact_contracts: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        smact_rows[arm], smact_contracts[arm] = load_smact4_audit(
            smact_paths[arm], raw_path=raw_paths[arm], arm=arm, denominator=denominator
        )

    summaries: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ARMS:
        summaries[arm], records[arm] = classify_arm(
            raw_rows[arm], smact_rows[arm], arm=arm, denominator=denominator
        )

    seed_mismatches: list[int] = []
    candidate_prompt_mismatches: list[int] = []
    for ordinal in range(denominator):
        arm_seeds = {records[arm][ordinal]["sampling_seed"] for arm in ARMS}
        if arm_seeds != {ledger_seeds[ordinal]}:
            seed_mismatches.append(ordinal)
        if (
            records["c0"][ordinal]["prompt_sha256"] != records["c1"][ordinal]["prompt_sha256"]
            or records["c0"][ordinal]["input_ids_sha256"] != records["c1"][ordinal]["input_ids_sha256"]
        ):
            candidate_prompt_mismatches.append(ordinal)
    contract_shas = {contract.get("contract_sha256") for contract in smact_contracts.values()}
    engineering_gates = {
        "all_arms_exact_all_attempt_denominator": True,
        "common_registered_ordinal_seeds_100pct": not seed_mismatches,
        "c0_c1_prompt_and_input_identity_100pct": not candidate_prompt_mismatches,
        "exact_common_smact4_contract": len(contract_shas) == 1 and None not in contract_shas,
        "all_arm_identity_checks_zero": all(
            all(int(value) == 0 for value in summaries[arm]["identity_failures"].values())
            for arm in ARMS
        ),
        "c0_c1_generated_charge_field_zero": summaries["c0"]["generated_charge_field_count"] == 0 and summaries["c1"]["generated_charge_field_count"] == 0,
        "latency_complete": all(summaries[arm]["planner_generation_latency"]["count"] == denominator for arm in ARMS),
    }
    gates, deltas = scientific_gates(summaries, stage=denominator)
    engineering_passed = all(engineering_gates.values())
    scientific_passed = all(gates.values())
    if not engineering_passed:
        status = "engineering_failure"
        decision = "stop_nocharge_ion_aux_sft_route"
        exit_code = 2
    elif scientific_passed:
        status = "planner_gate_pass"
        decision = "eligible_for_next_preregistered_stage"
        exit_code = 0
    else:
        status = "scientific_stop"
        decision = "stop_nocharge_ion_aux_sft_route_no_rl"
        exit_code = 0
    paired = {
        field: paired_binary_report(
            records["c0"], records["c1"], field=field, denominator=denominator
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
        "status": status,
        "decision": decision,
        "engineering_passed": engineering_passed,
        "scientific_passed": scientific_passed,
        "gate_passed": engineering_passed and scientific_passed,
        "paper_comparable_primary_evaluator": "frozen_legacy_crysllmgen_smact",
        "secondary_evaluator": f"SMACT_{SMACT4_VERSION}_ICSD24_frozen_contract",
        "arms": summaries,
        "primary_contrast": "c1_explicit_oxidation_aux_minus_c0_neutral_atom_aux",
        "deltas": deltas,
        "paired_c1_vs_c0": paired,
        "engineering_gates": engineering_gates,
        "scientific_gates": gates,
        "identity_failure_ordinals": {
            "seed": seed_mismatches,
            "c0_c1_prompt_or_input": candidate_prompt_mismatches,
        },
        "science_ledger_sha256": sha256_file(args.ledger),
        "raw_sha256": {arm: sha256_file(raw_paths[arm]) for arm in ARMS},
        "smact4_audit_sha256": {arm: sha256_file(smact_paths[arm]) for arm in ARMS},
        "automatic_downstream": False,
        "automatic_rl": False,
    }
    return result, exit_code


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        f"# H1 no-charge ion-aux Planner {result['stage']} gate",
        "",
        f"- status: `{result['status']}`",
        f"- decision: `{result['decision']}`",
        f"- engineering_passed: `{result['engineering_passed']}`",
        f"- scientific_passed: `{result['scientific_passed']}`",
        "",
        "| arm | parse | completion | legacy comp_valid | legacy primary | SMACT4 valid | SMACT4 uniform | unique | mean N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = result["arms"][arm]
        lines.append(
            f"| {arm} | {row['parse_count']}/{row['denominator']} | {row['completion_count']}/{row['denominator']} | "
            f"{row['legacy_comp_valid_count']}/{row['denominator']} | {row['legacy_primary_nonshortcut_count']}/{row['denominator']} | "
            f"{row['smact4_valid_count']}/{row['denominator']} | {row['smact4_uniform_primary_count']}/{row['denominator']} | "
            f"{row['unique_formula_count']}/{row['denominator']} | {row['mean_N_when_parsed']} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in {**result["engineering_gates"], **result["scientific_gates"]}.items()
    )
    lines.extend(["", "## Paired C1 vs C0", "", "```json", json.dumps(result["paired_c1_vs_c0"], ensure_ascii=False, indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(64, 256), required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    for arm in ARMS:
        parser.add_argument(f"--{arm}-raw", type=Path, required=True)
        parser.add_argument(f"--{arm}-smact4", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result, exit_code = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_md)
    print(json.dumps({"status": result["status"], "gate_passed": result["gate_passed"]}))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
