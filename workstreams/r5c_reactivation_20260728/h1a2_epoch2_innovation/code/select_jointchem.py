#!/usr/bin/env python3
"""Outcome-blind checkpoint and Plan-arm selection for H1-A2C JointChem."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


TVD_KEYS = (
    "n_tvd",
    "num_elements_tvd",
    "element_presence_tvd",
    "anion_framework_tvd",
    "charge_bucket_tvd",
    "lattice_system_tvd",
    "spacegroup_bucket_tvd",
    "volume_per_atom_bin_tvd",
)
MAX_RELATIVE_NLL_DEGRADATION = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def checkpoint_margins(event: Mapping[str, Any]) -> dict[str, float | None]:
    metrics = event.get("eval") if isinstance(event.get("eval"), Mapping) else {}
    return {
        "chemistry": (
            None
            if metrics.get("chemistry_paired_margin") is None
            else float(metrics["chemistry_paired_margin"])
        ),
        "joint": (
            None
            if metrics.get("joint_paired_margin") is None
            else float(metrics["joint_paired_margin"])
        ),
    }


def select_checkpoint(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("ok") is not True or int(report.get("global_step", -1)) != 400:
        raise ValueError("training report is not a complete 400-update run")
    arm = str(report.get("arm"))
    initial_eval = (
        report.get("initial_epoch2_eval")
        if isinstance(report.get("initial_epoch2_eval"), Mapping)
        else {}
    )
    if "positive_nll" not in initial_eval:
        raise ValueError("training report lacks initial epoch-2 positive_nll")
    initial_positive_nll = float(initial_eval["positive_nll"])
    if initial_positive_nll <= 0:
        raise ValueError(f"invalid initial epoch-2 positive_nll={initial_positive_nll}")
    checkpoints = list(report.get("checkpoints") or [])
    expected_steps = list(range(50, 401, 50))
    observed_steps = [int(event["step"]) for event in checkpoints]
    if observed_steps != expected_steps:
        raise ValueError(f"checkpoint steps mismatch: expected {expected_steps}, observed {observed_steps}")

    candidates: list[dict[str, Any]] = []
    for event in checkpoints:
        metrics = event.get("eval") if isinstance(event.get("eval"), Mapping) else {}
        if "positive_nll" not in metrics:
            raise ValueError(f"checkpoint {event['step']} lacks positive_nll")
        margins = checkpoint_margins(event)
        eligible = True
        reasons: list[str] = []
        relative_nll_change = (
            float(metrics["positive_nll"]) - initial_positive_nll
        ) / initial_positive_nll
        if relative_nll_change > MAX_RELATIVE_NLL_DEGRADATION:
            eligible = False
            reasons.append("positive_nll_degradation_above_1pct")
        if arm == "jointchem":
            for key in ("chemistry", "joint"):
                if margins[key] is None or float(margins[key]) <= 0:
                    eligible = False
                    reasons.append(f"{key}_margin_not_positive")
        candidates.append(
            {
                "step": int(event["step"]),
                "checkpoint_dir": str(event["checkpoint_dir"]),
                "checkpoint_manifest_sha256": str(event["checkpoint_manifest_sha256"]),
                "positive_nll": float(metrics["positive_nll"]),
                "initial_epoch2_positive_nll": initial_positive_nll,
                "relative_nll_change_vs_epoch2": relative_nll_change,
                "nll_noninferior": relative_nll_change <= MAX_RELATIVE_NLL_DEGRADATION,
                "diagnostic_loss": float(metrics.get("loss", metrics["positive_nll"])),
                "margins": margins,
                "eligible": eligible,
                "reasons": reasons,
            }
        )
    eligible = [value for value in candidates if value["eligible"]]
    if not eligible:
        return {
            "schema": "h1a2_jointchem_checkpoint_selection_v1",
            "arm": arm,
            "execution_manifest_sha256": str(report.get("execution_manifest_sha256")),
            "initial_adapter_sha256": str(report.get("initial_adapter_sha256")),
            "decision": "no_eligible_checkpoint",
            "selected": None,
            "candidates": candidates,
            "automatic_crystal_evaluation_authorized": False,
        }
    if arm == "jointchem":
        selected = min(
            eligible,
            key=lambda value: (
                value["diagnostic_loss"],
                -float(value["margins"]["joint"]),
                value["positive_nll"],
                value["step"],
            ),
        )
    else:
        selected = min(eligible, key=lambda value: (value["positive_nll"], value["step"]))
    return {
        "schema": "h1a2_jointchem_checkpoint_selection_v1",
        "arm": arm,
        "execution_manifest_sha256": str(report.get("execution_manifest_sha256")),
        "initial_adapter_sha256": str(report.get("initial_adapter_sha256")),
        "decision": "selected_for_plan_only_sampling",
        "selected": selected,
        "candidates": candidates,
        "automatic_crystal_evaluation_authorized": False,
        "sun_or_mlip_used_for_selection": False,
        "max_relative_nll_degradation": MAX_RELATIVE_NLL_DEGRADATION,
    }


def plan_eligibility(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    checkpoint_selection: Mapping[str, Any],
) -> tuple[bool, list[str], dict[str, float]]:
    if int(baseline.get("denominator", -1)) != 512 or int(candidate.get("denominator", -1)) != 512:
        raise ValueError("Plan reports must both use denominator=512")
    bcomp = baseline["composition"]
    ccomp = candidate["composition"]
    reasons: list[str] = []
    deltas = {
        "parse_rate": float(ccomp["parse_rate"]) - float(bcomp["parse_rate"]),
        "composition_valid_rate": float(ccomp["composition_valid_rate"])
        - float(bcomp["composition_valid_rate"]),
        "anion_match_rate": float(ccomp["anion_match_rate"]) - float(bcomp["anion_match_rate"]),
        "charge_match_rate": float(ccomp["charge_match_rate"]) - float(bcomp["charge_match_rate"]),
        "all_metal_rate": float(ccomp["all_metal_rate"]) - float(bcomp["all_metal_rate"]),
        "mean_N": float(candidate["generated_distribution"]["mean_N"])
        - float(baseline["generated_distribution"]["mean_N"]),
    }
    if deltas["parse_rate"] < -0.005:
        reasons.append("parse_rate_below_noninferiority")
    if deltas["composition_valid_rate"] < 0.02:
        reasons.append("composition_valid_gain_below_2pp")
    if deltas["anion_match_rate"] < -0.005:
        reasons.append("anion_match_below_noninferiority")
    if deltas["charge_match_rate"] < -0.005:
        reasons.append("charge_match_below_noninferiority")
    if deltas["all_metal_rate"] > 0.02:
        reasons.append("all_metal_inflation_above_2pp")
    if abs(deltas["mean_N"]) > 0.5:
        reasons.append("mean_N_drift_above_0.5")

    baseline_tvd = baseline["distribution_comparison"]
    candidate_tvd = candidate["distribution_comparison"]
    tvd_excess: dict[str, float] = {}
    for key in TVD_KEYS:
        if key not in baseline_tvd or key not in candidate_tvd:
            reasons.append(f"missing_{key}")
            continue
        tvd_excess[key] = float(candidate_tvd[key]) - float(baseline_tvd[key])
        if tvd_excess[key] > 0.02:
            reasons.append(f"{key}_worsened_above_0.02")

    selected_checkpoint = checkpoint_selection.get("selected")
    if not isinstance(selected_checkpoint, Mapping):
        reasons.append("no_selected_checkpoint")
    else:
        margins = selected_checkpoint.get("margins")
        if not isinstance(margins, Mapping):
            reasons.append("missing_likelihood_margins")
        else:
            for key in ("chemistry", "joint"):
                value = margins.get(key)
                if value is None:
                    reasons.append(f"{key}_likelihood_margin_missing")
                elif float(value) <= 0:
                    reasons.append(f"{key}_likelihood_margin_not_positive")
        if selected_checkpoint.get("nll_noninferior") is not True:
            reasons.append("validation_nll_not_noninferior_to_epoch2")
    deltas["max_tvd_excess"] = max(tvd_excess.values(), default=float("inf"))
    return not reasons, reasons, deltas


def select_plan_arm(
    baseline: Mapping[str, Any],
    candidates: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    audited: list[dict[str, Any]] = []
    for report, checkpoint_selection in candidates:
        eligible, reasons, deltas = plan_eligibility(
            baseline,
            report,
            checkpoint_selection=checkpoint_selection,
        )
        audited.append(
            {
                "identity": report["identity"],
                "arm": report["arm"],
                "step": int(report["step"]),
                "eligible": eligible,
                "reasons": reasons,
                "deltas_vs_baseline": deltas,
                "composition_valid_rate": float(report["composition"]["composition_valid_rate"]),
                "checkpoint_selection": checkpoint_selection,
            }
        )
    eligible = [value for value in audited if value["eligible"]]
    selected = (
        None
        if not eligible
        else max(
            eligible,
            key=lambda value: (
                value["composition_valid_rate"],
                -float(value["deltas_vs_baseline"]["max_tvd_excess"]),
                -value["step"],
            ),
        )
    )
    return {
        "schema": "h1a2_jointchem_plan_selection_v1",
        "decision": (
            "stop_no_plan_candidate"
            if selected is None
            else "selected_for_paired256_crystal_screen"
        ),
        "selected": selected,
        "candidates": audited,
        "automatic_crystal_evaluation_authorized": False,
        "sun_or_mlip_used_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--training-report", type=Path, required=True)
    checkpoint_parser.add_argument("--output", type=Path, required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--baseline-report", type=Path, required=True)
    plan_parser.add_argument("--candidate-report", type=Path, action="append", required=True)
    plan_parser.add_argument("--checkpoint-selection", type=Path, action="append", required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "checkpoint":
        result = select_checkpoint(load_json(args.training_report))
        input_sha = {"training_report": sha256_file(args.training_report)}
    else:
        if len(args.candidate_report) != len(args.checkpoint_selection):
            raise ValueError("candidate-report and checkpoint-selection counts differ")
        baseline = load_json(args.baseline_report)
        candidates = [
            (load_json(report), load_json(selection))
            for report, selection in zip(args.candidate_report, args.checkpoint_selection)
        ]
        result = select_plan_arm(baseline, candidates)
        input_sha = {
            "baseline_report": sha256_file(args.baseline_report),
            "candidate_reports": [sha256_file(path) for path in args.candidate_report],
            "checkpoint_selections": [sha256_file(path) for path in args.checkpoint_selection],
        }
    result["input_sha256"] = input_sha
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
