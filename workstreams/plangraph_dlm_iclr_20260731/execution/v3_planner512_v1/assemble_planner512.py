#!/usr/bin/env python3
"""Assemble the preregistered H1-A2 V3 Planner-512 decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("P0", "P-control", "P-star")
TVD_KEYS = (
    "n_tvd",
    "num_elements_tvd",
    "element_presence_tvd",
    "family_tvd",
    "arity_tvd",
    "size_tvd",
    "anion_framework_tvd",
    "charge_bucket_tvd",
    "lattice_system_tvd",
    "spacegroup_bucket_tvd",
    "volume_per_atom_bin_tvd",
)
EXPECTED_CHECKPOINTS = {
    "P0": "65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a",
    "P-control": "8fa09305d22d113e7567daefaba886fe0774feb9ece7c13664214670d20d6c50",
    "P-star": "617aebf19e9b13ec84d7751bdc3824c13eb4fb639f8ccc5067dcf5fc3f84e950",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} is non-finite")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--planner-training-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for arm in ARMS:
        path = args.run_root / "arms" / arm / "plan_report.json"
        if not path.exists():
            reasons.append(f"{arm}:missing_plan_report")
            continue
        report = read_json(path)
        if (
            report.get("status") != "complete"
            or report.get("arm") != arm
            or report.get("denominator") != 512
        ):
            reasons.append(f"{arm}:report_identity_failed")
            continue
        if report.get("checkpoint_identity_sha256") != EXPECTED_CHECKPOINTS[arm]:
            reasons.append(f"{arm}:checkpoint_identity_changed")
            continue
        reports[arm] = report

    training = read_json(args.planner_training_terminal)
    if (
        training.get("status") != "complete"
        or training.get("selection_gate_passed") is not True
        or training.get("training_complete") is not True
    ):
        reasons.append("planner_likelihood_selection_not_complete")

    metrics: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    if set(reports) == set(ARMS):
        p0 = reports["P0"]["attempt_audit"]
        pc = reports["P-control"]["attempt_audit"]
        ps = reports["P-star"]["attempt_audit"]

        p0_nll = finite(training["p0_initial_target_nll"], "P0 target NLL")
        pstar_candidates = training["selections"]["pstar"]["candidates"]
        pstar_endpoint = [
            item
            for item in pstar_candidates
            if int(item["step"]) == 400
            and item["checkpoint_manifest_sha256"] == EXPECTED_CHECKPOINTS["P-star"]
        ]
        if len(pstar_endpoint) != 1:
            raise ValueError("registered P-star step-400 endpoint is missing")
        pstar_nll = finite(
            pstar_endpoint[0]["target_nll"],
            "P-star step-400 target NLL",
        )
        metrics = {
            "P0": p0,
            "P-control": pc,
            "P-star": ps,
            "Pstar_composition_gain_points": 100.0
            * (
                finite(ps["composition_valid_rate"], "P-star comp")
                - finite(p0["composition_valid_rate"], "P0 comp")
            ),
            "Pstar_minus_Pcontrol_composition_points": 100.0
            * (
                finite(ps["composition_valid_rate"], "P-star comp")
                - finite(pc["composition_valid_rate"], "P-control comp")
            ),
            "Pstar_target_nll_relative_to_P0": pstar_nll / p0_nll,
            "Pstar_unique_formula_ratio_to_P0": (
                finite(ps["unique_formula_rate"], "P-star unique")
                / max(finite(p0["unique_formula_rate"], "P0 unique"), 1e-12)
            ),
            "Pstar_mean_N_delta": (
                finite(ps["mean_N"], "P-star mean N")
                - finite(p0["mean_N"], "P0 mean N")
            ),
        }
        gates = {
            "parse_drop_within_0_5pp": (
                finite(ps["parse_rate"], "P-star parse")
                >= finite(p0["parse_rate"], "P0 parse") - 0.005
            ),
            "completion_drop_within_0_5pp": (
                finite(ps["completion_rate"], "P-star completion")
                >= finite(p0["completion_rate"], "P0 completion") - 0.005
            ),
            "composition_gain_at_least_2pp": (
                metrics["Pstar_composition_gain_points"] >= 2.0 - 1e-12
            ),
            "composition_strictly_above_Pcontrol": (
                finite(ps["composition_valid_rate"], "P-star comp")
                > finite(pc["composition_valid_rate"], "P-control comp")
            ),
            "target_nll_within_plus_1pct": (
                metrics["Pstar_target_nll_relative_to_P0"] <= 1.01
            ),
            "unique_formula_at_least_95pct_of_P0": (
                metrics["Pstar_unique_formula_ratio_to_P0"] >= 0.95
            ),
            "mean_N_delta_within_0_5": abs(metrics["Pstar_mean_N_delta"]) <= 0.5,
            "all_metal_not_inflated": (
                finite(ps["all_metal_rate"], "P-star all metal")
                <= finite(p0["all_metal_rate"], "P0 all metal")
            ),
            "single_element_not_inflated": (
                finite(ps["single_element_rate"], "P-star single")
                <= finite(p0["single_element_rate"], "P0 single")
            ),
        }
        p0_dist = reports["P0"]["distribution_comparison"]
        pstar_dist = reports["P-star"]["distribution_comparison"]
        tvd_results = {}
        for key in TVD_KEYS:
            p0_value = finite(p0_dist[key], f"P0 {key}")
            pstar_value = finite(pstar_dist[key], f"P-star {key}")
            tvd_results[key] = {
                "P0": p0_value,
                "P-star": pstar_value,
                "worsening": pstar_value - p0_value,
                "passed": pstar_value <= p0_value + 0.02 + 1e-12,
            }
        metrics["registered_marginal_tvd"] = tvd_results
        gates["all_registered_marginal_tvd_within_plus_0_02"] = all(
            item["passed"] for item in tvd_results.values()
        )

    failed_gates = sorted(key for key, passed in gates.items() if not passed)
    reasons.extend(f"scientific_gate:{key}" for key in failed_gates)
    scheduler_or_identity_failure = any(
        not reason.startswith("scientific_gate:") for reason in reasons
    )
    gate_passed = bool(gates) and all(gates.values()) and not scheduler_or_identity_failure
    decision = (
        "select_Pstar_for_future_authorized_body_evaluation"
        if gate_passed
        else (
            "execution_failure"
            if scheduler_or_identity_failure
            else "scientific_stop_retain_P0"
        )
    )
    terminal = {
        "schema": "h1a2_v3_planner512_terminal_v1",
        "status": "failed" if scheduler_or_identity_failure else "complete",
        "decision": decision,
        "planner_gate_passed": gate_passed,
        "Pstar_selected": gate_passed,
        "reasons": reasons,
        "gates": gates,
        "metrics": metrics,
        "arm_report_sha256": {
            arm: sha256_file(args.run_root / "arms" / arm / "plan_report.json")
            for arm in reports
        },
        "planner_training_terminal_sha256": sha256_file(
            args.planner_training_terminal
        ),
        "raw_all_attempt_denominator": 512,
        "automatic_downstream": False,
        "automatic_promotion": False,
        "crystal_generation_or_sun_run": False,
    }
    args.output.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(terminal, sort_keys=True))
    return 3 if scheduler_or_identity_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
