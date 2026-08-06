#!/usr/bin/env python3
"""Assemble the immutable G1 report and scientific continue/stop decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from g1_protocol import G1_ATTEMPTS, evaluate_g1_gate, sha256_file, write_json


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--initial-adapter-sha256", required=True)
    parser.add_argument("--max-jsd-bits", type=float, default=0.15)
    parser.add_argument("--max-shortcut-rate-delta", type=float, default=0.05)
    args = parser.parse_args()

    ledger_report_path = args.run_root / "data" / "ledger_report.json"
    ledger = load_json(ledger_report_path)
    if ledger.get("status") != "complete":
        raise ValueError("G1 ledger is not complete")
    if int(ledger.get("attempts") or 0) != G1_ATTEMPTS:
        raise ValueError("G1 ledger denominator changed")
    ledger_sha = str(ledger["seed_ledger_sha256"])

    training: dict[str, dict] = {}
    training_evidence: dict[str, dict] = {}
    for arm in ("PG", "PG-shuffle"):
        path = args.run_root / "training" / arm / "training_report.json"
        report = load_json(path)
        if report.get("status") != "complete":
            raise ValueError(f"{arm} training is not complete")
        if (
            int(report.get("optimizer_updates") or 0) != 400
            or int(report.get("train_microbatches") or 0) != 3200
        ):
            raise ValueError(f"{arm} training schedule changed")
        if report.get("all_numeric_finite_positive") is not True:
            raise ValueError(f"{arm} training numerics failed")
        if report.get("checkpoint_selection") is not False:
            raise ValueError(f"{arm} unexpectedly selected a checkpoint")
        training[arm] = report
        training_evidence[arm] = {
            "training_report": str(path),
            "training_report_sha256": sha256_file(path),
            "checkpoint_dir": report["checkpoint_dir"],
            "checkpoint_manifest_sha256": report[
                "checkpoint_manifest_sha256"
            ],
            "optimizer_updates": report["optimizer_updates"],
            "train_microbatches": report["train_microbatches"],
        }

    reports: dict[str, dict] = {}
    sampling_evidence: dict[str, dict] = {}
    for arm in ("P0", "PG", "PG-shuffle"):
        path = args.run_root / "sampling" / arm / "planner_report.json"
        report = load_json(path)
        if report.get("status") != "complete":
            raise ValueError(f"{arm} sampling is not complete")
        if int(report.get("attempts") or 0) != G1_ATTEMPTS:
            raise ValueError(f"{arm} sampling denominator changed")
        if report.get("seed_ledger_sha256") != ledger_sha:
            raise ValueError(f"{arm} seed ledger identity mismatch")
        for key in ("retry", "replacement", "repair", "filter", "rerank"):
            if report.get(key) is not False:
                raise ValueError(f"{arm} violates {key}=false")
        expected_identity = (
            str(args.initial_adapter_sha256)
            if arm == "P0"
            else str(training[arm]["checkpoint_manifest_sha256"])
        )
        if report.get("checkpoint_identity_sha256") != expected_identity:
            raise ValueError(f"{arm} checkpoint identity mismatch")
        reports[arm] = report
        sampling_evidence[arm] = {
            "planner_report": str(path),
            "planner_report_sha256": sha256_file(path),
            "attempts_sha256": report["attempts_sha256"],
            "checkpoint_identity_sha256": report[
                "checkpoint_identity_sha256"
            ],
        }

    gate = evaluate_g1_gate(
        reports,
        max_jsd_bits=float(args.max_jsd_bits),
        max_shortcut_rate_delta=float(args.max_shortcut_rate_delta),
    )
    decision = {
        "schema": "plangraph-dlm-g1-decision@1",
        "gate": "G1",
        "decision": "continue_to_G2" if gate["passed"] else "scientific_stop",
        "g1_passed": bool(gate["passed"]),
        "automatic_submission_permitted_by_user_within_chain": True,
        "automatic_G4": False,
        "scientific_gate_failure_is_not_repairable": not bool(gate["passed"]),
    }
    terminal = {
        "schema": "plangraph-dlm-g1-terminal@1",
        "status": "complete",
        "execution_manifest_sha256": str(args.execution_manifest_sha256),
        "initial_adapter_sha256": str(args.initial_adapter_sha256),
        "seed_ledger_sha256": ledger_sha,
        "ledger_report_sha256": sha256_file(ledger_report_path),
        "training": training_evidence,
        "sampling": sampling_evidence,
        "metrics": {
            arm: {
                "parse_rate": report["parse_rate"],
                "plan_completion_rate": report["plan_completion_rate"],
                "composition_valid_rate": report["composition_valid_rate"],
                "unique_formula_count": report["unique_formula_count"],
                "unique_formula_rate_all_attempt": report[
                    "unique_formula_rate_all_attempt"
                ],
                "single_element_rate": report["single_element_rate"],
                "all_metal_rate": report["all_metal_rate"],
                "strict_schema_valid_count": report[
                    "strict_schema_valid_count"
                ],
                "parse_failures": report["parse_failures"],
                "chemistry_failures": report["chemistry_failures"],
            }
            for arm, report in reports.items()
        },
        "gate": gate,
        "decision": decision,
        "denominator": "all_attempt",
        "attempts_per_arm": G1_ATTEMPTS,
        "same_ordinal_seeds": True,
        "sample_id_in_prompt": False,
        "retry": False,
        "replacement": False,
        "repair": False,
        "filter": False,
        "rerank": False,
        "crystal_generation": False,
        "sun_evaluation": False,
        "automatic_G4": False,
    }
    write_json(args.run_root / "decision.json", decision)
    write_json(args.run_root / "terminal_report.json", terminal)
    print(json.dumps(terminal, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

