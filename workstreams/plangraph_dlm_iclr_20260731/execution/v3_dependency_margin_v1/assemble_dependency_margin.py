#!/usr/bin/env python3
"""Assemble the registered B1/B2 direct-dependency margin decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from evaluate_dependency_margin import bootstrap_mean_ci


EXPECTED_CHECKPOINTS = {
    "B1": "ace1a0d1f935e32881773228de5fcd5e91a76ca6e261ea132d609b84293bea8c",
    "B2": "e3451c94822ca83d1fa1f91680e24880ce9ced022379865b76705554f31f8cb8",
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


def paired_difference(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
) -> tuple[list[float], list[dict[str, Any]]]:
    left_map = {int(item["panel_ordinal"]): item for item in left}
    right_map = {int(item["panel_ordinal"]): item for item in right}
    if sorted(left_map) != list(range(100)) or sorted(right_map) != list(range(100)):
        raise ValueError("paired comparison requires panel ordinals 0..99")
    values: list[float] = []
    records: list[dict[str, Any]] = []
    for ordinal in range(100):
        b1 = left_map[ordinal]
        b2 = right_map[ordinal]
        if int(b1["group_pairs"]) != int(b2["group_pairs"]):
            raise ValueError(f"group-pair count changed at panel ordinal {ordinal}")
        delta = finite(b2["dependency_margin"], "B2 row margin") - finite(
            b1["dependency_margin"], "B1 row margin"
        )
        values.append(delta)
        records.append(
            {
                "panel_ordinal": ordinal,
                "B1_margin": float(b1["dependency_margin"]),
                "B2_margin": float(b2["dependency_margin"]),
                "B2_minus_B1": delta,
            }
        )
    return values, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dlm-training-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for arm in ("B1", "B2"):
        path = args.run_root / "arms" / arm / "dependency_report.json"
        if not path.exists():
            reasons.append(f"{arm}:missing_dependency_report")
            continue
        report = read_json(path)
        if (
            report.get("status") != "complete"
            or report.get("arm") != arm
            or report.get("result", {}).get("panel_rows") != 100
        ):
            reasons.append(f"{arm}:report_identity_failed")
            continue
        if report.get("checkpoint_adapter_sha256") != EXPECTED_CHECKPOINTS[arm]:
            reasons.append(f"{arm}:checkpoint_identity_changed")
            continue
        reports[arm] = report

    training = read_json(args.dlm_training_terminal)
    if (
        training.get("status") != "complete"
        or training.get("training_complete") is not True
        or training.get("B2_likelihood_gate_passed") is not True
    ):
        reasons.append("DLM_likelihood_gate_not_complete")

    gates: dict[str, bool] = {}
    metrics: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    if set(reports) == {"B1", "B2"}:
        b1_result = reports["B1"]["result"]
        b2_result = reports["B2"]["result"]
        b1_margin = finite(b1_result["arm_margin"]["mean"], "B1 arm margin")
        b2_margin = finite(b2_result["arm_margin"]["mean"], "B2 arm margin")
        deltas, paired_rows = paired_difference(
            b1_result["row_records"],
            b2_result["row_records"],
        )
        metrics = {
            "B1_arm_margin": b1_result["arm_margin"],
            "B2_arm_margin": b2_result["arm_margin"],
            "B2_minus_B1": bootstrap_mean_ci(deltas),
        }
        gates = {
            "B2_margin_strictly_positive": b2_margin > 0.0,
            "B2_margin_strictly_above_B1": b2_margin > b1_margin,
            "B2_fixed_panel_nll_within_plus_1pct": bool(
                training.get("B2_likelihood_gate_passed")
            ),
        }

    failed_gates = sorted(key for key, passed in gates.items() if not passed)
    reasons.extend(f"scientific_gate:{key}" for key in failed_gates)
    execution_failure = any(
        not reason.startswith("scientific_gate:") for reason in reasons
    )
    dependency_gate_passed = (
        bool(gates) and all(gates.values()) and not execution_failure
    )
    terminal = {
        "schema": "h1a2_v3_paired_direct_dependency_margin_terminal_v1",
        "status": "failed" if execution_failure else "complete",
        "decision": (
            "dependency_likelihood_gate_passed_completion_still_pending"
            if dependency_gate_passed
            else (
                "execution_failure"
                if execution_failure
                else "scientific_stop_retain_B0"
            )
        ),
        "dependency_gate_passed": dependency_gate_passed,
        "Bstar_selected": False,
        "conditional_body_completion_pending": dependency_gate_passed,
        "reasons": reasons,
        "gates": gates,
        "metrics": metrics,
        "paired_row_differences": paired_rows,
        "arm_report_sha256": {
            arm: sha256_file(
                args.run_root / "arms" / arm / "dependency_report.json"
            )
            for arm in reports
        },
        "dlm_training_terminal_sha256": sha256_file(args.dlm_training_terminal),
        "generation_sun_energy_or_hull_used": False,
        "shuffle_training_arm_used": False,
        "automatic_downstream": False,
        "automatic_promotion": False,
    }
    args.output.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(terminal, sort_keys=True))
    return 3 if execution_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
