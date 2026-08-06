#!/usr/bin/env python3
"""Select registered Planner likelihood checkpoints after both full epochs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object: {path}")
    return payload


def finite(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} is not finite and positive: {value}")
    return parsed


def select_arm(run_root: Path, arm: str, p0_target_nll: float) -> dict[str, Any]:
    arm_root = run_root / "arms" / arm
    report_path = arm_root / "training_report.json"
    report = read_json(report_path)
    if (
        report.get("status") != "complete"
        or report.get("arm") != arm
        or report.get("microbatches") != 3200
        or report.get("optimizer_updates") != 400
    ):
        raise ValueError(f"{arm} training report identity failed")
    checkpoints = report.get("checkpoints") or []
    if [item.get("step") for item in checkpoints] != list(range(50, 401, 50)):
        raise ValueError(f"{arm} checkpoint cadence changed")

    candidates = []
    for item in checkpoints:
        step = int(item["step"])
        checkpoint_root = arm_root / str(item["path"])
        manifest_path = checkpoint_root / "checkpoint_manifest.json"
        if sha256_file(manifest_path) != item["manifest_sha256"]:
            raise ValueError(f"{arm} checkpoint {step} manifest SHA mismatch")
        metrics = item["metrics"]
        target_nll = finite(metrics["target_nll"], f"{arm} step {step} target_nll")
        field_loss = finite(metrics["field_loss"], f"{arm} step {step} field_loss")
        candidates.append(
            {
                "step": step,
                "checkpoint_path": str(checkpoint_root / "adapter"),
                "checkpoint_manifest_sha256": item["manifest_sha256"],
                "target_nll": target_nll,
                "field_loss": field_loss,
                "nll_noninferior": target_nll <= 1.01 * p0_target_nll,
            }
        )
    eligible = [item for item in candidates if item["nll_noninferior"]]
    selected = min(eligible, key=lambda item: (item["field_loss"], item["step"])) if eligible else None
    result = {
        "schema": "h1a2_v3_planner_checkpoint_selection_v1",
        "status": "complete",
        "arm": arm,
        "p0_initial_target_nll": p0_target_nll,
        "candidates": candidates,
        "eligible_count": len(eligible),
        "selected": selected,
        "selection_uses_generation_sun_or_energy": False,
        "automatic_promotion": False,
    }
    output = arm_root / "checkpoint_selection.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["report_path"] = str(output)
    result["report_sha256"] = sha256_file(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        arm: read_json(args.run_root / "arms" / arm / "training_report.json")
        for arm in ("pcontrol", "pstar")
    }
    initial_values = [
        finite(reports[arm]["initial_validation"]["target_nll"], f"{arm} initial target_nll")
        for arm in ("pcontrol", "pstar")
    ]
    if abs(initial_values[0] - initial_values[1]) > 1e-6:
        raise ValueError(f"Planner initial P0 target NLL mismatch: {initial_values}")
    selections = {
        arm: select_arm(args.run_root, arm, initial_values[0])
        for arm in ("pcontrol", "pstar")
    }
    selection_gate = all(item["selected"] is not None for item in selections.values())
    terminal = {
        "schema": "h1a2_v3_planner_scientific400_terminal_v1",
        "status": "complete",
        "training_complete": True,
        "selection_gate_passed": selection_gate,
        "p0_initial_target_nll": initial_values[0],
        "selections": selections,
        "planner_512_authorized": False,
        "automatic_downstream": False,
        "automatic_promotion": False,
    }
    args.output.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
