#!/usr/bin/env python3
"""Assemble the fixed P0/P1/P2 Plan-only screen without crystal outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from select_jointchem import load_json, select_plan_arm  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--initial-adapter-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = args.run_root / "arms" / "P0" / "plan_report.json"
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    baseline = load_json(baseline_path)
    if baseline.get("execution_manifest_sha256") != args.execution_manifest_sha256:
        raise ValueError("P0 execution manifest identity mismatch")
    if baseline.get("initial_adapter_sha256") != args.initial_adapter_sha256:
        raise ValueError("P0 initial adapter identity mismatch")
    if baseline.get("checkpoint_identity_sha256") != args.initial_adapter_sha256:
        raise ValueError("P0 checkpoint is not the frozen epoch-2 adapter")
    candidate_pairs = []
    inputs: dict[str, Any] = {
        "P0_plan_report": sha256_file(baseline_path),
        "candidates": {},
    }
    unavailable = []
    for arm in ("P1", "P2"):
        arm_root = args.run_root / "arms" / arm
        selection_path = arm_root / "checkpoint_selection.json"
        report_path = arm_root / "plan_report.json"
        if not selection_path.exists():
            unavailable.append({"arm": arm, "reason": "missing_checkpoint_selection"})
            continue
        selection = load_json(selection_path)
        if selection.get("execution_manifest_sha256") != args.execution_manifest_sha256:
            raise ValueError(f"{arm} checkpoint-selection execution identity mismatch")
        if selection.get("initial_adapter_sha256") != args.initial_adapter_sha256:
            raise ValueError(f"{arm} checkpoint-selection adapter identity mismatch")
        if not isinstance(selection.get("selected"), dict):
            unavailable.append({"arm": arm, "reason": str(selection.get("decision"))})
            inputs["candidates"][arm] = {
                "checkpoint_selection": sha256_file(selection_path),
            }
            continue
        if not report_path.exists():
            raise FileNotFoundError(report_path)
        report = load_json(report_path)
        if report.get("execution_manifest_sha256") != args.execution_manifest_sha256:
            raise ValueError(f"{arm} plan-report execution identity mismatch")
        if report.get("initial_adapter_sha256") != args.initial_adapter_sha256:
            raise ValueError(f"{arm} plan-report adapter identity mismatch")
        if report.get("checkpoint_identity_sha256") != selection["selected"].get(
            "checkpoint_manifest_sha256"
        ):
            raise ValueError(f"{arm} sampled checkpoint identity mismatch")
        candidate_pairs.append((report, selection))
        inputs["candidates"][arm] = {
            "checkpoint_selection": sha256_file(selection_path),
            "plan_report": sha256_file(report_path),
        }

    selection = select_plan_arm(baseline, candidate_pairs)
    selection["identity"] = "h1a2c_jointchem_v1"
    selection["execution_manifest_sha256"] = args.execution_manifest_sha256
    selection["initial_adapter_sha256"] = args.initial_adapter_sha256
    selection["unavailable_arms"] = unavailable
    selection["input_sha256"] = inputs
    selection["all_attempt_denominator_per_sampled_arm"] = 512
    selection["automatic_downstream_authorized"] = False
    selection["automatic_crystal_evaluation_authorized"] = False
    selection["sun_chgnet_mlip_mp_api_or_energy_used"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
