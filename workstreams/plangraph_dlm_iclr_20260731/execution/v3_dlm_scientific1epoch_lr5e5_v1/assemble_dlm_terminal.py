#!/usr/bin/env python3
"""Audit B1/B2 one-epoch endpoints without automatic promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


VALIDATION_STEPS = [0, 212, 424, 636, 848, 1060, 1272, 1484, 1696]
POLICIES = {"B1": "d1", "B2": "d2"}


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


def read_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                value = json.loads(raw_line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def finite(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} is not finite and positive: {value}")
    return parsed


def inventory_final(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    ]


def audit_arm(run_root: Path, arm: str) -> dict[str, Any]:
    output = run_root / "arms" / arm / "output"
    config = read_json(output / "run_config.json")
    runtime = read_json(output / "distributed_runtime_report.json")
    validation = read_json(output / "validation_sampler_report.json")
    events = read_events(output / "training_log.jsonl")
    expected = {
        "planned_corruption_policy": POLICIES[arm],
        "epochs": 1,
        "max_train_steps": 1696,
        "batch_size": 1,
        "grad_accum": 8,
        "lr": 5e-05,
        "lr_scheduler": "cosine",
        "warmup_steps": 100,
        "min_lr_ratio": 0.2,
        "eval_before_train": True,
        "eval_steps": 212,
        "eval_max_batches": 50,
        "distributed": True,
        "world_size": 2,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{arm} run_config.{key} changed: {config.get(key)} != {value}")
    for key, value in {
        "status": "complete",
        "runtime_gate_passed": True,
        "world_size": 2,
        "global_effective_batch": 16,
        "global_train_sequences": 27136,
        "optimizer_updates": 1696,
        "final_checkpoint_saved": True,
    }.items():
        if runtime.get(key) != value:
            raise ValueError(f"{arm} runtime.{key} changed: {runtime.get(key)} != {value}")
    if (
        validation.get("gate_passed") is not True
        or validation.get("dataset_length") != 9047
        or validation.get("duplicate_count") != 0
        or validation.get("missing_count") != 0
        or [item.get("count") for item in validation.get("per_rank", [])] != [4524, 4523]
    ):
        raise ValueError(f"{arm} validation sampler audit failed")
    evaluations = [item for item in events if item.get("event") == "eval"]
    if [int(item["step"]) for item in evaluations] != VALIDATION_STEPS:
        raise ValueError(f"{arm} validation cadence changed")
    losses = [finite(item["val_loss"], f"{arm} step {item['step']} val_loss") for item in evaluations]
    final_dir = output / "final"
    if not final_dir.is_dir():
        raise ValueError(f"{arm} final checkpoint missing")
    result = {
        "schema": "h1a2_v3_dlm_one_epoch_arm_terminal_v1",
        "status": "complete",
        "arm": arm,
        "policy": POLICIES[arm],
        "optimizer_updates": 1696,
        "validation_steps": VALIDATION_STEPS,
        "validation_losses": losses,
        "initial_validation_loss": losses[0],
        "terminal_validation_loss": losses[-1],
        "terminal_relative_to_initial": losses[-1] / losses[0],
        "nll_noninferior_to_initial": losses[-1] <= 1.01 * losses[0],
        "final_checkpoint": str(final_dir),
        "final_inventory": inventory_final(final_dir),
        "automatic_promotion": False,
    }
    report_path = run_root / "arms" / arm / "terminal_training_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["report_sha256"] = sha256_file(report_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = {arm: audit_arm(args.run_root, arm) for arm in ("B1", "B2")}
    b0_initial_match = abs(
        arms["B1"]["initial_validation_loss"] - arms["B2"]["initial_validation_loss"]
    ) <= 1e-6
    likelihood_gate = b0_initial_match and arms["B2"]["nll_noninferior_to_initial"]
    terminal = {
        "schema": "h1a2_v3_dlm_one_epoch_lr5e5_terminal_v1",
        "status": "complete",
        "training_complete": True,
        "arms": arms,
        "B0_initial_panel_match": b0_initial_match,
        "B2_likelihood_gate_passed": likelihood_gate,
        "direct_dependency_margin_pending": True,
        "Bstar_selected": False,
        "scientific_stop_if_likelihood_gate_failed": not likelihood_gate,
        "automatic_downstream": False,
        "automatic_promotion": False,
    }
    args.output.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
