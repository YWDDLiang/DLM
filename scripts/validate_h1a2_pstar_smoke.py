#!/usr/bin/env python3
"""Fail-closed validator for one H1-A2 P-control/P* engineering arm."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MAX_REGISTERED_GPU_BYTES = 64 * 1024**3


def finite_positive(value: Any, *, label: str) -> float:
    observed = float(value)
    if not math.isfinite(observed) or observed <= 0.0:
        raise ValueError(f"{label} must be finite and positive, observed {value!r}")
    return observed


def load_events(path: Path) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not events:
        raise ValueError("smoke events are empty")
    return events


def validate(
    output_dir: Path,
    *,
    arm: str,
    gpu_csv: Path,
) -> dict[str, Any]:
    if arm not in {"pcontrol", "pstar"}:
        raise ValueError(f"unsupported smoke arm {arm!r}")
    report_path = output_dir / "training_report.json"
    event_path = output_dir / "events.jsonl"
    if not report_path.is_file() or not event_path.is_file():
        raise FileNotFoundError("training report or events are missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete"
        or report.get("arm") != arm
        or report.get("engineering_smoke") is not True
        or int(report.get("microbatches", -1)) != 32
        or int(report.get("optimizer_updates", -1)) != 4
        or int(report.get("batch_size", -1)) != 1
        or int(report.get("gradient_accumulation", -1)) != 8
        or int(report.get("max_length", -1)) != 768
        or int(report.get("seed", -1)) != 17
        or bool(report.get("shuffle"))
        or bool(report.get("generation_or_sun_selection"))
    ):
        raise ValueError("registered Planner smoke contract changed")
    checkpoints = report.get("checkpoints") or []
    if len(checkpoints) != 1 or int(checkpoints[0].get("step", -1)) != 4:
        raise ValueError("engineering smoke must publish only checkpoint step 4")

    initial = report.get("initial_validation") or {}
    final = (checkpoints[0].get("metrics") or {})
    for label, metrics in (("initial", initial), ("final", final)):
        finite_positive(metrics.get("target_nll"), label=f"{label}.target_nll")
        finite_positive(metrics.get("field_loss"), label=f"{label}.field_loss")
        finite_positive(metrics.get("total_loss"), label=f"{label}.total_loss")
    if arm == "pstar":
        finite_positive(initial.get("lookahead_loss"), label="initial.lookahead_loss")
        finite_positive(final.get("lookahead_loss"), label="final.lookahead_loss")
        if report.get("auxiliary_heads_discarded_for_inference") is not True:
            raise ValueError("P* auxiliary heads were not marked inference-discarded")

    events = load_events(event_path)
    training_events = [event for event in events if event.get("event") == "training"]
    if not training_events:
        raise ValueError("smoke has no training event")
    for index, event in enumerate(training_events):
        finite_positive(event.get("train_loss_recent"), label=f"train[{index}].loss")
        finite_positive(event.get("grad_norm"), label=f"train[{index}].grad_norm")

    cuda = report.get("cuda") or {}
    device_name = str(cuda.get("device_name") or "")
    if "A800" not in device_name:
        raise ValueError(f"smoke did not run on A800: {device_name!r}")
    allocated = int(
        finite_positive(
            cuda.get("peak_memory_allocated_bytes"),
            label="cuda.peak_memory_allocated_bytes",
        )
    )
    reserved = int(
        finite_positive(
            cuda.get("peak_memory_reserved_bytes"),
            label="cuda.peak_memory_reserved_bytes",
        )
    )
    if reserved > MAX_REGISTERED_GPU_BYTES:
        raise ValueError(
            f"reserved CUDA memory {reserved} exceeds 64 GiB smoke envelope"
        )
    if not gpu_csv.is_file() or not gpu_csv.read_text(encoding="utf-8").strip():
        raise ValueError("GPU utilization trace is missing or empty")

    result = {
        "schema": "h1a2_pstar_engineering_smoke_arm_v1",
        "status": "complete",
        "engineering_gate_passed": True,
        "arm": arm,
        "microbatches": 32,
        "optimizer_updates": 4,
        "checkpoint_step": 4,
        "initial_target_nll": float(initial["target_nll"]),
        "final_target_nll": float(final["target_nll"]),
        "initial_field_loss": float(initial["field_loss"]),
        "final_field_loss": float(final["field_loss"]),
        "cuda_device_name": device_name,
        "cuda_peak_memory_allocated_bytes": allocated,
        "cuda_peak_memory_reserved_bytes": reserved,
        "elapsed_sec": finite_positive(
            report.get("elapsed_sec"),
            label="elapsed_sec",
        ),
        "automatic_downstream": False,
        "scientific_training_authorized": False,
    }
    target = output_dir / "engineering_report.json"
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("pcontrol", "pstar"), required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            validate(args.output_dir, arm=args.arm, gpu_csv=args.gpu_csv),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
