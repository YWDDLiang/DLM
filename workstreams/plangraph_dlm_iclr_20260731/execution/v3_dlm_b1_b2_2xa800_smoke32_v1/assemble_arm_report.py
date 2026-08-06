#!/usr/bin/env python3
"""Assemble and fail-close one B1/B2 two-A800 engineering smoke report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ARMS = {
    "B1": {"planned_corruption_policy": "d1"},
    "B2": {"planned_corruption_policy": "d2"},
}


class ArmGateError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArmGateError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ArmGateError(f"expected JSONL objects: {path}")
                rows.append(payload)
    return rows


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ArmGateError(
            f"{label} changed: observed={observed!r} expected={expected!r}"
        )


def require_positive_finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ArmGateError(f"{label} is not numeric: {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ArmGateError(f"{label} is not finite and positive: {value!r}")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gpu_csv(path: Path) -> dict[str, Any]:
    indices: set[str] = set()
    names: set[str] = set()
    peak_memory_mib: dict[str, float] = {}
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 7:
                continue
            rows += 1
            index = row[1].strip()
            name = row[2].strip()
            indices.add(index)
            names.add(name)
            memory = float(row[3].strip())
            peak_memory_mib[index] = max(peak_memory_mib.get(index, 0.0), memory)
    if rows == 0:
        raise ArmGateError("GPU monitor emitted no rows")
    if len(indices) != 2:
        raise ArmGateError(f"expected two monitored GPUs, observed {sorted(indices)}")
    if any("A800" not in name for name in names):
        raise ArmGateError(f"non-A800 GPU observed: {sorted(names)}")
    if any(value <= 0.0 for value in peak_memory_mib.values()):
        raise ArmGateError(f"zero GPU memory peak: {peak_memory_mib}")
    return {
        "rows": rows,
        "indices": sorted(indices),
        "names": sorted(names),
        "peak_memory_mib": peak_memory_mib,
    }


def assemble(arm: str, output_dir: Path, gpu_csv: Path) -> dict[str, Any]:
    if arm not in ARMS:
        raise ArmGateError(f"unknown arm: {arm}")
    run_config = read_json(output_dir / "run_config.json")
    validation = read_json(output_dir / "validation_sampler_report.json")
    runtime = read_json(output_dir / "distributed_runtime_report.json")
    events = read_jsonl(output_dir / "training_log.jsonl")
    gpu = parse_gpu_csv(gpu_csv)

    expected_config = {
        "representation": "dynamic_v1",
        "planned_corruption_policy": ARMS[arm]["planned_corruption_policy"],
        "iid_fraction": 2.0,
        "planned_fraction": 1.0,
        "corruption_seed": 20260731,
        "data_seed": 20260515,
        "max_length": 382,
        "answer_token_count": 87,
        "limit_train": 32,
        "limit_val": 32,
        "epochs": 1,
        "max_train_steps": 2,
        "batch_size": 1,
        "grad_accum": 8,
        "lr": 5e-05,
        "lr_scheduler": "constant",
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "logging_steps": 1,
        "eval_before_train": True,
        "eval_steps": 2,
        "eval_max_batches": 16,
        "dataloader_num_workers": 0,
        "engineering_only": True,
        "skip_final_save": True,
        "distributed": True,
        "world_size": 2
    }
    for key, expected in expected_config.items():
        require_equal(run_config.get(key), expected, f"run_config.{key}")

    for key, expected in {
        "gate_passed": True,
        "dataset_length": 32,
        "world_size": 2,
        "total_assigned": 32,
        "unique_assigned": 32,
        "duplicate_count": 0,
        "missing_count": 0,
        "rank_mapping_exact": True,
        "sampler": "DistributedNoPaddingSampler",
    }.items():
        require_equal(validation.get(key), expected, f"validation.{key}")
    require_equal(
        [item.get("count") for item in validation.get("per_rank", [])],
        [16, 16],
        "validation per-rank counts",
    )

    for key, expected in {
        "runtime_gate_passed": True,
        "distributed": True,
        "world_size": 2,
        "batch_size_per_rank": 1,
        "gradient_accumulation": 8,
        "global_effective_batch": 16,
        "global_train_sequences": 32,
        "optimizer_updates": 2,
        "rank0_only_checkpoint_and_report_publication": True,
        "final_checkpoint_saved": False,
        "engineering_only": True,
        "eligible_for_checkpoint_selection": False,
        "eligible_for_later_initialization": False,
        "automatic_downstream": False,
        "scientific_training_authorized": False,
    }.items():
        require_equal(runtime.get(key), expected, f"runtime.{key}")
    rank_runtime = runtime.get("rank_runtime") or []
    require_equal(
        [item.get("rank") for item in rank_runtime],
        [0, 1],
        "runtime ranks",
    )
    for item in rank_runtime:
        rank = item["rank"]
        require_equal(item.get("train_microbatches"), 16, f"rank {rank} microbatches")
        require_equal(item.get("optimizer_updates"), 2, f"rank {rank} updates")
        require_equal(item.get("task_loss_count"), 16, f"rank {rank} task losses")
        require_equal(item.get("gradient_norm_count"), 2, f"rank {rank} gradients")
        require_equal(item.get("evaluation_loss_count"), 2, f"rank {rank} evals")
        for key in (
            "task_loss_min",
            "task_loss_max",
            "gradient_norm_min",
            "gradient_norm_max",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ):
            require_positive_finite(item.get(key), f"rank {rank} {key}")
        for idx, value in enumerate(item.get("evaluation_losses") or []):
            require_positive_finite(value, f"rank {rank} eval {idx}")

    train_events = [item for item in events if item.get("event") == "train"]
    eval_events = [item for item in events if item.get("event") == "eval"]
    require_equal(
        [item.get("step") for item in train_events],
        [1, 2],
        "training log steps",
    )
    require_equal(
        [item.get("step") for item in eval_events],
        [0, 2],
        "validation log steps",
    )
    for idx, item in enumerate(train_events):
        require_positive_finite(item.get("task_loss"), f"train event {idx} task loss")
        require_positive_finite(
            item.get("pre_clip_gradient_norm"),
            f"train event {idx} gradient norm",
        )
    for idx, item in enumerate(eval_events):
        require_positive_finite(item.get("val_loss"), f"eval event {idx}")

    if (output_dir / "final").exists() or (output_dir / "checkpoints").exists():
        raise ArmGateError("engineering smoke serialized an ineligible checkpoint")

    return {
        "schema": "h1a2_dlm_b1_b2_2xa800_arm_report_v1",
        "status": "complete",
        "engineering_gate_passed": True,
        "arm": arm,
        "planned_corruption_policy": ARMS[arm]["planned_corruption_policy"],
        "world_size": 2,
        "a800_count": 2,
        "global_training_sequences": 32,
        "optimizer_updates": 2,
        "global_effective_batch": 16,
        "validation_rows": 32,
        "validation_duplicate_count": 0,
        "validation_missing_count": 0,
        "initial_validation_loss": eval_events[0]["val_loss"],
        "final_validation_loss": eval_events[-1]["val_loss"],
        "minimum_rank_task_loss": min(
            item["task_loss_min"] for item in rank_runtime
        ),
        "minimum_rank_gradient_norm": min(
            item["gradient_norm_min"] for item in rank_runtime
        ),
        "maximum_cuda_peak_reserved_bytes": max(
            item["cuda_peak_reserved_bytes"] for item in rank_runtime
        ),
        "gpu_monitor": gpu,
        "run_config_sha256": sha256_file(output_dir / "run_config.json"),
        "validation_sampler_report_sha256": sha256_file(
            output_dir / "validation_sampler_report.json"
        ),
        "distributed_runtime_report_sha256": sha256_file(
            output_dir / "distributed_runtime_report.json"
        ),
        "training_log_sha256": sha256_file(output_dir / "training_log.jsonl"),
        "scientific_result": False,
        "eligible_for_checkpoint_selection": False,
        "eligible_for_later_initialization": False,
        "automatic_downstream": False,
        "scientific_training_authorized": False,
        "crystal_generation": False,
        "sun_evaluation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.arm, args.output_dir, args.gpu_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
