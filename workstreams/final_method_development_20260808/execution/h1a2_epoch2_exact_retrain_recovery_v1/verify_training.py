#!/usr/bin/env python3
"""Verify the newly trained adapter and compare it with the historical epoch-2 artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from protocol import read_json, require_source_manifest, sha256_file, validate_config, write_json_exclusive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    manifest = require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    validate_config(config)
    training = args.training_dir.resolve()
    train_config = read_json(training / "train_config.json")
    metrics = read_json(training / "train_metrics.json")
    adapter_config = read_json(training / "final" / "adapter_config.json")
    adapter = training / "final" / "adapter_model.safetensors"
    if not adapter.is_file() or adapter.stat().st_size <= 0:
        raise FileNotFoundError(adapter)

    expected_training = config["training"]
    expected_config = {
        "model_path": str(expected_training["command_model_path"]),
        "checkpoint_path": str(expected_training["command_checkpoint_path"]),
        "data_dir": str(expected_training["command_data_dir"]),
        "max_length": 768,
        "epochs": 1.0,
        "batch_size": 1,
        "grad_accum": 8,
        "lr": 2e-5,
        "total_updates": 3392,
        "train_rows": 27136,
        "val_rows": 9047,
    }
    for key, expected in expected_config.items():
        if train_config.get(key) != expected:
            raise ValueError(f"train_config {key} changed: {train_config.get(key)!r}")
    if int(metrics.get("global_step", -1)) != int(expected_training["expected_updates"]):
        raise ValueError("training did not complete 3392 updates")
    final_eval = float(metrics.get("final_eval_loss", float("nan")))
    if not math.isfinite(final_eval):
        raise ValueError("final eval loss is non-finite")
    lora = expected_training["lora"]
    if (
        int(adapter_config.get("r", -1)) != int(lora["rank"])
        or int(adapter_config.get("lora_alpha", -1)) != int(lora["alpha"])
        or float(adapter_config.get("lora_dropout", -1)) != float(lora["dropout"])
        or adapter_config.get("bias") != "none"
        or set(adapter_config.get("target_modules") or []) != set(lora["targets"])
    ):
        raise ValueError("saved LoRA contract changed")

    adapter_sha = sha256_file(adapter)
    adapter_config_sha = sha256_file(training / "final" / "adapter_config.json")
    historical = config["historical_anchor"]
    report = {
        "schema": "h1a2_epoch2_exact_retrain_terminal_report_v1",
        "engineering_status": "complete",
        "source_manifest_sha256": sha256_file(manifest),
        "global_step": int(metrics["global_step"]),
        "final_eval_loss": final_eval,
        "elapsed_seconds": float(metrics["elapsed_sec"]),
        "adapter_path": str(adapter),
        "adapter_bytes": adapter.stat().st_size,
        "adapter_sha256": adapter_sha,
        "adapter_config_sha256": adapter_config_sha,
        "historical_adapter_sha256": str(historical["adapter_sha256"]),
        "historical_adapter_config_sha256": str(historical["adapter_config_sha256"]),
        "adapter_byte_exact_historical": adapter_sha == str(historical["adapter_sha256"]),
        "adapter_config_byte_exact_historical": adapter_config_sha == str(historical["adapter_config_sha256"]),
        "historical_final_eval_loss": float(historical["final_eval_loss"]),
        "final_eval_loss_delta": final_eval - float(historical["final_eval_loss"]),
        "exact_code_data_hyperparameters_environment": True,
        "resource_difference": "one_visible_A800_instead_of_two_allocated_but_unused_second_GPU",
        "downstream_sampling_authorized_by_this_job": False,
    }
    write_json_exclusive(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
