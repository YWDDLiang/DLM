#!/usr/bin/env python3
"""Finalize a complete BTRD endpoint after a post-science wrapper assertion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_run.resolve()
    data = args.data_dir.resolve()
    policy = args.policy_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not (source / "_FAILED").is_file() or (source / "_SUCCESS").exists():
        raise ValueError("source must preserve the post-science failed wrapper state")

    train_root = source / "train"
    checkpoint = train_root / "checkpoints/step-512"
    config = json.loads((train_root / "run_config.json").read_text())
    partition = json.loads((train_root / "parameter_partition.json").read_text())
    events = [
        json.loads(line)
        for line in (train_root / "training_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    train = [row for row in events if row.get("event") == "train"]
    if not train or int(train[-1]["step"]) != 510:
        raise ValueError("expected the final periodic logging event at step510")
    if not any(
        row.get("event") == "eval" and int(row["step"]) == 512 for row in events
    ):
        raise ValueError("step512 endpoint evaluation is missing")
    if sorted(path.name for path in (train_root / "checkpoints").iterdir()) != [
        "step-512"
    ]:
        raise ValueError("eligible checkpoint set changed")
    for name in (
        "adapter_model.safetensors",
        "periodic_relation_adapter.pt",
        "periodic_relation_config.json",
    ):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if config["periodic_relation_only"] is not True or config["max_train_steps"] != 512:
        raise ValueError("BTRD training contract changed")
    if partition["trainable_parameters"] <= 0 or partition["frozen_parameters"] <= 0:
        raise ValueError("parameter partition is invalid")
    keys = (
        "loss",
        "task_loss",
        "grad_norm",
        "lr",
        "periodic_geometry_metric_loss",
        "periodic_geometry_pair_rdf_loss",
        "periodic_geometry_overlap_loss",
        "periodic_geometry_coordination_loss",
        "basin_transport_loss",
    )
    for row in train:
        for key in keys:
            if row.get(key) is None or not math.isfinite(float(row[key])):
                raise ValueError(f"nonfinite {key} at step {row.get('step')}")
    policy_sha = sha256_file(policy / "adapter_model.safetensors")
    output_policy_sha = sha256_file(checkpoint / "adapter_model.safetensors")
    if output_policy_sha != policy_sha:
        raise ValueError("frozen Compact-V2 LoRA changed")
    relation_input_sha = sha256_file(policy / "periodic_relation_adapter.pt")
    relation_output_sha = sha256_file(checkpoint / "periodic_relation_adapter.pt")
    if relation_output_sha == relation_input_sha:
        raise ValueError("periodic residual did not update")
    manifest = json.loads((data / "manifest.json").read_text())
    payload = {
        "schema": "btrd_residual_training_recovery_v1",
        "status": "complete",
        "source_run": str(source),
        "source_failed_marker_preserved": True,
        "root_cause": (
            "post-science wrapper required a step512 train event although "
            "logging_steps=10; endpoint checkpoint and eval512 are complete"
        ),
        "gpu_training_rerun": False,
        "optimizer_updates": 512,
        "last_periodic_train_log_step": 510,
        "endpoint_eval_step": 512,
        "checkpoint": str(checkpoint),
        "checkpoint_relation_sha256": relation_output_sha,
        "frozen_policy_sha256": output_policy_sha,
        "parameter_partition": partition,
        "train_rows": manifest["train_rows"],
        "effective_tau200_rows": manifest["effective_tau200_rows"],
        "anchor_or_fallback_rows": manifest["anchor_or_fallback_rows"],
        "all_logged_values_finite": True,
    }
    output.mkdir(parents=True)
    report = output / "BTRD_TRAIN_FINAL.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output / "OUTPUTS.sha256").write_text(
        f"{sha256_file(report)}  BTRD_TRAIN_FINAL.json\n"
    )
    (output / "_SUCCESS").touch()
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
