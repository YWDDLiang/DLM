#!/usr/bin/env python3
"""Recover completed A/B science from a post-check wrapper assertion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(method: str, run: Path) -> dict:
    if not (run / "_FAILED").is_file() or (run / "_SUCCESS").exists():
        raise ValueError(f"{method} is not the preserved post-check failure")
    checkpoint = run / "train/checkpoints/step-1696"
    checkpoints = sorted(path.name for path in (run / "train/checkpoints").glob("step-*"))
    if checkpoints != ["step-1696"]:
        raise ValueError(f"{method} checkpoint selection changed: {checkpoints}")
    required = (
        checkpoint / "adapter_model.safetensors",
        checkpoint / "periodic_relation_adapter.pt",
        checkpoint / "periodic_relation_config.json",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"{method} endpoint files incomplete")
    config = json.loads((run / "train/run_config.json").read_text())
    relation = json.loads((checkpoint / "periodic_relation_config.json").read_text())
    if not (
        config["max_train_steps"] == 1696
        and config["world_size"] == 2
        and config["batch_size"] == 1
        and config["grad_accum"] == 8
        and config["periodic_image_radius"] == 2
        and config["periodic_relation_uncertainty_gate"] is (method == "B")
        and relation["step0_checked"] is True
        and relation["step0_max_logit_delta"] == 0.0
        and relation["config"]["uncertainty_gate"] is (method == "B")
    ):
        raise ValueError(f"{method} frozen config mismatch")
    events = [
        json.loads(line)
        for line in (run / "train/training_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    train = [row for row in events if row.get("event") == "train"]
    evaluation = [row for row in events if row.get("event") == "eval"]
    if max(int(row["step"]) for row in train) != 1690:
        raise ValueError(f"{method} expected last logged multiple-of-10 step1690")
    if [int(row["step"]) for row in evaluation] != [1696]:
        raise ValueError(f"{method} endpoint evaluation missing")
    numeric = (
        "loss", "task_loss", "grad_norm", "lr",
        "periodic_relation_grad_norm", "periodic_relation_output_grad_norm",
        "periodic_geometry_metric_loss", "periodic_geometry_pair_rdf_loss",
        "periodic_geometry_overlap_loss", "periodic_geometry_coordination_loss",
    )
    for row in train:
        for key in numeric:
            if not math.isfinite(float(row[key])):
                raise ValueError(f"{method} nonfinite {key}")
    failures = sorted(run.glob("ENGINEERING_FAILURE*.tsv"))
    return {
        "method": method,
        "source_run": str(run.resolve()),
        "source_failed_marker_preserved": True,
        "root_cause": "post-science wrapper expected a train log event at step1696 although logging_steps=10; endpoint eval/checkpoint are complete",
        "optimizer_updates": 1696,
        "last_periodic_train_log_step": 1690,
        "endpoint_eval_step": 1696,
        "endpoint_val_loss": float(evaluation[0]["val_loss"]),
        "step0_max_logit_delta": 0.0,
        "uncertainty_gate": method == "B",
        "policy_path": str(checkpoint.resolve()),
        "adapter_sha256": sha256(required[0]),
        "relation_sha256": sha256(required[1]),
        "relation_config_sha256": sha256(required[2]),
        "failure_records": [str(path.resolve()) for path in failures],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-run", type=Path, required=True)
    parser.add_argument("--b-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report = {
        "schema": "g2_full_epoch_training_final_v1",
        "status": "complete_postcheck_recovery_without_retraining",
        "checkpoint_selection": False,
        "methods": [audit("A", args.a_run), audit("B", args.b_run)],
    }
    args.output_dir.mkdir(parents=True)
    result = args.output_dir / "G2_FULL_EPOCH_TRAINING_FINAL.json"
    result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "OUTPUTS.sha256").write_text(
        f"{sha256(result)}  {result.name}\n"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
