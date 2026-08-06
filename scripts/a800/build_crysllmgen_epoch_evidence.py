#!/usr/bin/env python3
"""Build the exclusive three-epoch validation evidence manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from crystal_dlm.wqcodiff.contracts import write_json_exclusive
from crystal_dlm.wqcodiff.crysllmgen.epoch_training import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--training-amendment", type=Path, required=True)
    parser.add_argument("--base-source-bundle-sha256", required=True)
    parser.add_argument("--execution-patch-sha256")
    parser.add_argument("--adapter-training-execution-patch-sha256")
    parser.add_argument("--refiner-training-execution-patch-sha256")
    parser.add_argument("--evaluation-execution-patch-sha256")
    parser.add_argument("--supersedes-epoch-evaluation-array-job-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    legacy_patch = args.execution_patch_sha256
    explicit_patches = (
        args.adapter_training_execution_patch_sha256,
        args.refiner_training_execution_patch_sha256,
        args.evaluation_execution_patch_sha256,
    )
    if legacy_patch is not None and any(value is not None for value in explicit_patches):
        raise ValueError(
            "legacy execution patch cannot be mixed with separated execution identities"
        )
    if legacy_patch is not None:
        adapter_training_patch = legacy_patch
        refiner_training_patch = legacy_patch
        evaluation_patch = legacy_patch
        schema = "crysllmgen_epoch_selection_evidence_v1"
    else:
        if any(value is None for value in explicit_patches):
            raise ValueError("all separated execution identities are required")
        if not args.supersedes_epoch_evaluation_array_job_id:
            raise ValueError("evaluation supersession job identity is required")
        adapter_training_patch = str(
            args.adapter_training_execution_patch_sha256
        )
        refiner_training_patch = str(
            args.refiner_training_execution_patch_sha256
        )
        evaluation_patch = str(args.evaluation_execution_patch_sha256)
        schema = "crysllmgen_epoch_selection_evidence_v2"
    for label, value in (
        ("source", args.base_source_bundle_sha256),
        ("adapter training execution patch", adapter_training_patch),
        ("refiner training execution patch", refiner_training_patch),
        ("evaluation execution patch", evaluation_patch),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{label} must be one lowercase SHA256")
    training = args.training_root.resolve()
    panel = args.panel_root.resolve()
    epochs = []
    for epoch in (1, 2, 3):
        tag = f"epoch_{epoch:02d}"
        training_report = training / tag / "training_report.json"
        evaluation = panel / tag
        entry = {
            "logical_epoch": epoch,
            "training_report": str(training_report),
            "nll_report": str(evaluation / "nll_report.json"),
            "generation_jsonl": str(evaluation / "generation.jsonl"),
            "crysllmgen_metrics_report": str(
                evaluation / "crysllmgen_metrics" / "report.json"
            ),
            "a100_attempt_jsonl": str(
                evaluation / "r5c_a100_sun" / "attempt_results.jsonl"
            ),
            "a100_summary": str(
                evaluation / "r5c_a100_sun" / "attempt_summary.json"
            ),
        }
        for key, value in entry.items():
            if key != "logical_epoch" and not Path(str(value)).is_file():
                raise FileNotFoundError(value)
        report = json.loads(training_report.read_text(encoding="utf-8"))
        if (
            report.get("logical_epoch") != epoch
            or report.get("source_bundle_sha256")
            != args.base_source_bundle_sha256
            or report.get("execution_patch_sha256")
            != adapter_training_patch
        ):
            raise ValueError(f"{tag} training evidence identity changed")
        epochs.append(entry)
    result = {
        "schema": schema,
        "training_seed": 11,
        "training_amendment_sha256": sha256_file(args.training_amendment),
        "base_source_bundle_sha256": args.base_source_bundle_sha256,
        "execution_patch_sha256": evaluation_patch,
        "adapter_training_execution_patch_sha256": adapter_training_patch,
        "refiner_training_execution_patch_sha256": refiner_training_patch,
        "evaluation_execution_patch_sha256": evaluation_patch,
        "epochs": epochs,
        "execution_supersession": (
            None
            if schema.endswith("_v1")
            else {
                "supersedes_epoch_evaluation_array_job_id": (
                    args.supersedes_epoch_evaluation_array_job_id
                ),
                "reason": "pre-scientific execution-identity contract correction",
            }
        ),
        "retry_or_replacement_used": not schema.endswith("_v1"),
        "attempt_retry_or_replacement_used": False,
    }
    write_json_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
