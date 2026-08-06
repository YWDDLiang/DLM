#!/usr/bin/env python3
"""Immutable Slurm entry point for registered CrysLLMGen/WQ sampling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _require_runtime() -> None:
    values = []
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        value = int(os.environ.get(name, "0"))
        if value not in (4, 8, 16):
            raise RuntimeError(f"{name} must be one of 4, 8, or 16")
        values.append(value)
    if len(set(values)) != 1:
        raise RuntimeError("sampling numerical thread settings must agree")
    if values[0] > int(os.environ.get("SLURM_CPUS_PER_TASK", "0")):
        raise RuntimeError("sampling threads exceed allocated Slurm CPUs")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("registered CUDA sampling must run through Slurm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-a-lock", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--llama-adapter", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--pairing-id", required=True)
    parser.add_argument(
        "--method",
        choices=(
            "C-WQ-HANDOFF",
            "C-WQ-CONFEDIT",
            "C-WQ-GEOREV",
            "C-WQ-BIRTH-DEATH-ONLY",
            "C-WQ-RANDOM-MATCHED-COUNT",
            "C-WQ-SHUFFLED-GEOMETRY",
            "C-WQ-EXTRA-CALL-IGNORED",
        ),
        required=True,
    )
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--adapter-training-execution-patch-sha256")
    parser.add_argument("--refiner-training-execution-patch-sha256")
    parser.add_argument("--start-ordinal", type=int, default=0)
    parser.add_argument("--reverse-steps", type=int, default=32)
    parser.add_argument(
        "--handoff-tau",
        type=float,
        choices=(0.25, 0.5, 0.75, 1.0),
        default=1.0,
    )
    parser.add_argument("--revision-threshold", type=float, default=0.7)
    parser.add_argument("--revision-calibration-lock", type=Path)
    parser.add_argument("--reference-generation-jsonl", type=Path)
    args = parser.parse_args()

    _require_runtime()
    from crystal_dlm.wqcodiff.crysllmgen.wq_sampling import (
        CrysLLMGenWQSamplingConfig,
        sample,
    )

    report = sample(
        CrysLLMGenWQSamplingConfig(
            protocol_path=str(args.protocol.resolve()),
            gate_a_lock=str(args.gate_a_lock.resolve()),
            refiner_checkpoint=str(args.refiner_checkpoint.resolve()),
            llama_root=str(args.llama_root.resolve()),
            llama_adapter=str(args.llama_adapter.resolve()),
            output_jsonl=str(args.output_jsonl.resolve()),
            attempt_ledger=str(args.attempt_ledger.resolve()),
            report_path=str(args.report.resolve()),
            experiment_id=args.experiment_id,
            pairing_id=args.pairing_id,
            method=args.method,
            training_seed=args.training_seed,
            sampling_seed=args.sampling_seed,
            attempts=args.attempts,
            execution_patch_sha256=args.execution_patch_sha256,
            adapter_training_execution_patch_sha256=(
                args.adapter_training_execution_patch_sha256
            ),
            refiner_training_execution_patch_sha256=(
                args.refiner_training_execution_patch_sha256
            ),
            start_ordinal=args.start_ordinal,
            reverse_steps=args.reverse_steps,
            handoff_tau=args.handoff_tau,
            revision_threshold=args.revision_threshold,
            revision_calibration_lock=(
                None
                if args.revision_calibration_lock is None
                else str(args.revision_calibration_lock.resolve())
            ),
            reference_generation_jsonl=(
                None
                if args.reference_generation_jsonl is None
                else str(args.reference_generation_jsonl.resolve())
            ),
        )
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
