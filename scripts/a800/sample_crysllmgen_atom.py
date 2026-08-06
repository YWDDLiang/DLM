#!/usr/bin/env python3
"""Immutable Slurm entry point for registered CrysLLMGen atom baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _require_runtime() -> None:
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("registered CUDA sampling must run through Slurm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-a-lock", type=Path, required=True)
    parser.add_argument("--csp-checkpoint", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--llama-adapter", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--pairing-id", required=True)
    parser.add_argument(
        "--method",
        choices=("C-ATOM-OFFICIAL", "C-ATOM-MATCHED"),
        required=True,
    )
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--start-ordinal", type=int, default=0)
    args = parser.parse_args()
    _require_runtime()

    from crystal_dlm.wqcodiff.crysllmgen.atom_sampling import (
        CrysLLMGenAtomSamplingConfig,
        sample,
    )

    report = sample(
        CrysLLMGenAtomSamplingConfig(
            protocol_path=str(args.protocol.resolve()),
            gate_a_lock=str(args.gate_a_lock.resolve()),
            csp_checkpoint=str(args.csp_checkpoint.resolve()),
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
            start_ordinal=args.start_ordinal,
        )
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
