#!/usr/bin/env python3
"""Slurm entry point for validation-only CrysLLMGen revision calibration."""

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
        raise RuntimeError("revision calibration must run through Slurm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-a-lock", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--llama-adapter", type=Path, required=True)
    parser.add_argument("--validation", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    args = parser.parse_args()
    _require_runtime()
    from crystal_dlm.wqcodiff.crysllmgen.revision_calibration import calibrate

    result = calibrate(
        protocol_path=args.protocol.resolve(),
        gate_a_lock_path=args.gate_a_lock.resolve(),
        refiner_checkpoint=args.refiner_checkpoint.resolve(),
        llama_root=args.llama_root.resolve(),
        llama_adapter=args.llama_adapter.resolve(),
        validation_paths=[value.resolve() for value in args.validation],
        output_dir=args.output_dir.resolve(),
        training_seed=args.training_seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
