#!/usr/bin/env python3
"""CPU-Slurm entry point for immutable CrysLLMGen Gate-B/C locks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _require_runtime() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("screening locks must be written through Slurm CPU")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("B", "C"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-a-lock", type=Path, required=True)
    parser.add_argument("--gate-b-lock", type=Path)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require_runtime()
    from crystal_dlm.wqcodiff.crysllmgen.screening import (
        freeze_gate_b,
        freeze_gate_c,
    )

    common = {
        "protocol_path": args.protocol.resolve(),
        "gate_a_lock_path": args.gate_a_lock.resolve(),
        "evidence_manifest_path": args.evidence_manifest.resolve(),
        "output": args.output.resolve(),
    }
    if args.gate == "B":
        if args.gate_b_lock is not None:
            raise ValueError("Gate B cannot consume itself")
        result = freeze_gate_b(**common)
    else:
        if args.gate_b_lock is None:
            raise ValueError("Gate C requires --gate-b-lock")
        result = freeze_gate_c(
            **common,
            gate_b_lock_path=args.gate_b_lock.resolve(),
        )
    print(json.dumps(result, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
