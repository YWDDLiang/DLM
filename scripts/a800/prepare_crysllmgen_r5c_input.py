#!/usr/bin/env python3
"""Prepare an attempt-preserving extxyz input for the frozen R5-C evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _require_threads() -> None:
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--generation-ledger", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--structure-stage",
        choices=("raw", "common_refiner", "mlip_relaxed"),
        required=True,
    )
    args = parser.parse_args()
    _require_threads()

    from crystal_dlm.wqcodiff.crysllmgen.evaluation_adapter import prepare_r5c_input

    report = prepare_r5c_input(
        generation_jsonl=args.generation_jsonl.resolve(),
        generation_ledger=args.generation_ledger.resolve(),
        structures_path=args.structures.resolve(),
        manifest_path=args.manifest.resolve(),
        method=args.method,
        structure_stage=args.structure_stage,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
