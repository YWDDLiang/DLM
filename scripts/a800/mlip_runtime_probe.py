#!/usr/bin/env python3
"""Load exactly one frozen MLIP in its registered process/runtime scope."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.wqcodiff.mlip import EvaluatorLock, MLIPCalculator  # noqa: E402
from crystal_dlm.wqcodiff.dependency_waiver import (  # noqa: E402
    MLIP_ASSET_LOCK_FILENAME,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", choices=("chgnet", "mattersim", "mace"), required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/public/home/jiaosz/ywliang/models/wqcodiff"),
    )
    parser.add_argument("--asset-lock", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--relax-smoke", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA MLIP probes are allowed only inside Slurm")
    model_root = args.model_root.resolve()
    lock_path = (args.asset_lock or model_root / MLIP_ASSET_LOCK_FILENAME).resolve()
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5.43),
        [14, 14],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    calculator = MLIPCalculator(
        evaluator=args.evaluator,
        asset_lock=EvaluatorLock.load(lock_path),
        model_root=model_root,
        device=args.device,
    )
    single_point = calculator.single_point(structure)
    payload: dict[str, Any] = {
        "schema": "wqcodiff_mlip_runtime_probe_v1",
        "ok": True,
        "evaluator": args.evaluator,
        "device": args.device,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "asset_lock": str(lock_path),
        "contract_hash": calculator.contract_hash,
        "single_point": single_point,
    }
    if args.relax_smoke:
        relaxed = calculator.relax(structure)
        payload["relaxation"] = {
            key: value for key, value in relaxed.items() if key != "structure"
        }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
