#!/usr/bin/env python3
"""Write the immutable validation-only mixed-edit epoch selection lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _require_runtime() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("epoch selection must run through Slurm CPU")
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
        raise RuntimeError("epoch-selection numerical thread settings must agree")
    if values[0] > int(os.environ.get("SLURM_CPUS_PER_TASK", "0")):
        raise RuntimeError("epoch-selection threads exceed allocated Slurm CPUs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-amendment", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require_runtime()

    from crystal_dlm.wqcodiff.crysllmgen.epoch_selection import (
        select_epoch_checkpoint,
    )
    from crystal_dlm.wqcodiff.crysllmgen.epoch_training import (
        load_mixed_edit_epoch_contract,
    )
    from crystal_dlm.wqcodiff.crysllmgen.protocol import load_protocol_v4

    protocol = load_protocol_v4(args.protocol.resolve())
    contract = load_mixed_edit_epoch_contract(
        args.training_amendment.resolve(),
        base_protocol_name=protocol.name,
        base_protocol_sha256=protocol.sha256,
    )
    result = select_epoch_checkpoint(
        contract=contract,
        evidence_manifest_path=args.evidence_manifest.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
