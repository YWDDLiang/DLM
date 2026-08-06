#!/usr/bin/env python3
"""Aggregate immutable Gate-A evidence and write the training-unblock lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.gate import build_gate_a_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-sync-record", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--parity-audit", type=Path, required=True)
    parser.add_argument("--llama-report", type=Path, required=True)
    parser.add_argument("--grammar-report", type=Path, required=True)
    parser.add_argument("--atom-smoke-report", type=Path, required=True)
    parser.add_argument("--wq-smoke-report", type=Path, required=True)
    parser.add_argument("--constrained-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate A aggregation must run through Slurm CPU")
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
    payload = build_gate_a_lock(
        project_root=args.project_root,
        source_sync_record=args.source_sync_record,
        protocol_path=args.protocol,
        registry_path=args.registry,
        parity_audit_path=args.parity_audit,
        llama_report_path=args.llama_report,
        grammar_report_path=args.grammar_report,
        atom_smoke_report_path=args.atom_smoke_report,
        wq_smoke_report_path=args.wq_smoke_report,
        constrained_report_path=args.constrained_report,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
