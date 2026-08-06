#!/usr/bin/env python3
"""Run the frozen WQ LoRA smoke and constrained-generation Gate A in one lane."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _require_runtime() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the WQ Gate A lane must run through Slurm")
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


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=(11,), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    args = parser.parse_args()
    _require_runtime()

    project_root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Gate A lane output is immutable: {output}")
    # Child entry points create their own exclusive subdirectories.  Creating
    # only the parent here keeps partial evidence visible if either stage fails.
    output.mkdir(parents=True, exist_ok=False)
    smoke = output / "wq_lora_smoke"
    constrained = output / "constrained_256"
    _run(
        [
            sys.executable,
            str(project_root / "scripts/a800/train_crysllmgen_lora.py"),
            "--protocol",
            str(args.protocol.resolve()),
            "--llama-root",
            str(args.llama_root.resolve()),
            "--data",
            str(args.data.resolve()),
            "--data-manifest",
            str(args.data_manifest.resolve()),
            "--token-audit",
            str(args.token_audit.resolve()),
            "--representation",
            "wyckoff",
            "--training-stage",
            "coarse",
            "--training-seed",
            str(args.training_seed),
            "--output-dir",
            str(smoke),
            "--source-bundle-sha256",
            args.source_bundle_sha256,
            "--microbatch",
            str(args.microbatch),
            "--gradient-accumulation",
            str(args.gradient_accumulation),
            "--max-steps",
            "100",
            "--run-role",
            "smoke",
        ]
    )
    _run(
        [
            sys.executable,
            str(project_root / "scripts/a800/run_crysllmgen_constrained_gate.py"),
            "--llama-root",
            str(args.llama_root.resolve()),
            "--adapter",
            str(smoke / "adapter_final"),
            "--output-dir",
            str(constrained),
            "--attempts",
            "256",
        ]
    )
    report = {
        "schema": "crysllmgen_wq_gate_a_lane_report_v1",
        "ok": True,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "wq_smoke_report": str(smoke / "training_report.json"),
        "constrained_report": str(constrained / "report.json"),
        "retry_or_replacement_used": False,
    }
    report_path = output / "lane_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
