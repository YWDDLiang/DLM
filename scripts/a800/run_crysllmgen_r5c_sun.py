#!/usr/bin/env python3
"""Run the unchanged R5-C SUN script, then restore the attempt denominator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
        raise RuntimeError("R5-C MatterSim evaluation must run through Slurm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-a-lock", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument("--matter-sim-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--structure-matcher",
        choices=("ordered", "disordered"),
        default="disordered",
    )
    parser.add_argument("--no-relax", action="store_true")
    args = parser.parse_args()
    _require_runtime()

    from crystal_dlm.wqcodiff.crysllmgen.evaluation_adapter import (
        R5C_SCRIPT_SHA256,
        aggregate_r5c_output,
        write_terminal_evaluator_failure,
    )
    from crystal_dlm.wqcodiff.crysllmgen.gate import GateALock, sha256_file
    from crystal_dlm.wqcodiff.crysllmgen.protocol import load_protocol_v4

    protocol = load_protocol_v4(args.protocol.resolve())
    project_root = args.protocol.resolve().parents[3]
    gate = GateALock.load(
        args.gate_a_lock.resolve(),
        project_root=project_root,
        protocol_path=args.protocol.resolve(),
    )
    script = project_root / "scripts/run_mattergen_sun_eval.py"
    if sha256_file(script) != R5C_SCRIPT_SHA256:
        raise RuntimeError("the frozen R5-C SUN script changed")
    checkpoint = args.matter_sim_checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = json.loads(args.input_manifest.resolve().read_text(encoding="utf-8"))
    if manifest.get("schema") != "crysllmgen_r5c_input_manifest_v1":
        raise ValueError("invalid R5-C input manifest")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "metrics": output / "r5c_metrics.json",
        "detailed": output / "r5c_detailed.json",
        "relaxed": output / "r5c_relaxed.extxyz",
        "summary": output / "r5c_summary.json",
        "relax_failures": output / "r5c_relax_failures.json",
        "unsupported": output / "r5c_unsupported.json",
        "metric_errors": output / "r5c_metric_errors.json",
        "attempt_results": output / "attempt_results.jsonl",
        "attempt_summary": output / "attempt_summary.json",
    }
    command = [
        sys.executable,
        str(script),
        "--structures-path",
        str(Path(manifest["structures_path"]).resolve()),
        "--reference-dataset",
        str(args.reference_dataset.resolve()),
        "--save-as",
        str(paths["metrics"]),
        "--save-detailed-as",
        str(paths["detailed"]),
        "--structures-output-path",
        str(paths["relaxed"]),
        "--summary-json",
        str(paths["summary"]),
        "--device",
        "cuda",
        "--potential-load-path",
        str(checkpoint),
        "--relax-max-steps",
        "500",
        "--relax-fmax",
        "0.05",
        "--max-natoms-per-batch",
        "512",
        "--relax-failures-json",
        str(paths["relax_failures"]),
        "--unsupported-failures-json",
        str(paths["unsupported"]),
        "--metric-errors-json",
        str(paths["metric_errors"]),
        "--structure-matcher",
        args.structure_matcher,
    ]
    if args.no_relax:
        command.append("--no-relax")
    contract = {
        "schema": "crysllmgen_r5c_run_contract_v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": gate.sha256,
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256_file(args.input_manifest),
        "r5c_script": str(script),
        "r5c_script_sha256": R5C_SCRIPT_SHA256,
        "matter_sim_checkpoint": str(checkpoint),
        "matter_sim_checkpoint_sha256": sha256_file(checkpoint),
        "reference_dataset": str(args.reference_dataset.resolve()),
        "reference_dataset_sha256": sha256_file(args.reference_dataset),
        "structure_matcher": args.structure_matcher,
        "no_relax": bool(args.no_relax),
        "command": command,
        "threads": 1,
        "offline": True,
        "retry_or_replacement_used": False,
    }
    from crystal_dlm.wqcodiff.contracts import write_json_exclusive

    write_json_exclusive(output / "run_contract.json", contract)
    started = time.monotonic()
    try:
        subprocess.run(command, check=True)
        result = aggregate_r5c_output(
            input_manifest_path=args.input_manifest.resolve(),
            r5c_summary_path=paths["summary"],
            detailed_metrics_path=paths["detailed"],
            unsupported_path=paths["unsupported"],
            relax_failures_path=paths["relax_failures"],
            output_jsonl=paths["attempt_results"],
            output_summary=paths["attempt_summary"],
            evaluator="MatterSim-v1.0.0-5M",
            evaluator_checkpoint=checkpoint,
            r5c_script=script,
        )
    except Exception as exc:
        if not paths["attempt_results"].exists() and not paths["attempt_summary"].exists():
            result = write_terminal_evaluator_failure(
                input_manifest_path=args.input_manifest.resolve(),
                output_jsonl=paths["attempt_results"],
                output_summary=paths["attempt_summary"],
                reason=f"{type(exc).__name__}:{exc}",
                evaluator="MatterSim-v1.0.0-5M",
            )
        (output / "executor_failure.json").write_text(
            json.dumps(
                {
                    "schema": "crysllmgen_r5c_executor_failure_v1",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "walltime_s": time.monotonic() - started,
                    "retry_or_replacement_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True))
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
