#!/usr/bin/env python3
"""Select FlashAttention or the frozen SDPA fallback exactly once."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path

from crystal_dlm.wqcodiff.contracts import write_json_exclusive
from crystal_dlm.wqcodiff.crysllmgen.formal_execution import (
    AUTHORIZATION_RECORD,
    AUTHORIZATION_SHA256,
    CACHE_MANIFEST_SHA256,
    FLASH_PROFILE_ID,
    SDPA_PROFILE_ID,
    sha256_file,
)


FLASH_MAX_SECONDS_PER_UPDATE = 4.0797596
FLASH_MINIMUM_RELATIVE_SPEEDUP = 0.02
SDPA_SECONDS_PER_UPDATE = 4.16302
SDPA_REPORT_SHA256 = (
    "d3d1948088439f8ea0da0d89cbee7551a734308aa51748cfbca88b9de6d57960"
)
SDPA_SUMMARY_SHA256 = (
    "c7505653b717d0e072f4d5394347aa9b6b8fc8e1280c376d8fccffe4212170f1"
)
ANOMALY = re.compile(
    r"(?:traceback|out of memory|cuda (?:error|failure)|runtimeerror.*cuda|"
    r"cublas.*error|cudnn.*error|nccl.*(?:error|abort)|\bnan\b|non[- ]finite)",
    re.IGNORECASE,
)


def _sacct(job_id: str) -> tuple[str, str, str]:
    completed = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ExitCode",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = [line.split("|") for line in completed.stdout.splitlines() if line.strip()]
    exact = [row for row in rows if len(row) >= 3 and row[0] == job_id]
    if len(exact) != 1:
        raise RuntimeError(f"cannot identify exact FlashAttention Slurm row: {rows}")
    return exact[0][0], exact[0][1].split()[0], exact[0][2]


def _gpu_csv(path: Path) -> tuple[int, list[int]]:
    indices: set[int] = set()
    samples = 0
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) >= 5:
                    samples += 1
                    indices.add(int(row[1].strip()))
    return samples, sorted(indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--flash-job-id", required=True)
    parser.add_argument("--flash-profile-matrix", type=Path, required=True)
    parser.add_argument("--flash-report", type=Path, required=True)
    parser.add_argument("--flash-stderr", type=Path, required=True)
    parser.add_argument("--flash-gpu-csv", type=Path, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--protocol-v4-sha256", required=True)
    parser.add_argument("--gate-a-lock", type=Path, required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal execution selection must run through Slurm CPU")
    for label, value in (
        ("Flash job ID", args.flash_job_id),
        ("selector job ID", os.environ["SLURM_JOB_ID"]),
    ):
        if not value.isdigit():
            raise ValueError(f"{label} must be numeric")
    for label, value in (
        ("source", args.source_bundle_sha256),
        ("protocol", args.protocol_v4_sha256),
        ("execution patch", args.execution_patch_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{label} must be one lowercase SHA256")

    gate = json.loads(args.gate_a_lock.read_text(encoding="utf-8"))
    gate_sha = sha256_file(args.gate_a_lock)
    matrix_sha = sha256_file(args.flash_profile_matrix)
    errors: list[str] = []
    try:
        job_id, state, exit_code = _sacct(args.flash_job_id)
    except Exception as exc:
        job_id, state, exit_code = args.flash_job_id, "ACCOUNTING_UNAVAILABLE", "unknown"
        errors.append(f"sacct:{type(exc).__name__}:{exc}")
    if (state, exit_code) != ("COMPLETED", "0:0"):
        errors.append(f"slurm:{state}:{exit_code}")
    stderr_text = (
        args.flash_stderr.read_text(encoding="utf-8", errors="replace")
        if args.flash_stderr.is_file()
        else ""
    )
    if not args.flash_stderr.is_file():
        errors.append("stderr_missing")
    elif ANOMALY.search(stderr_text):
        errors.append("stderr_anomaly")
    gpu_samples, gpu_indices = _gpu_csv(args.flash_gpu_csv)
    if gpu_samples <= 0 or len(gpu_indices) != 2:
        errors.append("gpu_telemetry_incomplete")

    report: dict[str, object] | None = None
    report_sha: str | None = None
    seconds_per_update: float | None = None
    if args.flash_report.is_file():
        report = json.loads(args.flash_report.read_text(encoding="utf-8"))
        report_sha = sha256_file(args.flash_report)
        optimizer = report.get("optimizer") or {}
        execution = report.get("execution") or {}
        matrix = report.get("profile_matrix") or {}
        distributed = report.get("distributed") or {}
        metrics = report.get("metrics") or {}
        runtime = report.get("runtime") or {}
        seconds_per_update = float(
            runtime.get("train_seconds_per_optimizer_update", float("nan"))
        )
        checks = {
            "schema": report.get("schema")
            == "crysllmgen_lora_ddp_performance_profile_v1",
            "pass": report.get("pass") is True,
            "nonscientific": report.get("scientific_attempt") is False,
            "source": report.get("source_bundle_sha256")
            == args.source_bundle_sha256,
            "protocol": report.get("protocol_sha256") == args.protocol_v4_sha256,
            "patch": report.get("execution_patch_sha256")
            == args.execution_patch_sha256,
            "matrix": matrix.get("sha256") == matrix_sha,
            "profile": matrix.get("profile_id") == FLASH_PROFILE_ID,
            "updates": optimizer.get("completed_global_step") == 50,
            "world": optimizer.get("world_size") == 2,
            "batch": optimizer.get("global_effective_batch") == 64,
            "microbatch": optimizer.get("per_device_microbatch") == 8,
            "accumulation": optimizer.get("gradient_accumulation") == 4,
            "nccl": distributed.get("backend") == "nccl",
            "ranks": len(distributed.get("rank_runtimes") or ()) == 2,
            "attention": execution.get("attention_implementation")
            == "flash_attention_2",
            "gc_off": execution.get("gradient_checkpointing") is False,
            "cache": (execution.get("data_loading") or {}).get(
                "cache_manifest_sha256"
            )
            == CACHE_MANIFEST_SHA256,
            "loss": math.isfinite(float(metrics.get("train_loss", float("nan")))),
            "timing": math.isfinite(seconds_per_update),
            "speedup": seconds_per_update <= FLASH_MAX_SECONDS_PER_UPDATE,
        }
        errors.extend(key for key, value in checks.items() if not value)
    else:
        errors.append("profile_report_missing")

    eligible = not errors
    selected_attention = "flash_attention_2" if eligible else "sdpa"
    selected_profile = FLASH_PROFILE_ID if eligible else SDPA_PROFILE_ID
    result = {
        "schema": "crysllmgen_formal_execution_selection_v1",
        "status": "selected_for_formal_training",
        "scientific_attempt": False,
        "run_id": args.run_id,
        "source_bundle_sha256": args.source_bundle_sha256,
        "protocol_v4_sha256": args.protocol_v4_sha256,
        "gate_a_lock_sha256": gate_sha,
        "gate_a_lock_declared_sha256": gate.get("gate_a_lock_sha256"),
        "execution_patch_sha256": args.execution_patch_sha256,
        "authorization": {
            "record": AUTHORIZATION_RECORD,
            "sha256": AUTHORIZATION_SHA256,
        },
        "selected_attention_implementation": selected_attention,
        "selected_profile_id": selected_profile,
        "decision": (
            "eligible_flash_exceeded_two_percent_speedup_threshold"
            if eligible
            else "frozen_sdpa_fallback_after_failed_or_ineligible_flash"
        ),
        "execution": {
            "world_size": 2,
            "per_device_microbatch": 8,
            "gradient_accumulation": 4,
            "global_effective_batch": 64,
            "data_mode": "pretokenized_memmap",
            "dataloader_num_workers_per_rank": 4,
            "pretokenized_cache_manifest_sha256": CACHE_MANIFEST_SHA256,
            "gradient_checkpointing": False,
            "gradient_checkpointing_use_reentrant": None,
        },
        "sdpa_reference": {
            "profile_id": SDPA_PROFILE_ID,
            "seconds_per_update": SDPA_SECONDS_PER_UPDATE,
            "report_sha256": SDPA_REPORT_SHA256,
            "summary_sha256": SDPA_SUMMARY_SHA256,
        },
        "flash_candidate": {
            "profile_id": FLASH_PROFILE_ID,
            "profile_matrix": str(args.flash_profile_matrix.resolve()),
            "profile_matrix_sha256": matrix_sha,
            "profile_report": str(args.flash_report.resolve()),
            "profile_report_sha256": report_sha,
            "slurm_job_id": job_id,
            "slurm_state": state,
            "slurm_exit_code": exit_code,
            "seconds_per_update": seconds_per_update,
            "minimum_relative_speedup": FLASH_MINIMUM_RELATIVE_SPEEDUP,
            "maximum_seconds_per_update": FLASH_MAX_SECONDS_PER_UPDATE,
            "eligible": eligible,
            "ineligibility_reasons": errors,
            "stderr": str(args.flash_stderr.resolve()),
            "stderr_sha256": (
                sha256_file(args.flash_stderr)
                if args.flash_stderr.is_file()
                else None
            ),
            "gpu_csv": str(args.flash_gpu_csv.resolve()),
            "gpu_csv_sha256": (
                sha256_file(args.flash_gpu_csv)
                if args.flash_gpu_csv.is_file()
                else None
            ),
            "gpu_samples": gpu_samples,
            "gpu_indices": gpu_indices,
        },
        "selector_slurm_job_id": os.environ["SLURM_JOB_ID"],
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
