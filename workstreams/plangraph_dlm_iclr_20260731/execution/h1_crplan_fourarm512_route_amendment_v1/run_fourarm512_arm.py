#!/usr/bin/env python3
"""Run one immutable arm of the H1 CR-Plan four-arm 512 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from workstreams.plangraph_dlm_iclr_20260731.execution.h1_crplan_e1_physical_performance_probe_v1 import (
    run_crplan_e1_probe as e1,
)


IDENTITY = "h1_crplan_fourarm512_route_amendment_v1"
ATTEMPTS = 512
MODES = ("off", "grammar_only", "terminal_only", "full_prefix")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def validate_contract(
    *,
    config: Mapping[str, Any],
    ledger: Mapping[str, Any],
    ledger_sha256: str,
    mode: str,
) -> dict[int, int]:
    if config.get("identity") != IDENTITY:
        raise ValueError("config identity mismatch")
    if int(config.get("attempts_per_arm", -1)) != ATTEMPTS:
        raise ValueError("config denominator mismatch")
    if tuple(config.get("modes") or ()) != MODES:
        raise ValueError("config arm order mismatch")
    if mode not in MODES:
        raise ValueError(f"unknown arm mode {mode!r}")
    if config.get("science_ledger_sha256") != ledger_sha256:
        raise ValueError("science ledger SHA mismatch")
    if ledger.get("identity") != IDENTITY:
        raise ValueError("ledger identity mismatch")
    if int(ledger.get("attempts_per_arm", -1)) != ATTEMPTS:
        raise ValueError("ledger denominator mismatch")
    if int(ledger.get("base_seed", -1)) != int(
        config.get("science_base_seed", -2)
    ):
        raise ValueError("ledger base seed mismatch")
    if (
        ledger.get("seed_derivation_phrase_sha256")
        != config.get("seed_derivation_phrase_sha256")
    ):
        raise ValueError("ledger derivation identity mismatch")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != ATTEMPTS:
        raise ValueError("ledger rows are incomplete")
    seeds: dict[int, int] = {}
    for expected, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"ledger row {expected} is not an object")
        ordinal = int(row.get("ordinal", -1))
        if ordinal != expected or ordinal in seeds:
            raise ValueError("ledger ordinal mapping is not exactly 0..511")
        if row.get("stage") != "planner_sampling" or row.get("role") != "shared":
            raise ValueError(f"ledger role mismatch at ordinal {ordinal}")
        seeds[ordinal] = int(row["planner_sampling_seed"])
    return seeds


def configure_frozen_e1_primitives(config: Mapping[str, Any]) -> None:
    """Bind the already-audited sampler primitives to the new frozen contract."""

    e1.ATTEMPTS = ATTEMPTS
    e1.BASE_SEED = int(config["science_base_seed"])
    e1.MODES = MODES
    e1.PROMPT_STYLE = str(config["prompt_style"])
    e1.MAX_NEW_TOKENS = int(config["max_new_tokens"])
    e1.TEMPERATURE = float(config["temperature"])
    e1.TOP_P = float(config["top_p"])
    e1.TOP_K = int(config["top_k"])
    e1.MAX_ATOMS = int(config["max_atoms"])
    e1.MISSING_STATE_POLICY = str(config["missing_state_policy"])


def run(args: argparse.Namespace) -> None:
    import torch

    run_root = args.run_root.resolve()
    arm_root = run_root / "arms" / args.mode
    if arm_root.exists():
        raise FileExistsError(f"refusing to overwrite {arm_root}")

    config = read_json(args.config)
    ledger = read_json(args.ledger)
    ledger_sha256 = sha256_file(args.ledger)
    ledger_seeds = validate_contract(
        config=config,
        ledger=ledger,
        ledger_sha256=ledger_sha256,
        mode=args.mode,
    )
    source_manifest_sha256 = sha256_file(args.source_manifest)
    configure_frozen_e1_primitives(config)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one CUDA device, observed {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "A800" not in gpu_name:
        raise RuntimeError(f"expected A800, observed {gpu_name!r}")

    arm_root.mkdir(parents=True)
    job_started = time.perf_counter()
    load_started = time.perf_counter()
    model, tokenizer = e1.load_planner(
        args.model_path,
        args.checkpoint_path,
        device,
    )
    e1.cuda_sync(torch, device)
    model_load_seconds = time.perf_counter() - load_started
    warmup_seconds = e1.common_model_warmup(model, tokenizer, device)
    context = None if args.mode == "off" else e1.mode_context(tokenizer)

    setup = {
        "schema": "h1_crplan_fourarm512_arm_setup_v1",
        "identity": IDENTITY,
        "mode": args.mode,
        "attempts": ATTEMPTS,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "model_path": str(args.model_path),
        "checkpoint_path": str(args.checkpoint_path),
        "adapter_model_sha256": args.expected_adapter_sha256,
        "model_load_seconds": model_load_seconds,
        "common_model_forward_warmup_seconds": warmup_seconds,
        "support_setup_seconds": (
            None if context is None else context["setup_seconds"]
        ),
        "token_fragment_sha256": (
            None
            if context is None
            else context["vocabulary"].fragment_sha256
        ),
        "oxidation_table_sha256": (
            None if context is None else context["reachability"].table_sha256
        ),
        "constraint_contract_sha256": (
            None
            if context is None
            else context["reachability"].constraint_contract_sha256
        ),
        "source_manifest_sha256": source_manifest_sha256,
        "science_ledger_sha256": ledger_sha256,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mode_specific_sampling_warmup": False,
        "network": False,
    }
    e1.atomic_write_json(arm_root / "setup.json", setup)
    e1.atomic_write_json(
        arm_root / "run_config.json",
        {
            "schema": "h1_crplan_fourarm512_arm_config_v1",
            "identity": IDENTITY,
            "mode": args.mode,
            "attempts": ATTEMPTS,
            "science_base_seed": int(config["science_base_seed"]),
            "science_ledger_sha256": ledger_sha256,
            "prompt_style": config["prompt_style"],
            "include_sample_id": config["include_sample_id"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "max_new_tokens": config["max_new_tokens"],
            "max_atoms": config["max_atoms"],
            "seed_mode": config["seed_mode"],
            "seed_stage": config["seed_stage"],
            "seed_role": config["seed_role"],
            "missing_state_policy": config["missing_state_policy"],
            "effective_generation_batch_size": (
                config["effective_generation_batch_size"]
            ),
            "source_manifest_sha256": source_manifest_sha256,
            "adapter_model_sha256": args.expected_adapter_sha256,
        },
    )

    parse_count = 0
    completion_count = 0
    error_count = 0
    for ordinal in range(ATTEMPTS):
        row = e1.sample_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            mode=args.mode,
            ordinal=ordinal,
            sequence=ordinal,
            context=context,
            scalar_reference=False,
        )
        if int(row["planner_sampling_seed"]) != ledger_seeds[ordinal]:
            raise RuntimeError(f"ledger seed mismatch at ordinal {ordinal}")
        row.update(
            {
                "schema": "h1_crplan_fourarm512_attempt_v1",
                "experiment_identity": IDENTITY,
                "science_ledger_sha256": ledger_sha256,
                "ledger_ordinal": ordinal,
                "ledger_role": "shared",
                "execution_variant": "optimized_scientific_primary",
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "body_rerun": False,
                "refiner_rerun": False,
                "direct_rerun": False,
                "sun_rerun": False,
                "network": False,
                "training": False,
                "checkpoint_reselection": False,
                "automatic_downstream": False,
            }
        )
        e1.append_jsonl(arm_root / "raw_generations.jsonl", row)
        parse_count += int(row.get("parsed") is True)
        completion_count += int(row.get("plan_end_marker_present") is True)
        error_count += int(row.get("generation_error") is True)
        print(
            json.dumps(
                {
                    "event": "fourarm512_attempt_complete",
                    "mode": args.mode,
                    "ordinal": ordinal,
                    "parsed": row.get("parsed"),
                    "completion": row.get("plan_end_marker_present"),
                    "reason": row.get("reason"),
                    "latency_sec": row.get(
                        "planner_generation_latency_sec"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    runner_report = {
        "schema": "h1_crplan_fourarm512_arm_runner_report_v1",
        "status": "runner_complete",
        "identity": IDENTITY,
        "mode": args.mode,
        "attempt_count": ATTEMPTS,
        "parse_count": parse_count,
        "completion_count": completion_count,
        "generation_error_count": error_count,
        "setup": setup,
        "job_wall_seconds_through_runner": time.perf_counter() - job_started,
        "v4_state_gate_reused": False,
        "semantic_states_report_only": True,
        "body_rerun": False,
        "refiner_rerun": False,
        "direct_rerun": False,
        "sun_rerun": False,
        "network": False,
        "training": False,
        "checkpoint_reselection": False,
        "automatic_downstream": False,
    }
    e1.atomic_write_json(arm_root / "runner_report.json", runner_report)
    print(json.dumps(runner_report, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
