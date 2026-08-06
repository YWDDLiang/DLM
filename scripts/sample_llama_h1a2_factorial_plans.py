#!/usr/bin/env python3
"""Sample strict model-proposed H1-A2 Plans for the two-factor experiment.

Every ordinal is attempted exactly once with a rank-independent seed.  Failed
Planner attempts remain in the output denominator; this runner never retries,
repairs, replaces, filters, or reranks a model continuation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import StoppingCriteriaList

from crystal_dlm.fixed_slot import write_json  # noqa: E402
from crystal_dlm.h1a2_factorial_contract import (  # noqa: E402
    PLANNER_ARMS,
    build_factorial_ordinal_record,
    build_planner_input_contract,
    persist_model_sampled_plan,
)
from crystal_dlm.h1a2_factorial_runtime import (  # noqa: E402
    ordered_planner_attempts,
    read_jsonl_objects,
)
from scripts.sample_llama_h1_formula_plans import (  # noqa: E402
    GeneratedPlanEndStoppingCriteria,
    load_planner,
    model_device,
)
from scripts.sample_llada_dynamic_crystals import init_distributed, rank_path  # noqa: E402


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _merge_rank_attempts(
    output_dir: Path,
    *,
    world_size: int,
    expected_count: int,
    planner_arm: str,
) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for rank in range(int(world_size)):
        path = rank_path(output_dir, "planner_attempts.jsonl", rank, True)
        if not path.exists():
            raise FileNotFoundError(f"missing Planner rank output: {path}")
        records.extend(read_jsonl_objects(path))
    ordered = ordered_planner_attempts(
        records,
        expected_count=int(expected_count),
        expected_planner_arm=planner_arm,
    )
    _write_jsonl(output_dir / "planner_attempts.jsonl", ordered)
    _write_jsonl(
        output_dir / "plans_for_body.jsonl",
        [record for record in ordered if record.get("attempt_status") == "complete"],
    )
    return ordered


def _metrics(
    attempts: list[Mapping[str, Any]],
    *,
    requested_samples: int,
    elapsed: float,
    rank: int,
    world_size: int,
    distributed: bool,
) -> dict[str, Any]:
    failures: dict[str, int] = {}
    complete = 0
    for record in attempts:
        if record.get("attempt_status") == "complete":
            complete += 1
        else:
            reason = str(record.get("failure_reason") or "unknown")
            failures[reason] = failures.get(reason, 0) + 1
    denominator = int(requested_samples)
    return {
        "schema": "h1a2_factorial_planner_metrics_v1",
        "requested_samples": denominator,
        "attempt_rows": len(attempts),
        "complete_plans": complete,
        "failed_plans": denominator - complete,
        "plan_completion_rate_all_attempt": complete / max(1, denominator),
        "failures": failures,
        "time_sec": float(elapsed),
        "rank": int(rank),
        "world_size": int(world_size),
        "distributed": bool(distributed),
        "retry": False,
        "replacement": False,
        "repair": False,
        "filter": False,
        "rerank": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/public/home/jiaosz/ywliang/models/Llama-3.1-8B-Instruct/",
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--planner-arm", choices=PLANNER_ARMS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--base-seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-atoms", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be positive")

    dist_info = init_distributed()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    distributed = bool(dist_info["distributed"])
    is_main = bool(dist_info["is_main"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_planner(
        args.model_path,
        args.checkpoint_path,
        dist_info["device"],
    )
    input_contract = build_planner_input_contract(
        tokenizer,
        planner_arm=args.planner_arm,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    if is_main:
        write_json(
            str(args.output_dir / "planner_input_contract.json"),
            input_contract,
        )
        write_json(
            str(args.output_dir / "run_config.json"),
            {
                "schema": "h1a2_factorial_planner_run_v1",
                "model_path": args.model_path,
                "checkpoint_path": args.checkpoint_path,
                "checkpoint_sha256": args.checkpoint_sha256,
                "planner_arm": args.planner_arm,
                "num_samples": int(args.num_samples),
                "base_seed": int(args.base_seed),
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "max_atoms": int(args.max_atoms),
                "prompt_style": input_contract["prompt_style"],
                "include_sample_id": False,
                "effective_batch_size": 1,
                "seed_mode": "stateless_ordinal_v1",
                "distributed": distributed,
                "world_size": world_size,
                "retry": False,
                "replacement": False,
                "repair": False,
                "filter": False,
                "rerank": False,
            },
        )

    assigned = [
        sample_idx
        for sample_idx in range(int(args.num_samples))
        if sample_idx % world_size == rank
    ]
    attempts_path = rank_path(
        args.output_dir,
        "planner_attempts.jsonl",
        rank,
        distributed,
    )
    local_attempts: list[dict[str, Any]] = []
    start = time.time()
    progress = tqdm(
        assigned,
        desc=f"H1-A2 {args.planner_arm} Planner rank{rank}",
        disable=distributed and not is_main,
    )
    for sample_idx in progress:
        ordinal = build_factorial_ordinal_record(
            int(args.base_seed),
            sample_idx=int(sample_idx),
        )
        sampling_seed = int(ordinal["planner_sampling_seed"])
        torch.manual_seed(sampling_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sampling_seed)

        prompt = str(input_contract["prompt_text"])
        encoded = tokenizer(
            [prompt],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        observed_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
        if observed_ids != list(input_contract["input_ids"]):
            raise ValueError("runtime Planner input IDs changed from frozen contract")
        input_ids = encoded["input_ids"].to(model_device(model))
        attention_mask = encoded["attention_mask"].to(model_device(model))
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [GeneratedPlanEndStoppingCriteria(tokenizer, input_ids.shape[1])]
                ),
            )
        generated_ids = outputs[0, input_ids.shape[1] :]
        decoded = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        strict_text = (
            str(decoded)
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )
        try:
            record = persist_model_sampled_plan(
                strict_text,
                planner_arm=args.planner_arm,
                sample_idx=int(sample_idx),
                planner_sampling_seed=sampling_seed,
                planner_input_contract=input_contract,
                max_atoms=int(args.max_atoms),
            )
            record.update(
                {
                    "attempt_status": "complete",
                    "earliest_failure_stage": None,
                    "raw_model_continuation_text": decoded,
                    "raw_model_continuation_token_ids": [
                        int(value) for value in generated_ids.tolist()
                    ],
                    "evaluation_order": int(ordinal["evaluation_order"]),
                    "body_sampling_seed": int(ordinal["body_sampling_seed"]),
                    "refiner_sampling_seed": int(
                        ordinal["refiner_sampling_seed"]
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            record = {
                "schema": "h1a2_factorial_contract_v1",
                "sample_idx": int(sample_idx),
                "evaluation_order": int(ordinal["evaluation_order"]),
                "planner_arm": args.planner_arm,
                "checkpoint_sha256": args.checkpoint_sha256,
                "attempt_status": "failed",
                "earliest_failure_stage": "planner",
                "failure_reason": type(exc).__name__,
                "failure_message": str(exc),
                "raw_model_continuation_text": decoded,
                "raw_model_continuation_token_ids": [
                    int(value) for value in generated_ids.tolist()
                ],
                "planner_prompt_sha256": input_contract["prompt_sha256"],
                "planner_input_ids_sha256": input_contract["input_ids_sha256"],
                "planner_tokenizer_identity_sha256": input_contract[
                    "tokenizer_identity_sha256"
                ],
                "planner_sampling_seed": sampling_seed,
                "body_sampling_seed": int(ordinal["body_sampling_seed"]),
                "refiner_sampling_seed": int(ordinal["refiner_sampling_seed"]),
                "retry_used": False,
                "replacement_used": False,
                "repair_used": False,
                "filter_used": False,
                "rerank_used": False,
            }
        local_attempts.append(record)

    _write_jsonl(attempts_path, local_attempts)
    elapsed = time.time() - start
    write_json(
        str(rank_path(args.output_dir, "planner_metrics.json", rank, distributed)),
        _metrics(
            local_attempts,
            requested_samples=len(assigned),
            elapsed=elapsed,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        ),
    )

    if distributed:
        dist.barrier()
        if is_main:
            merged = _merge_rank_attempts(
                args.output_dir,
                world_size=world_size,
                expected_count=int(args.num_samples),
                planner_arm=args.planner_arm,
            )
            write_json(
                str(args.output_dir / "planner_metrics.json"),
                _metrics(
                    list(merged),
                    requested_samples=int(args.num_samples),
                    elapsed=elapsed,
                    rank=0,
                    world_size=world_size,
                    distributed=True,
                ),
            )
        dist.barrier()
        dist.destroy_process_group()
    else:
        ordered = ordered_planner_attempts(
            local_attempts,
            expected_count=int(args.num_samples),
            expected_planner_arm=args.planner_arm,
        )
        _write_jsonl(
            args.output_dir / "plans_for_body.jsonl",
            [
                record
                for record in ordered
                if record.get("attempt_status") == "complete"
            ],
        )


if __name__ == "__main__":
    main()
