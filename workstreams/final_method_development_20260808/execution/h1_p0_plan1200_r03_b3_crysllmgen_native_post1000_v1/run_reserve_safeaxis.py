#!/usr/bin/env python3
"""Generate every frozen reserve plan once with the V3 D2 safe-axis body."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from tqdm import tqdm

from native_protocol import (
    PREFIX_COUNT,
    candidate_seed,
    canonical_sha256,
    ordered_candidate_rows,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    validate_arm,
    validate_repeat,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from protocol import require_file, validate_config
from run_body_safeaxis1000 import _model_device, _runtime
from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.h1a2_factorial_runtime import assert_body_tokenizer_identity
from crystal_dlm.r5_dynamic_length import (
    count_prefill_for_batch,
    exact_body_token_count,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
)
from crystal_dlm.r5_plan_state import build_body_prompt
from paired_llada import generate_paired_exact_plan
from safe_axis_schedule import (
    h1a2_safe_axis_generation_schedule,
    require_safe_axis_schedule,
)
from scripts.sample_llada_dynamic_crystals import (
    build_dynamic_lightweight_constraints,
    graph_from_arrays,
    import_process_one,
    load_model_and_tokenizer,
)
from scripts.sample_llada_r5_exact_length import (
    element_prefill_for_batch,
    merge_prefill_maps,
)


def _load_tasks(
    arm: str, repeat: int, candidate_pool: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = ordered_candidate_rows(read_jsonl(candidate_pool.resolve()))
    reserve = candidates[PREFIX_COUNT:]
    if not reserve:
        raise ValueError("frozen planner draw has no reserve parse-success plans")
    tasks: list[dict[str, Any]] = []
    invariants: list[dict[str, Any]] = []
    for row in reserve:
        rank = int(row["candidate_rank"])
        state = row.get("plan_state")
        if (
            rank < PREFIX_COUNT
            or not isinstance(state, Mapping)
            or int(row.get("repeat", -1)) != repeat
            or row.get("parsed") is not True
            or row.get("candidate_partition") != "frozen_reserve"
            or row.get("raw_rich_seven_line_forwarded") is not False
            or row.get("canonical_charge_bucket_visible") is not True
            or row.get("body_prompt_contract")
            != "historical_r5c_plan_state_json_exact_length"
            or int(row.get("body_noise_seed", -1))
            != candidate_seed(repeat, rank, "body")
            or int(row.get("refiner_noise_seed", -1))
            != candidate_seed(repeat, rank, "refiner")
        ):
            raise ValueError(f"reserve candidate contract changed at rank {rank}")
        plan = dict(state)
        prompt = build_body_prompt(plan)
        if row.get("body_prompt") != prompt or '"charge_bucket"' not in prompt:
            raise ValueError(f"body prompt changed at reserve rank {rank}")
        schedule = h1a2_safe_axis_generation_schedule(plan)
        invariant = require_safe_axis_schedule(schedule, num_atoms=int(plan["N"]))
        if (
            invariant.get("gate_passed") is not True
            or int(invariant.get("z_before_xy_count", -1)) != 0
            or invariant.get("all_xy_precede_all_z") is not True
            or int(invariant.get("mixed_axis_coordinate_groups", -1)) != 0
        ):
            raise ValueError(f"safe-axis invariant failed at reserve rank {rank}")
        plan_sha = canonical_sha256(plan)
        schedule_sha = canonical_sha256(schedule)
        tasks.append(
            {
                "ordinal": rank,
                "candidate_rank": rank,
                "sample_idx": rank,
                "planner": "P0",
                "body": arm,
                "arm": arm,
                "repeat": repeat,
                "pair_id": f"h1-plan1200-native-candidate-r{repeat}:{rank:04d}",
                "planner_attempt_id": str(row["candidate_id"]),
                "planner_candidate_ordinal": int(row["planner_candidate_ordinal"]),
                "planner_sampling_seed": row.get("planner_sampling_seed"),
                "body_noise_seed": candidate_seed(repeat, rank, "body"),
                "refiner_noise_seed": candidate_seed(repeat, rank, "refiner"),
                "plan_state": plan,
                "plan_state_sha256": plan_sha,
                "body_prompt": prompt,
                "body_prompt_sha256": sha256_text(prompt),
                "schedule": schedule,
                "schedule_sha256": schedule_sha,
                "schedule_invariant": invariant,
                "schedule_invariant_sha256": canonical_sha256(invariant),
            }
        )
        invariants.append(
            {
                "candidate_rank": rank,
                "plan_state_sha256": plan_sha,
                "schedule_sha256": schedule_sha,
                "invariant_sha256": canonical_sha256(invariant),
            }
        )
    return tasks, {
        "schema": "h1_plan1200_native_reserve_input_report_v1",
        "arm": arm,
        "repeat": repeat,
        "candidate_pool_sha256": sha256_file(candidate_pool),
        "reserve_attempts": len(tasks),
        "candidate_ranks": [int(task["candidate_rank"]) for task in tasks],
        "candidate_order": "planner_parse_success_rank_by_planner_ordinal",
        "seed_mode": "extended_frozen_candidate_rank_v1",
        "all_safe_axis_invariants_passed": True,
        "schedule_invariants": invariants,
        "same_plan_retry": False,
    }


def _base_record(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "h1_plan1200_native_reserve_body_attempt_v1",
        "arm": task["arm"],
        "repeat": int(task["repeat"]),
        "ordinal": int(task["candidate_rank"]),
        "candidate_rank": int(task["candidate_rank"]),
        "sample_idx": int(task["candidate_rank"]),
        "evaluation_order": int(task["candidate_rank"]),
        "pair_id": task["pair_id"],
        "planner_attempt_id": task["planner_attempt_id"],
        "planner_candidate_ordinal": int(task["planner_candidate_ordinal"]),
        "planner_arm": "P0",
        "body_checkpoint_arm": task["arm"],
        "schedule_arm": "D2_SAFE_AXIS",
        "generation_policy": "d2_safe_axis",
        "body_noise_seed": int(task["body_noise_seed"]),
        "refiner_noise_seed": int(task["refiner_noise_seed"]),
        "planner_sampling_seed": task.get("planner_sampling_seed"),
        "plan_state_sha256": task["plan_state_sha256"],
        "num_atoms": int(task["plan_state"]["N"]),
        "expected_body_token_count": exact_body_token_count(task["plan_state"]),
        "actual_body_token_count": None,
        "exact_length_match": None,
        "body_prompt_sha256": task["body_prompt_sha256"],
        "schedule_sha256": task["schedule_sha256"],
        "schedule_invariant_sha256": task["schedule_invariant_sha256"],
        "status": "failed",
        "reason": "",
        "body_generation_complete": False,
        "body_plan_match": False,
        "body_graph_complete": False,
        "retry_used": False,
        "replacement_used": False,
        "repair_used": False,
        "filter_used": False,
        "rerank_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--body-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    arm = validate_arm(args.arm)
    repeat = validate_repeat(args.repeat)
    config = read_json(args.body_config.resolve())
    validate_config(config)
    device = _runtime()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tasks, input_report = _load_tasks(arm, repeat, args.candidate_pool)
    write_json_exclusive(output / "input_report.json", input_report)

    body = config["body"]
    model_spec = body["models"][arm]
    checkpoint = Path(model_spec["checkpoint"]).resolve()
    adapter = checkpoint / body["adapter_file"]
    if (
        not adapter.is_file()
        or adapter.stat().st_size != int(body["adapter_expected_bytes"])
    ):
        raise ValueError(f"{arm} adapter path or size changed")
    require_file(
        checkpoint / "tokenizer.json",
        body["tokenizer_json_sha256"],
        f"{arm} tokenizer.json",
    )
    require_file(
        checkpoint / "tokenizer_config.json",
        body["tokenizer_config_sha256"],
        f"{arm} tokenizer_config.json",
    )
    model, tokenizer = load_model_and_tokenizer(
        str(Path(body["base_model"]).resolve()), str(checkpoint), device
    )
    if tokenizer.pad_token_id == MASK_TOKEN_ID:
        raise RuntimeError("tokenizer pad token collides with LLaDA mask token")
    tokenizer_identity = assert_body_tokenizer_identity(
        tokenizer, expected_vocab_sha256=body["tokenizer_vocab_sha256"]
    )
    if int(tokenizer_identity["vocab_size"]) != int(body["tokenizer_size"]):
        raise ValueError("body tokenizer size changed")
    write_json_exclusive(output / "body_tokenizer_identity.json", tokenizer_identity)

    process_one = import_process_one(args.crysllmgen_dir.resolve())
    lightweight = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=bool(body["duplicate_coordinate_mask"]),
        lattice_volume_mask=bool(body["lattice_volume_mask"]),
        min_lattice_rad=float(body["min_lattice_rad"]),
    )
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        buckets[(int(task["plan_state"]["N"]), task["schedule_sha256"])].append(task)
    batches: list[list[dict[str, Any]]] = []
    max_batch = int(body["max_batch_size"])
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda item: int(item["candidate_rank"]))
        for offset in range(0, len(bucket), max_batch):
            batches.append(bucket[offset : offset + max_batch])
    partition = [
        [int(task["candidate_rank"]) for task in batch] for batch in batches
    ]
    expected_ranks = sorted(int(task["candidate_rank"]) for task in tasks)
    if sorted(value for batch in partition for value in batch) != expected_ranks:
        raise ValueError("reserve batch partition lost or duplicated a candidate")
    write_json_exclusive(
        output / "batch_partition.json",
        {
            "schema": "h1_plan1200_native_reserve_partition_v1",
            "arm": arm,
            "repeat": repeat,
            "partition": partition,
            "partition_sha256": canonical_sha256(partition),
            "candidate_ranks": expected_ranks,
        },
    )

    attempts: dict[int, dict[str, Any]] = {}
    proposal_graphs: list[dict[str, Any]] = []
    started = time.monotonic()
    progress = tqdm(total=len(tasks), desc=f"{arm} repeat{repeat} native reserve body")
    for batch in batches:
        schedule = batch[0]["schedule"]
        if any(task["schedule"] != schedule for task in batch):
            raise ValueError("safe-axis generation batch is not homogeneous")
        num_atoms = int(batch[0]["plan_state"]["N"])
        prompts = [str(task["body_prompt"]) for task in batch]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(_model_device(model))
        attention_mask = encoded["attention_mask"].to(_model_device(model))
        allowed = exact_dynamic_schema_constraints(tokenizer, num_atoms)
        prefill = merge_prefill_maps(
            count_prefill_for_batch(tokenizer, num_atoms, len(batch)),
            element_prefill_for_batch(
                tokenizer, [task["plan_state"] for task in batch]
            ),
        )
        outputs = generate_paired_exact_plan(
            model,
            input_ids,
            base_seeds=[int(task["body_noise_seed"]) for task in batch],
            attention_mask=attention_mask,
            gen_length=exact_body_token_count(num_atoms),
            temperature=float(body["temperature"]),
            cfg_scale=float(body["cfg_scale"]),
            remasking=str(body["remasking"]),
            mask_id=MASK_TOKEN_ID,
            allowed_token_ids_by_generation_pos=allowed,
            prefill_token_ids_by_generation_pos=prefill,
            generation_position_groups=schedule,
            lightweight_decoding_constraints=lightweight,
        )
        generated = outputs[:, input_ids.shape[1] :]
        texts = tokenizer.batch_decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for task, token_ids, text in zip(batch, generated, texts, strict=True):
            rank = int(task["candidate_rank"])
            record = _base_record(task)
            record.update(
                {
                    "body_generation_complete": True,
                    "raw_body_token_ids": [
                        int(value) for value in token_ids.detach().cpu().tolist()
                    ],
                    "text": text,
                    "raw_body_text_sha256": sha256_text(text),
                }
            )
            record["actual_body_token_count"] = len(record["raw_body_token_ids"])
            record["exact_length_match"] = (
                record["actual_body_token_count"]
                == record["expected_body_token_count"]
            )
            if record["exact_length_match"] is not True:
                raise ValueError(f"exact-length mismatch at reserve rank {rank}")
            stage = "body_parse"
            try:
                arrays = validate_answer_matches_plan(task["plan_state"], text)
                record["body_plan_match"] = True
                stage = "body_graph"
                graph, cif = graph_from_arrays(arrays, process_one)
                graph["h1_plan1200_prepost_metadata"] = {
                    "arm": arm,
                    "repeat": repeat,
                    "ordinal": rank,
                    "candidate_rank": rank,
                    "planner": "P0",
                    "body": arm,
                    "schedule_arm": "D2_SAFE_AXIS",
                    "generation_policy": "d2_safe_axis",
                    "body_noise_seed": int(task["body_noise_seed"]),
                    "plan_state_sha256": task["plan_state_sha256"],
                    "body_prompt_sha256": task["body_prompt_sha256"],
                    "schedule_sha256": task["schedule_sha256"],
                }
                proposal_graphs.append(
                    {
                        "ordinal": rank,
                        "candidate_rank": rank,
                        "arm": arm,
                        "repeat": repeat,
                        "graph": graph,
                    }
                )
                record.update(
                    {
                        "status": "succeeded",
                        "reason": "",
                        "earliest_failure_stage": None,
                        "body_graph_complete": True,
                        "arrays": arrays,
                        "proposal_cif_sha256": canonical_sha256(cif),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                record.update(
                    {
                        "status": "failed",
                        "reason": f"body:{type(exc).__name__}:{exc}",
                        "earliest_failure_stage": stage,
                        "arrays": None,
                    }
                )
            attempts[rank] = record
            progress.update(1)
    progress.close()
    if sorted(attempts) != expected_ranks:
        raise ValueError("reserve generation lost or duplicated a candidate")
    ordered = [attempts[rank] for rank in expected_ranks]
    attempts_path = output / "reserve_body_attempts.jsonl"
    graphs_path = output / "reserve_proposal_graphs.pt"
    write_jsonl_exclusive(attempts_path, ordered)
    with graphs_path.open("xb") as handle:
        torch.save(
            sorted(proposal_graphs, key=lambda row: int(row["candidate_rank"])),
            handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    report = {
        "schema": "h1_plan1200_native_reserve_body_report_v1",
        "status": "complete",
        "arm": arm,
        "repeat": repeat,
        "attempts": len(ordered),
        "succeeded": sum(row["status"] == "succeeded" for row in ordered),
        "failed": sum(row["status"] != "succeeded" for row in ordered),
        "candidate_ranks": expected_ranks,
        "body_attempts_sha256": sha256_file(attempts_path),
        "proposal_graphs_sha256": sha256_file(graphs_path),
        "body_checkpoint": str(checkpoint),
        "body_adapter_sha256_recorded": model_spec["adapter_sha256"],
        "schedule": "D2_SAFE_AXIS",
        "all_safe_axis_invariants_passed": True,
        "walltime_s": time.monotonic() - started,
        "same_plan_retry": False,
        "stochastic_replacement": False,
        "repair_filter_rerank": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_exclusive(output / "reserve_generation_report.json", report)
    del model, tokenizer, process_one
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
