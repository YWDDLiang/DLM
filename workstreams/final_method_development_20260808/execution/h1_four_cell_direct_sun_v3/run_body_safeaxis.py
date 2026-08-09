#!/usr/bin/env python3
"""Generate one current Planner/Body cell with the frozen R03 safe-axis path."""

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

from protocol import (
    DENOMINATOR,
    canonical_sha256,
    ordered_rows,
    read_json,
    read_jsonl,
    require_file,
    sha256_file,
    sha256_text,
    validate_cell,
    validate_config,
    write_json_exclusive,
    write_jsonl_exclusive,
)
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


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _runtime() -> torch.device:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("registered body generation must run through Slurm")
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu":
        raise RuntimeError("body generation requires the gpu partition")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 8:
        raise RuntimeError("body generation requires exactly eight CPUs")
    if os.environ.get("CONDA_DEFAULT_ENV") != "diff_meets_diff":
        raise RuntimeError("body generation requires diff_meets_diff")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("body generation requires exactly one CUDA device")
    name = torch.cuda.get_device_name(0)
    if "A800" not in name:
        raise RuntimeError(f"body generation requires A800, observed {name}")
    return torch.device("cuda", 0)


def _load_tasks(
    config: Mapping[str, Any], cell: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cell_spec = config["cells"][cell]
    planner = str(cell_spec["planner"])
    planner_spec = config["planner_sources"][planner]
    raw_path = require_file(
        planner_spec["raw_generations"],
        planner_spec["raw_generations_sha256"],
        f"{planner} planner raw256",
    )
    raw_rows = ordered_rows(read_jsonl(raw_path), ordinal_field="sample_idx")
    ledger_spec = config["attempt_ledger"]
    ledger_path = require_file(
        ledger_spec["path"], ledger_spec["sha256"], "common H1 attempt ledger"
    )
    seed_rows = ordered_rows(read_jsonl(ledger_path), ordinal_field="sample_idx")

    tasks: list[dict[str, Any]] = []
    parsed = 0
    ineligible: list[int] = []
    invariants: list[dict[str, Any]] = []
    for ordinal, (row, seed_row) in enumerate(zip(raw_rows, seed_rows, strict=True)):
        if int(seed_row.get("ordinal", -1)) != ordinal:
            raise ValueError(f"seed ledger ordinal changed at {ordinal}")
        body_seed = seed_row.get(ledger_spec["body_seed_field"])
        refiner_seed = seed_row.get(ledger_spec["refiner_seed_field"])
        if not isinstance(body_seed, int) or not isinstance(refiner_seed, int):
            raise ValueError(f"seed fields changed at ordinal {ordinal}")
        base = {
            "ordinal": ordinal,
            "sample_idx": ordinal,
            "planner": planner,
            "body": str(cell_spec["body"]),
            "cell": cell,
            "pair_id": f"h1-ef-fourcell:{ordinal:04d}",
            "planner_attempt_id": str(
                row.get("attempt_id") or f"{planner.lower()}-{ordinal:04d}"
            ),
            "planner_sampling_seed": row.get("planner_sampling_seed"),
            "body_noise_seed": int(body_seed),
            "refiner_noise_seed": int(refiner_seed),
        }
        if row.get("parsed") is not True:
            ineligible.append(ordinal)
            tasks.append(
                {
                    **base,
                    "eligible": False,
                    "reason": str(
                        row.get("parse_error")
                        or row.get("failure_reason")
                        or "planner_parse_failed"
                    ),
                    "plan_state_sha256": None,
                }
            )
            continue
        plan_state = row.get("plan_state")
        if not isinstance(plan_state, Mapping):
            raise ValueError(f"parsed planner row lacks plan_state at {ordinal}")
        parsed += 1
        plan = dict(plan_state)
        prompt = build_body_prompt(plan).rstrip() + "\n"
        if row.get("prompt") is not None and str(row["prompt"]) != prompt:
            raise ValueError(f"frozen body prompt changed at ordinal {ordinal}")
        schedule = h1a2_safe_axis_generation_schedule(plan)
        invariant = require_safe_axis_schedule(schedule, num_atoms=int(plan["N"]))
        if (
            invariant.get("gate_passed") is not True
            or int(invariant.get("z_before_xy_count", -1)) != 0
            or invariant.get("all_xy_precede_all_z") is not True
            or int(invariant.get("mixed_axis_coordinate_groups", -1)) != 0
        ):
            raise ValueError(f"safe-axis invariant failed at ordinal {ordinal}")
        plan_sha = canonical_sha256(plan)
        schedule_sha = canonical_sha256(schedule)
        tasks.append(
            {
                **base,
                "eligible": True,
                "reason": "",
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
                "ordinal": ordinal,
                "plan_state_sha256": plan_sha,
                "schedule_sha256": schedule_sha,
                "invariant": invariant,
                "invariant_sha256": canonical_sha256(invariant),
            }
        )
    if parsed != int(planner_spec["expected_parsed"]):
        raise ValueError(
            f"{planner} parsed count changed: expected "
            f"{planner_spec['expected_parsed']} observed {parsed}"
        )
    report = {
        "schema": "h1_ef_fourcell_input_report_v1",
        "cell": cell,
        "planner": planner,
        "body": cell_spec["body"],
        "attempts": DENOMINATOR,
        "parsed": parsed,
        "planner_ineligible": DENOMINATOR - parsed,
        "ineligible_ordinals": ineligible,
        "planner_raw_sha256": sha256_file(raw_path),
        "attempt_ledger_sha256": sha256_file(ledger_path),
        "schedule_invariants": invariants,
        "all_safe_axis_invariants_passed": True,
    }
    return tasks, report


def _base_record(task: Mapping[str, Any], cell_spec: Mapping[str, Any]) -> dict[str, Any]:
    eligible = bool(task["eligible"])
    return {
        "schema": "h1_ef_fourcell_body_attempt_v1",
        "cell": task["cell"],
        "ordinal": int(task["ordinal"]),
        "sample_idx": int(task["ordinal"]),
        "evaluation_order": int(task["ordinal"]),
        "pair_id": task["pair_id"],
        "planner_attempt_id": task["planner_attempt_id"],
        "planner_arm": cell_spec["planner"],
        "body_checkpoint_arm": cell_spec["body"],
        "schedule_arm": "D2_SAFE_AXIS",
        "generation_policy": "d2_safe_axis",
        "body_noise_seed": int(task["body_noise_seed"]),
        "refiner_noise_seed": int(task["refiner_noise_seed"]),
        "planner_sampling_seed": task.get("planner_sampling_seed"),
        "plan_state_sha256": task.get("plan_state_sha256"),
        "body_eligible": eligible,
        "num_atoms": int(task["plan_state"]["N"]) if eligible else None,
        "expected_body_token_count": (
            exact_body_token_count(task["plan_state"]) if eligible else None
        ),
        "actual_body_token_count": None,
        "exact_length_match": None,
        "body_prompt_sha256": task.get("body_prompt_sha256"),
        "schedule_sha256": task.get("schedule_sha256"),
        "schedule_invariant_sha256": task.get("schedule_invariant_sha256"),
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
    parser.add_argument("--cell", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cell = validate_cell(args.cell)
    config = read_json(args.config.resolve())
    validate_config(config)
    device = _runtime()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tasks, input_report = _load_tasks(config, cell)
    write_json_exclusive(output / "input_report.json", input_report)

    cell_spec = config["cells"][cell]
    body = config["body"]
    model_spec = body["models"][cell_spec["body"]]
    checkpoint = Path(model_spec["checkpoint"]).resolve()
    adapter = checkpoint / body["adapter_file"]
    if (
        not adapter.is_file()
        or adapter.stat().st_size != int(body["adapter_expected_bytes"])
    ):
        raise ValueError(f"{cell_spec['body']} adapter path or size changed")
    require_file(
        checkpoint / "tokenizer.json",
        body["tokenizer_json_sha256"],
        f"{cell_spec['body']} tokenizer.json",
    )
    require_file(
        checkpoint / "tokenizer_config.json",
        body["tokenizer_config_sha256"],
        f"{cell_spec['body']} tokenizer_config.json",
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

    crysllmgen_dir = Path(
        "/public/home/jiaosz/ywliang/ai4s/"
        "diffsion_language_model_meets_diffusion/reference/crysllmgen"
    )
    process_one = import_process_one(crysllmgen_dir)
    lightweight = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=bool(body["duplicate_coordinate_mask"]),
        lattice_volume_mask=bool(body["lattice_volume_mask"]),
        min_lattice_rad=float(body["min_lattice_rad"]),
    )

    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task["eligible"]:
            buckets[(int(task["plan_state"]["N"]), task["schedule_sha256"])].append(task)
    batches: list[list[dict[str, Any]]] = []
    max_batch = int(body["max_batch_size"])
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda item: int(item["ordinal"]))
        for offset in range(0, len(bucket), max_batch):
            batches.append(bucket[offset : offset + max_batch])
    partition = [[int(task["ordinal"]) for task in batch] for batch in batches]
    eligible_ordinals = [value for batch in partition for value in batch]
    ineligible_ordinals = [int(task["ordinal"]) for task in tasks if not task["eligible"]]
    if sorted([*eligible_ordinals, *ineligible_ordinals]) != list(range(DENOMINATOR)):
        raise ValueError("body batch partition lost or duplicated an ordinal")
    write_json_exclusive(
        output / "batch_partition.json",
        {
            "schema": "h1_ef_fourcell_body_partition_v1",
            "cell": cell,
            "partition": partition,
            "partition_sha256": canonical_sha256(partition),
            "eligible_ordinals": eligible_ordinals,
            "ineligible_ordinals": ineligible_ordinals,
        },
    )

    attempts: dict[int, dict[str, Any]] = {}
    proposal_graphs: list[dict[str, Any]] = []
    for task in tasks:
        if task["eligible"]:
            continue
        record = _base_record(task, cell_spec)
        record.update(
            {
                "reason": str(task["reason"]),
                "earliest_failure_stage": "planner",
            }
        )
        attempts[int(task["ordinal"])] = record

    started = time.monotonic()
    progress = tqdm(total=len(eligible_ordinals), desc=f"{cell} safe-axis body256")
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
            ordinal = int(task["ordinal"])
            record = _base_record(task, cell_spec)
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
                raise ValueError(f"exact-length mismatch at ordinal {ordinal}")
            stage = "body_parse"
            try:
                arrays = validate_answer_matches_plan(task["plan_state"], text)
                record["body_plan_match"] = True
                stage = "body_graph"
                graph, cif = graph_from_arrays(arrays, process_one)
                graph["h1_ef_fourcell_metadata"] = {
                    "cell": cell,
                    "ordinal": ordinal,
                    "planner": cell_spec["planner"],
                    "body": cell_spec["body"],
                    "schedule_arm": "D2_SAFE_AXIS",
                    "generation_policy": "d2_safe_axis",
                    "body_noise_seed": int(task["body_noise_seed"]),
                    "plan_state_sha256": task["plan_state_sha256"],
                    "body_prompt_sha256": task["body_prompt_sha256"],
                    "schedule_sha256": task["schedule_sha256"],
                }
                proposal_graphs.append(
                    {
                        "ordinal": ordinal,
                        "cell": cell,
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
            attempts[ordinal] = record
            progress.update(1)
    progress.close()

    if sorted(attempts) != list(range(DENOMINATOR)):
        raise ValueError("body generation lost or duplicated an ordinal")
    ordered = [attempts[ordinal] for ordinal in range(DENOMINATOR)]
    attempts_path = output / "body_attempts.jsonl"
    graphs_path = output / "proposal_graphs.pt"
    write_jsonl_exclusive(attempts_path, ordered)
    with graphs_path.open("xb") as handle:
        torch.save(
            sorted(proposal_graphs, key=lambda row: int(row["ordinal"])), handle
        )
        handle.flush()
        os.fsync(handle.fileno())
    failures: dict[str, int] = {}
    for record in ordered:
        if record["status"] == "succeeded":
            continue
        reason = str(record.get("reason") or "unknown")
        key = ":".join(reason.split(":")[:2])
        failures[key] = failures.get(key, 0) + 1
    report = {
        "schema": "h1_ef_fourcell_body_generation_report_v1",
        "status": "complete",
        "cell": cell,
        "planner": cell_spec["planner"],
        "body": cell_spec["body"],
        "attempts": DENOMINATOR,
        "body_eligible": len(eligible_ordinals),
        "planner_ineligible": len(ineligible_ordinals),
        "succeeded": sum(row["status"] == "succeeded" for row in ordered),
        "failed": sum(row["status"] != "succeeded" for row in ordered),
        "failure_classes": failures,
        "body_attempts_sha256": sha256_file(attempts_path),
        "proposal_graphs_sha256": sha256_file(graphs_path),
        "body_checkpoint": str(checkpoint),
        "body_adapter_sha256_recorded": model_spec["adapter_sha256"],
        "schedule": "D2_SAFE_AXIS",
        "generation_policy": "d2_safe_axis",
        "all_safe_axis_invariants_passed": True,
        "batch_partition_sha256": canonical_sha256(partition),
        "walltime_s": time.monotonic() - started,
        "retry_replacement_repair_filter_rerank": False,
        "automatic_downstream": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_exclusive(output / "generation_report.json", report)
    del model, tokenizer, process_one
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
