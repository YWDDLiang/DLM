#!/usr/bin/env python3
"""Two-stage chemical-plan then fixed-slot LLaDA crystal sampler."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.chemical_plan import PLAN_PROMPT, build_plan_conditioned_prompt, truncate_generated_plan
from crystal_dlm.fixed_slot import ANSWER_TOKEN_COUNT, MASK_TOKEN_ID, PROMPT_POOL, arrays_to_structure, arrays_to_torch_payload, parse_fixed_slot_answer, write_json
from crystal_dlm.generation_schedule import n_elements_coords_lattice_schedule, n_elements_sequential_rest_schedule
from crystal_dlm.llada_generation import generate
from scripts.sample_llada_crystals import (
    build_atom_count_grammar,
    build_lightweight_decoding_constraints,
    build_schema_generation_constraints,
    graph_from_arrays,
    import_process_one,
    init_distributed,
    load_atom_count_prior,
    load_model_and_tokenizer,
    merge_distributed_outputs,
    rank_path,
    sample_atom_count,
    atom_count_token_id,
    write_valid_arrays,
)


def sample_plan_batch(
    *,
    model,
    tokenizer,
    prompts: List[str],
    plan_gen_length: int,
    plan_block_length: int,
    plan_steps: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    allowed_token_ids_by_generation_pos: List[List[int]] | None = None,
) -> List[str]:
    encoded = tokenizer(
        [prompt.rstrip() + "\n" for prompt in prompts],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    outputs = generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        steps=plan_steps,
        gen_length=plan_gen_length,
        block_length=plan_block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=MASK_TOKEN_ID,
        allowed_token_ids_by_generation_pos=allowed_token_ids_by_generation_pos,
    )
    generated_ids = outputs[:, input_ids.shape[1] :]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    return [truncate_generated_plan(text) for text in decoded]


def build_plan_allowed_token_ids(tokenizer, gen_length: int, ban_crystal_special_tokens: bool) -> List[List[int]] | None:
    if not ban_crystal_special_tokens:
        return None
    banned: set[int] = set()
    for token, token_id in tokenizer.get_vocab().items():
        if (
            token.startswith("<N_")
            or token.startswith("<E_")
            or token.startswith("<X_")
            or token.startswith("<Y_")
            or token.startswith("<Z_")
            or token.startswith("<LA_")
            or token.startswith("<LB_")
            or token.startswith("<LC_")
            or token.startswith("<AA_")
            or token.startswith("<AB_")
            or token.startswith("<AG_")
            or token.startswith("<S")
            or token in {"<EMPTY>", "<X_PAD>", "<Y_PAD>", "<Z_PAD>"}
        ):
            banned.add(int(token_id))
    allowed = [token_id for token_id in range(len(tokenizer)) if token_id not in banned]
    return [allowed] * int(gen_length)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=ANSWER_TOKEN_COUNT)
    parser.add_argument("--gen-length", type=int, default=ANSWER_TOKEN_COUNT)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--plan-prompt", default=PLAN_PROMPT)
    parser.add_argument("--plan-gen-length", type=int, default=96)
    parser.add_argument("--plan-block-length", type=int, default=16)
    parser.add_argument("--plan-steps", type=int, default=96)
    parser.add_argument("--plan-temperature", type=float, default=0.7)
    parser.add_argument("--plan-ban-crystal-special-tokens", action="store_true", default=True)
    parser.add_argument("--no-plan-ban-crystal-special-tokens", dest="plan_ban_crystal_special_tokens", action="store_false")
    parser.add_argument(
        "--generation-schedule",
        choices=["default", "n-elements-coords-lattice", "n-elements-sequential-rest"],
        default="n-elements-sequential-rest",
    )
    parser.add_argument("--schema-logit-mask", action="store_true", default=True)
    parser.add_argument("--no-schema-logit-mask", dest="schema_logit_mask", action="store_false")
    parser.add_argument("--prefill-slot-tokens", action="store_true", default=True)
    parser.add_argument("--no-prefill-slot-tokens", dest="prefill_slot_tokens", action="store_false")
    parser.add_argument("--duplicate-coordinate-mask", action="store_true", default=True)
    parser.add_argument("--no-duplicate-coordinate-mask", dest="duplicate_coordinate_mask", action="store_false")
    parser.add_argument("--lattice-volume-mask", action="store_true", default=True)
    parser.add_argument("--no-lattice-volume-mask", dest="lattice_volume_mask", action="store_false")
    parser.add_argument("--min-lattice-rad", type=float, default=1e-4)
    parser.add_argument("--atom-count-grammar-mask", action="store_true", default=True)
    parser.add_argument("--no-atom-count-grammar-mask", dest="atom_count_grammar_mask", action="store_false")
    parser.add_argument("--prefill-atom-count-prior", choices=["none", "uniform", "train", "val", "test"], default="none")
    parser.add_argument("--atom-count-stats-json", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20/stats.json")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--target-graph-success", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args()

    if args.plan_gen_length % args.plan_block_length != 0:
        raise RuntimeError("--plan-gen-length must be divisible by --plan-block-length")
    if args.plan_steps % (args.plan_gen_length // args.plan_block_length) != 0:
        raise RuntimeError("--plan-steps must be divisible by the number of plan blocks")
    if args.generation_schedule != "default" and not args.prefill_slot_tokens:
        raise RuntimeError(f"--generation-schedule {args.generation_schedule} requires --prefill-slot-tokens.")
    if args.generation_schedule != "default" and not args.atom_count_grammar_mask:
        raise RuntimeError(f"--generation-schedule {args.generation_schedule} requires --atom-count-grammar-mask.")
    if (args.duplicate_coordinate_mask or args.lattice_volume_mask) and args.block_length != 1:
        raise RuntimeError("--duplicate-coordinate-mask/--lattice-volume-mask require --block-length 1.")

    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_one = import_process_one(args.crysllmgen_dir)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, dist_info["device"])
    plan_allowed_token_ids_by_generation_pos = build_plan_allowed_token_ids(
        tokenizer,
        args.plan_gen_length,
        args.plan_ban_crystal_special_tokens,
    )

    allowed_token_ids_by_generation_pos = None
    prefill_token_ids_by_generation_pos = None
    if args.schema_logit_mask or args.prefill_slot_tokens:
        allowed_token_ids_by_generation_pos, slot_prefill = build_schema_generation_constraints(tokenizer)
        if not args.schema_logit_mask:
            allowed_token_ids_by_generation_pos = None
        if args.prefill_slot_tokens:
            prefill_token_ids_by_generation_pos = slot_prefill
    atom_count_grammar = build_atom_count_grammar(tokenizer) if args.atom_count_grammar_mask else None
    lightweight_decoding_constraints = build_lightweight_decoding_constraints(
        tokenizer,
        duplicate_coordinate_mask=args.duplicate_coordinate_mask,
        lattice_volume_mask=args.lattice_volume_mask,
        min_lattice_rad=args.min_lattice_rad,
    )
    generation_position_groups = None
    if args.generation_schedule == "n-elements-coords-lattice":
        generation_position_groups = n_elements_coords_lattice_schedule()
    elif args.generation_schedule == "n-elements-sequential-rest":
        generation_position_groups = n_elements_sequential_rest_schedule()

    atom_count_values, atom_count_weights = load_atom_count_prior(
        args.prefill_atom_count_prior,
        args.atom_count_stats_json,
    )

    if is_main:
        write_json(
            str(args.output_dir / "run_config.json"),
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            | {"distributed": distributed, "world_size": world_size, "sampler": "chemical_plan_two_stage"},
        )
        write_json(str(args.output_dir / "prompt_pool.json"), {"prompt_pool": PROMPT_POOL, "plan_prompt": args.plan_prompt})
        write_json(
            str(args.output_dir / "tokenizer_report.json"),
            {
                "model_path": args.model_path,
                "checkpoint_path": args.checkpoint_path,
                "vocab_size": len(tokenizer),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "mask_token_id": MASK_TOKEN_ID,
            },
        )
        with (args.output_dir / "training_log.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "not_applicable", "stage": "chemical_plan_sampling"}) + "\n")

    raw_path = rank_path(args.output_dir, "raw_generations.jsonl", rank, distributed)
    failure_path = rank_path(args.output_dir, "failure_cases.jsonl", rank, distributed)
    valid_arrays_path = rank_path(args.output_dir, "valid_arrays.jsonl", rank, distributed)
    valid_arrays: List[Dict[str, Any]] = []
    proposal_graphs: List[Dict[str, Any]] = []

    target_mode = args.target_graph_success is not None
    if target_mode:
        target_graph_success = int(args.target_graph_success or 0)
        max_attempts = int(args.max_attempts or args.num_samples)
        base_target = target_graph_success // world_size
        target_remainder = target_graph_success % world_size
        local_target_graph_success = base_target + int(rank < target_remainder)
        sample_indices = list(range(rank, max_attempts, world_size))
    else:
        target_graph_success = None
        max_attempts = None
        local_target_graph_success = None
        sample_indices = list(range(rank, args.num_samples, world_size))
    metrics = {
        "requested_samples": len(sample_indices),
        "decoded_samples": 0,
        "parse_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "target_mode": target_mode,
        "target_graph_success": target_graph_success,
        "target_graph_success_assigned": local_target_graph_success,
        "target_reached": False,
        "max_attempts": max_attempts,
        "prefill_atom_count_prior": args.prefill_atom_count_prior,
        "target_atom_count_histogram": {},
        "failures": {},
        "time_sec": None,
        "rank": rank,
        "world_size": world_size,
        "assigned_samples": len(sample_indices),
    }

    start = time.time()
    with raw_path.open("w", encoding="utf-8") as raw_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        progress = tqdm(total=len(sample_indices), desc=f"ChemPlan sampling rank{rank}", disable=distributed and not is_main)
        for batch_start in range(0, len(sample_indices), args.batch_size):
            if local_target_graph_success is not None and metrics["graph_success"] >= local_target_graph_success:
                break
            current_indices = sample_indices[batch_start : batch_start + args.batch_size]
            current_batch = len(current_indices)
            plan_prompts = [args.plan_prompt] * current_batch
            plans = sample_plan_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=plan_prompts,
                plan_gen_length=args.plan_gen_length,
                plan_block_length=args.plan_block_length,
                plan_steps=args.plan_steps,
                temperature=args.plan_temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                allowed_token_ids_by_generation_pos=plan_allowed_token_ids_by_generation_pos,
            )
            prompts = [build_plan_conditioned_prompt(plan).rstrip() + "\n" for plan in plans]
            current_prefill = dict(prefill_token_ids_by_generation_pos or {})
            current_target_atom_counts: List[int | None] = [None] * current_batch
            if atom_count_values:
                current_target_atom_counts = [
                    sample_atom_count(sample_idx, atom_count_values, atom_count_weights, args.seed)
                    for sample_idx in current_indices
                ]
                current_prefill[0] = [atom_count_token_id(tokenizer, int(atom_count)) for atom_count in current_target_atom_counts]
                for atom_count in current_target_atom_counts:
                    key = str(int(atom_count))
                    metrics["target_atom_count_histogram"][key] = metrics["target_atom_count_histogram"].get(key, 0) + 1
            encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            outputs = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed_token_ids_by_generation_pos,
                prefill_token_ids_by_generation_pos=current_prefill,
                atom_count_grammar=atom_count_grammar,
                generation_position_groups=generation_position_groups,
                lightweight_decoding_constraints=lightweight_decoding_constraints,
            )
            generated_ids = outputs[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            for sample_idx, plan, text, target_atom_count in zip(current_indices, plans, decoded, current_target_atom_counts):
                metrics["decoded_samples"] += 1
                combined_text = plan.rstrip() + "\n" + text
                raw_record: Dict[str, Any] = {
                    "sample_idx": sample_idx,
                    "plan": plan,
                    "text": text,
                    "combined_text": combined_text,
                }
                if target_atom_count is not None:
                    raw_record["target_num_atoms"] = int(target_atom_count)
                try:
                    arrays = parse_fixed_slot_answer(text)
                    metrics["parse_success"] += 1
                    structure = arrays_to_structure(arrays)
                    metrics["pymatgen_success"] += 1
                    graph, cif = graph_from_arrays(arrays, process_one)
                    metrics["graph_success"] += 1
                    valid_arrays.append(arrays)
                    proposal_graphs.append(graph)
                    raw_record.update({"parsed": True, "cif": cif, "num_atoms": arrays["num_atoms"]})
                except Exception as exc:  # noqa: BLE001
                    reason = type(exc).__name__
                    metrics["failures"][reason] = metrics["failures"].get(reason, 0) + 1
                    failure_handle.write(
                        json.dumps(
                            {
                                "sample_idx": sample_idx,
                                "reason": reason,
                                "message": str(exc),
                                "plan": plan,
                                "text": combined_text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    raw_record.update({"parsed": False, "reason": reason, "message": str(exc)})
                raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                progress.update(1)

    metrics["time_sec"] = time.time() - start
    metrics["parse_rate"] = metrics["parse_success"] / max(1, metrics["decoded_samples"])
    metrics["graph_rate"] = metrics["graph_success"] / max(1, metrics["decoded_samples"])
    metrics["graph_acceptance_rate"] = metrics["graph_rate"]
    metrics["valid_array_count"] = len(valid_arrays)
    metrics["target_reached"] = local_target_graph_success is not None and metrics["graph_success"] >= local_target_graph_success
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), metrics)
    if valid_arrays:
        write_valid_arrays(valid_arrays_path, valid_arrays)
        if distributed:
            torch.save(proposal_graphs, rank_path(args.output_dir, "proposal_graphs.pt", rank, True))
        else:
            payload = arrays_to_torch_payload(valid_arrays)
            payload["time"] = metrics["time_sec"]
            torch.save(payload, args.output_dir / "raw_dlm_samples.pt")
            torch.save(proposal_graphs, args.output_dir / "proposal_graphs.pt")
    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_outputs(args.output_dir, world_size)
            merged_metrics = json.loads((args.output_dir / "sample_metrics.json").read_text(encoding="utf-8"))
            with (args.output_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "sampling_complete",
                            "decoded_samples": merged_metrics["decoded_samples"],
                            "parse_rate": merged_metrics["parse_rate"],
                            "graph_rate": merged_metrics["graph_rate"],
                            "target_reached": merged_metrics["target_reached"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
