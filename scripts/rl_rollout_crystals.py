#!/usr/bin/env python3
"""Online rollout collection for MP-20 fixed-slot LLaDA TraceRL."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.fixed_slot import ANSWER_TOKEN_COUNT, CANONICAL_PROMPT, MASK_TOKEN_ID, write_json
from crystal_dlm.generation_schedule import (
    n_elements_coords_lattice_schedule,
    n_elements_sequential_rest_schedule,
)
from crystal_dlm.llada_generation import generate
from scripts.sample_llada_crystals import (
    build_atom_count_grammar,
    build_lightweight_decoding_constraints,
    build_schema_generation_constraints,
    init_distributed,
    load_model_and_tokenizer,
    rank_path,
)


def merge_rank_jsonl(output_dir: Path, filename: str, world_size: int) -> int:
    total = 0
    with (output_dir / filename).open("w", encoding="utf-8") as out:
        for rank in range(world_size):
            path = rank_path(output_dir, filename, rank, True)
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line)
                        total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--responses-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=ANSWER_TOKEN_COUNT)
    parser.add_argument("--gen-length", type=int, default=ANSWER_TOKEN_COUNT)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument(
        "--generation-schedule",
        choices=["default", "n-elements-coords-lattice", "n-elements-sequential-rest"],
        default="default",
    )
    parser.add_argument("--prompt", default=CANONICAL_PROMPT)
    parser.add_argument("--schema-logit-mask", action="store_true", default=True)
    parser.add_argument("--no-schema-logit-mask", dest="schema_logit_mask", action="store_false")
    parser.add_argument("--prefill-slot-tokens", action="store_true", default=True)
    parser.add_argument("--no-prefill-slot-tokens", dest="prefill_slot_tokens", action="store_false")
    parser.add_argument("--atom-count-grammar-mask", action="store_true", default=True)
    parser.add_argument("--no-atom-count-grammar-mask", dest="atom_count_grammar_mask", action="store_false")
    parser.add_argument("--duplicate-coordinate-mask", action="store_true", default=True)
    parser.add_argument("--no-duplicate-coordinate-mask", dest="duplicate_coordinate_mask", action="store_false")
    parser.add_argument("--lattice-volume-mask", action="store_true", default=True)
    parser.add_argument("--no-lattice-volume-mask", dest="lattice_volume_mask", action="store_false")
    parser.add_argument("--min-lattice-rad", type=float, default=1e-4)
    args = parser.parse_args()

    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    device = dist_info["device"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, device)
    allowed, prefill = build_schema_generation_constraints(tokenizer)
    allowed_token_ids_by_generation_pos = allowed if args.schema_logit_mask else None
    prefill_token_ids_by_generation_pos = prefill if args.prefill_slot_tokens else None
    atom_count_grammar = build_atom_count_grammar(tokenizer) if args.atom_count_grammar_mask else None
    lightweight_constraints = build_lightweight_decoding_constraints(
        tokenizer,
        duplicate_coordinate_mask=args.duplicate_coordinate_mask,
        lattice_volume_mask=args.lattice_volume_mask,
        min_lattice_rad=args.min_lattice_rad,
    )
    if args.generation_schedule == "n-elements-coords-lattice":
        generation_position_groups = n_elements_coords_lattice_schedule()
    elif args.generation_schedule == "n-elements-sequential-rest":
        generation_position_groups = n_elements_sequential_rest_schedule()
    else:
        generation_position_groups = None

    total_samples = int(args.num_prompts) * int(args.responses_per_prompt)
    sample_indices = list(range(rank, total_samples, world_size))
    raw_path = rank_path(args.output_dir, "rollout_raw.jsonl", rank, distributed)
    metrics_path = rank_path(args.output_dir, "rollout_metrics.json", rank, distributed)
    start = time.time()
    decoded_samples = 0
    prompt_text = args.prompt.rstrip() + "\n"
    with raw_path.open("w", encoding="utf-8") as handle:
        progress = tqdm(sample_indices, desc=f"RL rollout rank{rank}", disable=distributed and not is_main)
        for batch_start in range(0, len(sample_indices), args.batch_size):
            current_indices = sample_indices[batch_start : batch_start + args.batch_size]
            prompts = [prompt_text] * len(current_indices)
            encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            generated, step_maps = generate(
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
                prefill_token_ids_by_generation_pos=prefill_token_ids_by_generation_pos,
                atom_count_grammar=atom_count_grammar,
                generation_position_groups=generation_position_groups,
                lightweight_decoding_constraints=lightweight_constraints,
                return_step_map=True,
            )
            generated_ids = generated[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            for local_idx, sample_idx in enumerate(current_indices):
                decoded_samples += 1
                handle.write(
                    json.dumps(
                        {
                            "sample_idx": int(sample_idx),
                            "prompt_id": int(sample_idx) // int(args.responses_per_prompt),
                            "response_id": int(sample_idx) % int(args.responses_per_prompt),
                            "prompt": args.prompt,
                            "response": decoded[local_idx],
                            "step_map": [int(v) for v in step_maps[local_idx].detach().cpu().tolist()],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            progress.update(len(current_indices))

    metrics = {
        "decoded_samples": decoded_samples,
        "num_prompts": args.num_prompts,
        "responses_per_prompt": args.responses_per_prompt,
        "total_samples": total_samples,
        "rank": rank,
        "world_size": world_size,
        "time_sec": time.time() - start,
    }
    write_json(str(metrics_path), metrics)
    if distributed:
        dist.barrier()
        if is_main:
            merged_count = merge_rank_jsonl(args.output_dir, "rollout_raw.jsonl", world_size)
            merged = {
                "decoded_samples": merged_count,
                "num_prompts": args.num_prompts,
                "responses_per_prompt": args.responses_per_prompt,
                "total_samples": total_samples,
                "world_size": world_size,
                "distributed": True,
            }
            write_json(str(args.output_dir / "rollout_metrics.json"), merged)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
