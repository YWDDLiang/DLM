#!/usr/bin/env python3
"""Replay materialized closure actions through the real DLM continuation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.r5_dynamic_length import (
    exact_dynamic_schema_constraints,
    validate_dynamic_tokenizer_contract,
)
from crystal_dlm.spad_generation import (
    continue_spad_species_blocks_from_cursor,
    revise_spad_species_blocks,
)
from crystal_dlm.spad_program import (
    program_from_element_order,
    reverse_species_block_revision_slots,
)
from scripts.sample_llada_dynamic_crystals import (
    build_dynamic_lightweight_constraints,
    load_model_and_tokenizer,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("state JSONL row is not an object")
                yield value


def encoded_ids(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return ids.to(device=device, dtype=torch.long)


def row_tokens(
    tokenizer: Any, row: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    prompt = str(row["prompt"]).rstrip() + "\n"
    prompt_ids = encoded_ids(tokenizer, prompt, device)
    predictor = encoded_ids(tokenizer, str(row["predictor_body"]), device)
    state = encoded_ids(tokenizer, str(row["state_body"]), device)
    entry_text = (
        str(row["predictor_body"])
        if row["state_type"] == "cell"
        else str(row["block_entry_snapshot"]["body"])
    )
    entry = encoded_ids(tokenizer, entry_text, device)
    final = encoded_ids(tokenizer, str(row["final_body"]), device)
    gen_length = 7 + 4 * int(row["N"])
    if any(int(value.shape[1]) != gen_length for value in (predictor, state, entry, final)):
        raise ValueError("a materialized body is not exact 7+4N")
    return prompt_ids, predictor, state, entry, int(prompt_ids.shape[1]), gen_length


def replay_row(model: Any, tokenizer: Any, row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    prompt_ids, predictor, state, entry, prompt_length, gen_length = row_tokens(
        tokenizer, row, device
    )
    final_ids = encoded_ids(tokenizer, str(row["final_body"]), device)
    attention = torch.ones_like(prompt_ids)
    plan = row["plan_state"]
    program = program_from_element_order(
        plan, row["species_program"], order_source="frozen_planner_llama_pointer"
    )
    blocks = [list(value) for value in reverse_species_block_revision_slots(program)]
    allowed = exact_dynamic_schema_constraints(tokenizer, int(row["N"]))
    constraints = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=True,
        lattice_volume_mask=True,
        min_lattice_rad=1.0e-4,
        canonicalize_periodic_alias=True,
        pbc_min_distance_mask=True,
        pbc_min_distance_A=0.5,
        pbc_image_radius=2,
    )
    action = [int(value) for value in row["reference_action"]["token_ids"]]
    if row["state_type"] == "cell":
        complete = torch.cat((prompt_ids, predictor), dim=1)
        for position, token_id in zip(
            row["active_generation_positions"], action, strict=True
        ):
            complete[0, prompt_length + int(position)] = int(token_id)
        output, _logs = revise_spad_species_blocks(
            model,
            complete,
            prompt_length=prompt_length,
            gen_length=gen_length,
            revision_blocks_by_batch=[blocks],
            attention_mask=attention,
            temperature=0.7,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=MASK_TOKEN_ID,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=constraints,
            sampling_seeds_by_batch=[
                int(row["continuation_seeds"]["species_blocks"])
            ],
        )
        continuation_report = {"state_type": "cell", "later_blocks": len(blocks)}
    else:
        complete = torch.cat((prompt_ids, state), dim=1)
        block_entry = torch.cat((prompt_ids, entry), dim=1)
        output, continuation_report = continue_spad_species_blocks_from_cursor(
            model,
            complete,
            block_entry_tokens=block_entry,
            prompt_length=prompt_length,
            gen_length=gen_length,
            revision_blocks=blocks,
            block_index=int(row["cursor"]["block_index"]),
            site_order_index=int(row["cursor"]["site_order_index"]),
            action_token_ids=action,
            attention_mask=attention,
            temperature=0.7,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=MASK_TOKEN_ID,
            allowed_token_ids_by_generation_pos=allowed,
            atom_count_grammar=None,
            lightweight_decoding_constraints=constraints,
            sampling_seed=int(row["continuation_seeds"]["species_blocks"]),
        )
    observed = output[:, prompt_length : prompt_length + gen_length]
    mismatches = torch.nonzero(observed[0] != final_ids[0], as_tuple=False).flatten()
    return {
        "schema": "spad_basin_preflight_model_replay_attempt_v1",
        "sample_idx": int(row["sample_idx"]),
        "state_type": str(row["state_type"]),
        "cursor_bucket": row.get("cursor_bucket"),
        "matches_final": int(mismatches.numel()) == 0,
        "mismatch_positions": [int(value) for value in mismatches.cpu().tolist()],
        "continuation_report": continuation_report,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    if world <= 0 or not 0 <= rank < world:
        raise ValueError("invalid rank/world size")
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(
        str(args.model_path), str(args.checkpoint_path), device
    )
    contract = validate_dynamic_tokenizer_contract(tokenizer)
    states = list(iter_jsonl(args.states_jsonl.resolve()))
    if len(states) != 128 or {int(row["sample_idx"]) for row in states} != set(range(128)):
        raise ValueError("preflight states must cover sample_idx 0..127")
    selected = [row for row in states if int(row["sample_idx"]) % world == rank]
    started = time.time()
    attempts: list[dict[str, Any]] = []
    for row in selected:
        try:
            attempts.append(replay_row(model, tokenizer, row, device))
        except Exception as error:  # noqa: BLE001
            attempts.append(
                {
                    "schema": "spad_basin_preflight_model_replay_attempt_v1",
                    "sample_idx": int(row["sample_idx"]),
                    "state_type": str(row["state_type"]),
                    "cursor_bucket": row.get("cursor_bucket"),
                    "matches_final": False,
                    "mismatch_positions": [],
                    "error": f"{type(error).__name__}:{error}",
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = args.output_dir / f"attempts_rank{rank}.jsonl"
    if attempt_path.exists():
        raise FileExistsError(attempt_path)
    with attempt_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in attempts:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema": "spad_basin_preflight_model_replay_rank_v1",
        "rank": rank,
        "world_size": world,
        "assigned": len(attempts),
        "matches": sum(row["matches_final"] for row in attempts),
        "mismatches": sum(not row["matches_final"] for row in attempts),
        "elapsed_seconds": time.time() - started,
        "tokenizer_contract": contract,
        "outcomes_read": False,
    }
    (args.output_dir / f"report_rank{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
