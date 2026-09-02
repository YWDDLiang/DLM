#!/usr/bin/env python3
"""Capture real exact-axis rollout states for active-group DLM training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from crystal_dlm.ctv_rollout import collect_ctv_branch_states
from crystal_dlm.fixed_slot import MASK_TOKEN_ID, tokenize_answer_text
from crystal_dlm.periodic_relation_runtime import wrap_with_periodic_relation
from crystal_dlm.r5_dynamic_length import (
    count_prefill_for_batch,
    exact_body_token_count,
    exact_dynamic_generation_schedule,
    exact_dynamic_schema_constraints,
    validate_answer_matches_plan,
)
from crystal_dlm.sgtc_sampling import matched_base_noise_group
from scripts.sample_llada_dynamic_crystals import (
    build_dynamic_lightweight_constraints,
    load_model_and_tokenizer,
)
from scripts.sample_llada_r5_exact_length import element_prefill_for_batch


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group_midpoint_milestones(num_atoms: int) -> tuple[float, float, float, float]:
    count = int(num_atoms)
    total = 6 + 3 * count
    half_sites = max(1, math.ceil(count / 2))
    visible = (3, 6 + half_sites, 6 + count + half_sites, 6 + 2 * count + half_sites)
    values = tuple(float(value) / float(total) for value in visible)
    if not all(0.0 < value < 1.0 for value in values):
        raise ValueError("rollout milestones must lie strictly inside the trajectory")
    return values


def ids_to_answer(tokenizer, token_ids: list[int]) -> str:
    tokens = tokenizer.convert_ids_to_tokens([int(value) for value in token_ids])
    answer = "".join(str(token) for token in tokens)
    if len(tokenize_answer_text(answer)) != len(token_ids):
        raise ValueError("dynamic token round-trip changed during rollout capture")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--periodic-relation-checkpoint", required=True)
    parser.add_argument("--periodic-relation-rank", type=int, default=64)
    parser.add_argument("--plans-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("rollout capture requires one CUDA device")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    plans = read_jsonl(args.plans_jsonl.resolve())
    if len(plans) != 128 or [int(row["sample_idx"]) for row in plans] != list(range(128)):
        raise ValueError("rollout pilot requires ordered 128 train Plans")

    device = torch.device("cuda", 0)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.checkpoint_path, device
    )
    model = wrap_with_periodic_relation(
        model,
        tokenizer,
        rank=int(args.periodic_relation_rank),
        checkpoint=args.periodic_relation_checkpoint,
    ).to(device)
    model.eval()
    lightweight = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=True,
        lattice_volume_mask=True,
        min_lattice_rad=1.0e-4,
    )

    transitions: list[dict] = []
    trajectories: list[dict] = []
    for ordinal, row in enumerate(plans):
        plan = dict(row["plan_state"])
        count = int(plan["N"])
        prompt_text = str(row["prompt"]).rstrip() + "\n"
        encoded = tokenizer(
            [prompt_text], add_special_tokens=False, padding=True, return_tensors="pt"
        )
        prompt = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        target_answer = str(row["target_answer"])
        validate_answer_matches_plan(plan, target_answer)
        target_ids = tokenizer(
            target_answer, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].tolist()
        expected = exact_body_token_count(count)
        if len(target_ids) != expected:
            raise ValueError("target answer token count differs from exact 7+4N")
        schedule = exact_dynamic_generation_schedule(count)
        final_tokens, snapshots = collect_ctv_branch_states(
            model,
            prompt,
            attention_mask=attention,
            num_atoms=count,
            gen_length=expected,
            temperature=float(args.temperature),
            mask_id=MASK_TOKEN_ID,
            allowed_token_ids_by_generation_pos=exact_dynamic_schema_constraints(
                tokenizer, count
            ),
            prefill_token_ids_by_generation_pos={
                **count_prefill_for_batch(tokenizer, count, 1),
                **element_prefill_for_batch(tokenizer, [plan]),
            },
            generation_position_groups=schedule,
            lightweight_decoding_constraints=lightweight,
            base_noise_group=matched_base_noise_group(
                seed=int(args.seed),
                composition_id=str(row["reduced_composition_identity"]),
                sample_idx=ordinal,
            ),
            milestones=group_midpoint_milestones(count),
        )
        if len(snapshots) != 4:
            raise RuntimeError("rollout capture did not return four states")
        observed_groups = [int(snapshot["group_index"]) for snapshot in snapshots]
        if observed_groups != [2, 3, 4, 5]:
            raise RuntimeError(
                f"rollout stages changed: expected [2,3,4,5], got {observed_groups}"
            )
        prompt_length = int(prompt.shape[1])
        for stage_index, snapshot in enumerate(snapshots):
            suffix = snapshot["tokens"][0, prompt_length:].detach().cpu().tolist()
            forced_mask = [
                position
                for position, token_id in enumerate(suffix)
                if int(token_id) == int(MASK_TOKEN_ID)
            ]
            group_index = int(snapshot["group_index"])
            active_group = [int(value) for value in schedule[group_index]]
            loss_positions = sorted(set(forced_mask) & set(active_group))
            if not forced_mask or not loss_positions:
                raise RuntimeError("captured rollout state lacks active masked positions")
            prior_positions = {
                int(position)
                for group in schedule[:group_index]
                for position in group
            }
            future_positions = {
                int(position)
                for group in schedule[group_index + 1 :]
                for position in group
            }
            if set(forced_mask) & prior_positions:
                raise RuntimeError("rollout capture remasked a previously committed group")
            if not future_positions <= set(forced_mask):
                raise RuntimeError("rollout capture exposed a future schedule group")
            source_ids = [
                int(target_ids[position])
                if int(token_id) == int(MASK_TOKEN_ID)
                else int(token_id)
                for position, token_id in enumerate(suffix)
            ]
            source_answer = ids_to_answer(tokenizer, source_ids)
            validate_answer_matches_plan(plan, source_answer)
            committed_error_positions = [
                position
                for position, (source_id, target_id) in enumerate(
                    zip(source_ids, target_ids, strict=True)
                )
                if position not in set(forced_mask) and int(source_id) != int(target_id)
            ]
            group_name = {2: "lattice", 3: "x", 4: "y", 5: "z"}[group_index]
            transitions.append(
                {
                    "schema": "rollout_matched_transition_v1",
                    "prompt": str(row["prompt"]),
                    "source_answer": source_answer,
                    "answer": target_answer,
                    "num_atoms": count,
                    "loss_profile": "fixed_slot",
                    "sample_weight": 0.25,
                    "sample_idx": ordinal,
                    "source_row_idx": int(row["source_row_idx"]),
                    "stage_index": stage_index,
                    "active_group": group_name,
                    "schedule_group_index": group_index,
                    "step_in_group": int(snapshot["step_in_group"]),
                    "forced_mask_positions": forced_mask,
                    "loss_positions": loss_positions,
                    "committed_error_count": len(committed_error_positions),
                    "committed_error_positions": committed_error_positions,
                    "visible_free_geometry_fraction": float(
                        snapshot["visible_free_geometry_fraction"]
                    ),
                }
            )
        final_suffix = final_tokens[0, prompt_length:].detach().cpu().tolist()
        trajectories.append(
            {
                "sample_idx": ordinal,
                "source_row_idx": int(row["source_row_idx"]),
                "answer": ids_to_answer(tokenizer, final_suffix),
            }
        )

    if len(transitions) != 512:
        raise RuntimeError("rollout transition count changed")
    train = [row for row in transitions if int(row["sample_idx"]) % 4 != 0]
    validation = [row for row in transitions if int(row["sample_idx"]) % 4 == 0]
    if len(train) != 384 or len(validation) != 128:
        raise RuntimeError("rollout transition train/validation split changed")
    output.mkdir(parents=True)
    (output / "transitions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transitions)
    )
    (output / "trajectories.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in trajectories)
    )
    (output / "train.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train)
    )
    (output / "val.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in validation)
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "rollout_matched_capture_v1",
                "status": "complete",
                "plans": 128,
                "stages_per_plan": 4,
                "transitions": 512,
                "train_transitions": 384,
                "validation_transitions": 128,
                "train_plans": 96,
                "validation_plans": 32,
                "seed": int(args.seed),
                "temperature": float(args.temperature),
                "schedule": "exact_axis",
                "retry_or_replacement_used": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
