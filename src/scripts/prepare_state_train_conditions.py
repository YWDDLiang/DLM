#!/usr/bin/env python3
"""Frozen Planner predictions for train compositions; never resample chemistry.

The existing typed chemical transcript fixes the MP20 composition.  Structural
soft fields and the species pointer are predicted from the frozen deployed
models, not copied from the MP20 structure labels.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import torch.distributed as dist

from crystal_dlm.c3fd_llama_typed_planner import SOFT_FIELDS, unit_weight_poe_log_probs
from crystal_dlm.c3fd_native_plan import build_native_body_prompt
from crystal_dlm.species_program_pointer import PlanConditionedSpeciesPointer, SpeciesPointerConfig
from scripts.train_c3fd_llama_typed_planner import load_frozen_c3fd, frozen_c3fd_logits, sha256_file
from scripts.train_spad_species_pointer import collate_pointer_rows, load_frozen_planner, move_batch


def select_conditions(pointer_rows, sft_rows, bundle, *, count, seed):
    by_source = {int(row["source_row_idx"]): row for row in sft_rows}
    eligible, excluded = [], 0
    for row in pointer_rows:
        proposal = row["proposal_target"]
        stratum = (int(proposal["family_id"]), int(proposal["N"]), int(proposal["arity"]))
        if stratum not in bundle.stratum_to_index or int(row["source_row_idx"]) not in by_source:
            excluded += 1
            continue
        eligible.append(row)
    order = np.random.default_rng(seed).permutation(len(eligible))
    selected, seen = [], set()
    for index in order:
        row = eligible[int(index)]
        identity = tuple(zip(row["canonical_atomic_numbers"], row["canonical_element_counts"]))
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("insufficient distinct train compositions in the retained typed pool")
    return selected, by_source, {"typed_rows": len(pointer_rows), "eligible_metadata_rows": len(eligible),
                                  "metadata_unavailable_or_unsupported": excluded}


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    p = argparse.ArgumentParser()
    for name in ("pointer-data", "sft-data", "c3fd-checkpoint", "vocabulary", "llama-model", "planner-final", "pointer-state", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--count", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260905)
    args = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Planner preparation must run inside its declared allocation")
    world, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if world > 1 else 0
    bundle = load_frozen_c3fd(
        checkpoint_path=args.c3fd_checkpoint, vocabulary_path=args.vocabulary,
        checkpoint_sha256=sha256_file(args.c3fd_checkpoint), vocabulary_sha256=sha256_file(args.vocabulary),
    )
    selected, sft, source_report = select_conditions(
        read_rows(args.pointer_data), read_rows(args.sft_data), bundle, count=args.count, seed=args.seed,
    )
    bundle.model.to(device)
    bundle = replace(bundle, context=bundle.context.to(device))
    llama, typed, typed_config = load_frozen_planner(
        llama_model=args.llama_model, planner_final=args.planner_final, device=device,
    )
    payload = torch.load(args.pointer_state, map_location="cpu", weights_only=False)
    pointer = PlanConditionedSpeciesPointer(SpeciesPointerConfig(**payload["config"])).to(device).eval()
    pointer.load_state_dict(payload["state_dict"])
    for parameter in pointer.parameters():
        parameter.requires_grad_(False)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    indexed = [(i, row) for i, row in enumerate(selected) if i % world == rank]
    output = []
    with torch.inference_mode():
        for offset in range(0, len(indexed), args.batch_size):
            chunk = indexed[offset:offset + args.batch_size]
            batch = move_batch(collate_pointer_rows([row for _, row in chunk], bundle=bundle), device)
            embeddings = typed.typed_inputs_embeds(
                stability_goal_ids=batch["stability_goal_ids"], proposal_state_ids=batch["proposal_state_ids"],
                previous_species_indices=batch["previous_species_indices"], previous_count_values=batch["previous_count_values"],
                ledger_features=batch["ledger_features"],
            ).to(dtype=next(llama.parameters()).dtype)
            hidden = llama(inputs_embeds=embeddings, attention_mask=batch["attention_mask"],
                           output_hidden_states=True, use_cache=False, return_dict=True).hidden_states[-1]
            residual = typed(hidden.to(next(typed.parameters()).dtype), soft_position_indices=batch["soft_position_indices"])
            _, _, base_soft = frozen_c3fd_logits(bundle, batch)
            predicted = torch.stack([
                unit_weight_poe_log_probs(base_soft[field].float(), residual.soft_fields[field].float(),
                                          batch[f"soft_mask:{field}"]).argmax(-1)
                for field in SOFT_FIELDS
            ], -1)
            terminal = hidden[torch.arange(len(chunk), device=device), batch["soft_position_indices"]].float()
            orders = pointer.decode(terminal, batch["pointer_atomic_numbers"], batch["pointer_counts"],
                                    batch["pointer_valid_mask"], predicted).cpu()
            predicted = predicted.cpu()
            for row_index, (ordinal, row) in enumerate(chunk):
                source_id = int(row["source_row_idx"])
                plan = dict(sft[source_id]["plan_state"])
                for field_index, field in enumerate(SOFT_FIELDS):
                    plan[field] = str(bundle.vocabulary["soft_vocabulary"][field][int(predicted[row_index, field_index])])
                indices = orders[row_index, :len(plan["elements"])].tolist()
                if sorted(indices) != list(range(len(plan["elements"]))):
                    raise RuntimeError("frozen pointer changed composition")
                output.append({
                    "group_id": str(source_id), "sample_idx": ordinal, "source_row_idx": source_id,
                    "source_split": "train", "plan_state": plan,
                    "prompt": build_native_body_prompt(plan).rstrip() + "\n",
                    "species_program": [plan["elements"][i] for i in indices],
                    "species_program_source": "frozen_llama_pointer_predicted_soft",
                    "soft_plan_source": "frozen_C3FD_Llama_PoE_terminal_MAP_given_train_composition",
                    "composition_resampled": False, "outcomes_read": False,
                })
    shard = args.output_dir / f"conditions.rank{rank}.jsonl"
    with shard.open("x", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if world > 1:
        dist.barrier()
    if rank == 0:
        combined = sorted([row for r in range(world) for row in read_rows(args.output_dir / f"conditions.rank{r}.jsonl")],
                          key=lambda row: row["sample_idx"])
        with (args.output_dir / "plans_for_dlm.jsonl").open("x", encoding="utf-8") as handle:
            for row in combined:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = {**source_report, "conditions": len(combined), "unique_compositions": len(combined),
                  "seed": args.seed, "composition_resampled": False, "outcomes_read": False,
                  "soft_fields_predicted": True, "pointer_uses_predicted_soft_ids": True,
                  "planner_training": False, "slurm_job_id": os.environ["SLURM_JOB_ID"]}
        (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
