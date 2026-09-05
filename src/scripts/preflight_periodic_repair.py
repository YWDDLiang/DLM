#!/usr/bin/env python3
"""Real two-GPU engineering checks; no eligible model is produced."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.periodic_repair_model import load_repair_model
from crystal_dlm.programmed_path_data import compile_condition, load_path_model, read_jsonl, trace_terminal_body
from crystal_dlm.programmed_path_runtime import full_cell_transaction_positions, replay_scalar_states
from crystal_dlm.state_training import enable_native_checkpointing, materialize_state_batch
from crystal_dlm.terminal_repair_data import CONDITION_KEYS
from crystal_dlm.terminal_repair_objective import RepairObjective
from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
from scripts.sample_state_programmed_paths import make_sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--paths-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    if world != 2 or "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("preflight requires its two-GPU job")
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("nccl")
    torch.manual_seed(20260906)
    device = torch.device("cuda", rank)
    model, tokenizer = load_repair_model(args.model_path, args.checkpoint_path, device, trainable=True)
    enable_native_checkpointing(model.base_model)
    reference, _ = load_path_model(args.model_path, args.checkpoint_path, device)
    reference.requires_grad_(False).eval()
    wrapped = DistributedDataParallel(model, device_ids=[rank])
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True,
        min_lattice_rad=1e-4, canonicalize_periodic_alias=True,
        pbc_min_distance_mask=True, pbc_min_distance_A=.5, pbc_image_radius=2,
    )
    records = [row for path in args.paths_jsonl for row in read_jsonl(path)
               if row["success"] and row["num_atoms"] >= 2]
    rows = [records[rank], records[rank + world]]
    objective = RepairObjective(tokenizer, constraints, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    diagnostics, maximum_zero_delta = [], 0.
    started = time.monotonic()
    for step in range(3):
        examples, actual_states = [], []
        for record in rows:
            compiled = compile_condition(record, tokenizer, mask_id=MASK_TOKEN_ID)
            positions = full_cell_transaction_positions(compiled["program"])
            chosen = positions[(0, 3, 6)[step]]
            old = list(record["final_body_token_ids"])
            current = old.copy()
            for pos in positions:
                current[pos] = MASK_TOKEN_ID
            for pos in positions:
                if pos == chosen:
                    break
                current[pos] = old[pos]
            metadata = {key: record[key] for key in CONDITION_KEYS}
            examples.append({
                **metadata, "num_atoms": record["num_atoms"], "input_body": current, "old_body": old,
                "transaction_positions": list(positions), "position": chosen, "target_token": old[chosen],
                "phase": "full_cell_repair", "sample_weight": 1.,
            })
            states = list(replay_scalar_states(record["trace"]))
            actual_states.append({**metadata, **states[min(step, len(states) - 1)],
                                  "num_atoms": record["num_atoms"], "sample_weight": 1.})
        batch = materialize_state_batch(examples, tokenizer, device=device)
        replay_batch = materialize_state_batch(actual_states, tokenizer, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref_logits = reference(
                replay_batch["input_ids"], attention_mask=replay_batch["attention_mask"],
                geometry_context=replay_batch["geometry_context"],
                attention_bias=torch.zeros(len(rows), 1, replay_batch["input_ids"].shape[1],
                                           replay_batch["input_ids"].shape[1], device=device),
            ).logits
        with torch.autocast("cuda", dtype=torch.bfloat16):
            replay_logits = wrapped(
                replay_batch["input_ids"], attention_mask=replay_batch["attention_mask"],
                geometry_context=replay_batch["geometry_context"],
            ).logits
            if step == 0:
                maximum_zero_delta = float((ref_logits - replay_logits).abs().max())
                if maximum_zero_delta > 1e-5:
                    raise RuntimeError(f"zero increment changed real model logits: {maximum_zero_delta}")
            kl = objective.policy_kl(replay_logits, ref_logits, replay_batch)
        (.01 * kl).backward()
        del ref_logits, replay_logits
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = wrapped(batch["input_ids"], attention_mask=batch["attention_mask"],
                             geometry_context=batch["geometry_context"]).logits
            loss, stats = objective.terminal_ce(logits, batch)
        loss.backward()
        if not bool(torch.isfinite(loss) & torch.isfinite(kl)):
            raise FloatingPointError("nonfinite engineering objective")
        norms = {}
        for name_prefix in ("geometry_attention", "numeric_adapter", "repair_task_projection", "state_conditioner"):
            grads = [p.grad for name, p in model.named_parameters()
                     if name.startswith(name_prefix + ".") and p.grad is not None]
            if not grads or not all(torch.isfinite(value).all() for value in grads):
                raise RuntimeError(f"missing/nonfinite gradients in {name_prefix}")
            norms[name_prefix] = float(sum(value.float().abs().sum() for value in grads))
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        diagnostics.append(dict(step=step + 1, loss=float(loss.detach()), kl=float(kl.detach()),
                                logits_dtype=str(logits.dtype), gradient_abs_sum=norms, **stats))
        print(json.dumps(dict(rank=rank, **diagnostics[-1])), flush=True)
    if diagnostics[-1]["gradient_abs_sum"]["geometry_attention"] == 0:
        raise RuntimeError("internal geometry attention received no learning signal")
    model.eval()
    record = rows[0]
    compiled = compile_condition(record, tokenizer, mask_id=MASK_TOKEN_ID)
    prefix = compiled["prompt_token_ids"]
    x = torch.tensor([prefix + record["final_body_token_ids"]], device=device)
    sampler = make_sampler(model, tokenizer, constraints, [compiled], [2026090600 + rank], .7)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        result, traces = sampler.run(x, torch.ones_like(x), construct=False,
                                     cooperative=False, closure=False, full_cell_repair=True)
    trace = traces[0]
    if trace_terminal_body(trace) != result[0, len(prefix):].tolist():
        raise RuntimeError("full-cell attempted trace does not reproduce its endpoint")
    maximum_replay_delta = 0.
    for state in replay_scalar_states(trace):
        current = torch.tensor([prefix + state["input_body"]], device=device)
        old = torch.tensor([prefix + state["old_body"]], device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, bad = sampler.processed_logits(
                current, old, {0: state["position"]}, {0: state["transaction_positions"]},
                torch.ones_like(current), phase=state["phase"],
            )
            logp = torch.log_softmax(logits[0, len(prefix) + state["position"]].double() / .7, -1)
        delta = abs(float(logp[state["target_token"]]) - state["recorded_log_probability"])
        if bad or delta > 1e-6:
            raise RuntimeError(f"full-cell likelihood replay differs: {delta}")
        maximum_replay_delta = max(maximum_replay_delta, delta)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "rank": rank, "eligible_policy": False, "engineering_only": True,
        "engineering_identity_targets_are_not_repair_supervision": True,
        "updated_weights_discarded": True, "source_checkpoint": args.checkpoint_path,
        "zero_increment_max_logit_delta": maximum_zero_delta,
        "full_cell_replay_max_delta": maximum_replay_delta,
        "sampled_repair_decisions": sum(event["op"] == "draw" for event in trace["events"]),
        "optimizer_steps": 3, "diagnostics": diagnostics,
        "peak_memory_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / f"PREFLIGHT_FINAL.rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    if rank == 0:
        combined = [json.loads((args.output_dir / f"PREFLIGHT_FINAL.rank{r}.json").read_text()) for r in range(world)]
        (args.output_dir / "PREFLIGHT_FINAL.json").write_text(
            json.dumps({"eligible_policy": False, "checks_passed": True, "ranks": combined}, indent=2) + "\n")
        (args.output_dir / "_SUCCESS").touch()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
