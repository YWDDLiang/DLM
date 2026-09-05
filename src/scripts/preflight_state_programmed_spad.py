#!/usr/bin/env python3
"""Allocated-GPU integration check, not an eligible scientific checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
from crystal_dlm.programmed_path_runtime import ProgrammedPathSampler, replay_scalar_states
from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
from crystal_dlm.spad_program import program_from_element_order
from crystal_dlm.state_conditioned_model import (
    StateConditionedDLM, context_from_programs, set_state_lora_trainable,
)
from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints, load_model_and_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--data-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260905)
    return p.parse_args()


def main():
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("the real-model preflight must run inside its GPU allocation")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if world > 1 else 0
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    base, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, device)
    config = PeriodicStateConfig(int(base.get_input_embeddings().weight.shape[1]))
    model = StateConditionedDLM(base, tokenizer, config).to(device)
    counts = set_state_lora_trainable(model)
    model.train()
    # LLaDA exposes a native activation-checkpoint strategy rather than requiring
    # the generic HF gradient_checkpointing_enable protocol.
    checkpoint_modules = []
    for name, module in base.named_modules():
        method = getattr(module, "set_activation_checkpointing", None)
        if callable(method) and hasattr(module, "transformer"):
            method("whole_layer")
            checkpoint_modules.append(name)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    rows = []
    with args.data_jsonl.open(encoding="utf-8") as handle:
        for _ in range(max(args.steps * world, world)):
            rows.append(json.loads(next(handle)))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    diagnostics = []
    context = None
    x = None
    started = time.monotonic()
    for step in range(args.steps):
        row = rows[step * world + rank]
        prompt = row["prompt"].rstrip() + "\n"
        prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        body = tokenizer(row["answer"], add_special_tokens=False)["input_ids"]
        joined = tokenizer(prompt + row["answer"], add_special_tokens=False)["input_ids"]
        if joined != prefix + body or len(body) != 7 + 4 * int(row["num_atoms"]):
            raise ValueError("retained prompt/body tokenization no longer matches")
        old = torch.tensor([joined], dtype=torch.long, device=device)
        x = old.clone()
        positions = list(row["forced_mask_positions"])
        x[:, [len(prefix) + p for p in positions]] = MASK_TOKEN_ID
        target_position = int(row["loss_positions"][0])
        program = program_from_element_order(row["plan_state"], row["species_program"], order_source=row["species_program_source"])
        context = context_from_programs(
            old, prompt_length=len(prefix), num_sites=program.num_atoms,
            programs=[program], active_positions={0: positions},
        )
        attention = torch.ones_like(x)
        if step == 0:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                reference = base(x, attention_mask=attention).logits[:, len(prefix) + target_position]
                candidate = model(x, attention_mask=attention, geometry_context=context).logits[:, len(prefix) + target_position]
                zero_delta = float((reference.float() - candidate.float()).abs().max())
            if zero_delta != 0.0:
                raise RuntimeError(f"zero conditioner changed identical-state logits: {zero_delta}")
        optimizer.zero_grad(set_to_none=True)
        tic = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = wrapped(x, attention_mask=attention, geometry_context=context).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, len(prefix) + target_position].float(),
                old[:, len(prefix) + target_position],
            )
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        diagnostics.append({"step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(grad),
                            "seconds": time.monotonic() - tic, "tokens": x.shape[1]})
        print(json.dumps({"rank": rank, **diagnostics[-1]}), flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            before_reload = model(x, attention_mask=attention, geometry_context=context).logits[:, len(prefix) + target_position].float().cpu()
        # Same module reload isolates state-serialization errors without loading
        # a second 8B backbone. A fresh process loader is also used before rollout.
        saved = args.output_dir / "reload_check"
        model.save_pretrained(saved)
        tokenizer.save_pretrained(saved)
        model.load_state_conditioner(saved)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            after_reload = model(x, attention_mask=attention, geometry_context=context).logits[:, len(prefix) + target_position].float().cpu()
        if not torch.equal(before_reload, after_reload):
            raise RuntimeError("conditioner reload changed logits")
        constraints = build_dynamic_lightweight_constraints(
            tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True,
            min_lattice_rad=1e-4,
            canonicalize_periodic_alias=True, pbc_min_distance_mask=True,
            pbc_min_distance_A=.5, pbc_image_radius=2,
        )
        sampler = ProgrammedPathSampler(
            model, prompt_length=len(prefix), gen_length=len(body), mask_id=MASK_TOKEN_ID,
            programs=[program], allowed_token_ids=exact_dynamic_schema_constraints(tokenizer, program.num_atoms),
            atom_count_grammar=None, constraints=constraints, temperature=.7, sampling_seeds=[args.seed],
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output, traces = sampler.run(old, attention, construct=False, closure=True)
        scalar_states = list(replay_scalar_states(traces[0]))
        replay_errors = []
        for state in scalar_states[:3]:
            current = torch.tensor([prefix + state["input_body"]], device=device)
            previous = torch.tensor([prefix + state["old_body"]], device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                values, bad = sampler.processed_logits(current, previous, {0: state["position"]},
                                                        {0: state["transaction_positions"]}, torch.ones_like(current))
                actual = float(torch.log_softmax(values[0, len(prefix) + state["position"]].double() / .7, -1)[state["target_token"]])
            replay_errors.append(abs(actual - state["recorded_log_probability"]))
            if bad or replay_errors[-1] > 1e-6:
                raise RuntimeError("real attempted-path replay differs from sampling")
        if not traces[0]["success"] or not scalar_states:
            raise RuntimeError("real joint/closure integration produced no supported attempted path")
        report = {
            "kind": "engineering_preflight_only", "eligible_policy": False,
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "world_size": world,
            "trainable_parameters": counts, "checkpoint_modules": checkpoint_modules,
            "same_state_zero_delta": zero_delta, "steps": diagnostics,
            "maximum_replay_error": max(replay_errors, default=0),
            "sampled_decisions": len(scalar_states), "trace_success": traces[0]["success"],
            "elapsed_seconds": time.monotonic() - started,
        }
        (args.output_dir / "PREFLIGHT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(report), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
