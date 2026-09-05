#!/usr/bin/env python3
"""Independent native paths and fresh-process replay through the same runtime."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.programmed_path_data import (
    compile_condition, load_path_model, path_seed, read_jsonl,
    trace_summary, trace_terminal_body, validate_completed_body,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--condition-start", type=int, default=0)
    p.add_argument("--condition-stop", type=int, default=1024)
    p.add_argument("--collection-round", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--purpose", choices=("train", "evaluation"), default="train")
    p.add_argument("--temperature", type=float, default=.7)
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--replay-jsonl", type=Path)
    p.add_argument("--replay-tolerance", type=float, default=1e-6)
    return p.parse_args()


def conditions_for_run(args):
    rows = read_jsonl(args.conditions)
    if not 0 <= args.condition_start < args.condition_stop <= len(rows):
        raise ValueError("condition range is outside the frozen source pool")
    selected = list(enumerate(rows))[args.condition_start:args.condition_stop]
    if len({str(row["group_id"]) for _, row in selected}) != len(selected):
        raise ValueError("duplicate condition identity")
    if args.purpose == "train" and any(row.get("source_split") != "train" for _, row in selected):
        raise ValueError("training collection cannot use heldout conditions")
    return selected


def merge(args):
    expected = {(str(row["group_id"]), j) for _, row in conditions_for_run(args) for j in range(args.candidates)}
    records = [row for rank in range(args.world_size)
               for row in read_jsonl(args.output_dir / f"records.rank{rank}.jsonl")]
    keys = [(str(row["group_id"]), row["candidate_index"]) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("missing or duplicate requested path occurrences")
    if len({row["trajectory_id"] for row in records}) != len(records):
        raise ValueError("duplicate trajectory identity")
    records.sort(key=lambda row: (row["condition_ordinal"], row["candidate_index"]))
    with (args.output_dir / "paths.jsonl").open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"requested": len(records), "successful": sum(row["success"] for row in records),
              "conditions": len(expected) // args.candidates, "candidates": args.candidates,
              "condition_start": args.condition_start, "condition_stop": args.condition_stop,
              "collection_round": args.collection_round, "checkpoint": args.checkpoint_path,
              "seed": args.seed, "purpose": args.purpose, "inference_mlip": False,
              "outcome_selection": False, "failures_retained": True}
    (args.output_dir / "SAMPLE_FINAL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


def make_sampler(model, tokenizer, constraints, compiled, seeds, temperature):
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    from crystal_dlm.programmed_path_runtime import ProgrammedPathSampler
    from crystal_dlm.r5_dynamic_length import exact_dynamic_schema_constraints
    first = compiled[0]
    return ProgrammedPathSampler(
        model, prompt_length=len(first["prompt_token_ids"]), gen_length=len(first["initial_body"]),
        mask_id=MASK_TOKEN_ID, programs=[c["program"] for c in compiled],
        allowed_token_ids=exact_dynamic_schema_constraints(tokenizer, first["program"].num_atoms),
        atom_count_grammar=None, constraints=constraints, temperature=temperature, sampling_seeds=seeds,
    )


def replay(args, model, tokenizer, constraints, device):
    import torch
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    from crystal_dlm.programmed_path_runtime import replay_scalar_states
    maximum, total, state_effect = 0., 0, 0.
    rows = read_jsonl(args.replay_jsonl)
    for record in rows:
        compiled = compile_condition(record, tokenizer, mask_id=MASK_TOKEN_ID, purpose=args.purpose)
        prefix = compiled["prompt_token_ids"]
        if prefix != record["prompt_token_ids"]:
            raise ValueError("fresh tokenizer changed the native prompt")
        sampler = make_sampler(model, tokenizer, constraints, [compiled], [record["sampling_seed"]], record["trace"]["temperature"])
        for state in replay_scalar_states(record["trace"]):
            x = torch.tensor([prefix + state["input_body"]], device=device)
            old = torch.tensor([prefix + state["old_body"]], device=device)
            sampler._prepare(x)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, bad = sampler.processed_logits(x, old, {0: state["position"]},
                                                      {0: state["transaction_positions"]}, torch.ones_like(x))
                actual = float(torch.log_softmax(logits[0, len(prefix) + state["position"]].double() / state["temperature"], -1)[state["target_token"]])
            difference = abs(actual - state["recorded_log_probability"])
            if bad or not torch.isfinite(torch.tensor(actual)) or difference > args.replay_tolerance:
                raise RuntimeError(f"fresh-process replay differs at {record['trajectory_id']}:{state['decision_index']}: {difference}, bad={bad}")
            if state["phase"] == "cooperative" and state_effect == 0:
                from crystal_dlm.state_conditioned_model import context_from_programs
                context = context_from_programs(old, prompt_length=len(prefix), num_sites=compiled["program"].num_atoms,
                    programs=[compiled["program"]], active_positions={0: state["transaction_positions"]})
                with torch.no_grad():
                    encoded = model.state_conditioner(**model.geometry_inputs(context))
                    state_effect = float(encoded["cell_embedding"].abs().sum() + encoded["site_embeddings"].abs().sum())
            maximum, total = max(maximum, difference), total + 1
        if trace_terminal_body(record["trace"]) != record["final_body_token_ids"]:
            raise RuntimeError("trace replay does not reconstruct the deployed endpoint")
    if total == 0 or state_effect == 0:
        raise RuntimeError("probe did not exercise the trained periodic state input")
    report = {"kind": "fresh_process_base_lora_conditioner_replay", "paths": len(rows),
              "all_recorded_decisions_checked": total, "maximum_logp_error": maximum,
              "trained_state_residual_l1": state_effect, "checkpoint": args.checkpoint_path,
              "scope": "all decisions of the recorded engineering probe paths"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "REPLAY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report), flush=True)


def main():
    args = parse_args()
    if args.candidates < 1 or args.batch_size < 1 or args.temperature <= 0:
        raise ValueError("positive occurrence count, batch size and likelihood temperature required")
    if args.merge_only:
        merge(args)
        return
    if "SLURM_JOB_ID" not in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("model sampling/replay requires its declared GPU allocation")
    import torch
    from crystal_dlm.fixed_slot import MASK_TOKEN_ID
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    rank, local = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", local)
    model, tokenizer = load_path_model(args.model_path, args.checkpoint_path, device)
    constraints = build_dynamic_lightweight_constraints(
        tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4,
        canonicalize_periodic_alias=True, pbc_min_distance_mask=True, pbc_min_distance_A=.5, pbc_image_radius=2,
    )
    if args.replay_jsonl:
        replay(args, model, tokenizer, constraints, device)
        return
    buckets = defaultdict(list)
    for ordinal, row in conditions_for_run(args):
        if ordinal % args.world_size != rank:
            continue
        c = compile_condition(row, tokenizer, mask_id=MASK_TOKEN_ID, purpose=args.purpose)
        for j in range(args.candidates):
            buckets[(c["program"].num_atoms, len(c["prompt_token_ids"]))].append((ordinal, j, c))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started, completed, successful = time.monotonic(), 0, 0
    with (args.output_dir / f"records.rank{rank}.jsonl").open("x", encoding="utf-8") as handle:
        for key in sorted(buckets):
            items = buckets[key]
            for offset in range(0, len(items), args.batch_size):
                batch = items[offset:offset + args.batch_size]
                compiled = [c for _, _, c in batch]
                seeds = [path_seed(args.seed, c["record"]["group_id"], args.collection_round, j) for _, j, c in batch]
                x = torch.tensor([c["prompt_token_ids"] + c["initial_body"] for c in compiled], device=device)
                sampler = make_sampler(model, tokenizer, constraints, compiled, seeds, args.temperature)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result, traces = sampler.run(x, torch.ones_like(x))
                for row_index, (ordinal, candidate, c) in enumerate(batch):
                    body_ids = result[row_index, len(c["prompt_token_ids"]):].tolist()
                    trace = traces[row_index]
                    if trace_terminal_body(trace) != body_ids:
                        raise RuntimeError("recorded attempted trace does not reconstruct endpoint")
                    body = tokenizer.decode(body_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    if trace["success"]:
                        validate_completed_body(body, c)
                    record = {**c["record"], "prompt": c["prompt"], "prompt_token_ids": c["prompt_token_ids"],
                              "condition_ordinal": ordinal, "candidate_index": candidate,
                              "trajectory_id": f"{c['record']['group_id']}:{args.collection_round}:{candidate}",
                              "collection_round": args.collection_round, "sampling_seed": seeds[row_index],
                              "num_atoms": c["program"].num_atoms, "checkpoint": args.checkpoint_path,
                              "success": trace["success"], "body": body, "final_body_token_ids": body_ids,
                              "trace": trace, "trace_summary": trace_summary(trace)}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed += 1
                    successful += int(trace["success"])
                handle.flush()
                print(json.dumps({"rank": rank, "completed": completed, "successful": successful,
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    (args.output_dir / f"_SUCCESS.rank{rank}").touch()


if __name__ == "__main__":
    main()
