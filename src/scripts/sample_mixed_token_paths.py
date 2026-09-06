#!/usr/bin/env python3
"""Independent T construction sampling and fresh scalar replay of a mixed final policy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from crystal_dlm.mixed_geometry_sampling import validate_final_policy
from crystal_dlm.mixed_token_sampling import (
    BASE_SEED, BATCH_CAP, DEPLOYMENT, LAYOUT_WORLD_SIZE, METHOD, TEMPERATURE, TRAINING_METHOD,
    load_final_mixed_token_policy, replay_token_batch, sample_token_batch, validate_protocol, validate_record_identity,
)
from crystal_dlm.programmed_path_data import compile_condition, read_jsonl
from crystal_dlm.sampling_layout import sampling_batches
from scripts.sample_state_programmed_paths import conditions_for_run


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--condition-start", type=int, default=0)
    p.add_argument("--condition-stop", type=int)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    p.add_argument("--collection-round", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=BATCH_CAP)
    p.add_argument("--sampling-layout-world-size", type=int, default=LAYOUT_WORLD_SIZE)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    p.add_argument("--max-length", type=int, default=382)
    p.add_argument("--cpu-threads", type=int, default=2)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--merge-only", action="store_true")
    mode.add_argument("--replay-jsonl", type=Path)
    p.add_argument("--replay-tolerance", type=float, default=1e-6)
    return p


def expected_conditions(args):
    if args.condition_stop is None:
        args.condition_stop = len(read_jsonl(args.conditions))
    args.purpose = "evaluation"
    return conditions_for_run(args)


def protocol(args):
    return {"method": METHOD, "training_method": TRAINING_METHOD, "deployment": DEPLOYMENT,
            "conditions_sha256": file_sha256(args.conditions), "checkpoint": str(args.checkpoint_path),
            "condition_start": args.condition_start, "condition_stop": args.condition_stop,
            "seed": args.seed, "collection_round": args.collection_round, "temperature": args.temperature,
            "logical_batch_cap": args.batch_size, "sampling_layout_world_size": args.sampling_layout_world_size,
            "max_length": args.max_length, "execution_world_size": args.world_size}


def merge_outputs(args):
    validate_final_policy(args.checkpoint_path)
    selected = expected_conditions(args)
    expected = {ordinal: source for ordinal, source in selected}
    contract = protocol(args)
    stats, records = [], []
    for rank in range(args.world_size):
        if not (args.output_dir/f"_SUCCESS.rank{rank}").is_file():
            raise ValueError("a T sampling rank has not completed")
        stat = json.loads((args.output_dir/f"SAMPLING.rank{rank}.json").read_text(encoding="utf-8"))
        if stat.get("rank") != rank or any(stat.get(k) != v for k,v in contract.items()):
            raise ValueError("T sampling rank changed the source, policy or protocol")
        shard = read_jsonl(args.output_dir/f"records.rank{rank}.jsonl")
        if stat["requested"] != len(shard) or stat["successful"] != sum(row["success"] for row in shard):
            raise ValueError("T rank totals differ from its complete ledger")
        if stat["model_row_evaluations"] != stat["live_row_model_evaluations"]+stat["failed_row_padding_evaluations"]:
            raise ValueError("T forward and failed-row padding totals differ")
        stats.append(stat)
        records.extend(shard)
    keyed = {int(row["condition_ordinal"]): row for row in records}
    if len(keyed) != len(records) or set(keyed) != set(expected):
        raise ValueError("T sampling duplicated or omitted a frozen request")
    if len({row["trajectory_id"] for row in records}) != len(records):
        raise ValueError("T sampling duplicated a trajectory identity")
    records.sort(key=lambda row: row["condition_ordinal"])
    for row in records:
        ordinal = row["condition_ordinal"]
        validate_record_identity(row, expected[ordinal], ordinal, checkpoint_path=args.checkpoint_path,
                                 condition_start=args.condition_start)
        if row["success"] and (not row["parseable"] or row["sampling_nfe"] != 6+3*row["num_atoms"]):
            raise ValueError("healthy T request has an incomplete construction endpoint")
    layout_items = [(row["condition_ordinal"], 0,
                     {"program": SimpleNamespace(num_atoms=row["num_atoms"]), "prompt_token_ids": row["prompt_token_ids"],
                      "row": row}) for row in records]
    for batch in sampling_batches(layout_items, batch_size=BATCH_CAP, rank=0, world_size=1, layout_world_size=LAYOUT_WORLD_SIZE):
        members = [item[0] for item in batch]
        for index, (_, _, c) in enumerate(batch):
            row = c["row"]
            if (row["sampling_batch_ordinals"] != members or row["sampling_row_index"] != index
                    or row["sampling_batch_size"] != len(batch)):
                raise ValueError("T sampling changed the original logical batch membership/order")
    destination = args.output_dir/"native"
    destination.mkdir(exist_ok=False)
    with (destination/"paths.jsonl").open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    (destination/"_SUCCESS").touch()
    report = {**contract, "requested": len(records), "successful": sum(row["success"] for row in records),
              "failed": sum(not row["success"] for row in records), "candidates": 1,
              "path_mode": "construction_only", "primary": "native_token_construction",
              "outcome_selection": False, "failures_retained": True, "inference_mlip": False,
              "failure_reasons": dict(Counter(row["failure"] for row in records if not row["success"])),
              **{key: sum(stat[key] for stat in stats) for key in ("model_forward_calls", "model_row_evaluations",
                 "live_row_model_evaluations", "failed_row_padding_evaluations", "geometry_forward_calls")},
              "rank_statistics": stats}
    write_json(args.output_dir/"SAMPLE_FINAL.json", report)
    (args.output_dir/"_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return report


def replay_records(args, model, tokenizer, constraints, items):
    if args.world_size != 1 or args.rank != 0:
        raise ValueError("fresh scalar replay is one process preserving each original logical batch")
    if not (args.replay_jsonl.parent/"_SUCCESS").is_file():
        raise ValueError("replay requires a completed merged T ledger")
    records = read_jsonl(args.replay_jsonl)
    keyed = {int(row["condition_ordinal"]): row for row in records}
    if len(keyed) != len(records) or set(keyed) != {item[0] for item in items}:
        raise ValueError("replay input does not cover exactly the declared condition range")
    batches = []
    started = time.monotonic()
    for batch in sampling_batches(items, batch_size=BATCH_CAP, rank=0, world_size=1, layout_world_size=LAYOUT_WORLD_SIZE):
        report = replay_token_batch(model, tokenizer, constraints, batch, [keyed[item[0]] for item in batch],
                    device=model.device, checkpoint_path=args.checkpoint_path, condition_start=args.condition_start,
                    tolerance=args.replay_tolerance, max_length=args.max_length)
        batches.append(report)
        print(json.dumps({"replayed_requests": sum(row["requests"] for row in batches),
                          "elapsed_seconds": time.monotonic()-started}), flush=True)
    final = {**protocol(args), "kind": "fresh_process_T_full_logical_batch_scalar_replay", "status": "PASS",
             "source": str(args.replay_jsonl), "source_sha256": file_sha256(args.replay_jsonl),
             "tolerance": args.replay_tolerance, "requests": len(records),
             "maximum_logp_error": max(row["maximum_logp_error"] for row in batches),
             **{key: sum(row[key] for row in batches) for key in ("checked_draws", "checked_no_support",
                   "seeded_token_matches", "model_forward_calls", "model_row_evaluations", "geometry_forward_calls")},
             "batches": batches, "physics_or_SUN_evaluation": False, "optimizer_steps": 0,
             "elapsed_seconds": time.monotonic()-started}
    write_json(args.output_dir/"REPLAY.json", final)
    (args.output_dir/"_SUCCESS").touch()
    print(json.dumps(final, ensure_ascii=False, allow_nan=False), flush=True)
    return final


def main():
    args = parser().parse_args()
    validate_protocol(seed=args.seed, collection_round=args.collection_round, batch_cap=args.batch_size,
                      layout_world_size=args.sampling_layout_world_size, temperature=args.temperature)
    if (min(args.world_size, args.max_length, args.cpu_threads) < 1 or not 0 <= args.rank < args.world_size
            or not math.isfinite(args.replay_tolerance) or args.replay_tolerance < 0):
        raise ValueError("invalid allocation or replay tolerance")
    if args.merge_only:
        merge_outputs(args)
        return
    import torch
    from scripts.sample_llada_dynamic_crystals import build_dynamic_lightweight_constraints
    if not torch.cuda.is_available():
        raise RuntimeError("real mixed T policy sampling/replay requires its CUDA allocation")
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", local)
    selected = expected_conditions(args)
    model, tokenizer, policy = load_final_mixed_token_policy(args.model_path, args.checkpoint_path, device)
    constraints = build_dynamic_lightweight_constraints(tokenizer, duplicate_coordinate_mask=True,
        lattice_volume_mask=True, min_lattice_rad=1e-4, canonicalize_periodic_alias=True,
        pbc_min_distance_mask=True, pbc_min_distance_A=.5, pbc_image_radius=2)
    items = [(ordinal, 0, compile_condition(source, tokenizer, mask_id=model.mask_id, purpose="evaluation"))
             for ordinal, source in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {**protocol(args), "final_policy": policy, "torch_version": str(torch.__version__),
              "device_name": torch.cuda.get_device_name(device), "inference_autocast": "bfloat16",
              "state_representation": "original_integer_tokens", "geometry_heads_called": False,
              "support": {"duplicate_coordinate_mask": True, "lattice_volume_mask": True, "min_lattice_rad": 1e-4,
                          "canonicalize_periodic_alias": True, "pbc_min_distance_A": .5, "pbc_image_radius": 2}}
    write_json(args.output_dir/f"CONFIG.rank{args.rank}.json", config)
    if args.replay_jsonl:
        replay_records(args, model, tokenizer, constraints, items)
        return
    totals = Counter(requested=0, successful=0, model_forward_calls=0, model_row_evaluations=0,
                     live_row_model_evaluations=0, failed_row_padding_evaluations=0, geometry_forward_calls=0)
    batches = []
    started = time.monotonic()
    with (args.output_dir/f"records.rank{args.rank}.jsonl").open("x", encoding="utf-8") as stream:
        for batch in sampling_batches(items, batch_size=BATCH_CAP, rank=args.rank, world_size=args.world_size,
                                      layout_world_size=LAYOUT_WORLD_SIZE):
            records, stat = sample_token_batch(model, tokenizer, constraints, batch, device=device,
                checkpoint_path=args.checkpoint_path, condition_start=args.condition_start, max_length=args.max_length)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            batches.append(stat)
            for key in tuple(totals):
                totals[key] += stat[key]
            print(json.dumps({"rank": args.rank, **totals, "elapsed_seconds": time.monotonic()-started}), flush=True)
    write_json(args.output_dir/f"SAMPLING.rank{args.rank}.json", {
        **protocol(args), "rank": args.rank, **totals, "logical_batches": batches,
        "elapsed_seconds": time.monotonic()-started})
    (args.output_dir/f"_SUCCESS.rank{args.rank}").touch()


if __name__ == "__main__":
    main()
