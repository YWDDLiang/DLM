#!/usr/bin/env python3
"""One H-P33 float-native sample and its fixed Q output for each frozen condition."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from crystal_dlm.mixed_geometry_sampling import METHOD, sample_compiled_batch, validate_final_policy
from crystal_dlm.programmed_path_data import compile_condition, path_seed, read_jsonl
from crystal_dlm.sampling_layout import sampling_batches
from scripts.sample_state_programmed_paths import conditions_for_run


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-path", required=True)
    result.add_argument("--checkpoint-path", type=Path, required=True)
    result.add_argument("--conditions", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--condition-start", type=int, default=0)
    result.add_argument("--condition-stop", type=int)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--sampling-layout-world-size", type=int, default=2)
    result.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    result.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    result.add_argument("--seed", type=int, default=20260905)
    result.add_argument("--collection-round", type=int, default=0)
    result.add_argument("--max-length", type=int, default=382)
    result.add_argument("--cpu-threads", type=int, default=2)
    result.add_argument("--merge-only", action="store_true")
    return result


def expected_conditions(args):
    if args.condition_stop is None:
        args.condition_stop = len(read_jsonl(args.conditions))
    # Reuse the original evaluation identity and prompt-condition contract.
    args.purpose = "evaluation"
    return conditions_for_run(args)


def merge_outputs(args):
    selected = expected_conditions(args)
    expected = {(str(record["group_id"]), 0) for _, record in selected}
    conditions_hash = file_sha256(args.conditions)
    outputs, stats = {}, []
    for rank in range(args.world_size):
        if not (args.output_dir/f"_SUCCESS.rank{rank}").is_file():
            raise ValueError("a sampling rank has not completed its request ledger")
        statistic = json.loads((args.output_dir/f"SAMPLING.rank{rank}.json").read_text(encoding="utf-8"))
        if (statistic["rank"] != rank or statistic["conditions_sha256"] != conditions_hash
                or statistic["seed"] != args.seed or statistic["collection_round"] != args.collection_round
                or statistic["checkpoint"] != str(args.checkpoint_path)
                or statistic["sampling_layout_world_size"] != args.sampling_layout_world_size
                or statistic["logical_batch_cap"] != args.batch_size):
            raise ValueError("sampling shards used different conditions, seeds, policy or logical layout")
        stats.append(statistic)
    for endpoint in ("native", "quantized"):
        records = [record for rank in range(args.world_size)
                   for record in read_jsonl(args.output_dir/f"{endpoint}.records.rank{rank}.jsonl")]
        keys = [(str(row["group_id"]), row["candidate_index"]) for row in records]
        if len(keys) != len(set(keys)) or set(keys) != expected or len({r["trajectory_id"] for r in records}) != len(records):
            raise ValueError(f"{endpoint} is missing or duplicating frozen requests")
        records.sort(key=lambda row: (row["condition_ordinal"], row["candidate_index"]))
        if [row["evaluation_ordinal"] for row in records] != list(range(len(records))):
            raise ValueError("merged evaluation ordinals changed")
        if any(row["sampling_seed"] != path_seed(args.seed, row["group_id"], args.collection_round, 0)
               or row["collection_round"] != args.collection_round for row in records):
            raise ValueError("a sampling shard changed the registered path seed")
        outputs[endpoint] = records
    paired_fields = ("sample_idx", "group_id", "condition_ordinal", "evaluation_ordinal", "prompt", "plan_state",
                     "species_program", "species_program_source", "num_atoms", "sampling_seed", "sampling_batch_size",
                     "candidate_index", "sampling_layout_world_size", "initial_geometry_prior")
    for native, quantized in zip(outputs["native"], outputs["quantized"], strict=True):
        if any(native[field] != quantized[field] for field in paired_fields):
            raise ValueError("float and Q do not describe the same frozen draw")
        for record in (native, quantized):
            if not record["success"] and (record.get("body") is not None or record.get("structure") is not None):
                raise ValueError("a failed sample left an active geometry preview")
    for endpoint, records in outputs.items():
        directory = args.output_dir/endpoint
        directory.mkdir(exist_ok=False)
        with (directory/"paths.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
        (directory/"_SUCCESS").touch()
    report = {"method": METHOD, "requested": len(selected), "conditions": len(selected), "candidates": 1,
              "condition_start": args.condition_start, "condition_stop": args.condition_stop,
              "seed": args.seed, "collection_round": args.collection_round,
              "checkpoint": str(args.checkpoint_path), "conditions_sha256": file_sha256(args.conditions),
              "execution_world_size": args.world_size, "sampling_layout_world_size": args.sampling_layout_world_size,
              "logical_batch_cap": args.batch_size, "outcome_selection": False, "failures_retained": True,
              "inference_mlip": False, "primary": "continuous_float_native", "secondary": "same_samples_Q_raw",
              "counts": {name: {"requested": len(rows), "successful": sum(row["success"] for row in rows),
                                 "failed": sum(not row["success"] for row in rows),
                                 "failure_reasons": dict(Counter(row.get("failure") for row in rows if not row["success"]))}
                         for name, rows in outputs.items()},
              "model_forward_calls": sum(s["model_forward_calls"] for s in stats),
              "model_row_evaluations": sum(s["model_row_evaluations"] for s in stats),
              "live_row_model_evaluations": sum(s["live_row_model_evaluations"] for s in stats),
              "failed_row_padding_evaluations": sum(s["failed_row_padding_evaluations"] for s in stats),
              "rank_statistics": stats, "Q_additional_neural_nfe": 0}
    write_json(args.output_dir/"SAMPLE_FINAL.json", report)
    (args.output_dir/"_SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return report


def main():
    args = parser().parse_args()
    if (min(args.batch_size, args.sampling_layout_world_size, args.world_size, args.max_length, args.cpu_threads) < 1
            or not 0 <= args.rank < args.world_size or min(args.seed, args.collection_round) < 0):
        raise ValueError("invalid sampling allocation or seed")
    if args.merge_only:
        merge_outputs(args)
        return
    import torch
    from crystal_dlm.mixed_geometry_model import load_mixed_geometry_model

    if not torch.cuda.is_available():
        raise RuntimeError("real H-P33 checkpoint sampling requires CUDA")
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda", local)
    policy = validate_final_policy(args.checkpoint_path)
    selected = expected_conditions(args)
    model, tokenizer = load_mixed_geometry_model(args.model_path, args.checkpoint_path, device, trainable=False)
    model.requires_grad_(False).eval()
    if model.normalizer.source_count != policy["training"]["source_count"]:
        raise ValueError("saved normalizer and final training source counts differ")
    items = [(ordinal, 0, compile_condition(record, tokenizer, mask_id=model.mask_id, purpose="evaluation"))
             for ordinal, record in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {**vars(args), "purpose": "evaluation", "method": METHOD, "final_policy": policy,
                     "normalizer": model.normalizer.to_dict(), "conditions_sha256": file_sha256(args.conditions),
                     "torch_version": str(torch.__version__), "device_name": torch.cuda.get_device_name(device),
                     "prior_device": "cpu", "prior_dtype": "float64", "integration_dtype": "float64",
                     "network_state_dtype": "float32", "inference_autocast": "bfloat16"}
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in configuration.items()}
    write_json(args.output_dir/f"CONFIG.rank{args.rank}.json", configuration)
    totals = Counter(model_forward_calls=0, model_row_evaluations=0, live_row_model_evaluations=0,
                     failed_row_padding_evaluations=0, requested=0, native_successful=0, quantized_successful=0)
    batches = []
    started = time.monotonic()
    with (args.output_dir/f"native.records.rank{args.rank}.jsonl").open("x", encoding="utf-8") as native_stream, \
         (args.output_dir/f"quantized.records.rank{args.rank}.jsonl").open("x", encoding="utf-8") as q_stream:
        for batch in sampling_batches(items, batch_size=args.batch_size, rank=args.rank, world_size=args.world_size,
                                      layout_world_size=args.sampling_layout_world_size):
            native, quantized, statistics = sample_compiled_batch(model, tokenizer, batch, device=device,
                    checkpoint_path=args.checkpoint_path, seed=args.seed, collection_round=args.collection_round,
                    condition_start=args.condition_start, layout_world_size=args.sampling_layout_world_size, max_length=args.max_length)
            for records, stream in ((native, native_stream), (quantized, q_stream)):
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush()
            batches.append(statistics)
            for key in tuple(totals):
                totals[key] += len(batch) if key == "requested" else statistics.get(key, 0)
            print(json.dumps({"rank": args.rank, **totals, "elapsed_seconds": time.monotonic()-started}), flush=True)
    write_json(args.output_dir/f"SAMPLING.rank{args.rank}.json", {
        "rank": args.rank, **totals, "logical_batches": batches, "elapsed_seconds": time.monotonic()-started,
        "conditions_sha256": configuration["conditions_sha256"], "seed": args.seed,
        "collection_round": args.collection_round,
        "checkpoint": str(args.checkpoint_path), "sampling_layout_world_size": args.sampling_layout_world_size,
        "logical_batch_cap": args.batch_size})
    (args.output_dir/f"_SUCCESS.rank{args.rank}").touch()


if __name__ == "__main__":
    main()
