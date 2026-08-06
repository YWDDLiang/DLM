#!/usr/bin/env python3
"""Sample H2 plain-text DLM proposals conditioned on generated rich plans."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm

from crystal_dlm.crysllmgen_text import (  # noqa: E402
    arrays_to_structure,
    arrays_to_torch_payload,
    parse_crysllmgen_text,
    write_json,
)
from crystal_dlm.h2_plaintext_dlm import (  # noqa: E402
    H2_PLAINTEXT_DLM_PROMPT_VERSION,
    H2_PLAINTEXT_DLM_REPRESENTATION,
    build_h2_plaintext_prompt,
    composition_signature_from_arrays,
    composition_signature_from_plan,
    proposal_matches_plan_composition,
)
from scripts.sample_llada_crysllmgen_text import (  # noqa: E402
    generate_texts,
    graph_from_arrays,
    import_process_one,
    init_distributed,
    load_model_and_tokenizer,
    rank_path,
    read_valid_arrays,
    write_valid_arrays,
)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_plan_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(path)):
        plan = row.get("plan_state") or row.get("parsed_plan")
        if not isinstance(plan, Mapping):
            continue
        rows.append(
            {
                "plan_state": dict(plan),
                "plan_text": row.get("plan_text"),
                "sample_idx": row.get("sample_idx", idx),
                "source_idx": idx,
                "source_record": row,
            }
        )
    if not rows:
        raise ValueError(f"No executable plan records found in {path}")
    return rows


def add_failure(metrics: Dict[str, Any], failure_handle, sample_idx: int, stage: str, exc: Exception, record: Dict[str, Any]) -> None:
    reason = f"{stage}:{type(exc).__name__}"
    metrics["failures"][reason] = int(metrics["failures"].get(reason, 0)) + 1
    message = str(exc)
    if "duplicate" in message.lower() or "pbc" in message.lower():
        metrics["pbc_duplicate_failures"] += 1
    failure_handle.write(
        json.dumps(
            {
                "sample_idx": int(sample_idx),
                "stage": stage,
                "reason": type(exc).__name__,
                "message": message,
                "raw_text": record.get("raw_text", ""),
                "plan_state": record.get("plan_state"),
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def merge_distributed_outputs(output_dir: Path, world_size: int) -> None:
    merged_metrics = {
        "requested_samples": 0,
        "decoded_samples": 0,
        "parse_success": 0,
        "composition_match_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "pbc_duplicate_failures": 0,
        "failures": {},
        "time_sec": 0.0,
        "distributed": True,
        "world_size": world_size,
    }
    valid_arrays: List[Dict[str, Any]] = []
    proposal_graphs: List[Dict[str, Any]] = []
    with (output_dir / "raw_generations.jsonl").open("w", encoding="utf-8") as raw_out, (
        output_dir / "failure_cases.jsonl"
    ).open("w", encoding="utf-8") as failure_out:
        for rank in range(world_size):
            metrics_path = rank_path(output_dir, "sample_metrics.json", rank, True)
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in (
                    "requested_samples",
                    "decoded_samples",
                    "parse_success",
                    "composition_match_success",
                    "pymatgen_success",
                    "graph_success",
                    "pbc_duplicate_failures",
                ):
                    merged_metrics[key] += int(metrics.get(key, 0))
                merged_metrics["time_sec"] = max(float(merged_metrics["time_sec"]), float(metrics.get("time_sec") or 0.0))
                for reason, count in metrics.get("failures", {}).items():
                    merged_metrics["failures"][reason] = int(merged_metrics["failures"].get(reason, 0)) + int(count)
            for filename, handle in (("raw_generations.jsonl", raw_out), ("failure_cases.jsonl", failure_out)):
                path = rank_path(output_dir, filename, rank, True)
                if path.exists():
                    handle.write(path.read_text(encoding="utf-8"))
            valid_arrays.extend(read_valid_arrays(rank_path(output_dir, "valid_arrays.jsonl", rank, True)))
            graph_path = rank_path(output_dir, "proposal_graphs.pt", rank, True)
            if graph_path.exists():
                proposal_graphs.extend(torch.load(graph_path, map_location="cpu"))
    decoded = max(1, int(merged_metrics["decoded_samples"]))
    merged_metrics["parse_rate"] = float(merged_metrics["parse_success"]) / decoded
    merged_metrics["composition_match_rate"] = float(merged_metrics["composition_match_success"]) / decoded
    merged_metrics["graph_acceptance_rate"] = float(merged_metrics["graph_success"]) / decoded
    merged_metrics["pbc_duplicate_rate"] = float(merged_metrics["pbc_duplicate_failures"]) / decoded
    merged_metrics["valid_array_count"] = len(valid_arrays)
    write_json(str(output_dir / "sample_metrics.json"), merged_metrics)
    if valid_arrays:
        payload = arrays_to_torch_payload(valid_arrays)
        payload["time"] = merged_metrics["time_sec"]
        torch.save(payload, output_dir / "raw_dlm_samples.pt")
        torch.save(proposal_graphs, output_dir / "proposal_graphs.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--prompt-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gen-length", type=int, default=360)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    args = parser.parse_args()

    if args.gen_length % args.block_length != 0:
        raise RuntimeError("--gen-length must be divisible by --block-length")
    dist_info = init_distributed()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    distributed = bool(dist_info["distributed"])
    is_main = bool(dist_info["is_main"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_one = import_process_one(args.crysllmgen_dir)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, dist_info["device"])
    plan_records = read_plan_records(args.prompt_jsonl)
    sample_indices = [idx for idx in range(int(args.num_samples)) if idx % world_size == rank]

    if is_main:
        run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        run_config.update(
            {
                "representation": "crysllmgen_text",
                "h2_representation": H2_PLAINTEXT_DLM_REPRESENTATION,
                "prompt_version": H2_PLAINTEXT_DLM_PROMPT_VERSION,
                "distributed": distributed,
                "world_size": world_size,
                "available_plan_records": len(plan_records),
            }
        )
        write_json(str(args.output_dir / "run_config.json"), run_config)

    raw_path = rank_path(args.output_dir, "raw_generations.jsonl", rank, distributed)
    failure_path = rank_path(args.output_dir, "failure_cases.jsonl", rank, distributed)
    valid_arrays_path = rank_path(args.output_dir, "valid_arrays.jsonl", rank, distributed)
    valid_arrays: List[Dict[str, Any]] = []
    proposal_graphs: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "requested_samples": len(sample_indices),
        "decoded_samples": 0,
        "parse_success": 0,
        "composition_match_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "pbc_duplicate_failures": 0,
        "failures": {},
        "time_sec": None,
        "rank": rank,
        "world_size": world_size,
    }
    start = time.time()
    with raw_path.open("w", encoding="utf-8") as raw_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        progress = tqdm(total=len(sample_indices), desc=f"H2 plaintext DLM rank{rank}", disable=distributed and not is_main)
        for batch_start in range(0, len(sample_indices), int(args.batch_size)):
            current_indices = sample_indices[batch_start : batch_start + int(args.batch_size)]
            batch_records = [plan_records[idx % len(plan_records)] for idx in current_indices]
            prompts = [build_h2_plaintext_prompt(item["plan_state"]) for item in batch_records]
            raw_texts = generate_texts(
                model,
                tokenizer,
                prompts,
                gen_length=int(args.gen_length),
                block_length=int(args.block_length),
                temperature=float(args.temperature),
                cfg_scale=float(args.cfg_scale),
                remasking=args.remasking,
            )
            for sample_idx, plan_record, prompt_text, raw_text in zip(current_indices, batch_records, prompts, raw_texts):
                plan_state = dict(plan_record["plan_state"])
                record: Dict[str, Any] = {
                    "sample_idx": int(sample_idx),
                    "prompt_record_idx": int(plan_record["source_idx"]),
                    "source_planner_sample_idx": plan_record.get("sample_idx"),
                    "representation": "crysllmgen_text",
                    "h2_representation": H2_PLAINTEXT_DLM_REPRESENTATION,
                    "prompt_version": H2_PLAINTEXT_DLM_PROMPT_VERSION,
                    "plan_state": plan_state,
                    "plan_signature": composition_signature_from_plan(plan_state),
                    "conditioning_prompt": prompt_text.rstrip(),
                    "raw_text": raw_text,
                    "text": raw_text,
                    "parsed": False,
                    "composition_match": False,
                }
                metrics["decoded_samples"] += 1
                try:
                    arrays = parse_crysllmgen_text(raw_text)
                    metrics["parse_success"] += 1
                    record["proposal_signature"] = composition_signature_from_arrays(arrays)
                    if not proposal_matches_plan_composition(arrays, plan_state):
                        raise ValueError(
                            f"proposal composition {record['proposal_signature']} does not match plan {record['plan_signature']}"
                        )
                    metrics["composition_match_success"] += 1
                    structure = arrays_to_structure(arrays)
                    metrics["pymatgen_success"] += 1
                    graph, cif = graph_from_arrays(arrays, process_one)
                    metrics["graph_success"] += 1
                    valid_arrays.append(arrays)
                    proposal_graphs.append(graph)
                    record.update(
                        {
                            "parsed": True,
                            "composition_match": True,
                            "text": arrays["answer"],
                            "cif": cif,
                            "num_atoms": int(arrays["num_atoms"]),
                            "pymatgen_formula": structure.composition.formula,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    add_failure(metrics, failure_handle, int(sample_idx), "decode_match_or_graph", exc, record)
                    record.update({"reason": type(exc).__name__, "message": str(exc)})
                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.update(1)

    metrics["time_sec"] = time.time() - start
    decoded = max(1, int(metrics["decoded_samples"]))
    metrics["parse_rate"] = float(metrics["parse_success"]) / decoded
    metrics["composition_match_rate"] = float(metrics["composition_match_success"]) / decoded
    metrics["graph_acceptance_rate"] = float(metrics["graph_success"]) / decoded
    metrics["pbc_duplicate_rate"] = float(metrics["pbc_duplicate_failures"]) / decoded
    metrics["valid_array_count"] = len(valid_arrays)
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), metrics)
    if valid_arrays:
        write_valid_arrays(valid_arrays_path, valid_arrays)
        torch.save(proposal_graphs, rank_path(args.output_dir, "proposal_graphs.pt", rank, distributed))
        torch.save(arrays_to_torch_payload(valid_arrays), rank_path(args.output_dir, "raw_dlm_samples.pt", rank, distributed))
    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_outputs(args.output_dir, world_size)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
