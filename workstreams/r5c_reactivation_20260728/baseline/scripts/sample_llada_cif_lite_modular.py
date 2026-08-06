#!/usr/bin/env python3
"""Sample CIF-lite modular MP-20 proposals from a LLaDA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from crystal_dlm.cif_lite import (
    arrays_to_structure,
    arrays_to_torch_payload,
    assert_no_pbc_duplicate,
    build_module_prompt,
    parse_cif_lite_modules,
    parse_composition_block,
    parse_lattice_block,
    parse_sites_block,
    truncate_module_text,
    write_json,
)
from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.llada_generation import generate
from crystal_dlm.llada_resize import ensure_llada_vocab_size
from crystal_dlm.transformers_compat import (
    ensure_create_bidirectional_mask,
    ensure_llada2_rope_parameters,
)


def init_distributed() -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed sampling requires CUDA.")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "is_main": rank == 0,
    }


def import_process_one(crysllmgen_dir: Path):
    sys.path.insert(0, str(crysllmgen_dir.resolve()))
    from data_utils import process_one

    return process_one


def model_class_for(config):
    return AutoModelForCausalLM if getattr(config, "model_type", None) == "llada2_moe" else AutoModel


def load_model_and_tokenizer(base_model_path: str, checkpoint_path: Optional[str], device: torch.device):
    tokenizer_source = checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if checkpoint_path and (Path(checkpoint_path) / "adapter_config.json").exists():
        from peft import PeftModel

        ensure_create_bidirectional_mask()
        model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        ensure_llada2_rope_parameters(model_config)
        model = model_class_for(model_config).from_pretrained(
            base_model_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        ensure_llada_vocab_size(model, len(tokenizer))
        model = PeftModel.from_pretrained(model, checkpoint_path)
    elif checkpoint_path:
        ensure_create_bidirectional_mask()
        model_config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        ensure_llada2_rope_parameters(model_config)
        model = model_class_for(model_config).from_pretrained(
            checkpoint_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    else:
        ensure_create_bidirectional_mask()
        model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        ensure_llada2_rope_parameters(model_config)
        model = model_class_for(model_config).from_pretrained(
            base_model_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        ensure_llada_vocab_size(model, len(tokenizer))
    model.to(device).eval()
    return model, tokenizer


def model_device(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def generate_module_texts(
    model,
    tokenizer,
    prompts: List[str],
    *,
    gen_length: int,
    steps: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
) -> List[str]:
    encoded = tokenizer(
        [prompt.rstrip() + "\n" for prompt in prompts],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(model_device(model))
    attention_mask = encoded["attention_mask"].to(model_device(model))
    outputs = generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=MASK_TOKEN_ID,
    )
    generated_ids = outputs[:, input_ids.shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def rank_path(output_dir: Path, filename: str, rank: int, distributed: bool) -> Path:
    if not distributed:
        return output_dir / filename
    path = Path(filename)
    return output_dir / f"{path.stem}.rank{rank}{path.suffix}"


def write_valid_arrays(path: Path, arrays_list: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for arrays in arrays_list:
            handle.write(json.dumps(arrays, ensure_ascii=False) + "\n")


def read_valid_arrays(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def graph_from_arrays(arrays: Dict[str, Any], process_one) -> Tuple[Dict[str, Any], str]:
    assert_no_pbc_duplicate(arrays["frac_coords"])
    structure = arrays_to_structure(arrays)
    cif = structure.to(fmt="cif")
    (
        frac_coords,
        atom_types,
        lengths,
        angles,
        num_atoms,
        edge_indices,
        to_jimages,
        data_dict,
    ) = process_one(cif, True, False, "crystalnn", False, 0.01)
    return data_dict, cif


def merge_distributed_outputs(output_dir: Path, world_size: int) -> None:
    merged_metrics = {
        "requested_samples": 0,
        "decoded_samples": 0,
        "composition_parse_success": 0,
        "lattice_parse_success": 0,
        "sites_parse_success": 0,
        "parse_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "target_mode": False,
        "target_graph_success": None,
        "target_graph_success_assigned": 0,
        "target_reached_ranks": 0,
        "max_attempts": None,
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
                    "composition_parse_success",
                    "lattice_parse_success",
                    "sites_parse_success",
                    "parse_success",
                    "pymatgen_success",
                    "graph_success",
                ):
                    merged_metrics[key] += int(metrics.get(key, 0))
                merged_metrics["time_sec"] = max(merged_metrics["time_sec"], float(metrics.get("time_sec") or 0.0))
                if metrics.get("target_mode"):
                    merged_metrics["target_mode"] = True
                    merged_metrics["target_graph_success"] = metrics.get("target_graph_success")
                    merged_metrics["max_attempts"] = metrics.get("max_attempts")
                    merged_metrics["target_graph_success_assigned"] += int(metrics.get("target_graph_success_assigned") or 0)
                    merged_metrics["target_reached_ranks"] += int(bool(metrics.get("target_reached")))
                for reason, count in metrics.get("failures", {}).items():
                    merged_metrics["failures"][reason] = merged_metrics["failures"].get(reason, 0) + int(count)
            raw_path = rank_path(output_dir, "raw_generations.jsonl", rank, True)
            if raw_path.exists():
                raw_out.write(raw_path.read_text(encoding="utf-8"))
            failure_path = rank_path(output_dir, "failure_cases.jsonl", rank, True)
            if failure_path.exists():
                failure_out.write(failure_path.read_text(encoding="utf-8"))
            valid_arrays.extend(read_valid_arrays(rank_path(output_dir, "valid_arrays.jsonl", rank, True)))
            graph_path = rank_path(output_dir, "proposal_graphs.pt", rank, True)
            if graph_path.exists():
                proposal_graphs.extend(torch.load(graph_path, map_location="cpu"))

    merged_metrics["parse_rate"] = merged_metrics["parse_success"] / max(1, merged_metrics["decoded_samples"])
    merged_metrics["graph_rate"] = merged_metrics["graph_success"] / max(1, merged_metrics["decoded_samples"])
    merged_metrics["graph_acceptance_rate"] = merged_metrics["graph_rate"]
    if merged_metrics["target_mode"]:
        target = int(merged_metrics["target_graph_success"] or 0)
        merged_metrics["target_reached"] = (
            merged_metrics["target_reached_ranks"] == world_size
            and merged_metrics["graph_success"] >= target
        )
    else:
        merged_metrics["target_reached"] = False
    merged_metrics["valid_array_count"] = len(valid_arrays)
    write_json(str(output_dir / "sample_metrics.json"), merged_metrics)
    if valid_arrays:
        payload = arrays_to_torch_payload(valid_arrays)
        payload["time"] = merged_metrics["time_sec"]
        torch.save(payload, output_dir / "raw_dlm_samples.pt")
        torch.save(proposal_graphs, output_dir / "proposal_graphs.pt")


def add_failure(metrics: Dict[str, Any], failure_handle, sample_idx: int, stage: str, exc: Exception, record: Dict[str, Any]) -> None:
    reason = f"{stage}:{type(exc).__name__}"
    metrics["failures"][reason] = metrics["failures"].get(reason, 0) + 1
    failure_handle.write(
        json.dumps(
            {
                "sample_idx": sample_idx,
                "stage": stage,
                "reason": type(exc).__name__,
                "message": str(exc),
                **{key: value for key, value in record.items() if key.endswith("_text") or key == "text"},
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    record.update({"parsed": False, "reason": reason, "message": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--composition-gen-length", type=int, default=96)
    parser.add_argument("--lattice-gen-length", type=int, default=64)
    parser.add_argument("--sites-gen-length", type=int, default=384)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--target-graph-success", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args()

    if args.block_length != 1:
        raise RuntimeError("CIF-lite modular sampling currently requires --block-length 1")
    for name, length in (
        ("composition", args.composition_gen_length),
        ("lattice", args.lattice_gen_length),
        ("sites", args.sites_gen_length),
    ):
        if length % args.block_length != 0:
            raise RuntimeError(f"{name} gen length must be divisible by block length")

    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_one = import_process_one(args.crysllmgen_dir)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, dist_info["device"])
    if tokenizer.pad_token_id == MASK_TOKEN_ID:
        raise RuntimeError("tokenizer.pad_token_id must not equal LLaDA mask token id 126336")

    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config["representation"] = "cif_lite_modular"
    run_config["distributed"] = distributed
    run_config["world_size"] = world_size
    if is_main:
        write_json(str(args.output_dir / "run_config.json"), run_config)
        write_json(
            str(args.output_dir / "tokenizer_report.json"),
            {
                "model_path": args.model_path,
                "checkpoint_path": args.checkpoint_path,
                "vocab_size": len(tokenizer),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "mask_token_id": MASK_TOKEN_ID,
            },
        )
        with (args.output_dir / "training_log.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "not_applicable", "stage": "cif_lite_modular_sampling"}, ensure_ascii=False) + "\n")

    raw_path = rank_path(args.output_dir, "raw_generations.jsonl", rank, distributed)
    failure_path = rank_path(args.output_dir, "failure_cases.jsonl", rank, distributed)
    valid_arrays_path = rank_path(args.output_dir, "valid_arrays.jsonl", rank, distributed)
    valid_arrays: List[Dict[str, Any]] = []
    proposal_graphs: List[Dict[str, Any]] = []
    target_mode = args.target_graph_success is not None
    if target_mode:
        target_graph_success = int(args.target_graph_success or 0)
        max_attempts = int(args.max_attempts or args.num_samples)
        base_target = target_graph_success // world_size
        target_remainder = target_graph_success % world_size
        local_target_graph_success = base_target + int(rank < target_remainder)
        sample_indices = list(range(rank, max_attempts, world_size))
    else:
        target_graph_success = None
        max_attempts = None
        local_target_graph_success = None
        sample_indices = list(range(rank, args.num_samples, world_size))

    metrics = {
        "requested_samples": len(sample_indices),
        "decoded_samples": 0,
        "composition_parse_success": 0,
        "lattice_parse_success": 0,
        "sites_parse_success": 0,
        "parse_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "target_mode": target_mode,
        "target_graph_success": target_graph_success,
        "target_graph_success_assigned": local_target_graph_success,
        "target_reached": False,
        "max_attempts": max_attempts,
        "failures": {},
        "time_sec": None,
        "rank": rank,
        "world_size": world_size,
        "assigned_samples": len(sample_indices),
    }

    start = time.time()
    with raw_path.open("w", encoding="utf-8") as raw_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        progress = tqdm(total=len(sample_indices), desc=f"CIF-lite sampling rank{rank}", disable=distributed and not is_main)
        for batch_start in range(0, len(sample_indices), args.batch_size):
            if local_target_graph_success is not None and metrics["graph_success"] >= local_target_graph_success:
                break
            current_indices = sample_indices[batch_start : batch_start + args.batch_size]
            comp_texts = generate_module_texts(
                model,
                tokenizer,
                [build_module_prompt("composition")] * len(current_indices),
                gen_length=args.composition_gen_length,
                steps=args.composition_gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
            )
            comp_records: List[Dict[str, Any]] = []
            lattice_prompts: List[str] = []
            for sample_idx, comp_raw in zip(current_indices, comp_texts):
                metrics["decoded_samples"] += 1
                record: Dict[str, Any] = {
                    "sample_idx": sample_idx,
                    "representation": "cif_lite_modular",
                    "composition_text": truncate_module_text(comp_raw, "composition"),
                    "raw_composition_text": comp_raw,
                }
                try:
                    parse_composition_block(record["composition_text"])
                    metrics["composition_parse_success"] += 1
                    comp_records.append(record)
                    lattice_prompts.append(build_module_prompt("lattice", composition_block=record["composition_text"]))
                except Exception as exc:
                    add_failure(metrics, failure_handle, sample_idx, "composition", exc, record)
                    raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress.update(1)

            if not comp_records:
                continue

            lattice_texts = generate_module_texts(
                model,
                tokenizer,
                lattice_prompts,
                gen_length=args.lattice_gen_length,
                steps=args.lattice_gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
            )
            site_prompts: List[str] = []
            site_records: List[Dict[str, Any]] = []
            for record, lattice_raw in zip(comp_records, lattice_texts):
                record["lattice_text"] = truncate_module_text(lattice_raw, "lattice")
                record["raw_lattice_text"] = lattice_raw
                try:
                    parse_lattice_block(record["lattice_text"])
                    metrics["lattice_parse_success"] += 1
                    site_records.append(record)
                    site_prompts.append(
                        build_module_prompt(
                            "sites",
                            composition_block=record["composition_text"],
                            lattice_block=record["lattice_text"],
                        )
                    )
                except Exception as exc:
                    add_failure(metrics, failure_handle, int(record["sample_idx"]), "lattice", exc, record)
                    raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress.update(1)

            if not site_records:
                continue

            sites_texts = generate_module_texts(
                model,
                tokenizer,
                site_prompts,
                gen_length=args.sites_gen_length,
                steps=args.sites_gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
            )
            for record, sites_raw in zip(site_records, sites_texts):
                sample_idx = int(record["sample_idx"])
                record["sites_text"] = truncate_module_text(sites_raw, "sites")
                record["raw_sites_text"] = sites_raw
                try:
                    parse_sites_block(record["sites_text"])
                    metrics["sites_parse_success"] += 1
                    arrays = parse_cif_lite_modules(
                        record["composition_text"],
                        record["lattice_text"],
                        record["sites_text"],
                    )
                    metrics["parse_success"] += 1
                    structure = arrays_to_structure(arrays)
                    metrics["pymatgen_success"] += 1
                    graph, cif = graph_from_arrays(arrays, process_one)
                    metrics["graph_success"] += 1
                    valid_arrays.append(arrays)
                    proposal_graphs.append(graph)
                    record.update(
                        {
                            "parsed": True,
                            "cif": cif,
                            "num_atoms": arrays["num_atoms"],
                            "text": arrays["answer"],
                        }
                    )
                except Exception as exc:
                    add_failure(metrics, failure_handle, sample_idx, "sites_or_graph", exc, record)
                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.update(1)

    metrics["time_sec"] = time.time() - start
    metrics["parse_rate"] = metrics["parse_success"] / max(1, metrics["decoded_samples"])
    metrics["graph_rate"] = metrics["graph_success"] / max(1, metrics["decoded_samples"])
    metrics["graph_acceptance_rate"] = metrics["graph_rate"]
    metrics["valid_array_count"] = len(valid_arrays)
    metrics["target_reached"] = local_target_graph_success is not None and metrics["graph_success"] >= local_target_graph_success
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), metrics)
    if valid_arrays:
        write_valid_arrays(valid_arrays_path, valid_arrays)
        if distributed:
            torch.save(proposal_graphs, rank_path(args.output_dir, "proposal_graphs.pt", rank, True))
        else:
            payload = arrays_to_torch_payload(valid_arrays)
            payload["time"] = metrics["time_sec"]
            torch.save(payload, args.output_dir / "raw_dlm_samples.pt")
            torch.save(proposal_graphs, args.output_dir / "proposal_graphs.pt")

    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_outputs(args.output_dir, world_size)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
