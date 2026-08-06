#!/usr/bin/env python3
"""Sample MP-20 proposals with CrysLLMGen-answer-module LLaDA generation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from crystal_dlm.crysllmgen_text import (
    arrays_to_structure,
    arrays_to_torch_payload,
    assemble_crysllmgen_text,
    composition_prompt,
    coords_prompt,
    lattice_prompt,
    parse_composition_text,
    parse_coords_text,
    parse_lattice_text,
    parse_species_text,
    species_prompt,
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


def model_class_for(config):
    return AutoModelForCausalLM if getattr(config, "model_type", None) == "llada2_moe" else AutoModel


def import_process_one(crysllmgen_dir: Path):
    sys.path.insert(0, str(crysllmgen_dir.resolve()))
    from data_utils import process_one

    return process_one


def load_model_and_tokenizer(base_model_path: str, checkpoint_path: Optional[str], device: torch.device):
    tokenizer_source = checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ensure_create_bidirectional_mask()
    if checkpoint_path and (Path(checkpoint_path) / "adapter_config.json").exists():
        from peft import PeftModel

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
        model_config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        ensure_llada2_rope_parameters(model_config)
        model = model_class_for(model_config).from_pretrained(
            checkpoint_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    else:
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
def generate_texts(
    model,
    tokenizer,
    prompts: List[str],
    *,
    gen_length: int,
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
        steps=gen_length,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=MASK_TOKEN_ID,
    )
    generated_ids = outputs[:, input_ids.shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def rounded_length(raw_length: int, block_length: int) -> int:
    block_length = max(1, int(block_length))
    return max(block_length, int(math.ceil(max(1, int(raw_length)) / block_length) * block_length))


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
        "lattice_parse_success": 0,
        "composition_parse_success": 0,
        "species_parse_success": 0,
        "coords_parse_success": 0,
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
                    "lattice_parse_success",
                    "composition_parse_success",
                    "species_parse_success",
                    "coords_parse_success",
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
            for filename, handle in (("raw_generations.jsonl", raw_out), ("failure_cases.jsonl", failure_out)):
                path = rank_path(output_dir, filename, rank, True)
                if path.exists():
                    handle.write(path.read_text(encoding="utf-8"))
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
                "raw_lattice_text": record.get("raw_lattice_text", ""),
                "lattice_text": record.get("lattice_text", ""),
                "raw_composition_text": record.get("raw_composition_text", ""),
                "composition_text": record.get("composition_text", ""),
                "raw_species_text": record.get("raw_species_text", ""),
                "species_text": record.get("species_text", ""),
                "raw_coords_text": record.get("raw_coords_text", ""),
                "coords_text": record.get("coords_text", ""),
                "text": record.get("text", ""),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    record.update({"parsed": False, "reason": reason, "message": str(exc)})


def parse_lattice_batch(
    raw_texts: List[str],
    current_indices: List[int],
    metrics,
    failure_handle,
    raw_handle,
    *,
    module_style: str,
) -> tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    prompts: List[str] = []
    for sample_idx, raw_text in zip(current_indices, raw_texts):
        metrics["decoded_samples"] += 1
        record: Dict[str, Any] = {
            "sample_idx": int(sample_idx),
            "representation": "crysllmgen_text",
            "generation_mode": "crysllmgen_answer_modules",
            "raw_lattice_text": raw_text,
        }
        try:
            parsed = parse_lattice_text(raw_text)
            metrics["lattice_parse_success"] += 1
            record["lattice_text"] = parsed["text"]
            records.append(record)
            if module_style == "composition":
                prompts.append(composition_prompt(parsed["text"]))
            else:
                prompts.append(species_prompt(parsed["text"]))
        except Exception as exc:
            add_failure(metrics, failure_handle, int(sample_idx), "lattice", exc, record)
            raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records, prompts


def parse_composition_batch(raw_texts: List[str], records: List[Dict[str, Any]], metrics, failure_handle, raw_handle) -> tuple[List[Dict[str, Any]], List[str]]:
    out_records: List[Dict[str, Any]] = []
    prompts: List[str] = []
    for record, raw_text in zip(records, raw_texts):
        record["raw_composition_text"] = raw_text
        try:
            parsed = parse_composition_text(raw_text)
            metrics["composition_parse_success"] += 1
            record["composition_text"] = parsed["text"]
            record["species_text"] = parsed["species_text"]
            record["num_atoms"] = int(len(parsed["species"]))
            out_records.append(record)
            prompts.append(coords_prompt(record["lattice_text"], parsed["species_text"]))
        except Exception as exc:
            add_failure(metrics, failure_handle, int(record["sample_idx"]), "composition", exc, record)
            raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_records, prompts


def parse_species_batch(raw_texts: List[str], records: List[Dict[str, Any]], metrics, failure_handle, raw_handle) -> tuple[List[Dict[str, Any]], List[str]]:
    out_records: List[Dict[str, Any]] = []
    prompts: List[str] = []
    for record, raw_text in zip(records, raw_texts):
        record["raw_species_text"] = raw_text
        try:
            parsed = parse_species_text(raw_text)
            metrics["species_parse_success"] += 1
            record["species_text"] = parsed["text"]
            record["num_atoms"] = int(len(parsed["species"]))
            out_records.append(record)
            prompts.append(coords_prompt(record["lattice_text"], parsed["text"]))
        except Exception as exc:
            add_failure(metrics, failure_handle, int(record["sample_idx"]), "species", exc, record)
            raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_records, prompts


def finish_records(
    records: List[Dict[str, Any]],
    raw_texts: List[str],
    *,
    metrics,
    failure_handle,
    raw_handle,
    valid_arrays: List[Dict[str, Any]],
    proposal_graphs: List[Dict[str, Any]],
    process_one,
) -> None:
    for record, raw_text in zip(records, raw_texts):
        sample_idx = int(record["sample_idx"])
        record["raw_coords_text"] = raw_text
        try:
            coords = parse_coords_text(
                raw_text,
                expected_count=int(record["num_atoms"]),
                skip_invalid=True,
                deduplicate_pbc=True,
            )
            metrics["coords_parse_success"] += 1
            record["coords_text"] = coords["text"]
            arrays = assemble_crysllmgen_text(
                record["lattice_text"],
                record["species_text"],
                coords["text"],
                require_no_pbc_duplicate=True,
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
                    "text": arrays["answer"],
                    "cif": cif,
                    "num_atoms": arrays["num_atoms"],
                    "composition": arrays["composition"],
                }
            )
        except Exception as exc:
            add_failure(metrics, failure_handle, sample_idx, "coords_or_graph", exc, record)
        raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lattice-gen-length", type=int, default=48)
    parser.add_argument("--lattice-block-length", type=int, default=16)
    parser.add_argument("--module-style", choices=["composition", "species"], default="composition")
    parser.add_argument("--composition-gen-length", type=int, default=64)
    parser.add_argument("--composition-block-length", type=int, default=4)
    parser.add_argument("--species-gen-length", type=int, default=96)
    parser.add_argument("--species-block-length", type=int, default=4)
    parser.add_argument("--coords-tokens-per-site", type=int, default=16)
    parser.add_argument("--coords-extra-tokens", type=int, default=8)
    parser.add_argument("--coords-block-length", type=int, default=16)
    parser.add_argument("--coords-max-gen-length", type=int, default=352)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--target-graph-success", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args()

    for name, length, block in (
        ("lattice", args.lattice_gen_length, args.lattice_block_length),
        ("composition", args.composition_gen_length, args.composition_block_length),
        ("species", args.species_gen_length, args.species_block_length),
    ):
        if int(length) % int(block) != 0:
            raise RuntimeError(f"{name} gen length must be divisible by its block length")

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
    run_config["representation"] = "crysllmgen_text"
    run_config["generation_mode"] = f"crysllmgen_answer_modules_{args.module_style}"
    run_config["distributed"] = distributed
    run_config["world_size"] = world_size
    if is_main:
        write_json(str(args.output_dir / "run_config.json"), run_config)

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
        "lattice_parse_success": 0,
        "composition_parse_success": 0,
        "species_parse_success": 0,
        "coords_parse_success": 0,
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
        progress = tqdm(total=len(sample_indices), desc=f"CrysLLMGen modular sampling rank{rank}", disable=distributed and not is_main)
        for batch_start in range(0, len(sample_indices), args.batch_size):
            if local_target_graph_success is not None and metrics["graph_success"] >= local_target_graph_success:
                break
            current_indices = sample_indices[batch_start : batch_start + args.batch_size]
            lattice_texts = generate_texts(
                model,
                tokenizer,
                [lattice_prompt()] * len(current_indices),
                gen_length=args.lattice_gen_length,
                block_length=args.lattice_block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
            )
            lattice_records, species_prompts = parse_lattice_batch(
                lattice_texts,
                current_indices,
                metrics,
                failure_handle,
                raw_handle,
                module_style=args.module_style,
            )
            if not lattice_records:
                progress.update(len(current_indices))
                continue

            mid_texts = generate_texts(
                model,
                tokenizer,
                species_prompts,
                gen_length=args.composition_gen_length if args.module_style == "composition" else args.species_gen_length,
                block_length=args.composition_block_length if args.module_style == "composition" else args.species_block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
            )
            if args.module_style == "composition":
                species_records, coords_prompts = parse_composition_batch(
                    mid_texts,
                    lattice_records,
                    metrics,
                    failure_handle,
                    raw_handle,
                )
            else:
                species_records, coords_prompts = parse_species_batch(
                    mid_texts,
                    lattice_records,
                    metrics,
                    failure_handle,
                    raw_handle,
                )
            if not species_records:
                progress.update(len(current_indices))
                continue

            groups: Dict[int, List[tuple[Dict[str, Any], str]]] = defaultdict(list)
            for record, prompt in zip(species_records, coords_prompts):
                groups[int(record["num_atoms"])].append((record, prompt))
            finished_count = 0
            for num_atoms, grouped in sorted(groups.items()):
                records = [item[0] for item in grouped]
                prompts = [item[1] for item in grouped]
                raw_length = int(num_atoms) * int(args.coords_tokens_per_site) + int(args.coords_extra_tokens)
                gen_length = min(
                    int(args.coords_max_gen_length),
                    rounded_length(raw_length, int(args.coords_block_length)),
                )
                if gen_length % int(args.coords_block_length) != 0:
                    raise RuntimeError("computed coords gen length is not divisible by coords block length")
                coords_texts = generate_texts(
                    model,
                    tokenizer,
                    prompts,
                    gen_length=gen_length,
                    block_length=args.coords_block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                )
                finish_records(
                    records,
                    coords_texts,
                    metrics=metrics,
                    failure_handle=failure_handle,
                    raw_handle=raw_handle,
                    valid_arrays=valid_arrays,
                    proposal_graphs=proposal_graphs,
                    process_one=process_one,
                )
                finished_count += len(records)
            progress.update(len(current_indices))

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
