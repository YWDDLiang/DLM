#!/usr/bin/env python3
"""Sample dynamic-v1 MP-20 crystal proposals from a LLaDA checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from crystal_dlm.dynamic_crystal import (
    CANONICAL_DYNAMIC_PROMPT,
    DYNAMIC_MAX_ANSWER_TOKEN_COUNT,
    Z_TO_SYMBOL,
    arrays_to_structure,
    arrays_to_torch_payload,
    parse_dynamic_answer,
    write_json,
)
from crystal_dlm.fixed_slot import FixedSlotConfig, MASK_TOKEN_ID
from crystal_dlm.llada_generation import generate
from crystal_dlm.llada_resize import ensure_llada_vocab_size
from crystal_dlm.transformers_compat import (
    ensure_create_bidirectional_mask,
    ensure_llada2_rope_parameters,
)


class DuplicateCoordinateError(ValueError):
    """Raised when decoded dynamic sites place more than one atom at the same PBC coordinate."""


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
        model.resize_token_embeddings(len(tokenizer))
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
    model.to(device).eval()
    return model, tokenizer


def _required_token_ids(tokenizer, tokens: List[str]) -> List[int]:
    vocab = tokenizer.get_vocab()
    missing = [token for token in tokens if token not in vocab]
    if missing:
        raise RuntimeError(f"Tokenizer is missing required dynamic crystal tokens: {missing[:10]}")
    return [int(vocab[token]) for token in tokens]


def atom_count_token_id(tokenizer, atom_count: int) -> int:
    token = f"<N_{atom_count:03d}>"
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        raise RuntimeError(f"Tokenizer is missing atom-count token {token}")
    return int(vocab[token])


def build_dynamic_schema_constraints(
    tokenizer,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> List[List[int]]:
    allowed: List[List[int]] = []
    allowed.append(_required_token_ids(tokenizer, [f"<N_{i:03d}>" for i in range(1, config.max_atoms + 1)]))
    for prefix in ("LA", "LB", "LC"):
        allowed.append(_required_token_ids(tokenizer, [f"<{prefix}_{i:03d}>" for i in range(config.length_min_bin, config.length_max_bin + 1)]))
    for prefix in ("AA", "AB", "AG"):
        allowed.append(_required_token_ids(tokenizer, [f"<{prefix}_{i:03d}>" for i in range(config.angle_min_bin, config.angle_max_bin + 1)]))
    element_tokens = [f"<E_{Z_TO_SYMBOL[z]}>" for z in range(1, config.max_atomic_number + 1)]
    coord_tokens = {
        axis: [f"<{axis}_{i:03d}>" for i in range(config.coord_min_bin, config.coord_max_bin + 1)]
        for axis in ("X", "Y", "Z")
    }
    eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id)
    for _slot_index in range(config.max_atoms):
        allowed.append(_required_token_ids(tokenizer, element_tokens) + [eos_id])
        allowed.append(_required_token_ids(tokenizer, coord_tokens["X"]) + [eos_id])
        allowed.append(_required_token_ids(tokenizer, coord_tokens["Y"]) + [eos_id])
        allowed.append(_required_token_ids(tokenizer, coord_tokens["Z"]) + [eos_id])
    expected = 7 + config.max_atoms * 4
    if len(allowed) != expected:
        raise RuntimeError(f"Built {len(allowed)} dynamic positions, expected {expected}")
    return allowed


def build_dynamic_atom_count_grammar(
    tokenizer,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> Dict[str, Any]:
    element_tokens = [f"<E_{Z_TO_SYMBOL[z]}>" for z in range(1, config.max_atomic_number + 1)]
    coord_token_ids = {
        axis: _required_token_ids(tokenizer, [f"<{axis}_{i:03d}>" for i in range(config.coord_min_bin, config.coord_max_bin + 1)])
        for axis in ("X", "Y", "Z")
    }
    eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id)
    return {
        "representation": "dynamic_v1",
        "max_atoms": config.max_atoms,
        "count_token_to_n": {
            atom_count_token_id(tokenizer, atom_count): atom_count
            for atom_count in range(1, config.max_atoms + 1)
        },
        "element_token_ids": _required_token_ids(tokenizer, element_tokens),
        "coord_token_ids": coord_token_ids,
        "eos_token_id": eos_id,
    }


def build_dynamic_lightweight_constraints(
    tokenizer,
    *,
    duplicate_coordinate_mask: bool,
    lattice_volume_mask: bool,
    min_lattice_rad: float,
    config: FixedSlotConfig = FixedSlotConfig(),
) -> Dict[str, Any] | None:
    if not duplicate_coordinate_mask and not lattice_volume_mask:
        return None
    vocab = tokenizer.get_vocab()
    coord_token_to_bin = {
        axis: {
            int(vocab[f"<{axis}_{i:03d}>"]): i
            for i in range(config.coord_min_bin, config.coord_max_bin + 1)
        }
        for axis in ("X", "Y", "Z")
    }
    angle_token_to_bin = {
        prefix: {
            int(vocab[f"<{prefix}_{i:03d}>"]): i
            for i in range(config.angle_min_bin, config.angle_max_bin + 1)
        }
        for prefix in ("AA", "AB", "AG")
    }
    return {
        "representation": "dynamic_v1",
        "duplicate_coordinate_mask": bool(duplicate_coordinate_mask),
        "lattice_volume_mask": bool(lattice_volume_mask),
        "min_lattice_rad": float(min_lattice_rad),
        "max_atoms": config.max_atoms,
        "coord_period": config.coord_max_bin - config.coord_min_bin,
        "count_token_to_n": {
            atom_count_token_id(tokenizer, atom_count): atom_count
            for atom_count in range(1, config.max_atoms + 1)
        },
        "coord_token_to_bin": coord_token_to_bin,
        "z_bin_to_token_id": {
            i: int(vocab[f"<Z_{i:03d}>"])
            for i in range(config.coord_min_bin, config.coord_max_bin + 1)
        },
        "angle_token_to_bin": angle_token_to_bin,
        "gamma_bin_to_token_id": {
            i: int(vocab[f"<AG_{i:03d}>"])
            for i in range(config.angle_min_bin, config.angle_max_bin + 1)
        },
        "zero_length_token_ids_by_position": {
            1: int(vocab["<LA_000>"]),
            2: int(vocab["<LB_000>"]),
            3: int(vocab["<LC_000>"]),
        },
    }


def dynamic_generation_schedule(config: FixedSlotConfig = FixedSlotConfig()) -> List[List[int]]:
    element_positions = [7 + 4 * slot_index for slot_index in range(config.max_atoms)]
    x_positions = [8 + 4 * slot_index for slot_index in range(config.max_atoms)]
    y_positions = [9 + 4 * slot_index for slot_index in range(config.max_atoms)]
    z_positions = [10 + 4 * slot_index for slot_index in range(config.max_atoms)]
    return [[0], element_positions, [1, 2, 3, 4, 5, 6], x_positions, y_positions, z_positions]


def atom_count_histogram_from_csv(split: str, config: FixedSlotConfig = FixedSlotConfig()) -> Dict[str, int]:
    from pymatgen.core import Structure

    csv_path = PROJECT_ROOT / "reference/crysllmgen/data/mp_20" / f"{split}.csv"
    histogram: Counter[str] = Counter()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            structure = Structure.from_str(row["cif"], fmt="cif")
            if 1 <= len(structure) <= config.max_atoms:
                histogram[str(len(structure))] += 1
    return dict(histogram)


def load_atom_count_prior(mode: str, stats_json: Path, config: FixedSlotConfig = FixedSlotConfig()) -> Tuple[List[int], List[int]]:
    if mode == "none":
        return [], []
    if mode == "uniform":
        return list(range(1, config.max_atoms + 1)), [1] * config.max_atoms
    if stats_json.exists():
        data = json.loads(stats_json.read_text(encoding="utf-8"))
        histogram = data["splits"][mode]["atom_count_histogram"]
    else:
        histogram = atom_count_histogram_from_csv(mode, config=config)
    values_weights = sorted(
        (int(atom_count), int(count))
        for atom_count, count in histogram.items()
        if 1 <= int(atom_count) <= config.max_atoms and int(count) > 0
    )
    if not values_weights:
        raise RuntimeError(f"Atom-count prior {mode!r} is empty")
    values, weights = zip(*values_weights)
    return list(values), list(weights)


def sample_atom_count(sample_idx: int, values: List[int], weights: List[int], seed: int) -> int:
    rng = random.Random(seed + int(sample_idx) * 1009)
    return int(rng.choices(values, weights=weights, k=1)[0])


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
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge_distributed_outputs(output_dir: Path, world_size: int) -> None:
    merged_metrics = {
        "requested_samples": 0,
        "decoded_samples": 0,
        "parse_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "target_mode": False,
        "target_graph_success": None,
        "target_graph_success_assigned": 0,
        "target_reached_ranks": 0,
        "max_attempts": None,
        "prefill_atom_count_prior": "none",
        "target_atom_count_histogram": {},
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
                for key in ("requested_samples", "decoded_samples", "parse_success", "pymatgen_success", "graph_success"):
                    merged_metrics[key] += int(metrics.get(key, 0))
                merged_metrics["time_sec"] = max(merged_metrics["time_sec"], float(metrics.get("time_sec") or 0.0))
                if metrics.get("target_mode"):
                    merged_metrics["target_mode"] = True
                    merged_metrics["target_graph_success"] = metrics.get("target_graph_success")
                    merged_metrics["max_attempts"] = metrics.get("max_attempts")
                    merged_metrics["target_graph_success_assigned"] += int(metrics.get("target_graph_success_assigned") or 0)
                    merged_metrics["target_reached_ranks"] += int(bool(metrics.get("target_reached")))
                for atom_count, count in metrics.get("target_atom_count_histogram", {}).items():
                    merged_metrics["target_atom_count_histogram"][str(atom_count)] = (
                        merged_metrics["target_atom_count_histogram"].get(str(atom_count), 0) + int(count)
                    )
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


def graph_from_arrays(arrays: Dict[str, Any], process_one) -> Tuple[Dict[str, Any], str]:
    seen: Dict[Tuple[int, int, int], str] = {}
    for species, coord in zip(arrays["species"], arrays["frac_coords"]):
        key = tuple(int(round(float(value) * 100.0)) % 100 for value in coord)
        if key in seen:
            raise DuplicateCoordinateError(
                f"duplicate/PBC-equivalent fractional coordinate {key} for species {seen[key]} and {species}"
            )
        seen[key] = species
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=DYNAMIC_MAX_ANSWER_TOKEN_COUNT)
    parser.add_argument("--gen-length", type=int, default=DYNAMIC_MAX_ANSWER_TOKEN_COUNT)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--generation-schedule", choices=["dynamic-n-elements-lattice-coords"], default="dynamic-n-elements-lattice-coords")
    parser.add_argument("--prompt", default=CANONICAL_DYNAMIC_PROMPT)
    parser.add_argument("--schema-logit-mask", action="store_true", default=True)
    parser.add_argument("--no-schema-logit-mask", dest="schema_logit_mask", action="store_false")
    parser.add_argument("--atom-count-grammar-mask", action="store_true", default=True)
    parser.add_argument("--no-atom-count-grammar-mask", dest="atom_count_grammar_mask", action="store_false")
    parser.add_argument("--duplicate-coordinate-mask", action="store_true", default=True)
    parser.add_argument("--no-duplicate-coordinate-mask", dest="duplicate_coordinate_mask", action="store_false")
    parser.add_argument("--lattice-volume-mask", action="store_true", default=True)
    parser.add_argument("--no-lattice-volume-mask", dest="lattice_volume_mask", action="store_false")
    parser.add_argument("--min-lattice-rad", type=float, default=1e-4)
    parser.add_argument("--prefill-atom-count-prior", choices=["none", "uniform", "train", "val", "test"], default="none")
    parser.add_argument("--atom-count-stats-json", type=Path, default=PROJECT_ROOT / "data/dlm_sft/mp_20_dynamic_v1/stats.json")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--target-graph-success", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args()

    if args.block_length != 1:
        raise RuntimeError("dynamic-v1 sampling currently requires --block-length 1")
    if args.gen_length != DYNAMIC_MAX_ANSWER_TOKEN_COUNT:
        raise RuntimeError(f"dynamic-v1 gen length must be {DYNAMIC_MAX_ANSWER_TOKEN_COUNT}")
    if args.steps != args.gen_length:
        raise RuntimeError("dynamic-v1 uses one denoising step per generation position")
    if args.duplicate_coordinate_mask and not args.atom_count_grammar_mask:
        raise RuntimeError("--duplicate-coordinate-mask requires --atom-count-grammar-mask")

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

    allowed_token_ids_by_generation_pos = build_dynamic_schema_constraints(tokenizer) if args.schema_logit_mask else None
    atom_count_grammar = build_dynamic_atom_count_grammar(tokenizer) if args.atom_count_grammar_mask else None
    lightweight_decoding_constraints = build_dynamic_lightweight_constraints(
        tokenizer,
        duplicate_coordinate_mask=args.duplicate_coordinate_mask,
        lattice_volume_mask=args.lattice_volume_mask,
        min_lattice_rad=args.min_lattice_rad,
    )
    generation_position_groups = dynamic_generation_schedule()
    atom_count_values, atom_count_weights = load_atom_count_prior(args.prefill_atom_count_prior, args.atom_count_stats_json)

    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config["representation"] = "dynamic_v1"
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
            handle.write(json.dumps({"event": "not_applicable", "stage": "dynamic_sampling"}, ensure_ascii=False) + "\n")

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
        "parse_success": 0,
        "pymatgen_success": 0,
        "graph_success": 0,
        "target_mode": target_mode,
        "target_graph_success": target_graph_success,
        "target_graph_success_assigned": local_target_graph_success,
        "target_reached": False,
        "max_attempts": max_attempts,
        "prefill_atom_count_prior": args.prefill_atom_count_prior,
        "target_atom_count_histogram": {},
        "failures": {},
        "time_sec": None,
        "rank": rank,
        "world_size": world_size,
        "assigned_samples": len(sample_indices),
    }
    prompt_text = args.prompt.rstrip() + "\n"
    start = time.time()
    with raw_path.open("w", encoding="utf-8") as raw_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        progress = tqdm(total=len(sample_indices), desc=f"Dynamic sampling rank{rank}", disable=distributed and not is_main)
        for batch_start in range(0, len(sample_indices), args.batch_size):
            if local_target_graph_success is not None and metrics["graph_success"] >= local_target_graph_success:
                break
            current_indices = sample_indices[batch_start : batch_start + args.batch_size]
            current_batch = len(current_indices)
            current_prefill: Dict[int, List[int]] = {}
            current_target_atom_counts: List[int | None] = [None] * current_batch
            if atom_count_values:
                current_target_atom_counts = [
                    sample_atom_count(sample_idx, atom_count_values, atom_count_weights, args.seed)
                    for sample_idx in current_indices
                ]
                current_prefill[0] = [atom_count_token_id(tokenizer, int(atom_count)) for atom_count in current_target_atom_counts]
                for atom_count in current_target_atom_counts:
                    key = str(int(atom_count))
                    metrics["target_atom_count_histogram"][key] = metrics["target_atom_count_histogram"].get(key, 0) + 1
            encoded = tokenizer([prompt_text] * current_batch, add_special_tokens=False, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            outputs = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=allowed_token_ids_by_generation_pos,
                prefill_token_ids_by_generation_pos=current_prefill,
                atom_count_grammar=atom_count_grammar,
                generation_position_groups=generation_position_groups,
                lightweight_decoding_constraints=lightweight_decoding_constraints,
            )
            generated_ids = outputs[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            for sample_idx, text, target_atom_count in zip(current_indices, decoded, current_target_atom_counts):
                metrics["decoded_samples"] += 1
                raw_record: Dict[str, Any] = {"sample_idx": sample_idx, "text": text, "representation": "dynamic_v1"}
                if target_atom_count is not None:
                    raw_record["target_num_atoms"] = int(target_atom_count)
                try:
                    arrays = parse_dynamic_answer(text)
                    metrics["parse_success"] += 1
                    structure = arrays_to_structure(arrays)
                    metrics["pymatgen_success"] += 1
                    graph, cif = graph_from_arrays(arrays, process_one)
                    metrics["graph_success"] += 1
                    valid_arrays.append(arrays)
                    proposal_graphs.append(graph)
                    raw_record.update({"parsed": True, "cif": cif, "num_atoms": arrays["num_atoms"]})
                except Exception as exc:
                    reason = type(exc).__name__
                    metrics["failures"][reason] = metrics["failures"].get(reason, 0) + 1
                    failure_handle.write(
                        json.dumps(
                            {
                                "sample_idx": sample_idx,
                                "reason": reason,
                                "message": str(exc),
                                "text": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    raw_record.update({"parsed": False, "reason": reason, "message": str(exc)})
                raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
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
