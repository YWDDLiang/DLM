#!/usr/bin/env python3
"""Sample fixed-slot MP-20 crystal proposals from a LLaDA checkpoint."""

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

from crystal_dlm.fixed_slot import (
    ANSWER_TOKEN_COUNT,
    CANONICAL_PROMPT,
    FixedSlotConfig,
    MASK_TOKEN_ID,
    PROMPT_POOL,
    Z_TO_SYMBOL,
    arrays_to_structure,
    arrays_to_torch_payload,
    parse_fixed_slot_answer,
    write_json,
)
from crystal_dlm.fixed_slot_compressed import (
    CompressedFixedSlotConfig,
    parse_compressed_fixed_slot_answer,
    schema_allowed_token_strings,
)
from crystal_dlm.physical_header import (
    PHYSICAL_HEADER_ANSWER_TOKEN_COUNT,
    PHYSICAL_HEADER_BODY_OFFSET,
    PHYSICAL_HEADER_CANONICAL_PROMPT,
    PHYSICAL_HEADER_PROMPT_POOL,
    parse_physical_header_answer,
    physical_header_allowed_token_strings,
)
from crystal_dlm.generation_schedule import (
    n_elements_coords_lattice_schedule,
    n_elements_sequential_rest_schedule,
)
from crystal_dlm.llada_generation import generate
from crystal_dlm.llada_resize import ensure_llada_vocab_size
from crystal_dlm.transformers_compat import (
    ensure_create_bidirectional_mask,
    ensure_llada2_rope_parameters,
)


class DuplicateCoordinateError(ValueError):
    """Raised when decoded atom slots place more than one atom at the same coordinate."""


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


def load_model_and_tokenizer(base_model_path: str, checkpoint_path: Optional[str], device: torch.device):
    if checkpoint_path and not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"--checkpoint-path does not exist: {checkpoint_path}")
    tokenizer_source = checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def model_class_for(config):
        return (
            AutoModelForCausalLM
            if getattr(config, "model_type", None) == "llada2_moe"
            else AutoModel
        )

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
        raise RuntimeError(f"Tokenizer is missing required crystal tokens: {missing[:10]}")
    return [int(vocab[token]) for token in tokens]


def build_schema_generation_constraints(
    tokenizer,
    config: FixedSlotConfig | CompressedFixedSlotConfig = FixedSlotConfig(),
    *,
    header_allowed_strings: List[List[str]] | None = None,
    body_offset: int = 0,
) -> Tuple[List[List[int]], Dict[int, int]]:
    if header_allowed_strings is not None:
        header_allowed = [
            _required_token_ids(tokenizer, allowed_strings)
            for allowed_strings in header_allowed_strings
        ]
        body_allowed, body_prefill = build_schema_generation_constraints(tokenizer, config=config)
        shifted_prefill = {
            int(position) + int(body_offset): token_id
            for position, token_id in body_prefill.items()
        }
        return header_allowed + body_allowed, shifted_prefill

    if isinstance(config, CompressedFixedSlotConfig):
        allowed_strings = schema_allowed_token_strings(config)
        allowed = [
            _required_token_ids(tokenizer, allowed_strings[position])
            for position in range(ANSWER_TOKEN_COUNT)
        ]
        prefill: Dict[int, int] = {}
        for slot_index in range(config.max_atoms):
            slot_pos = 7 + slot_index * 5
            slot_token = f"<S{slot_index:02d}>"
            prefill[slot_pos] = _required_token_ids(tokenizer, [slot_token])[0]
        return allowed, prefill

    allowed: List[List[int]] = []
    prefill: Dict[int, int] = {}

    allowed.append(_required_token_ids(tokenizer, [f"<N_{i:03d}>" for i in range(1, config.max_atoms + 1)]))
    for prefix in ("LA", "LB", "LC"):
        allowed.append(
            _required_token_ids(
                tokenizer,
                [f"<{prefix}_{i:03d}>" for i in range(config.length_min_bin, config.length_max_bin + 1)],
            )
        )
    for prefix in ("AA", "AB", "AG"):
        allowed.append(
            _required_token_ids(
                tokenizer,
                [f"<{prefix}_{i:03d}>" for i in range(config.angle_min_bin, config.angle_max_bin + 1)],
            )
        )

    element_tokens = [
        f"<E_{Z_TO_SYMBOL[z]}>" for z in range(1, config.max_atomic_number + 1)
    ]
    coord_tokens = {
        axis: [f"<{axis}_{i:03d}>" for i in range(config.coord_min_bin, config.coord_max_bin + 1)]
        for axis in ("X", "Y", "Z")
    }
    for slot_index in range(config.max_atoms):
        slot_pos = len(allowed)
        slot_token = f"<S{slot_index:02d}>"
        slot_token_id = _required_token_ids(tokenizer, [slot_token])[0]
        allowed.append([slot_token_id])
        prefill[slot_pos] = slot_token_id
        allowed.append(_required_token_ids(tokenizer, element_tokens + ["<EMPTY>"]))
        allowed.append(_required_token_ids(tokenizer, coord_tokens["X"] + ["<X_PAD>"]))
        allowed.append(_required_token_ids(tokenizer, coord_tokens["Y"] + ["<Y_PAD>"]))
        allowed.append(_required_token_ids(tokenizer, coord_tokens["Z"] + ["<Z_PAD>"]))

    expected_token_count = 1 + 6 + config.max_atoms * 5
    if len(allowed) != expected_token_count:
        raise RuntimeError(f"Built {len(allowed)} schema positions, expected {expected_token_count}")
    return allowed, prefill


def build_atom_count_grammar(
    tokenizer,
    config: FixedSlotConfig | CompressedFixedSlotConfig = FixedSlotConfig(),
    *,
    body_offset: int = 0,
) -> Dict[str, Any]:
    element_tokens = [
        f"<E_{Z_TO_SYMBOL[z]}>" for z in range(1, config.max_atomic_number + 1)
    ]
    if isinstance(config, CompressedFixedSlotConfig) and config.share_coordinates:
        coord_ids = _required_token_ids(
            tokenizer,
            [f"<C_{i:03d}>" for i in range(config.coord_min_bin, config.coord_max_bin + 1)],
        )
        coord_token_ids = {axis: coord_ids for axis in ("X", "Y", "Z")}
        pad_token_id = _required_token_ids(tokenizer, ["<C_PAD>"])[0]
        pad_coord_token_ids = {axis: pad_token_id for axis in ("X", "Y", "Z")}
    else:
        coord_token_ids = {
            axis: _required_token_ids(
                tokenizer,
                [f"<{axis}_{i:03d}>" for i in range(config.coord_min_bin, config.coord_max_bin + 1)],
            )
            for axis in ("X", "Y", "Z")
        }
        pad_coord_token_ids = {
            axis: _required_token_ids(tokenizer, [f"<{axis}_PAD>"])[0]
            for axis in ("X", "Y", "Z")
        }
    return {
        "body_offset": int(body_offset),
        "max_atoms": config.max_atoms,
        "count_token_to_n": {
            atom_count_token_id(tokenizer, atom_count): atom_count
            for atom_count in range(1, config.max_atoms + 1)
        },
        "element_token_ids": _required_token_ids(tokenizer, element_tokens),
        "empty_token_id": _required_token_ids(tokenizer, ["<EMPTY>"])[0],
        "coord_token_ids": coord_token_ids,
        "pad_coord_token_ids": pad_coord_token_ids,
    }


def build_lightweight_decoding_constraints(
    tokenizer,
    *,
    duplicate_coordinate_mask: bool,
    lattice_volume_mask: bool,
    min_lattice_rad: float,
    config: FixedSlotConfig | CompressedFixedSlotConfig = FixedSlotConfig(),
    body_offset: int = 0,
) -> Dict[str, Any] | None:
    if not duplicate_coordinate_mask and not lattice_volume_mask:
        return None
    vocab = tokenizer.get_vocab()
    if isinstance(config, CompressedFixedSlotConfig) and config.share_coordinates:
        shared_coord = {
            int(vocab[f"<C_{i:03d}>"]): i
            for i in range(config.coord_min_bin, config.coord_max_bin + 1)
        }
        coord_token_to_bin = {axis: shared_coord for axis in ("X", "Y", "Z")}
        z_bin_to_token_id = {
            i: int(vocab[f"<C_{i:03d}>"])
            for i in range(config.coord_min_bin, config.coord_max_bin + 1)
        }
    else:
        coord_token_to_bin = {
            axis: {
                int(vocab[f"<{axis}_{i:03d}>"]): i
                for i in range(config.coord_min_bin, config.coord_max_bin + 1)
            }
            for axis in ("X", "Y", "Z")
        }
        z_bin_to_token_id = {
            i: int(vocab[f"<Z_{i:03d}>"])
            for i in range(config.coord_min_bin, config.coord_max_bin + 1)
        }
    if isinstance(config, CompressedFixedSlotConfig) and config.share_angles:
        shared_angle = {
            int(vocab[f"<A_{i:03d}>"]): i
            for i in range(config.angle_min_bin, config.angle_max_bin + 1)
        }
        angle_token_to_bin = {prefix: shared_angle for prefix in ("AA", "AB", "AG")}
        gamma_bin_to_token_id = {
            i: int(vocab[f"<A_{i:03d}>"])
            for i in range(config.angle_min_bin, config.angle_max_bin + 1)
        }
    else:
        angle_token_to_bin = {
            prefix: {
                int(vocab[f"<{prefix}_{i:03d}>"]): i
                for i in range(config.angle_min_bin, config.angle_max_bin + 1)
            }
            for prefix in ("AA", "AB", "AG")
        }
        gamma_bin_to_token_id = {
            i: int(vocab[f"<AG_{i:03d}>"])
            for i in range(config.angle_min_bin, config.angle_max_bin + 1)
        }
    if isinstance(config, CompressedFixedSlotConfig) and config.share_lengths:
        zero_length_token_ids_by_position = {
            int(body_offset) + 1: int(vocab["<L_000>"]),
            int(body_offset) + 2: int(vocab["<L_000>"]),
            int(body_offset) + 3: int(vocab["<L_000>"]),
        }
    else:
        zero_length_token_ids_by_position = {
            int(body_offset) + 1: int(vocab["<LA_000>"]),
            int(body_offset) + 2: int(vocab["<LB_000>"]),
            int(body_offset) + 3: int(vocab["<LC_000>"]),
        }
    return {
        "body_offset": int(body_offset),
        "duplicate_coordinate_mask": bool(duplicate_coordinate_mask),
        "lattice_volume_mask": bool(lattice_volume_mask),
        "representation": (
            "fixed_slot_compressed_v1"
            if isinstance(config, CompressedFixedSlotConfig)
            else "fixed_slot"
        ),
        "min_lattice_rad": float(min_lattice_rad),
        "max_atoms": config.max_atoms,
        "coord_period": config.coord_max_bin - config.coord_min_bin,
        "count_token_to_n": {
            atom_count_token_id(tokenizer, atom_count): atom_count
            for atom_count in range(1, config.max_atoms + 1)
        },
        "coord_token_to_bin": coord_token_to_bin,
        "z_bin_to_token_id": z_bin_to_token_id,
        "angle_token_to_bin": angle_token_to_bin,
        "gamma_bin_to_token_id": gamma_bin_to_token_id,
        "zero_length_token_ids_by_position": zero_length_token_ids_by_position,
    }


def load_compressed_config(args) -> CompressedFixedSlotConfig | None:
    if args.representation != "fixed_slot_compressed_v1":
        return None
    candidates: List[Path] = []
    if args.compressed_token_config is not None:
        candidates.append(args.compressed_token_config)
    if args.checkpoint_path:
        candidates.append(Path(args.checkpoint_path) / "compressed_token_config.json")
    for path in candidates:
        if path.exists():
            return CompressedFixedSlotConfig.from_path(path)
    return CompressedFixedSlotConfig()


def atom_count_histogram_from_csv(split: str, config: FixedSlotConfig = FixedSlotConfig()) -> Dict[str, int]:
    from pymatgen.core import Structure

    csv_path = PROJECT_ROOT / "reference/crysllmgen/data/mp_20" / f"{split}.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Cannot build atom-count prior; missing {csv_path}")
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
        try:
            histogram = data["splits"][mode]["atom_count_histogram"]
        except KeyError as exc:
            raise RuntimeError(f"Cannot find atom_count_histogram for split {mode!r} in {stats_json}") from exc
    else:
        histogram = atom_count_histogram_from_csv(mode, config=config)
    values_weights = sorted(
        (int(atom_count), int(count))
        for atom_count, count in histogram.items()
        if 1 <= int(atom_count) <= config.max_atoms and int(count) > 0
    )
    if not values_weights:
        raise RuntimeError(f"Atom-count prior {mode!r} in {stats_json} is empty")
    values, weights = zip(*values_weights)
    return list(values), list(weights)


def sample_atom_count(sample_idx: int, values: List[int], weights: List[int], seed: int) -> int:
    rng = random.Random(seed + int(sample_idx) * 1009)
    return int(rng.choices(values, weights=weights, k=1)[0])


def read_prompt_jsonl(path: Path, prompt_field: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get(prompt_field) in (None, ""):
                raise ValueError(f"Prompt JSONL row {idx} has no field {prompt_field!r}")
            records.append(row)
    if not records:
        raise ValueError(f"Prompt JSONL is empty: {path}")
    return records


def prompt_record_for_index(
    prompt_records: List[Dict[str, Any]],
    sample_idx: int,
    prompt_field: str,
) -> Tuple[str, Dict[str, Any], int]:
    record_idx = int(sample_idx) % len(prompt_records)
    record = prompt_records[record_idx]
    return str(record[prompt_field]).rstrip() + "\n", record, record_idx


def atom_count_token_id(tokenizer, atom_count: int) -> int:
    token = f"<N_{atom_count:03d}>"
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        raise RuntimeError(f"Tokenizer is missing atom-count token {token}")
    return int(vocab[token])


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
    if not path.exists():
        return []
    rows = []
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
                    merged_metrics["target_graph_success_assigned"] += int(
                        metrics.get("target_graph_success_assigned") or 0
                    )
                    merged_metrics["target_reached_ranks"] += int(bool(metrics.get("target_reached")))
                if metrics.get("prefill_atom_count_prior") and metrics.get("prefill_atom_count_prior") != "none":
                    merged_metrics["prefill_atom_count_prior"] = metrics.get("prefill_atom_count_prior")
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
        write_valid_arrays(output_dir / "valid_arrays.jsonl", valid_arrays)
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
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument(
        "--representation",
        choices=["fixed_slot", "fixed_slot_compressed_v1", "fixed_slot_physical_header"],
        default="fixed_slot",
    )
    parser.add_argument(
        "--compressed-token-config",
        type=Path,
        default=None,
        help="compressed_token_config.json for fixed_slot_compressed_v1. Defaults to checkpoint_path/compressed_token_config.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=ANSWER_TOKEN_COUNT)
    parser.add_argument("--gen-length", type=int, default=ANSWER_TOKEN_COUNT)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument(
        "--generation-schedule",
        choices=["default", "n-elements-coords-lattice", "n-elements-sequential-rest"],
        default="default",
        help="Optional position-group denoising schedule. Default preserves the original generation path.",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--prompt-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL prompt pool. Each sample uses row sample_idx %% len(pool); "
            "useful for R5 prompt-side z conditioning while keeping the 107-token answer."
        ),
    )
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--schema-logit-mask", action="store_true", default=True)
    parser.add_argument("--no-schema-logit-mask", dest="schema_logit_mask", action="store_false")
    parser.add_argument("--prefill-slot-tokens", action="store_true", default=True)
    parser.add_argument("--no-prefill-slot-tokens", dest="prefill_slot_tokens", action="store_false")
    parser.add_argument(
        "--duplicate-coordinate-mask",
        action="store_true",
        default=True,
        help="For block_length=1 sampling, forbid Z tokens that exactly/PBC-duplicate a previous active slot coordinate.",
    )
    parser.add_argument("--no-duplicate-coordinate-mask", dest="duplicate_coordinate_mask", action="store_false")
    parser.add_argument(
        "--lattice-volume-mask",
        action="store_true",
        default=True,
        help="For block_length=1 sampling, forbid zero lattice lengths and gamma angles with invalid lattice volume factor.",
    )
    parser.add_argument("--no-lattice-volume-mask", dest="lattice_volume_mask", action="store_false")
    parser.add_argument("--min-lattice-rad", type=float, default=1e-4)
    parser.add_argument(
        "--atom-count-grammar-mask",
        action="store_true",
        default=True,
        help="After the model samples <N_...>, enforce slot occupancy/pad consistency with that sampled atom count.",
    )
    parser.add_argument("--no-atom-count-grammar-mask", dest="atom_count_grammar_mask", action="store_false")
    parser.add_argument(
        "--prefill-atom-count-prior",
        choices=["none", "uniform", "train", "val", "test"],
        default="none",
        help="Prefill <N_...> from a fixed atom-count prior without changing labels or schema.",
    )
    parser.add_argument(
        "--atom-count-stats-json",
        type=Path,
        default=PROJECT_ROOT / "data/dlm_sft/mp_20/stats.json",
    )
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument(
        "--target-graph-success",
        type=int,
        default=None,
        help="Keep sampling until this many graph-valid proposals are collected, subject to --max-attempts.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum decoded attempts when --target-graph-success is set. Defaults to --num-samples.",
    )
    args = parser.parse_args()
    if args.representation == "fixed_slot_physical_header":
        if args.prompt is None:
            args.prompt = PHYSICAL_HEADER_CANONICAL_PROMPT
        if args.gen_length == ANSWER_TOKEN_COUNT:
            args.gen_length = PHYSICAL_HEADER_ANSWER_TOKEN_COUNT
        if args.steps == ANSWER_TOKEN_COUNT:
            args.steps = PHYSICAL_HEADER_ANSWER_TOKEN_COUNT
    elif args.prompt is None:
        args.prompt = CANONICAL_PROMPT
    if (
        args.atom_count_grammar_mask
        and args.generation_schedule == "default"
        and args.block_length != 1
    ):
        raise RuntimeError(
            "--atom-count-grammar-mask currently requires --block-length 1 so <N_...> is generated before slots."
        )
    if args.generation_schedule != "default" and not args.prefill_slot_tokens:
        raise RuntimeError(f"--generation-schedule {args.generation_schedule} requires --prefill-slot-tokens.")
    if args.generation_schedule != "default" and not args.atom_count_grammar_mask:
        raise RuntimeError(f"--generation-schedule {args.generation_schedule} requires --atom-count-grammar-mask.")
    if (args.duplicate_coordinate_mask or args.lattice_volume_mask) and args.block_length != 1:
        raise RuntimeError("--duplicate-coordinate-mask/--lattice-volume-mask require --block-length 1.")
    if args.duplicate_coordinate_mask and not args.atom_count_grammar_mask:
        raise RuntimeError("--duplicate-coordinate-mask requires --atom-count-grammar-mask.")
    if (args.duplicate_coordinate_mask or args.lattice_volume_mask) and not args.schema_logit_mask:
        raise RuntimeError("--duplicate-coordinate-mask/--lattice-volume-mask require --schema-logit-mask.")

    dist_info = init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    distributed = dist_info["distributed"]
    is_main = dist_info["is_main"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_one = import_process_one(args.crysllmgen_dir)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.checkpoint_path, dist_info["device"])
    compressed_config = load_compressed_config(args)
    schema_config: FixedSlotConfig | CompressedFixedSlotConfig = compressed_config or FixedSlotConfig()
    body_offset = PHYSICAL_HEADER_BODY_OFFSET if args.representation == "fixed_slot_physical_header" else 0
    header_allowed_strings = (
        physical_header_allowed_token_strings()
        if args.representation == "fixed_slot_physical_header"
        else None
    )
    if tokenizer.pad_token_id == MASK_TOKEN_ID:
        raise RuntimeError("tokenizer.pad_token_id must not equal LLaDA mask token id 126336")
    allowed_token_ids_by_generation_pos = None
    prefill_token_ids_by_generation_pos = None
    atom_count_grammar = None
    if args.schema_logit_mask or args.prefill_slot_tokens:
        allowed_token_ids_by_generation_pos, slot_prefill = build_schema_generation_constraints(
            tokenizer,
            config=schema_config,
            header_allowed_strings=header_allowed_strings,
            body_offset=body_offset,
        )
        if not args.schema_logit_mask:
            allowed_token_ids_by_generation_pos = None
        if args.prefill_slot_tokens:
            prefill_token_ids_by_generation_pos = slot_prefill
    if args.atom_count_grammar_mask:
        atom_count_grammar = build_atom_count_grammar(tokenizer, config=schema_config, body_offset=body_offset)
    lightweight_decoding_constraints = build_lightweight_decoding_constraints(
        tokenizer,
        duplicate_coordinate_mask=args.duplicate_coordinate_mask,
        lattice_volume_mask=args.lattice_volume_mask,
        min_lattice_rad=args.min_lattice_rad,
        config=schema_config,
        body_offset=body_offset,
    )
    generation_position_groups = None
    if args.generation_schedule != "default":
        if args.generation_schedule == "n-elements-coords-lattice":
            generation_position_groups = n_elements_coords_lattice_schedule(offset=body_offset)
        elif args.generation_schedule == "n-elements-sequential-rest":
            generation_position_groups = n_elements_sequential_rest_schedule(offset=body_offset)
        else:
            raise RuntimeError(f"Unsupported generation schedule {args.generation_schedule!r}")
        if args.representation == "fixed_slot_physical_header":
            generation_position_groups = [[position] for position in range(body_offset)] + generation_position_groups
        scheduled_positions = {position for group in generation_position_groups for position in group}
        expected_positions = set(range(body_offset))
        expected_positions.add(body_offset)
        expected_positions.update(range(body_offset + 1, body_offset + 7))
        for slot_index in range(FixedSlotConfig().max_atoms):
            base = body_offset + 7 + slot_index * 5
            expected_positions.update([base + 1, base + 2, base + 3, base + 4])
        if scheduled_positions != expected_positions:
            raise RuntimeError("Internal generation schedule does not match fixed-slot non-prefill positions")
    atom_count_values, atom_count_weights = load_atom_count_prior(
        args.prefill_atom_count_prior,
        args.atom_count_stats_json,
    )
    prompt_records = None
    if args.prompt_jsonl is not None:
        prompt_records = read_prompt_jsonl(args.prompt_jsonl, args.prompt_field)

    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config["distributed"] = distributed
    run_config["world_size"] = world_size
    run_config["prompt_separator"] = "\\n"
    run_config["body_offset"] = body_offset
    if is_main:
        write_json(str(args.output_dir / "run_config.json"), run_config)
        if compressed_config is not None:
            write_json(str(args.output_dir / "compressed_token_config.json"), compressed_config.to_dict())
        write_json(
            str(args.output_dir / "prompt_pool.json"),
            {
                "prompt_pool": (
                    [
                        {
                            "source": str(args.prompt_jsonl),
                            "count": len(prompt_records or []),
                            "prompt_field": args.prompt_field,
                        }
                    ]
                    if prompt_records is not None
                    else (
                        PHYSICAL_HEADER_PROMPT_POOL
                        if args.representation == "fixed_slot_physical_header"
                        else PROMPT_POOL
                    )
                ),
                "active_prompt": args.prompt,
            },
        )
        write_json(
            str(args.output_dir / "tokenizer_report.json"),
            {
                "model_path": args.model_path,
                "checkpoint_path": args.checkpoint_path,
                "tokenizer_source": args.checkpoint_path
                if args.checkpoint_path and Path(args.checkpoint_path).exists()
                else args.model_path,
                "vocab_size": len(tokenizer),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "mask_token_id": MASK_TOKEN_ID,
                "pad_token_id_ne_mask_token_id": tokenizer.pad_token_id != MASK_TOKEN_ID,
                "representation": args.representation,
                "compressed_config": None if compressed_config is None else compressed_config.to_dict(),
            },
        )
        with (args.output_dir / "training_log.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "not_applicable",
                        "stage": "sampling",
                        "reason": "sampling run; no optimizer training is performed",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    raw_path = rank_path(args.output_dir, "raw_generations.jsonl", rank, distributed)
    failure_path = rank_path(args.output_dir, "failure_cases.jsonl", rank, distributed)
    valid_arrays_path = rank_path(args.output_dir, "valid_arrays.jsonl", rank, distributed)

    valid_arrays: List[Dict[str, Any]] = []
    proposal_graphs: List[Dict[str, Any]] = []
    target_mode = args.target_graph_success is not None
    if target_mode:
        target_graph_success = int(args.target_graph_success or 0)
        if target_graph_success <= 0:
            raise ValueError("--target-graph-success must be positive when provided")
        max_attempts = int(args.max_attempts or args.num_samples)
        if max_attempts < target_graph_success:
            raise ValueError("--max-attempts must be >= --target-graph-success")
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
    start = time.time()
    with raw_path.open("w", encoding="utf-8") as raw_handle, failure_path.open(
        "w", encoding="utf-8"
    ) as failure_handle:
        progress = tqdm(total=len(sample_indices), desc=f"LLaDA sampling rank{rank}", disable=distributed and not is_main)
        prompt_text = args.prompt.rstrip() + "\n"
        for batch_start in range(0, len(sample_indices), args.batch_size):
            if (
                local_target_graph_success is not None
                and metrics["graph_success"] >= local_target_graph_success
            ):
                break
            current_indices = sample_indices[batch_start : batch_start + args.batch_size]
            current_batch = len(current_indices)
            current_prefill = dict(prefill_token_ids_by_generation_pos or {})
            current_target_atom_counts: List[int | None] = [None] * current_batch
            if atom_count_values:
                current_target_atom_counts = [
                    sample_atom_count(sample_idx, atom_count_values, atom_count_weights, args.seed)
                    for sample_idx in current_indices
                ]
                current_prefill[body_offset] = [
                    atom_count_token_id(tokenizer, int(atom_count))
                    for atom_count in current_target_atom_counts
                ]
                for atom_count in current_target_atom_counts:
                    key = str(int(atom_count))
                    metrics["target_atom_count_histogram"][key] = (
                        metrics["target_atom_count_histogram"].get(key, 0) + 1
                    )
            batch_prompt_records: List[Dict[str, Any] | None] = [None] * current_batch
            batch_prompt_record_indices: List[int | None] = [None] * current_batch
            if prompt_records is None:
                prompts = [prompt_text] * current_batch
            else:
                prompts = []
                for local_i, sample_idx in enumerate(current_indices):
                    current_prompt, current_record, current_record_idx = prompt_record_for_index(
                        prompt_records,
                        sample_idx,
                        args.prompt_field,
                    )
                    prompts.append(current_prompt)
                    batch_prompt_records[local_i] = current_record
                    batch_prompt_record_indices[local_i] = current_record_idx
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
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
            for local_i, (sample_idx, text, target_atom_count) in enumerate(
                zip(current_indices, decoded, current_target_atom_counts)
            ):
                metrics["decoded_samples"] += 1
                raw_record = {"sample_idx": sample_idx, "text": text}
                if batch_prompt_records[local_i] is not None:
                    prompt_record = dict(batch_prompt_records[local_i] or {})
                    prompt_record.pop(args.prompt_field, None)
                    raw_record["prompt_record_idx"] = batch_prompt_record_indices[local_i]
                    raw_record["conditioning_prompt"] = prompts[local_i].rstrip()
                    raw_record["prompt_record"] = prompt_record
                if target_atom_count is not None:
                    raw_record["target_num_atoms"] = int(target_atom_count)
                try:
                    if args.representation == "fixed_slot_physical_header":
                        arrays = parse_physical_header_answer(text)
                    elif compressed_config is not None:
                        arrays = parse_compressed_fixed_slot_answer(text, config=compressed_config)
                    else:
                        arrays = parse_fixed_slot_answer(text)
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
    metrics["target_reached"] = (
        local_target_graph_success is not None
        and metrics["graph_success"] >= local_target_graph_success
    )
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), metrics)
    if is_main and not distributed:
        with (args.output_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                        {
                            "event": "sampling_complete",
                            "decoded_samples": metrics["decoded_samples"],
                            "parse_rate": metrics["parse_rate"],
                            "graph_rate": metrics["graph_rate"],
                            "target_reached": metrics["target_reached"],
                        },
                        ensure_ascii=False,
                    )
                + "\n"
            )

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
            merged_metrics = json.loads((args.output_dir / "sample_metrics.json").read_text(encoding="utf-8"))
            with (args.output_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "sampling_complete",
                            "decoded_samples": merged_metrics["decoded_samples"],
                            "parse_rate": merged_metrics["parse_rate"],
                            "graph_rate": merged_metrics["graph_rate"],
                            "target_reached": merged_metrics["target_reached"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
