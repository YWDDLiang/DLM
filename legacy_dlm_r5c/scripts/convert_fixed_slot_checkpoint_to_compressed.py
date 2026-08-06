#!/usr/bin/env python3
"""Convert a full-token fixed-slot PEFT checkpoint to compressed-token vocab.

Only tensors whose row/column dimension is the tokenizer vocabulary are
remapped. LoRA tensors are copied byte-for-byte, so the ablation isolates token
family compression while preserving the learned transformer adapter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

from crystal_dlm.fixed_slot_compressed import (  # noqa: E402
    CompressedFixedSlotConfig,
    build_compressed_special_tokens,
    compressed_token_sources,
)
from crystal_dlm.fixed_slot import write_json  # noqa: E402


TOKENIZER_FILENAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
}


def load_adapter_state(path: Path) -> tuple[Dict[str, torch.Tensor], str]:
    safetensors_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path)), "safetensors"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu"), "bin"
    raise FileNotFoundError(f"No adapter_model.safetensors/bin found in {path}")


def save_adapter_state(state: Mapping[str, torch.Tensor], output_dir: Path, fmt: str) -> None:
    if fmt == "safetensors":
        from safetensors.torch import save_file

        save_file(dict(state), str(output_dir / "adapter_model.safetensors"))
    else:
        torch.save(dict(state), output_dir / "adapter_model.bin")


def copy_checkpoint_sidecars(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        if source.name in TOKENIZER_FILENAMES:
            continue
        if source.name in {"adapter_model.safetensors", "adapter_model.bin"}:
            continue
        target = output_dir / source.name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        elif source.is_file():
            shutil.copy2(source, target)


def build_row_map(
    old_tokenizer,
    new_tokenizer,
    token_sources: Mapping[str, list[str]],
    allow_missing_source: bool,
) -> Dict[int, list[int]]:
    old_vocab = old_tokenizer.get_vocab()
    row_map: Dict[int, list[int]] = {}
    missing: Dict[str, list[str]] = {}
    for token, new_id in new_tokenizer.get_vocab().items():
        new_id = int(new_id)
        if token in old_vocab:
            row_map[new_id] = [int(old_vocab[token])]
            continue
        sources = token_sources.get(token, [token])
        source_ids = [int(old_vocab[src]) for src in sources if src in old_vocab]
        if source_ids:
            row_map[new_id] = source_ids
        else:
            missing[token] = list(sources)
    if missing and not allow_missing_source:
        examples = {key: missing[key] for key in list(missing)[:20]}
        raise RuntimeError(f"Missing source ids for {len(missing)} new tokens: {examples}")
    return row_map


def remap_vocab_tensor(
    tensor: torch.Tensor,
    *,
    old_vocab_size: int,
    new_vocab_size: int,
    row_map: Mapping[int, list[int]],
) -> torch.Tensor:
    if tensor.ndim < 2:
        return tensor
    if tensor.shape[0] == old_vocab_size:
        rows = []
        for new_id in range(new_vocab_size):
            source_ids = row_map.get(new_id)
            if not source_ids:
                rows.append(tensor[0])
            elif len(source_ids) == 1:
                rows.append(tensor[source_ids[0]])
            else:
                rows.append(tensor[source_ids].float().mean(dim=0).to(dtype=tensor.dtype))
        return torch.stack(rows, dim=0).contiguous()
    if tensor.shape[1] == old_vocab_size:
        cols = []
        for new_id in range(new_vocab_size):
            source_ids = row_map.get(new_id)
            if not source_ids:
                cols.append(tensor[:, 0])
            elif len(source_ids) == 1:
                cols.append(tensor[:, source_ids[0]])
            else:
                cols.append(tensor[:, source_ids].float().mean(dim=1).to(dtype=tensor.dtype))
        return torch.stack(cols, dim=1).contiguous()
    return tensor


def build_config(args: argparse.Namespace) -> CompressedFixedSlotConfig:
    if args.compressed_token_config is not None:
        return CompressedFixedSlotConfig.from_path(args.compressed_token_config)
    return CompressedFixedSlotConfig(
        share_coordinates=True,
        share_lengths=args.variant in {"abl2", "abl3"},
        share_angles=args.variant == "abl3",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--variant", choices=["abl1", "abl2", "abl3"], default="abl1")
    parser.add_argument("--compressed-token-config", type=Path, default=None)
    parser.add_argument("--allow-missing-source", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    special_tokens = build_compressed_special_tokens(config)
    token_sources = compressed_token_sources(config)

    old_tokenizer = AutoTokenizer.from_pretrained(args.source_checkpoint, trust_remote_code=True)
    new_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    new_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if new_tokenizer.pad_token_id is None:
        new_tokenizer.pad_token = new_tokenizer.eos_token

    state, fmt = load_adapter_state(args.source_checkpoint)
    old_vocab_size = len(old_tokenizer)
    new_vocab_size = len(new_tokenizer)
    row_map = build_row_map(
        old_tokenizer,
        new_tokenizer,
        token_sources=token_sources,
        allow_missing_source=args.allow_missing_source,
    )

    converted: Dict[str, torch.Tensor] = {}
    remapped_tensors: Dict[str, Dict[str, int]] = {}
    for key, tensor in state.items():
        new_tensor = remap_vocab_tensor(
            tensor,
            old_vocab_size=old_vocab_size,
            new_vocab_size=new_vocab_size,
            row_map=row_map,
        )
        converted[key] = new_tensor
        if tuple(new_tensor.shape) != tuple(tensor.shape):
            remapped_tensors[key] = {
                "old_rows": int(tensor.shape[0]) if tensor.ndim > 0 else 0,
                "new_rows": int(new_tensor.shape[0]) if new_tensor.ndim > 0 else 0,
            }

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    copy_checkpoint_sidecars(args.source_checkpoint, args.output_dir)
    save_adapter_state(converted, args.output_dir, fmt)
    new_tokenizer.save_pretrained(args.output_dir)
    write_json(str(args.output_dir / "compressed_token_config.json"), config.to_dict())
    write_json(str(args.output_dir / "token_map.json"), token_sources)
    write_json(
        str(args.output_dir / "conversion_report.json"),
        {
            "source_checkpoint": str(args.source_checkpoint),
            "output_dir": str(args.output_dir),
            "base_model_path": args.base_model_path,
            "old_vocab_size": old_vocab_size,
            "new_vocab_size": new_vocab_size,
            "special_token_count": len(special_tokens),
            "variant": args.variant,
            "config": config.to_dict(),
            "state_format": fmt,
            "remapped_tensors": remapped_tensors,
        },
    )


if __name__ == "__main__":
    main()
