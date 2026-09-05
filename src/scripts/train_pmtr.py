#!/usr/bin/env python3
"""Train a PMTR manifold head while keeping the retained SPAD DLM frozen."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.manifold_repair_head import (
    ManifoldRepairConfig,
    ManifoldRepairHead,
)
from crystal_dlm.manifold_repair_objective import ManifoldRepairLossConfig
from crystal_dlm.pmtr_runtime import PMTRRuntimeConfig
from crystal_dlm.pmtr_training import (
    PMTRDataCollator,
    PMTRHeadOnlyModule,
    PMTRJsonlDataset,
    build_training_examples,
    freeze_spad_model,
    frozen_spad_forward,
    materialize_transaction_start,
    move_tensor_batch,
    probe_component_gradient_scales,
)
from scripts import llada_sft as SFT


FINAL_STATE_NAME = "pmtr_head_state.pt"
FINAL_CONFIG_NAME = "pmtr_head_config.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--expected-rows", type=int, default=27_136)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=382)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20_260_905)
    parser.add_argument("--probe-batches", type=int, default=5)
    parser.add_argument("--head-width", type=int, default=128)
    parser.add_argument("--max-sites", type=int, default=20)
    parser.add_argument("--radial-basis-count", type=int, default=16)
    parser.add_argument("--radial-cutoff-A", type=float, default=8.0)
    parser.add_argument("--max-metric-tangent", type=float, default=0.20)
    parser.add_argument("--max-cartesian-step-A", type=float, default=0.75)
    parser.add_argument("--transport-gain", type=float, default=6.0)
    parser.add_argument("--image-radius", type=int, default=2)
    parser.add_argument("--lattice-target-scale", type=float, default=0.20)
    parser.add_argument("--cartesian-target-scale-A", type=float, default=0.75)
    parser.add_argument("--step-regularization", type=float, default=1.0e-3)
    return parser


def _distributed() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not 1 <= world_size <= 4:
        raise ValueError("PMTR supports one to four distributed ranks")
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed PMTR training requires CUDA")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "distributed": distributed,
        "world_size": world_size,
        "local_rank": local_rank,
        "rank": rank,
        "is_main": rank == 0,
        "device": device,
    }


def _seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + int(rank) * 1_000_003
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _load_frozen_model(
    args: argparse.Namespace,
    *,
    is_main: bool,
    model_loader: Callable[..., Any],
) -> tuple[Any, torch.nn.Module]:
    loader_args = SimpleNamespace(
        model_path=str(args.model_path),
        checkpoint_path=str(args.checkpoint_path),
        data_dir=Path(args.data_jsonl).parent,
        representation="dynamic_v1",
        skip_data_vocab_resize=True,
        semantic_init_element_tokens=False,
        use_lora=False,
    )
    loaded = model_loader(loader_args, is_main=is_main)
    if not isinstance(loaded, tuple) or len(loaded) < 2:
        raise TypeError("model loader must return tokenizer and model")
    tokenizer, model = loaded[:2]
    return tokenizer, model


def _hidden_size(model: torch.nn.Module) -> int:
    current: Any = model
    for _ in range(5):
        config = getattr(current, "config", None)
        for name in ("hidden_size", "d_model", "n_embd"):
            value = getattr(config, name, None)
            if value is not None and int(value) > 0:
                return int(value)
        next_model = getattr(current, "base_model", None)
        if next_model is None or next_model is current:
            break
        current = next_model
    output = model.get_output_embeddings()
    weight = getattr(output, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[1])
    raise ValueError("cannot infer retained DLM hidden size")


def _json_print(value: dict[str, Any], *, enabled: bool) -> None:
    if enabled:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _save_final(
    output_dir: Path,
    module: PMTRHeadOnlyModule,
    *,
    head_config: ManifoldRepairConfig,
    runtime_config: PMTRRuntimeConfig,
    loss_config: ManifoldRepairLossConfig,
    args: argparse.Namespace,
    steps: int,
    repair_rows: int,
) -> None:
    head = module.repair_head
    state = {name: value.detach().cpu() for name, value in head.state_dict().items()}
    torch.save(state, output_dir / FINAL_STATE_NAME)
    payload = {
        "schema": "pmtr_head_only_v1",
        "head": asdict(head_config),
        "runtime": asdict(runtime_config),
        "loss": asdict(loss_config),
        "training": {
            "epochs": int(args.epochs),
            "optimizer_steps": int(steps),
            "repair_rows": int(repair_rows),
            "alternation": ["clean_identity", "corrupt_repair"],
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "base_frozen": True,
            "full_transaction_supervision": True,
        },
        "base": {
            "model_path": str(args.model_path),
            "checkpoint_path": str(args.checkpoint_path),
        },
    }
    (output_dir / FINAL_CONFIG_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_training(
    args: argparse.Namespace,
    *,
    model_loader: Callable[..., Any] = SFT.load_tokenizer_and_model,
) -> dict[str, Any]:
    if int(args.epochs) <= 0 or int(args.batch_size) <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.limit is not None and int(args.limit) <= 0:
        raise ValueError("limit must be positive")
    if int(args.max_steps) < 0 or int(args.probe_batches) < 0:
        raise ValueError("max_steps and probe_batches must be non-negative")

    distributed = _distributed()
    device = distributed["device"]
    _seed_everything(int(args.seed), int(distributed["rank"]))
    tokenizer, base_model = _load_frozen_model(
        args,
        is_main=bool(distributed["is_main"]),
        model_loader=model_loader,
    )
    base_model.to(device)
    frozen_parameters = freeze_spad_model(base_model)

    dataset = PMTRJsonlDataset(
        Path(args.data_jsonl),
        tokenizer,
        int(args.max_length),
        limit=args.limit,
    )
    if args.limit is None and int(args.expected_rows) > 0:
        if dataset.source_row_count != int(args.expected_rows) or len(dataset) != int(
            args.expected_rows
        ):
            raise ValueError(
                "formal PMTR training requires every expected source row to carry "
                "a repair_target; use --limit only for preflight"
            )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=int(distributed["world_size"]),
            rank=int(distributed["rank"]),
            shuffle=True,
            seed=int(args.seed),
            drop_last=False,
        )
        if distributed["distributed"]
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=PMTRDataCollator(tokenizer),
    )

    head_config = ManifoldRepairConfig(
        hidden_size=_hidden_size(base_model),
        width=int(args.head_width),
        max_sites=int(args.max_sites),
        radial_basis_count=int(args.radial_basis_count),
        radial_cutoff_A=float(args.radial_cutoff_A),
        max_metric_tangent=float(args.max_metric_tangent),
        max_cartesian_step_A=float(args.max_cartesian_step_A),
    )
    runtime_config = PMTRRuntimeConfig(
        transport_gain=float(args.transport_gain),
        image_radius=int(args.image_radius),
    )
    loss_config = ManifoldRepairLossConfig(
        lattice_tangent_scale=float(args.lattice_target_scale),
        cartesian_step_scale_A=float(args.cartesian_target_scale_A),
        step_regularization=float(args.step_regularization),
    )
    train_module = PMTRHeadOnlyModule(
        ManifoldRepairHead(head_config),
        tokenizer,
        loss_config=loss_config,
        runtime_config=runtime_config,
    ).to(device)
    wrapped: torch.nn.Module = train_module
    if distributed["distributed"]:
        wrapped = DistributedDataParallel(
            train_module,
            device_ids=[int(distributed["local_rank"])],
            output_device=int(distributed["local_rank"]),
            find_unused_parameters=True,
        )
    optimizer = torch.optim.AdamW(
        wrapped.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    if distributed["is_main"]:
        Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    if distributed["distributed"]:
        dist.barrier()

    maximum_steps = int(args.max_steps) if int(args.max_steps) > 0 else math.inf
    totals = {name: 0.0 for name in ("total", "token", "spd", "torus", "step")}
    mode_steps = {"clean_identity": 0, "corrupt_repair": 0}
    optimizer_steps = 0
    wrapped.train()
    for epoch in range(int(args.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for raw_batch in loader:
            batch = move_tensor_batch(raw_batch, device)
            for mode in ("clean_identity", "corrupt_repair"):
                if optimizer_steps >= maximum_steps:
                    break
                materialized = materialize_transaction_start(
                    batch,
                    mode=mode,
                    mask_id=int(MASK_TOKEN_ID),
                )
                base_step = frozen_spad_forward(
                    base_model,
                    batch,
                    materialized,
                    mask_id=int(MASK_TOKEN_ID),
                )
                examples = build_training_examples(batch, materialized, base_step)
                if optimizer_steps < int(args.probe_batches):
                    scales = probe_component_gradient_scales(wrapped, examples)
                    _json_print(
                        {
                            "event": "gradient_scale",
                            "optimizer_step": optimizer_steps + 1,
                            "mode": mode,
                            **scales,
                            "automatic_reweighting": False,
                        },
                        enabled=bool(distributed["is_main"]),
                    )
                optimizer.zero_grad(set_to_none=True)
                losses = wrapped(examples)
                if not bool(torch.isfinite(losses.total).detach().item()):
                    raise FloatingPointError("PMTR total loss is not finite")
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    wrapped.parameters(), float(args.gradient_clip)
                )
                optimizer.step()
                optimizer_steps += 1
                mode_steps[mode] += 1
                for name in totals:
                    totals[name] += float(getattr(losses, name).detach().cpu())
                _json_print(
                    {
                        "event": "train_step",
                        "optimizer_step": optimizer_steps,
                        "mode": mode,
                        **{
                            name: float(getattr(losses, name).detach().cpu())
                            for name in totals
                        },
                    },
                    enabled=bool(distributed["is_main"]),
                )
            if optimizer_steps >= maximum_steps:
                break
        if optimizer_steps >= maximum_steps:
            break

    if optimizer_steps == 0:
        raise RuntimeError("PMTR training completed no optimizer steps")
    if distributed["distributed"]:
        dist.barrier()
    if distributed["is_main"]:
        _save_final(
            Path(args.output_dir),
            train_module,
            head_config=head_config,
            runtime_config=runtime_config,
            loss_config=loss_config,
            args=args,
            steps=optimizer_steps,
            repair_rows=len(dataset),
        )
    if distributed["distributed"]:
        dist.barrier()
        dist.destroy_process_group()
    report = {
        "optimizer_steps": optimizer_steps,
        "mode_steps": mode_steps,
        "repair_rows": len(dataset),
        "frozen_base_parameters": frozen_parameters,
        "mean_losses": {
            name: value / float(optimizer_steps) for name, value in totals.items()
        },
        "saved_final_only": True,
    }
    _json_print(
        {"event": "complete", **report}, enabled=bool(distributed["is_main"])
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
