#!/usr/bin/env python3
"""Measure atomic loss-gradient scales on hash-fixed WQ training batches.

This is a diagnostic-only Slurm entry point.  It never updates parameters and
writes one immutable JSON artifact containing every batch/term observation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.contracts import write_json_exclusive
from crystal_dlm.wqcodiff.losses import compute_wq_loss_terms
from crystal_dlm.wqcodiff.model import WQCoDenoiser, WQModelConfig, WQVariant
from crystal_dlm.wqcodiff.protocol import load_protocol
from crystal_dlm.wqcodiff.training import (
    EpochSampler,
    _corruption_seed,
    _dataset_identity,
)
from crystal_dlm.wqcodiff.training_data import JsonlRecordIndex, build_corrupted_batch


HEAD_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "space_group": ("space_group_head.",),
    "species": ("species_head.",),
    "wyckoff": ("wyckoff_head.",),
    "event": ("event_head.",),
    "event_pointer": ("event_orbit_head.",),
    "birth_species": ("birth_species_head.",),
    "birth_wyckoff": ("birth_wyckoff_head.",),
    "birth_coordinate": (
        "birth_coordinate_mean_head.",
        "birth_coordinate_log_scale_head.",
    ),
    "revision": ("revision_head.",),
    "coordinate_score": ("coordinate_head.",),
    "lattice_score": ("lattice_head.",),
    "bridge": ("bridge_mean_head.", "bridge_log_scale_head."),
    "prior_space_group": (
        "start_token",
        "prior_mlp.",
        "prior_space_group_head.",
    ),
    "prior_species": (
        "start_token",
        "prior_mlp.",
        "prior_species_head.",
    ),
    "prior_wyckoff": (
        "start_token",
        "prior_mlp.",
        "prior_wyckoff_head.",
    ),
    "prior_coordinate": (
        "start_token",
        "prior_mlp.",
        "prior_coordinate_mean_head.",
        "prior_coordinate_log_scale_head.",
    ),
    "prior_lattice": (
        "start_token",
        "prior_mlp.",
        "prior_lattice_mean_head.",
        "prior_lattice_log_scale_head.",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_shared_backbone(name: str) -> bool:
    return "_head." not in name and not name.startswith("prior_mlp.") and name != "start_token"


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(prefix) for prefix in prefixes)


def _parameter_group_norm(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    predicate: Callable[[str], bool],
) -> float:
    squared: torch.Tensor | None = None
    for name, parameter in named_parameters:
        if parameter.grad is None or not predicate(name):
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else math.sqrt(float(squared))


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    terms = sorted({str(record["term"]) for record in records})
    for term in terms:
        selected = [record for record in records if record["term"] == term]
        summary[term] = {}
        for field in ("loss", "global_grad_norm", "shared_backbone_grad_norm", "task_specific_grad_norm"):
            values = [float(record[field]) for record in selected]
            summary[term][field] = {
                "min": min(values),
                "median": statistics.median(values),
                "p95": _percentile(values, 0.95),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
    return summary


def _masked_scale_stats(log_scale: torch.Tensor, mask: torch.Tensor) -> dict[str, object]:
    values = log_scale.detach().float().exp()[mask]
    if not values.numel():
        return {"count": 0, "min": None, "median": None, "max": None}
    values = values.cpu().tolist()
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(value.value for value in WQVariant), required=True)
    parser.add_argument("--training-seed", type=int, choices=(11, 23, 47), required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--batches", type=int, choices=range(1, 33), default=16)
    parser.add_argument("--batch-size", type=int, choices=(128,), default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if not args.output.parent.is_dir():
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_bundle_sha256):
        raise ValueError("source bundle identity must be a lowercase SHA256")

    protocol = load_protocol(args.protocol)
    dataset_paths = tuple(str(path) for path in args.dataset)
    dataset_files = _dataset_identity(dataset_paths)
    dataset = JsonlRecordIndex(dataset_paths)
    sampler = EpochSampler(len(dataset), args.training_seed)
    variant = WQVariant(args.variant)
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("component-gradient audit requires registered Slurm CUDA")
    model = WQCoDenoiser(WQModelConfig()).to(device)
    checkpoint_step = 0
    target_updates = 100_000
    checkpoint_identity = None
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != "wqcodiff_checkpoint_v1":
            raise ValueError("unsupported diagnostic checkpoint schema")
        if payload.get("protocol_sha256") != protocol.sha256:
            raise ValueError("diagnostic checkpoint protocol mismatch")
        if payload.get("source_bundle_sha256") != args.source_bundle_sha256:
            raise ValueError("diagnostic checkpoint source mismatch")
        if payload.get("dataset_files") != [dict(item) for item in dataset_files]:
            raise ValueError("diagnostic checkpoint dataset mismatch")
        if payload.get("model_config") != dataclasses.asdict(model.config):
            raise ValueError("diagnostic checkpoint model mismatch")
        model.load_state_dict(payload["model"])
        sampler.load_state_dict(payload["sampler"])
        checkpoint_step = int(payload["step"])
        target_updates = int(payload["training_config"]["updates"])
        checkpoint_identity = {
            "path": str(args.checkpoint.resolve()),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": _sha256(args.checkpoint),
            "step": checkpoint_step,
        }
    shared_updates = int(round(0.6 * target_updates))
    active_variant = WQVariant.JOINT_NOREV if checkpoint_step < shared_updates else variant
    model.train()
    named_parameters = tuple(model.named_parameters())
    records: list[dict[str, object]] = []
    scale_records: list[dict[str, object]] = []
    started = time.monotonic()
    for batch_index in range(args.batches):
        indices = sampler.take(args.batch_size)
        payloads = [dataset[index] for index in indices]
        corrupted = build_corrupted_batch(
            payloads,
            seed=_corruption_seed(args.training_seed, checkpoint_step + batch_index, 0),
            variant=active_variant,
            representation_variant=variant,
            enable_revision_training=checkpoint_step >= shared_updates,
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(corrupted.inputs, variant=active_variant)
            prior_time = torch.ones_like(corrupted.inputs.time)
            masked_prior = model.forward_prior(
                prior_time,
                torch.zeros_like(corrupted.inputs.space_group),
            )
            conditioned_prior = model.forward_prior(
                prior_time,
                corrupted.prior_targets.space_group + 1,
            )
            terms = compute_wq_loss_terms(
                output,
                corrupted.targets,
                masked_prior=masked_prior,
                conditioned_prior=conditioned_prior,
                prior_target=corrupted.prior_targets,
            )
            losses = dict(terms._asdict())
            losses["registered_total"] = sum(losses.values())
        scale_records.append(
            {
                "batch_index": batch_index,
                "birth_coordinate": _masked_scale_stats(
                    output.birth_coordinate_log_scale,
                    corrupted.targets.birth_coordinate_mask,
                ),
                "bridge": _masked_scale_stats(
                    output.bridge_log_scale,
                    corrupted.targets.bridge_mask,
                ),
                "prior_coordinate": _masked_scale_stats(
                    conditioned_prior.first_coordinate_log_scale,
                    corrupted.prior_targets.first_coordinate_mask,
                ),
            }
        )
        loss_items = tuple(losses.items())
        for loss_index, (term, loss) in enumerate(loss_items):
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=loss_index + 1 < len(loss_items))
            prefixes = HEAD_PREFIXES.get(term, ())
            records.append(
                {
                    "batch_index": batch_index,
                    "term": term,
                    "loss": float(loss.detach()),
                    "global_grad_norm": _parameter_group_norm(named_parameters, lambda _name: True),
                    "shared_backbone_grad_norm": _parameter_group_norm(
                        named_parameters, _is_shared_backbone
                    ),
                    "task_specific_grad_norm": _parameter_group_norm(
                        named_parameters,
                        (lambda name, prefixes=prefixes: not _is_shared_backbone(name))
                        if term == "registered_total"
                        else (lambda name, prefixes=prefixes: _matches_prefix(name, prefixes)),
                    ),
                }
            )
    elapsed = time.monotonic() - started
    dataset.close()
    payload = {
        "schema": "wqcodiff_component_gradient_audit_v1",
        "ok": True,
        "paper_eligible": False,
        "diagnostic_only": True,
        "protocol_sha256": protocol.sha256,
        "source_bundle_sha256": args.source_bundle_sha256,
        "dataset_files": [dict(item) for item in dataset_files],
        "variant": variant.value,
        "active_variant": active_variant.value,
        "training_seed": args.training_seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "checkpoint": checkpoint_identity,
        "elapsed_s": elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "all_finite": all(
            math.isfinite(float(record[field]))
            for record in records
            for field in (
                "loss",
                "global_grad_norm",
                "shared_backbone_grad_norm",
                "task_specific_grad_norm",
            )
        ),
        "records": records,
        "periodic_scale_records": scale_records,
        "summary": summarize_records(records),
    }
    write_json_exclusive(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("schema", "ok", "all_finite", "elapsed_s", "summary")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
