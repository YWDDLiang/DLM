#!/usr/bin/env python3
"""Train the shared WQ refiner initialized from registered CrysLLMGen CSP."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch

from crystal_dlm.wqcodiff.crysllmgen.gate import GateALock, sha256_file
from crystal_dlm.wqcodiff.crysllmgen.protocol import load_protocol_v4
from crystal_dlm.wqcodiff.crysllmgen.wq_refiner import (
    CrysLLMGenWQRefiner,
    load_registered_csp_refiner,
)
from crystal_dlm.wqcodiff.losses import compute_wq_loss_terms
from crystal_dlm.wqcodiff.model import WQVariant
from crystal_dlm.wqcodiff.training import (
    EpochSampler,
    ExponentialMovingAverage,
    _dataset_identity,
    _learning_rate_multiplier,
)
from crystal_dlm.wqcodiff.training_data import JsonlRecordIndex
from crystal_dlm.wqcodiff.training_prefetch import DeterministicCorruptionPrefetcher


def _require_environment() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("WQ refiner training must run through Slurm")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")


def _seed(training_seed: int, update: int, microbatch: int) -> int:
    raw = json.dumps(
        ["crysllmgen_wq_refiner_v1", training_seed, update, microbatch],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _stage(step: int) -> str:
    if step < 20_000:
        return "chart_projection_bridge_warmup"
    if step < 60_000:
        return "shared_joint_geometry"
    return "event_revision_coupled_geometry"


def _stage_loss(terms: Any, stage: str) -> torch.Tensor:
    if stage == "chart_projection_bridge_warmup":
        selected = (terms.coordinate_score, terms.lattice_score, terms.bridge)
    elif stage == "shared_joint_geometry":
        selected = (
            terms.species,
            terms.wyckoff,
            terms.coordinate_score,
            terms.lattice_score,
            terms.bridge,
        )
    elif stage == "event_revision_coupled_geometry":
        selected = (
            terms.species,
            terms.wyckoff,
            terms.event,
            terms.event_pointer,
            terms.birth_species,
            terms.birth_wyckoff,
            terms.birth_coordinate,
            terms.revision,
            terms.coordinate_score,
            terms.lattice_score,
            terms.bridge,
        )
    else:  # pragma: no cover - internal closed enum
        raise ValueError(stage)
    return sum(selected)


def _ema_state(model: CrysLLMGenWQRefiner, ema: ExponentialMovingAverage) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in model.state_dict().items():
        selected = ema.shadow.get(name, value)
        result[name] = selected.detach().cpu()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-a-lock", type=Path)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--csp-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, choices=(64, 128), default=128)
    parser.add_argument("--run-role", choices=("smoke", "main"), required=True)
    parser.add_argument("--smoke-updates", type=int, default=100)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--replacement-of-job-id")
    parser.add_argument("--prefetch-workers", type=int, choices=range(1, 16), required=True)
    parser.add_argument("--prefetch-depth", type=int, required=True)
    args = parser.parse_args()
    _require_environment()
    if re.fullmatch(r"[0-9a-f]{64}", args.source_bundle_sha256) is None:
        raise ValueError("source bundle identity must be one lowercase SHA256")
    if re.fullmatch(r"[0-9a-f]{64}", args.execution_patch_sha256) is None:
        raise ValueError("execution patch identity must be one lowercase SHA256")
    if args.replacement_of_job_id is not None and not args.replacement_of_job_id.isdigit():
        raise ValueError("replacement job identity must be numeric")
    if args.training_seed not in {11, 23, 47}:
        raise ValueError("training seed is outside the registered set")
    if args.run_role == "smoke" and not 1 <= args.smoke_updates <= 100:
        raise ValueError("refiner smoke must contain 1-100 updates")
    if args.run_role == "main" and args.smoke_updates != 100:
        raise ValueError("main runs cannot alter the inert smoke argument")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the registered Slurm job")
    protocol = load_protocol_v4(args.protocol.resolve())
    contract = protocol.data["wq_refiner_training"]
    total_updates = int(contract["total_optimizer_updates"])
    updates = args.smoke_updates if args.run_role == "smoke" else total_updates
    effective_batch = int(contract["effective_batch_structures"])
    if effective_batch != 128 or effective_batch % args.microbatch:
        raise ValueError("microbatch violates the frozen effective batch")
    accumulation = effective_batch // args.microbatch
    project_root = Path(__file__).resolve().parents[2]
    gate = None
    if args.run_role == "main":
        if args.gate_a_lock is None:
            raise ValueError("main refiner training requires Gate A")
        gate = GateALock.load(
            args.gate_a_lock,
            project_root=project_root,
            protocol_path=args.protocol.resolve(),
            execution_patch_manifest_sha256=args.execution_patch_sha256,
        )
        if gate.source_bundle_sha256 != args.source_bundle_sha256:
            raise ValueError("refiner source differs from Gate A source")
    elif args.gate_a_lock is not None:
        raise ValueError("pre-Gate smoke cannot consume a future Gate A lock")
    if args.replacement_of_job_id is not None and args.run_role != "main":
        raise ValueError("only a main run can replace a terminal Slurm job")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_paths = tuple(str(value.resolve()) for value in args.dataset)
    dataset_files = _dataset_identity(dataset_paths)
    dataset = JsonlRecordIndex(dataset_paths)
    sampler = EpochSampler(len(dataset), args.training_seed)
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)
    device = torch.device("cuda", 0)
    model, mapping = load_registered_csp_refiner(
        snapshot_root=args.snapshot_root.resolve(),
        checkpoint=args.csp_checkpoint.resolve(),
    )
    if mapping["checkpoint_sha256"] != str(
        protocol.data["assets"]["cspdiffusion"]["sha256"]
    ):
        raise ValueError("CSP mapping checkpoint differs from protocol")
    model.set_inherited_backbone_trainable(False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["learning_rate"]),
        weight_decay=float(contract["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_multiplier(
            step,
            total_updates,
            float(contract["warmup_fraction"]),
        ),
    )
    ema = ExponentialMovingAverage(model, float(contract["ema"]))
    manifest = {
        "schema": "crysllmgen_wq_refiner_run_v1",
        "run_role": args.run_role,
        "training_seed": args.training_seed,
        "protocol": {"path": str(protocol.path), "sha256": protocol.sha256},
        "gate_a_lock": None if gate is None else {"path": str(gate.path), "sha256": gate.sha256},
        "source_bundle_sha256": args.source_bundle_sha256,
        "execution_patch_sha256": args.execution_patch_sha256,
        "replacement_of_job_id": args.replacement_of_job_id,
        "dataset_files": list(dataset_files),
        "dataset_records": len(dataset),
        "mapping": mapping,
        "optimizer": {
            "total_contract_updates": total_updates,
            "run_updates": updates,
            "effective_batch": effective_batch,
            "microbatch": args.microbatch,
            "accumulation": accumulation,
            "learning_rate": float(contract["learning_rate"]),
            "weight_decay": float(contract["weight_decay"]),
            "ema": float(contract["ema"]),
        },
        "model_config": dataclasses.asdict(model.config),
        "parameters": {
            "total": model.parameter_count(),
            "inherited": model.inherited_parameter_count(),
        },
        "runtime": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "host": platform.node(),
            "torch": torch.__version__,
            "python": sys.version,
            "threads": 1,
            "prefetch_workers": args.prefetch_workers,
            "prefetch_depth": args.prefetch_depth,
            "offline": True,
        },
    }
    _write_exclusive(output_dir / "run_manifest.json", manifest)
    _write_exclusive(output_dir / "csp_mapping_report.json", mapping)

    metrics = output_dir / "train_metrics.jsonl"
    started = time.monotonic()
    model.train()
    previous_stage = ""
    prefetch = DeterministicCorruptionPrefetcher(
        dataset=dataset,
        sampler=sampler,
        training_seed=args.training_seed,
        total_updates=updates,
        microbatch_size=args.microbatch,
        accumulation_steps=accumulation,
        seed_for=_seed,
        revision_start_update=min(60_000, updates),
        variant=WQVariant.STRAT_GEO,
        representation_variant=WQVariant.STRAT_GEO,
        mask_discrete_fields=False,
        # Warmup still exposes one legal topology corruption so the
        # target-stratum bridge receives supervision while inherited CSP
        # parameters remain frozen; event/revision losses stay off.
        enable_topology_corruption=True,
        workers=args.prefetch_workers,
        depth=args.prefetch_depth,
    )
    try:
        for step in range(updates):
            stage = _stage(step)
            if stage != previous_stage:
                model.set_inherited_backbone_trainable(
                    stage != "chart_projection_bridge_warmup"
                )
                previous_stage = stage
            optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            batch_epoch = 0
            for microbatch in range(accumulation):
                prepared = prefetch.take(step, microbatch)
                batch_epoch = prepared.task.epoch_after
                corrupted = prepared.batch.to(device)
                autocast = torch.autocast("cuda", dtype=torch.bfloat16)
                with autocast:
                    output = model(corrupted.inputs, use_geometry_evidence=True)
                    terms = compute_wq_loss_terms(output, corrupted.targets)
                    stage_loss = _stage_loss(terms, stage)
                    loss = stage_loss / accumulation
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"non-finite training loss at update {step + 1}")
                loss.backward()
                accumulated["total"] = accumulated.get("total", 0.0) + float(
                    stage_loss.detach()
                ) / accumulation
                for name, value in terms._asdict().items():
                    accumulated[name] = accumulated.get(name, 0.0) + float(
                        value.detach()
                    ) / accumulation
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(contract["gradient_clip"]),
                )
            )
            if not math.isfinite(gradient):
                raise RuntimeError(f"non-finite gradient at update {step + 1}")
            optimizer.step()
            scheduler.step()
            ema.update(model)
            completed = step + 1
            if completed == 1 or completed % 100 == 0 or completed == updates:
                elapsed = time.monotonic() - started
                _append(
                    metrics,
                    {
                        "schema": "crysllmgen_wq_refiner_metric_v1",
                        "update": completed,
                        "stage": stage,
                        "epoch": batch_epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "gradient_norm_pre_clip": gradient,
                        "backbone_trainable": stage != "chart_projection_bridge_warmup",
                        "updates_per_s": completed / max(elapsed, 1.0e-9),
                        "elapsed_s": elapsed,
                        **accumulated,
                    },
                )
    finally:
        prefetch.close()

    final_path = output_dir / "model_ema_final.pt"
    payload = {
        "schema": "crysllmgen_wq_refiner_ema_v1",
        "model": _ema_state(model, ema),
        "model_config": dataclasses.asdict(model.config),
        "mapping": mapping,
        "protocol_name": protocol.name,
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": None if gate is None else gate.sha256,
        "source_bundle_sha256": args.source_bundle_sha256,
        "execution_patch_sha256": args.execution_patch_sha256,
        "replacement_of_job_id": args.replacement_of_job_id,
        "training_seed": args.training_seed,
        "updates": updates,
        "paper_eligible": args.run_role == "main" and updates == total_updates,
        "dataset_files": list(dataset_files),
    }
    with final_path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "schema": "crysllmgen_wq_refiner_training_report_v1",
        "ok": True,
        "run_role": args.run_role,
        "training_seed": args.training_seed,
        "updates": updates,
        "paper_eligible": payload["paper_eligible"],
        "checkpoint": str(final_path),
        "checkpoint_sha256": sha256_file(final_path),
        "source_bundle_sha256": args.source_bundle_sha256,
        "execution_patch_sha256": args.execution_patch_sha256,
        "gate_a_lock_sha256": None if gate is None else gate.sha256,
        "walltime_s": time.monotonic() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
        "threads": 1,
        "prefetch_workers": args.prefetch_workers,
        "prefetch_depth": args.prefetch_depth,
        "offline": True,
        "retry_or_replacement_used": args.replacement_of_job_id is not None,
        "replacement_of_job_id": args.replacement_of_job_id,
    }
    _write_exclusive(output_dir / "training_report.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    dataset.close()


if __name__ == "__main__":
    main()
