"""Fixed-update, evidence-logged training loop for matched WQ variants."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .contracts import write_json_exclusive
from .losses import compute_wq_losses
from .model import WQCoDenoiser, WQModelConfig, WQVariant
from .protocol import RegisteredProtocol, load_protocol
from .training_data import JsonlRecordIndex, build_corrupted_batch


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_paths: tuple[str, ...]
    output_dir: str
    variant: WQVariant
    training_seed: int
    source_bundle_sha256: str
    updates: int = 100_000
    effective_batch_size: int = 128
    microbatch_size: int = 128
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-2
    warmup_fraction: float = 0.05
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    checkpoint_interval: int = 5_000
    log_interval: int = 100
    device: str = "cuda"
    allow_nonpaper_updates: bool = False
    resume_checkpoint: str | None = None
    shared_checkpoint: str | None = None
    stop_after_shared: bool = False
    stop_after_update: int | None = None

    def __post_init__(self) -> None:
        if not self.dataset_paths:
            raise ValueError("training dataset paths are required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_bundle_sha256):
            raise ValueError("source_bundle_sha256 must be a lowercase SHA256")
        if self.updates <= 0:
            raise ValueError("updates must be positive")
        if self.effective_batch_size != 128:
            raise ValueError("registered effective batch size is 128")
        if self.microbatch_size not in {64, 128}:
            raise ValueError("only registered microbatches 128 or OOM fallback 64 are allowed")
        if self.effective_batch_size % self.microbatch_size:
            raise ValueError("microbatch must divide effective batch")
        if self.updates != 100_000 and not self.allow_nonpaper_updates:
            raise ValueError("non-100k runs must be explicitly marked non-paper")
        if self.learning_rate != 2.0e-4 or self.weight_decay != 1.0e-2:
            raise ValueError("optimizer hyperparameters differ from protocol")
        if self.ema_decay != 0.999 or self.gradient_clip_norm != 1.0:
            raise ValueError("EMA/gradient clip differ from protocol")
        if self.resume_checkpoint and self.shared_checkpoint:
            raise ValueError("resume_checkpoint and shared_checkpoint are mutually exclusive")
        if self.stop_after_shared and self.shared_checkpoint:
            raise ValueError("a shared-stage run cannot itself fork from a shared checkpoint")
        if self.stop_after_shared and self.stop_after_update is not None:
            raise ValueError("stop_after_shared and stop_after_update are mutually exclusive")
        if self.stop_after_shared and self.variant not in {
            WQVariant.JOINT_NOREV,
            WQVariant.ATOM_JOINT,
        }:
            raise ValueError(
                "the reusable shared stage must use B-WQ-JOINT-NOREV or B-ATOM-JOINT"
            )
        if self.stop_after_update is not None:
            if not self.shared_updates < self.stop_after_update <= self.updates:
                raise ValueError(
                    "stop_after_update must be after the shared boundary and at most updates"
                )

    @property
    def accumulation_steps(self) -> int:
        return self.effective_batch_size // self.microbatch_size

    @property
    def shared_updates(self) -> int:
        return 60_000 if self.updates == 100_000 else int(round(0.6 * self.updates))

    @property
    def paper_eligible(self) -> bool:
        return (
            self.updates == 100_000
            and not self.allow_nonpaper_updates
            and not self.stop_after_shared
            and self.end_update == self.updates
        )

    @property
    def end_update(self) -> int:
        if self.stop_after_shared:
            return self.shared_updates
        return self.updates if self.stop_after_update is None else self.stop_after_update


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone().float()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach().float(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay mismatch")
        # Checkpoints are deliberately deserialized on CPU.  Relocate every
        # EMA tensor to the live model device before the next in-place update;
        # otherwise a shared-boundary fork fails on its first optimizer step.
        self.shadow = {
            name: value.detach().clone().float().to(device=device)
            for name, value in state["shadow"].items()
        }

    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            if name in state:
                state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))


class EpochSampler:
    """Deterministic no-replacement epochs independent of DataLoader workers."""

    def __init__(self, size: int, seed: int) -> None:
        if size <= 0:
            raise ValueError("dataset must be nonempty")
        self.size = size
        self.rng = random.Random(seed)
        self.order = list(range(size))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.epoch = 0

    def take(self, count: int) -> list[int]:
        result: list[int] = []
        while len(result) < count:
            available = min(count - len(result), self.size - self.cursor)
            result.extend(self.order[self.cursor : self.cursor + available])
            self.cursor += available
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor = 0
                self.rng.shuffle(self.order)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.order = list(state["order"])
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
        self.rng.setstate(state["rng_state"])


def _corruption_seed(training_seed: int, update: int, microbatch: int) -> int:
    value = f"wqcodiff:{training_seed}:{update}:{microbatch}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 63) - 1)


def _learning_rate_multiplier(step: int, updates: int, warmup_fraction: float) -> float:
    warmup = max(1, int(round(updates * warmup_fraction)))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(updates - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_identity(paths: Sequence[str]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        result.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return tuple(result)


def _save_checkpoint(
    path: Path,
    *,
    model: WQCoDenoiser,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
    sampler: EpochSampler,
    step: int,
    config: TrainingConfig,
    protocol: RegisteredProtocol,
    dataset_files: Sequence[Mapping[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    payload = {
            "schema": "wqcodiff_checkpoint_v1",
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "sampler": sampler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "training_config": dataclasses.asdict(config),
            "source_bundle_sha256": config.source_bundle_sha256,
            "dataset_files": [dict(item) for item in dataset_files],
            "model_config": dataclasses.asdict(model.config),
            "protocol_name": protocol.name,
            "protocol_sha256": protocol.sha256,
            "paper_eligible": config.paper_eligible,
        }
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    *,
    model: WQCoDenoiser,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
    sampler: EpochSampler,
    config: TrainingConfig,
    protocol: RegisteredProtocol,
    dataset_files: Sequence[Mapping[str, Any]],
    shared_fork: bool = False,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "wqcodiff_checkpoint_v1":
        raise ValueError("unsupported checkpoint schema")
    if payload["protocol_sha256"] != protocol.sha256 or payload["protocol_name"] != protocol.name:
        raise ValueError("checkpoint protocol hash mismatch")
    if payload.get("source_bundle_sha256") != config.source_bundle_sha256:
        raise ValueError("checkpoint source bundle hash mismatch")
    if payload.get("dataset_files") != [dict(item) for item in dataset_files]:
        raise ValueError("checkpoint dataset file identity mismatch")
    previous = payload["training_config"]
    previous_variant = (
        previous["variant"].value
        if isinstance(previous["variant"], WQVariant)
        else str(previous["variant"])
    )
    if shared_fork:
        expected_shared_variant = (
            WQVariant.ATOM_JOINT.value
            if config.variant is WQVariant.ATOM_JOINT
            else WQVariant.JOINT_NOREV.value
        )
        if previous_variant != expected_shared_variant:
            raise ValueError("shared fork source representation does not match target")
        if not bool(previous.get("stop_after_shared")):
            raise ValueError("shared fork source was not frozen at the shared boundary")
        if int(payload["step"]) != config.shared_updates:
            raise ValueError("shared fork checkpoint is not at the 60% boundary")
        if int(previous["updates"]) != config.updates:
            raise ValueError("shared fork and target update contracts differ")
    elif previous_variant != config.variant.value:
        raise ValueError("checkpoint variant mismatch")
    if int(previous["training_seed"]) != config.training_seed:
        raise ValueError("checkpoint training seed mismatch")
    matched_fields = (
        "dataset_paths",
        "updates",
        "effective_batch_size",
        "microbatch_size",
        "learning_rate",
        "weight_decay",
        "warmup_fraction",
        "gradient_clip_norm",
        "ema_decay",
        "source_bundle_sha256",
    )
    current = dataclasses.asdict(config)
    for field in matched_fields:
        previous_value = previous[field]
        current_value = current[field]
        if field == "dataset_paths":
            previous_value = tuple(previous_value)
            current_value = tuple(current_value)
        if previous_value != current_value:
            raise ValueError(f"checkpoint training contract mismatch: {field}")
    if payload.get("model_config") != dataclasses.asdict(model.config):
        raise ValueError("checkpoint model config mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    ema.load_state_dict(
        payload["ema"],
        device=next(model.parameters()).device,
    )
    sampler.load_state_dict(payload["sampler"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["step"])


def train(config: TrainingConfig, *, protocol_path: str | Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    dataset_files = _dataset_identity(config.dataset_paths)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "train_metrics.jsonl"
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered GPU training requires CUDA inside Slurm")

    random.seed(config.training_seed)
    torch.manual_seed(config.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training_seed)
    torch.use_deterministic_algorithms(False)

    dataset = JsonlRecordIndex(config.dataset_paths)
    sampler = EpochSampler(len(dataset), config.training_seed)
    model = WQCoDenoiser(WQModelConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_multiplier(
            step,
            config.updates,
            config.warmup_fraction,
        ),
    )
    ema = ExponentialMovingAverage(model, config.ema_decay)
    start_step = 0
    if config.resume_checkpoint:
        start_step = _load_checkpoint(
            Path(config.resume_checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            sampler=sampler,
            config=config,
            protocol=protocol,
            dataset_files=dataset_files,
        )
    elif config.shared_checkpoint:
        start_step = _load_checkpoint(
            Path(config.shared_checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            sampler=sampler,
            config=config,
            protocol=protocol,
            dataset_files=dataset_files,
            shared_fork=True,
        )

    run_manifest = {
        "schema": "wqcodiff_training_run_v1",
        "protocol_name": protocol.name,
        "protocol_sha256": protocol.sha256,
        "source_bundle_sha256": config.source_bundle_sha256,
        "training_config": {
            **dataclasses.asdict(config),
            "variant": config.variant.value,
        },
        "model_config": dataclasses.asdict(model.config),
        "parameter_count": model.parameter_count(),
        "dataset_records": len(dataset),
        "dataset_files": [dict(item) for item in dataset_files],
        "host": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "paper_eligible": config.paper_eligible,
        "lineage": {
            "resume_checkpoint": config.resume_checkpoint,
            "shared_checkpoint": config.shared_checkpoint,
            "shared_checkpoint_sha256": (
                hashlib.sha256(Path(config.shared_checkpoint).read_bytes()).hexdigest()
                if config.shared_checkpoint
                else None
            ),
        },
    }
    write_json_exclusive(output_dir / "run_manifest.json", run_manifest)

    model.train()
    started = time.monotonic()
    end_step = config.end_update
    if start_step >= end_step:
        raise ValueError("checkpoint is already at or beyond the requested training boundary")
    for step in range(start_step, end_step):
        completed = step + 1
        log_this_step = completed == 1 or completed % config.log_interval == 0
        optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        periodic_scale_observations: dict[str, list[torch.Tensor]] = {
            "birth_coordinate": [],
            "bridge": [],
            "prior_coordinate": [],
        }
        active_variant = (
            WQVariant.JOINT_NOREV
            if step < config.shared_updates
            else config.variant
        )
        for microbatch in range(config.accumulation_steps):
            indices = sampler.take(config.microbatch_size)
            records = [dataset[index] for index in indices]
            corrupted = build_corrupted_batch(
                records,
                seed=_corruption_seed(config.training_seed, step, microbatch),
                variant=active_variant,
                representation_variant=config.variant,
                enable_revision_training=step >= config.shared_updates,
            ).to(device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
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
                losses = compute_wq_losses(
                    output,
                    corrupted.targets,
                    masked_prior=masked_prior,
                    conditioned_prior=conditioned_prior,
                    prior_target=corrupted.prior_targets,
                )
                loss = losses.total / config.accumulation_steps
            if log_this_step:
                for name, log_scale, mask in (
                    (
                        "birth_coordinate",
                        output.birth_coordinate_log_scale,
                        corrupted.targets.birth_coordinate_mask,
                    ),
                    (
                        "bridge",
                        output.bridge_log_scale,
                        corrupted.targets.bridge_mask,
                    ),
                    (
                        "prior_coordinate",
                        conditioned_prior.first_coordinate_log_scale,
                        corrupted.prior_targets.first_coordinate_mask,
                    ),
                ):
                    values = log_scale.detach().float().exp()[mask]
                    if values.numel():
                        periodic_scale_observations[name].append(values.cpu())
            loss.backward()
            for name, value in losses._asdict().items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value.detach()) / config.accumulation_steps
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        )
        optimizer.step()
        scheduler.step()
        ema.update(model)

        if log_this_step:
            elapsed = time.monotonic() - started
            component_abs_sum = sum(
                abs(value)
                for name, value in aggregate.items()
                if name != "total"
            )
            periodic_scale_metrics: dict[str, float | int | None] = {}
            for name, chunks in periodic_scale_observations.items():
                if not chunks:
                    periodic_scale_metrics[f"{name}_scale_supervised_values"] = 0
                    periodic_scale_metrics[f"{name}_scale_min"] = None
                    periodic_scale_metrics[f"{name}_scale_mean"] = None
                    periodic_scale_metrics[f"{name}_scale_max"] = None
                    continue
                values = torch.cat(chunks)
                periodic_scale_metrics[f"{name}_scale_supervised_values"] = int(
                    values.numel()
                )
                periodic_scale_metrics[f"{name}_scale_min"] = float(values.min())
                periodic_scale_metrics[f"{name}_scale_mean"] = float(values.mean())
                periodic_scale_metrics[f"{name}_scale_max"] = float(values.max())
            _append_jsonl(
                metrics_path,
                {
                    "step": completed,
                    "phase": "shared_pretraining" if completed <= config.shared_updates else "method_specific",
                    "active_variant": active_variant.value,
                    "epoch": sampler.epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": gradient_norm,
                    "gradient_norm_pre_clip": gradient_norm,
                    "gradient_norm_pre_clip_per_sqrt_parameter": gradient_norm
                    / math.sqrt(model.parameter_count()),
                    "gradient_clip_threshold": config.gradient_clip_norm,
                    "gradient_was_clipped": gradient_norm > config.gradient_clip_norm,
                    "gradient_clip_scale": min(
                        1.0,
                        config.gradient_clip_norm / max(gradient_norm, 1.0e-12),
                    ),
                    "geometry_fraction_of_abs_components": (
                        abs(aggregate.get("geometry", 0.0))
                        / max(component_abs_sum, 1.0e-12)
                    ),
                    "elapsed_s": elapsed,
                    "updates_per_s": (completed - start_step) / max(elapsed, 1.0e-9),
                    **aggregate,
                    **periodic_scale_metrics,
                },
            )
        if completed % config.checkpoint_interval == 0 or completed == end_step:
            _save_checkpoint(
                output_dir / f"checkpoint_{completed:07d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                sampler=sampler,
                step=completed,
                config=config,
                protocol=protocol,
                dataset_files=dataset_files,
            )

    if config.stop_after_shared:
        shared_path = output_dir / f"checkpoint_{end_step:07d}.pt"
        result = {
            "ok": True,
            "schema": "wqcodiff_shared_training_complete_v1",
            "output_dir": str(output_dir),
            "shared_checkpoint": str(shared_path),
            "shared_checkpoint_sha256": hashlib.sha256(shared_path.read_bytes()).hexdigest(),
            "step": end_step,
            "target_updates": config.updates,
            "paper_eligible": False,
            "source_bundle_sha256": config.source_bundle_sha256,
            "dataset_files": [dict(item) for item in dataset_files],
            "elapsed_s": time.monotonic() - started,
        }
        write_json_exclusive(output_dir / "shared_training_complete.json", result)
        dataset.close()
        return result

    ema_model = WQCoDenoiser(model.config).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema.copy_to(ema_model)
    is_final = end_step == config.updates
    final_path = output_dir / (
        "model_ema_final.pt" if is_final else f"model_ema_at_{end_step:07d}.pt"
    )
    if final_path.exists():
        raise FileExistsError(f"immutable EMA checkpoint already exists: {final_path}")
    final_payload = {
            "schema": "wqcodiff_ema_model_v1",
            "model": ema_model.state_dict(),
            "model_config": dataclasses.asdict(model.config),
            "training_config": {
                **dataclasses.asdict(config),
                "variant": config.variant.value,
            },
            "source_bundle_sha256": config.source_bundle_sha256,
            "dataset_files": [dict(item) for item in dataset_files],
            "protocol_name": protocol.name,
            "protocol_sha256": protocol.sha256,
            "paper_eligible": config.paper_eligible,
        }
    with final_path.open("xb") as handle:
        torch.save(final_payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
    result = {
        "ok": True,
        "schema": (
            "wqcodiff_training_complete_v1"
            if is_final
            else "wqcodiff_partial_training_complete_v1"
        ),
        "output_dir": str(output_dir),
        "ema_checkpoint": str(final_path),
        "ema_checkpoint_sha256": digest,
        "updates_completed": end_step,
        "target_updates": config.updates,
        "paper_eligible": config.paper_eligible,
        "source_bundle_sha256": config.source_bundle_sha256,
        "dataset_files": [dict(item) for item in dataset_files],
        "elapsed_s": time.monotonic() - started,
    }
    if is_final:
        result["final_checkpoint"] = str(final_path)
        result["final_checkpoint_sha256"] = digest
        result["updates"] = config.updates
    write_json_exclusive(
        output_dir / ("training_complete.json" if is_final else "partial_training_complete.json"),
        result,
    )
    dataset.close()
    return result
