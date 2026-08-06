#!/usr/bin/env python3
"""CPU-only Slurm benchmark and bitwise audit for refiner batch prefetch."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from crystal_dlm.wqcodiff.model import WQVariant
from crystal_dlm.wqcodiff.training import EpochSampler, _dataset_identity
from crystal_dlm.wqcodiff.training_data import (
    CorruptedBatch,
    JsonlRecordIndex,
    build_corrupted_batch,
)
from crystal_dlm.wqcodiff.training_prefetch import DeterministicCorruptionPrefetcher


def _seed(training_seed: int, update: int, microbatch: int) -> int:
    raw = json.dumps(
        ["crysllmgen_wq_refiner_v1", training_seed, update, microbatch],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _tensors(batch: CorruptedBatch) -> Iterable[tuple[str, torch.Tensor]]:
    for field in dataclasses.fields(batch.inputs):
        yield f"inputs.{field.name}", getattr(batch.inputs, field.name)
    for name, value in zip(batch.targets._fields, batch.targets):
        yield f"targets.{name}", value
    for name, value in zip(batch.prior_targets._fields, batch.prior_targets):
        yield f"prior_targets.{name}", value


def _digest(batch: CorruptedBatch) -> str:
    digest = hashlib.sha256()
    for name, value in _tensors(batch):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    digest.update(
        json.dumps(batch.metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--training-seed", type=int, default=11)
    parser.add_argument("--microbatch", type=int, default=128)
    parser.add_argument("--workers", type=int, choices=(3, 7, 15), required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--parity-batches", type=int, default=4)
    parser.add_argument("--base-source-sha256", required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("prefetch benchmark must run through Slurm")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be 1 for process-level parallelism")
    for value in (args.base_source_sha256, args.execution_patch_sha256):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("source identities must be lowercase SHA256")
    if not 1 <= args.parity_batches < args.batches:
        raise ValueError("parity batches must be positive and smaller than all batches")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    paths = tuple(str(path.resolve()) for path in args.dataset)
    files = _dataset_identity(paths)
    dataset = JsonlRecordIndex(paths)

    reference_sampler = EpochSampler(len(dataset), args.training_seed)
    reference: list[str] = []
    direct_started = time.monotonic()
    for update in range(args.parity_batches):
        indices = reference_sampler.take(args.microbatch)
        records = [dataset[index] for index in indices]
        reference.append(
            _digest(
                build_corrupted_batch(
                    records,
                    seed=_seed(args.training_seed, update, 0),
                    variant=WQVariant.STRAT_GEO,
                    representation_variant=WQVariant.STRAT_GEO,
                    enable_revision_training=False,
                    mask_discrete_fields=False,
                    enable_topology_corruption=True,
                )
            )
        )
    direct_seconds = time.monotonic() - direct_started

    sampler = EpochSampler(len(dataset), args.training_seed)
    constructed = time.monotonic()
    prefetch = DeterministicCorruptionPrefetcher(
        dataset=dataset,
        sampler=sampler,
        training_seed=args.training_seed,
        total_updates=args.batches,
        microbatch_size=args.microbatch,
        accumulation_steps=1,
        seed_for=_seed,
        revision_start_update=args.batches,
        variant=WQVariant.STRAT_GEO,
        representation_variant=WQVariant.STRAT_GEO,
        mask_discrete_fields=False,
        enable_topology_corruption=True,
        workers=args.workers,
        depth=args.depth,
    )
    construction_seconds = time.monotonic() - constructed
    observed: list[str] = []
    warmup_started = time.monotonic()
    try:
        for update in range(args.parity_batches):
            observed.append(_digest(prefetch.take(update, 0).batch))
        warmup_seconds = time.monotonic() - warmup_started
        measured_started = time.monotonic()
        for update in range(args.parity_batches, args.batches):
            observed.append(_digest(prefetch.take(update, 0).batch))
        measured_seconds = time.monotonic() - measured_started
    finally:
        prefetch.close()
        dataset.close()
    parity = reference == observed[: args.parity_batches]
    if not parity:
        raise RuntimeError("prefetched corruptions differ from direct construction")
    aggregate = hashlib.sha256("".join(observed).encode("ascii")).hexdigest()
    measured_batches = args.batches - args.parity_batches
    report = {
        "schema": "wq_refiner_prefetch_benchmark_v1",
        "ok": True,
        "bitwise_parity": parity,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "host": platform.node(),
        "base_source_sha256": args.base_source_sha256,
        "execution_patch_sha256": args.execution_patch_sha256,
        "training_seed": args.training_seed,
        "dataset_files": list(files),
        "dataset_records": len(dataset),
        "microbatch": args.microbatch,
        "workers": args.workers,
        "depth": args.depth,
        "allocated_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
        "library_threads_per_process": 1,
        "batches": args.batches,
        "parity_batches": args.parity_batches,
        "direct_parity_seconds": direct_seconds,
        "construction_seconds": construction_seconds,
        "warmup_seconds": warmup_seconds,
        "measured_seconds": measured_seconds,
        "measured_batches": measured_batches,
        "batches_per_second": measured_batches / measured_seconds,
        "aggregate_batch_digest": aggregate,
        "parity_digests": reference,
    }
    _write_exclusive(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

