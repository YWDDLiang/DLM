"""Deterministic process prefetch for expensive ragged corruption batches."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import multiprocessing
import os
from collections import deque
from typing import Callable, Sequence

import torch

from .model import WQVariant
from .training import EpochSampler
from .training_data import CorruptedBatch, JsonlRecordIndex, build_corrupted_batch


@dataclasses.dataclass(frozen=True, slots=True)
class CorruptionTask:
    sequence: int
    update: int
    microbatch: int
    epoch_after: int
    indices: tuple[int, ...]
    seed: int
    variant: str
    representation_variant: str | None
    enable_revision_training: bool
    mask_discrete_fields: bool
    enable_topology_corruption: bool


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedCorruption:
    task: CorruptionTask
    batch: CorruptedBatch


_WORKER_DATASET: JsonlRecordIndex | None = None


def _initialize_worker(
    paths: tuple[str, ...],
    entries: tuple[tuple[int, int], ...],
) -> None:
    """Create one independent read-only JSONL view per spawned worker."""

    global _WORKER_DATASET
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_DATASET = JsonlRecordIndex.from_frozen_entries(paths, entries)


def _prepare(task: CorruptionTask) -> PreparedCorruption:
    if _WORKER_DATASET is None:  # pragma: no cover - guarded by executor initializer
        raise RuntimeError("corruption worker dataset is uninitialized")
    records = [_WORKER_DATASET[index] for index in task.indices]
    representation = (
        None
        if task.representation_variant is None
        else WQVariant(task.representation_variant)
    )
    batch = build_corrupted_batch(
        records,
        seed=task.seed,
        variant=WQVariant(task.variant),
        representation_variant=representation,
        enable_revision_training=task.enable_revision_training,
        mask_discrete_fields=task.mask_discrete_fields,
        enable_topology_corruption=task.enable_topology_corruption,
    )
    return PreparedCorruption(task=task, batch=batch)


class DeterministicCorruptionPrefetcher:
    """Submit independent batches ahead while consuming them in exact order.

    Only systems scheduling changes: the main process derives every index and
    corruption seed before submission.  Results are consumed by monotonically
    increasing ``sequence`` regardless of worker completion order.
    """

    def __init__(
        self,
        *,
        dataset: JsonlRecordIndex,
        sampler: EpochSampler,
        training_seed: int,
        total_updates: int,
        microbatch_size: int,
        accumulation_steps: int,
        seed_for: Callable[[int, int, int], int],
        revision_start_update: int,
        variant: WQVariant,
        representation_variant: WQVariant | None,
        mask_discrete_fields: bool,
        enable_topology_corruption: bool,
        workers: int,
        depth: int,
    ) -> None:
        if total_updates <= 0 or microbatch_size <= 0 or accumulation_steps <= 0:
            raise ValueError("prefetch training dimensions must be positive")
        if workers < 1:
            raise ValueError("deterministic process prefetch requires at least one worker")
        if depth < workers or depth > 4 * workers:
            raise ValueError("prefetch depth must be between workers and 4x workers")
        if revision_start_update < 0 or revision_start_update > total_updates:
            raise ValueError("invalid revision boundary")
        self._dataset = dataset
        self._sampler = sampler
        self._training_seed = int(training_seed)
        self._total_updates = int(total_updates)
        self._microbatch_size = int(microbatch_size)
        self._accumulation_steps = int(accumulation_steps)
        self._seed_for = seed_for
        self._revision_start_update = int(revision_start_update)
        self._variant = variant
        self._representation_variant = representation_variant
        self._mask_discrete_fields = bool(mask_discrete_fields)
        self._enable_topology_corruption = bool(enable_topology_corruption)
        self._depth = int(depth)
        self._next_sequence = 0
        self._next_submit_sequence = 0
        self._task_count = self._total_updates * self._accumulation_steps
        context = multiprocessing.get_context("spawn")
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(
                tuple(str(path) for path in dataset.paths),
                tuple(dataset.entries),
            ),
        )
        self._pending: deque[
            tuple[CorruptionTask, concurrent.futures.Future[PreparedCorruption]]
        ] = deque()
        self._fill()

    def _task(self, sequence: int) -> CorruptionTask:
        update, microbatch = divmod(sequence, self._accumulation_steps)
        indices = tuple(self._sampler.take(self._microbatch_size))
        return CorruptionTask(
            sequence=sequence,
            update=update,
            microbatch=microbatch,
            epoch_after=self._sampler.epoch,
            indices=indices,
            seed=self._seed_for(self._training_seed, update, microbatch),
            variant=self._variant.value,
            representation_variant=(
                None
                if self._representation_variant is None
                else self._representation_variant.value
            ),
            enable_revision_training=update >= self._revision_start_update,
            mask_discrete_fields=self._mask_discrete_fields,
            enable_topology_corruption=self._enable_topology_corruption,
        )

    def _fill(self) -> None:
        while (
            len(self._pending) < self._depth
            and self._next_submit_sequence < self._task_count
        ):
            task = self._task(self._next_submit_sequence)
            self._pending.append((task, self._executor.submit(_prepare, task)))
            self._next_submit_sequence += 1

    def take(self, update: int, microbatch: int) -> PreparedCorruption:
        expected = update * self._accumulation_steps + microbatch
        if expected != self._next_sequence or not self._pending:
            raise RuntimeError("prefetch consumer order differs from registered order")
        task, future = self._pending.popleft()
        if task.sequence != expected:
            raise RuntimeError("prefetch queue identity mismatch")
        prepared = future.result()
        if prepared.task != task:
            raise RuntimeError("prefetch worker returned a different task identity")
        self._next_sequence += 1
        self._fill()
        return prepared

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()

    def __enter__(self) -> "DeterministicCorruptionPrefetcher":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

