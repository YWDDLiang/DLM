"""Fail-closed checkpoint retention helpers.

Retention is evaluated within one training arm/checkpoint parent.  The kept
set is the union of the selected best checkpoint and the newest ``N``
checkpoints by numeric optimizer step.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_CHECKPOINT_PATTERNS = (
    re.compile(r"checkpoint-(\d+)$"),
    re.compile(r"step-(\d+)$"),
)


@dataclass(frozen=True)
class CheckpointEntry:
    path: str
    step: int
    bytes: int


@dataclass(frozen=True)
class RetentionPlan:
    checkpoint_root: str
    best_checkpoint: str
    keep_latest: int
    keep: tuple[CheckpointEntry, ...]
    delete: tuple[CheckpointEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "checkpoint-retention-plan@1",
            "checkpoint_root": self.checkpoint_root,
            "best_checkpoint": self.best_checkpoint,
            "keep_latest": self.keep_latest,
            "keep": [asdict(entry) for entry in self.keep],
            "delete": [asdict(entry) for entry in self.delete],
            "delete_bytes": sum(entry.bytes for entry in self.delete),
        }


def checkpoint_step(path: Path) -> int:
    """Return the numeric step encoded by a supported checkpoint basename."""

    for pattern in _CHECKPOINT_PATTERNS:
        match = pattern.fullmatch(path.name)
        if match:
            return int(match.group(1))
    raise ValueError(f"Unsupported checkpoint directory name: {path.name!r}")


def _allocated_bytes(path: Path) -> int:
    """Return allocated bytes without following symlinks."""

    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        current = Path(dirpath)
        if current.is_symlink():
            raise ValueError(f"Checkpoint tree contains a symlink directory: {current}")
        for name in filenames:
            item = current / name
            stat = item.lstat()
            if item.is_symlink():
                total += stat.st_size
            else:
                total += stat.st_blocks * 512
        for name in tuple(dirnames):
            item = current / name
            if item.is_symlink():
                raise ValueError(f"Checkpoint tree contains a symlink directory: {item}")
    return total


def discover_checkpoints(checkpoint_root: Path) -> tuple[CheckpointEntry, ...]:
    """Discover supported checkpoint directories directly below one root."""

    raw_root = checkpoint_root.expanduser()
    if raw_root.is_symlink():
        raise ValueError(f"Checkpoint root may not be a symlink: {raw_root}")
    root = raw_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Checkpoint root must be a real directory: {root}")

    entries: list[CheckpointEntry] = []
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        try:
            step = checkpoint_step(child)
        except ValueError:
            continue
        entries.append(
            CheckpointEntry(
                path=str(child.resolve(strict=True)),
                step=step,
                bytes=_allocated_bytes(child),
            )
        )
    entries.sort(key=lambda item: (item.step, item.path))
    if not entries:
        raise ValueError(f"No supported checkpoint directories found under {root}")
    if len({entry.step for entry in entries}) != len(entries):
        raise ValueError(f"Duplicate checkpoint steps found under {root}")
    return tuple(entries)


def build_retention_plan(
    checkpoint_root: Path,
    best_checkpoint: Path,
    *,
    keep_latest: int = 2,
) -> RetentionPlan:
    """Build a best-plus-latest retention plan without deleting anything."""

    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")

    raw_root = checkpoint_root.expanduser()
    if raw_root.is_symlink():
        raise ValueError(f"Checkpoint root may not be a symlink: {raw_root}")
    root = raw_root.resolve(strict=True)
    raw_best = best_checkpoint.expanduser()
    if raw_best.is_symlink():
        raise ValueError(f"Best checkpoint may not be a symlink: {raw_best}")
    best = raw_best.resolve(strict=True)

    entries = discover_checkpoints(root)
    by_path = {Path(entry.path): entry for entry in entries}
    if best not in by_path:
        raise ValueError(
            f"Best checkpoint must be a discovered direct child of {root}: {best}"
        )

    latest = entries[-keep_latest:]
    keep_paths = {best, *(Path(entry.path) for entry in latest)}
    keep = tuple(entry for entry in entries if Path(entry.path) in keep_paths)
    delete = tuple(entry for entry in entries if Path(entry.path) not in keep_paths)
    return RetentionPlan(
        checkpoint_root=str(root),
        best_checkpoint=str(best),
        keep_latest=keep_latest,
        keep=keep,
        delete=delete,
    )


def apply_retention_plan(plan: RetentionPlan) -> tuple[str, ...]:
    """Delete only the exact checkpoint directories registered by ``plan``."""

    root = Path(plan.checkpoint_root).resolve(strict=True)
    expected = {
        Path(entry.path).resolve(strict=True): entry for entry in (*plan.keep, *plan.delete)
    }
    current = {
        Path(entry.path).resolve(strict=True): entry for entry in discover_checkpoints(root)
    }
    if set(current) != set(expected):
        raise RuntimeError("Checkpoint inventory changed after the retention plan was built")

    deleted: list[str] = []
    for entry in plan.delete:
        target = Path(entry.path)
        resolved = target.resolve(strict=True)
        if target.is_symlink() or resolved.parent != root:
            raise RuntimeError(f"Refusing unsafe checkpoint deletion: {target}")
        if checkpoint_step(resolved) != entry.step:
            raise RuntimeError(f"Checkpoint identity changed before deletion: {target}")
        shutil.rmtree(resolved)
        deleted.append(str(resolved))
    return tuple(deleted)


def retained_paths(entries: Iterable[CheckpointEntry]) -> tuple[str, ...]:
    """Return stable path-only output for callers that write selection reports."""

    return tuple(entry.path for entry in entries)
