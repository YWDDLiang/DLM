"""Pure learning-rate helpers for the frozen two-epoch DLM schedule."""

from __future__ import annotations

import math


def _cosine_multiplier(step: int, total: int, warmup: int, minimum: float) -> float:
    step = max(0, int(step))
    total = max(1, int(total))
    warmup = max(0, min(int(warmup), total))
    minimum = float(minimum)
    if warmup > 0 and step < warmup:
        return max(1e-8, float(step + 1) / float(warmup))
    decay = max(1, total - warmup)
    progress = min(1.0, max(0.0, float(step - warmup) / float(decay)))
    return max(minimum, 0.5 * (1.0 + math.cos(math.pi * progress)))


def two_stage_lr_multiplier(
    step: int,
    *,
    total_steps: int,
    stage_boundary: int,
    stage1_warmup: int,
    stage1_min_ratio: float,
    stage2_base_ratio: float,
    stage2_warmup: int,
    stage2_min_ratio: float,
) -> float:
    """Return LR/epoch1-base-LR for one deterministic two-stage restart."""

    total_steps = int(total_steps)
    boundary = int(stage_boundary)
    if total_steps < 2 or boundary <= 0 or boundary >= total_steps:
        raise ValueError("two-stage LR boundary must lie strictly inside training")
    if not 0.0 < float(stage2_base_ratio) <= 1.0:
        raise ValueError("stage2 base ratio must be in (0,1]")
    if int(step) < boundary:
        return _cosine_multiplier(
            int(step),
            boundary,
            int(stage1_warmup),
            float(stage1_min_ratio),
        )
    return float(stage2_base_ratio) * _cosine_multiplier(
        int(step) - boundary,
        total_steps - boundary,
        int(stage2_warmup),
        float(stage2_min_ratio),
    )


__all__ = ["two_stage_lr_multiplier"]
