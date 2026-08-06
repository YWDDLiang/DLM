"""Independent schedule construction used to audit the upstream buffers."""

from __future__ import annotations

import math
from typing import Any


# The released MP20 checkpoint contains scheduler buffers with 1001 entries,
# i.e. a 1000-step parent schedule. CrysLLMGen refinement starts from t=800
# inside that schedule; these are deliberately separate contracts.
PARENT_SCHEDULER_TIMESTEPS = 1000
OFFICIAL_REVERSE_START_TIMESTEP = 800
PARENT_RUN_TYPE = "train"


def build_beta_tables(timesteps: int) -> dict[str, Any]:
    """Reconstruct CrysLLMGen's cosine beta buffers with torch operations."""

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - A800 runtime path.
        raise RuntimeError("torch is required for CrysLLMGen schedule parity") from exc
    smoothing = 0.008
    steps = timesteps + 1
    positions = torch.linspace(0, timesteps, steps)
    cumulative = torch.cos(
        ((positions / timesteps) + smoothing)
        / (1 + smoothing)
        * math.pi
        * 0.5
    ).square()
    cumulative = cumulative / cumulative[0]
    beta_body = 1 - cumulative[1:] / cumulative[:-1]
    beta_body = torch.clip(beta_body, 0.0001, 0.9999)
    betas = torch.cat([torch.zeros([1]), beta_body], dim=0)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    posterior_sigmas = torch.zeros_like(betas)
    posterior_sigmas[1:] = (
        betas[1:]
        * (1.0 - alphas_cumprod[:-1])
        / (1.0 - alphas_cumprod[1:])
    )
    posterior_sigmas = torch.sqrt(posterior_sigmas)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "posterior_sigmas": posterior_sigmas,
    }


def build_coordinate_sigmas(
    timesteps: int, *, sigma_begin: float = 0.005, sigma_end: float = 0.5
) -> Any:
    """Reconstruct the non-normalized wrapped-coordinate sigma grid."""

    if timesteps <= 0 or not 0.0 < sigma_begin < sigma_end:
        raise ValueError("invalid coordinate sigma schedule")
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - A800 runtime path.
        raise RuntimeError("numpy and torch are required for schedule parity") from exc
    body = torch.FloatTensor(
        np.exp(np.linspace(math.log(sigma_begin), math.log(sigma_end), timesteps))
    )
    return torch.cat([torch.zeros([1]), body], dim=0)
