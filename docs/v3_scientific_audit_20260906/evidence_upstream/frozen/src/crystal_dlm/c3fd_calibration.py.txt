"""Train-only proposal interaction and head-specific calibration for C³FD-v2.1."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    nll_before: float
    nll_after: float
    examples: int
    optimizer: str = "LBFGS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_temperature(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int = -100,
    max_iter: int = 100,
) -> TemperatureFit:
    """Fit one positive scalar temperature by validation NLL, without a grid."""

    values = torch.as_tensor(logits).detach().float()
    labels = torch.as_tensor(targets).detach().long().to(values.device)
    if values.ndim != 2 or labels.ndim != 1 or values.shape[0] != labels.shape[0]:
        raise ValueError("temperature fitting requires logits [N,C] and targets [N]")
    valid = labels != int(ignore_index)
    values = values[valid]
    labels = labels[valid]
    if values.shape[0] == 0:
        raise ValueError("temperature fitting has no valid examples")
    if bool(((labels < 0) | (labels >= values.shape[-1])).any().item()):
        raise ValueError("temperature targets outside class range")
    nll_before = float(F.cross_entropy(values, labels).item())
    log_temperature = torch.zeros((), device=values.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.25,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.clamp(
            min=math.log(0.05), max=math.log(20.0)
        ).exp()
        loss = F.cross_entropy(values / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(
        log_temperature.detach()
        .clamp(min=math.log(0.05), max=math.log(20.0))
        .exp()
        .item()
    )
    nll_after = float(F.cross_entropy(values / temperature, labels).item())
    return TemperatureFit(
        temperature=temperature,
        nll_before=nll_before,
        nll_after=nll_after,
        examples=int(values.shape[0]),
    )


@dataclass(frozen=True)
class StratumInteraction:
    """Train-only family/N/arity support and marginal-interaction correction."""

    strata: tuple[tuple[int, int, int], ...]
    log_corrections: tuple[float, ...]
    counts: tuple[int, ...]
    alpha: float

    @classmethod
    def fit(
        cls,
        rows: Iterable[Sequence[int]],
        *,
        alpha: float = 1.0,
    ) -> "StratumInteraction":
        if float(alpha) <= 0:
            raise ValueError("alpha must be positive")
        joint: Counter[tuple[int, int, int]] = Counter()
        for raw in rows:
            if len(raw) != 3:
                raise ValueError("stratum rows must contain family, N and arity")
            family, n_value, arity = (int(value) for value in raw)
            if family < 0 or not 1 <= n_value <= 20 or not 1 <= arity <= 7:
                raise ValueError("invalid proposal stratum")
            joint[(family, n_value, arity)] += 1
        if not joint:
            raise ValueError("cannot fit an empty stratum interaction")
        family_counts: Counter[int] = Counter()
        n_counts: Counter[int] = Counter()
        arity_counts: Counter[int] = Counter()
        for (family, n_value, arity), count in joint.items():
            family_counts[family] += count
            n_counts[n_value] += count
            arity_counts[arity] += count
        total = sum(joint.values())
        strata = tuple(sorted(joint))
        family_size = len(family_counts)
        n_size = len(n_counts)
        arity_size = len(arity_counts)
        joint_size = len(strata)
        corrections = []
        for family, n_value, arity in strata:
            p_joint = (joint[(family, n_value, arity)] + alpha) / (
                total + alpha * joint_size
            )
            p_family = (family_counts[family] + alpha) / (
                total + alpha * family_size
            )
            p_n = (n_counts[n_value] + alpha) / (total + alpha * n_size)
            p_arity = (arity_counts[arity] + alpha) / (
                total + alpha * arity_size
            )
            corrections.append(
                math.log(p_joint) - math.log(p_family) - math.log(p_n) - math.log(p_arity)
            )
        return cls(
            strata=strata,
            log_corrections=tuple(float(value) for value in corrections),
            counts=tuple(int(joint[value]) for value in strata),
            alpha=float(alpha),
        )

    def joint_scores(
        self,
        family_logits: Tensor,
        n_logits: Tensor,
        arity_logits: Tensor,
        *,
        family_temperature: float = 1.0,
        n_temperature: float = 1.0,
        arity_temperature: float = 1.0,
    ) -> Tensor:
        if family_logits.ndim != 1 or n_logits.ndim != 1 or arity_logits.ndim != 1:
            raise ValueError("joint proposal scoring expects one-dimensional logits")
        if n_logits.shape[0] < 20 or arity_logits.shape[0] < 7:
            raise ValueError("proposal logits do not cover N=1..20 and arity=1..7")
        for value in (family_temperature, n_temperature, arity_temperature):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("proposal temperatures must be finite and positive")
        scores = []
        for correction, (family, n_value, arity) in zip(
            self.log_corrections, self.strata
        ):
            if family >= family_logits.shape[0]:
                raise ValueError("family stratum outside logits")
            scores.append(
                family_logits[family] / float(family_temperature)
                + n_logits[n_value - 1] / float(n_temperature)
                + arity_logits[arity - 1] / float(arity_temperature)
                + family_logits.new_tensor(float(correction))
            )
        return torch.stack(scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "h1a2_c3fd_v21_stratum_interaction_v1",
            "alpha": float(self.alpha),
            "strata": [
                {
                    "family": family,
                    "N": n_value,
                    "arity": arity,
                    "count": count,
                    "log_correction": correction,
                }
                for (family, n_value, arity), count, correction in zip(
                    self.strata, self.counts, self.log_corrections
                )
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StratumInteraction":
        rows = payload.get("strata") or ()
        return cls(
            strata=tuple(
                (int(row["family"]), int(row["N"]), int(row["arity"]))
                for row in rows
            ),
            counts=tuple(int(row["count"]) for row in rows),
            log_corrections=tuple(float(row["log_correction"]) for row in rows),
            alpha=float(payload.get("alpha", 1.0)),
        )


def calibrated_top_p_probabilities(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
) -> Tensor:
    """Apply temperature and nucleus filtering only; no top-k truncation."""

    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must lie in (0,1]")
    values = torch.as_tensor(logits).float().clone()
    if values.ndim != 1 or not bool(torch.isfinite(values).any().item()):
        raise ValueError("sampling logits must contain a finite one-dimensional support")
    probabilities = torch.softmax(values / float(temperature), dim=-1)
    if float(top_p) < 1.0:
        sorted_prob, sorted_idx = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_prob, dim=-1)
        remove = cumulative - sorted_prob > float(top_p)
        sorted_prob[remove] = 0.0
        probabilities.zero_().scatter_(0, sorted_idx, sorted_prob)
        probabilities /= probabilities.sum().clamp_min(1e-12)
    return probabilities


__all__ = [
    "StratumInteraction",
    "TemperatureFit",
    "calibrated_top_p_probabilities",
    "fit_temperature",
]
