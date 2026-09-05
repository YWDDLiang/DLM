"""General transaction-scoped logit transforms for crystal DLM decoding.

The interfaces in this module deliberately know nothing about a particular
physics model.  A transform may prepare one proposal from the first model step
of a transaction and reuse it while the transaction's scalar components are
resolved.  The SPAD runtime remains responsible for schema/PBC support,
sampling, atomic commit, and rollback.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import torch


TransactionKind = Literal["cell", "site_xyz"]


def _immutable_metadata(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class TransactionContext:
    """Read-only description of one SPAD cell or site transaction.

    ``active_positions`` are absolute sequence positions.  The complete token
    snapshot is captured before the transaction is re-masked, so a transform
    can recover the previous geometry even while the model sees masks at the
    active positions.  Tensor storage is treated as read-only by this API.
    """

    kind: TransactionKind
    active_positions: tuple[int, ...]
    complete_pre_remask_tokens: torch.Tensor
    previous_active_token_ids: tuple[int, ...]
    prompt_length: int
    gen_length: int
    plan_metadata: Mapping[str, Any] | None = None
    program_metadata: Mapping[str, Any] | None = None
    batch_index: int = 0
    block_index: int | None = None
    site_index: int | None = None
    site_order_index: int | None = None
    component_indices: tuple[int, ...] = ()
    lattice_version: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("cell", "site_xyz"):
            raise ValueError("transaction kind must be 'cell' or 'site_xyz'")
        if int(self.prompt_length) < 0 or int(self.gen_length) <= 0:
            raise ValueError("transaction prompt/gen lengths are invalid")
        if self.complete_pre_remask_tokens.ndim != 1:
            raise ValueError("complete_pre_remask_tokens must be one-dimensional")
        expected_length = int(self.prompt_length) + int(self.gen_length)
        if int(self.complete_pre_remask_tokens.shape[0]) != expected_length:
            raise ValueError("pre-remask token snapshot has the wrong length")
        if len(self.active_positions) != len(self.previous_active_token_ids):
            raise ValueError("active positions and previous tokens must align")
        if len(self.active_positions) != len(self.component_indices):
            raise ValueError("active positions and component indices must align")
        if len(set(self.active_positions)) != len(self.active_positions):
            raise ValueError("active transaction positions must be unique")
        if any(not 0 <= int(position) < expected_length for position in self.active_positions):
            raise ValueError("active transaction position lies outside the sequence")
        if int(self.batch_index) < 0 or int(self.lattice_version) < 0:
            raise ValueError("batch index and lattice version must be non-negative")
        if self.kind == "cell":
            if len(self.active_positions) != 6 or self.component_indices != tuple(range(6)):
                raise ValueError("cell transactions require six ordered components")
            if self.site_index is not None or self.site_order_index is not None:
                raise ValueError("cell transactions cannot carry site indices")
        else:
            if len(self.active_positions) != 3 or self.component_indices != (0, 1, 2):
                raise ValueError("site_xyz transactions require X/Y/Z components")
            if self.site_index is None or self.site_order_index is None:
                raise ValueError("site_xyz transactions require site indices")
        object.__setattr__(self, "plan_metadata", _immutable_metadata(self.plan_metadata))
        object.__setattr__(
            self,
            "program_metadata",
            _immutable_metadata(self.program_metadata),
        )

    @property
    def generation_positions(self) -> tuple[int, ...]:
        return tuple(int(position) - int(self.prompt_length) for position in self.active_positions)


@dataclass(frozen=True)
class TransactionModelStep:
    """One model forward exposed to ``prepare`` without another forward pass."""

    token_ids: torch.Tensor
    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None

    def row(self, row_index: int) -> "TransactionModelStep":
        """Select one logical batch row while retaining sequence dimensions."""

        row = int(row_index)
        if self.token_ids.ndim != 2 or self.logits.ndim != 3:
            raise ValueError("batched model step must contain [B,L] ids and [B,L,V] logits")
        if not 0 <= row < int(self.token_ids.shape[0]):
            raise IndexError("model-step row lies outside the batch")
        hidden = self.hidden_states
        if hidden is not None:
            if hidden.ndim != 3 or int(hidden.shape[0]) != int(self.token_ids.shape[0]):
                raise ValueError("hidden states do not align with the logical model batch")
            hidden = hidden[row]
        return TransactionModelStep(
            token_ids=self.token_ids[row].detach().clone(),
            logits=self.logits[row],
            hidden_states=hidden,
        )


@runtime_checkable
class TransactionLogitTransform(Protocol):
    """Optional transaction proposal and component-logit transformation."""

    def prepare(
        self,
        context: TransactionContext,
        model_step: TransactionModelStep,
    ) -> Any:
        """Prepare one cached proposal before resolving transaction components."""

    def apply(
        self,
        component: int,
        logits: torch.Tensor,
        proposal: Any,
        context: TransactionContext,
    ) -> torch.Tensor:
        """Return transformed logits for one active component only."""


def apply_transaction_logit_transform(
    transform: TransactionLogitTransform,
    *,
    component: int,
    active_logits: torch.Tensor,
    proposal: Any,
    context: TransactionContext,
) -> torch.Tensor:
    """Apply a transform while preserving the runtime's existing hard support."""

    if active_logits.ndim != 1 or not active_logits.is_floating_point():
        raise ValueError("active transaction logits must be a floating-point vector")
    base = active_logits.clone()
    transformed = transform.apply(
        int(component),
        active_logits.clone(),
        proposal,
        context,
    )
    if not isinstance(transformed, torch.Tensor):
        raise TypeError("transaction logit transform must return a tensor")
    if transformed.shape != base.shape:
        raise ValueError("transaction logit transform changed the vocabulary shape")
    if transformed.device != base.device or transformed.dtype != base.dtype:
        raise ValueError("transaction logit transform changed logits device or dtype")
    if bool(torch.isnan(transformed).any()) or bool(torch.isposinf(transformed).any()):
        raise ValueError("transaction logit transform produced NaN or +inf")

    minimum = torch.finfo(base.dtype).min
    blocked = torch.isneginf(base) | (base == minimum)
    output = transformed.clone()
    output[blocked] = base[blocked]
    available = (~blocked) & torch.isfinite(output) & (output > minimum)
    if not bool(available.any()):
        raise ValueError("transaction logit transform removed every legal token")
    return output


__all__ = [
    "TransactionContext",
    "TransactionKind",
    "TransactionLogitTransform",
    "TransactionModelStep",
    "apply_transaction_logit_transform",
]
