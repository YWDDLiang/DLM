"""Typed residual Planner for a frozen C3FD distribution and Llama states.

This module deliberately has no dependency on ``transformers``.  A caller
builds ``inputs_embeds`` with :meth:`C3FDLlamaTypedResidualPlanner.typed_inputs_embeds`,
runs its causal language model, and passes the resulting hidden-state tensor
back to this module.  C3FD remains the authoritative masked base distribution;
the language model supplies only zero-initialized residual logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


SOFT_FIELDS = (
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
)


@dataclass(frozen=True)
class C3FDLlamaTypedPlannerConfig:
    """Dimensions and typed action conventions for the residual Planner."""

    llama_hidden_size: int
    typed_embedding_size: int
    num_stability_goals: int
    num_proposal_states: int
    num_proposal_strata: int
    num_species: int
    max_count: int
    ledger_feature_size: int
    num_lattice_systems: int
    num_spacegroup_buckets: int
    num_volume_per_atom_bins: int
    max_sequence_length: int = 16
    ledger_abs_bound: float = 1.0

    def __post_init__(self) -> None:
        integer_fields = {
            "llama_hidden_size": self.llama_hidden_size,
            "typed_embedding_size": self.typed_embedding_size,
            "num_stability_goals": self.num_stability_goals,
            "num_proposal_states": self.num_proposal_states,
            "num_proposal_strata": self.num_proposal_strata,
            "num_species": self.num_species,
            "max_count": self.max_count,
            "ledger_feature_size": self.ledger_feature_size,
            "num_lattice_systems": self.num_lattice_systems,
            "num_spacegroup_buckets": self.num_spacegroup_buckets,
            "num_volume_per_atom_bins": self.num_volume_per_atom_bins,
            "max_sequence_length": self.max_sequence_length,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not torch.isfinite(torch.tensor(float(self.ledger_abs_bound))):
            raise ValueError("ledger_abs_bound must be finite")
        if float(self.ledger_abs_bound) <= 0.0:
            raise ValueError("ledger_abs_bound must be positive")

    @property
    def eos_species_index(self) -> int:
        return int(self.num_species)

    @property
    def sentinel_species_index(self) -> int:
        return int(self.num_species) + 1

    @property
    def eos_action_index(self) -> int:
        return int(self.num_species) * int(self.max_count)

    @property
    def num_joint_actions(self) -> int:
        return self.eos_action_index + 1

    @property
    def soft_head_dims(self) -> dict[str, int]:
        return {
            "lattice_system": int(self.num_lattice_systems),
            "spacegroup_bucket": int(self.num_spacegroup_buckets),
            "volume_per_atom_bin": int(self.num_volume_per_atom_bins),
        }


@dataclass(frozen=True)
class TypedResidualLogits:
    """Residual logits emitted from caller-provided Llama hidden states."""

    proposal: Tensor
    actions: Tensor
    soft_fields: dict[str, Tensor]


def joint_action_index(
    species_index: int,
    count_value: int,
    *,
    num_species: int,
    max_count: int,
) -> int:
    """Encode a real species/count action or the explicit EOS action."""

    species = int(species_index)
    count = int(count_value)
    if species == int(num_species):
        if count != 0:
            raise ValueError("EOS teacher action requires count zero")
        return int(num_species) * int(max_count)
    if species < 0 or species >= int(num_species):
        raise ValueError("teacher species index is outside the real-species range")
    if count < 1 or count > int(max_count):
        raise ValueError("real teacher action requires count in 1..max_count")
    return species * int(max_count) + count - 1


def decode_joint_action(
    action_index: int,
    *,
    num_species: int,
    max_count: int,
) -> tuple[int, int]:
    """Inverse of :func:`joint_action_index`, failing on unknown classes."""

    action = int(action_index)
    eos = int(num_species) * int(max_count)
    if action == eos:
        return int(num_species), 0
    if action < 0 or action >= eos:
        raise ValueError("joint action index is outside the configured action space")
    return action // int(max_count), action % int(max_count) + 1


def _validate_logits_and_mask(logits: Tensor, legal_mask: Tensor, *, name: str) -> None:
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if logits.ndim < 1 or logits.shape[-1] <= 0:
        raise ValueError(f"{name} must have a non-empty class dimension")
    if not isinstance(legal_mask, Tensor) or legal_mask.dtype is not torch.bool:
        raise TypeError(f"{name} legal mask must be a bool tensor")
    if legal_mask.shape != logits.shape:
        raise ValueError(f"{name} legal mask must exactly match logits shape")
    if legal_mask.device != logits.device:
        raise ValueError(f"{name} legal mask and logits must share a device")
    if bool((~legal_mask.any(dim=-1)).any().item()):
        raise ValueError(f"{name} contains a row with no legal class")
    if bool(torch.isnan(logits).any().item()) or bool(torch.isposinf(logits).any().item()):
        raise ValueError(f"{name} contains NaN or positive infinity")
    if bool((legal_mask & torch.isneginf(logits)).any().item()):
        raise ValueError(f"{name} contains negative infinity at a legal class")


def masked_log_softmax(logits: Tensor, legal_mask: Tensor, *, name: str = "logits") -> Tensor:
    """Normalize over legal classes while returning ``-inf`` elsewhere."""

    _validate_logits_and_mask(logits, legal_mask, name=name)
    masked = logits.masked_fill(~legal_mask, -torch.inf)
    normalized = torch.log_softmax(masked, dim=-1)
    return normalized.masked_fill(~legal_mask, -torch.inf)


def unit_weight_poe_log_probs(
    c3fd_calibrated_logits: Tensor,
    llama_residual_logits: Tensor,
    legal_mask: Tensor,
) -> Tensor:
    """Return a normalized unit-weight product-of-experts distribution.

    Both experts are first normalized on the exact same legal support.  Their
    log probabilities are added and normalized once more.  Therefore an
    all-zero Llama residual is uniform on the support and the returned
    distribution is exactly the normalized C3FD distribution (up to floating
    point roundoff).
    """

    if c3fd_calibrated_logits.shape != llama_residual_logits.shape:
        raise ValueError("C3FD and Llama logits must have identical shapes")
    c3fd = masked_log_softmax(
        c3fd_calibrated_logits,
        legal_mask,
        name="C3FD calibrated logits",
    )
    residual = masked_log_softmax(
        llama_residual_logits,
        legal_mask,
        name="Llama residual logits",
    )
    return masked_log_softmax(c3fd + residual, legal_mask, name="PoE logits")


def _require_integer_targets(targets: Tensor, *, name: str) -> None:
    if not isinstance(targets, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if targets.dtype is torch.bool or targets.is_floating_point() or targets.is_complex():
        raise TypeError(f"{name} must use an integer dtype")


def _selected_nll(
    log_probs: Tensor,
    targets: Tensor,
    legal_mask: Tensor,
    *,
    name: str,
) -> Tensor:
    _validate_logits_and_mask(log_probs, legal_mask, name=name)
    _require_integer_targets(targets, name=f"{name} targets")
    if targets.shape != log_probs.shape[:-1]:
        raise ValueError(f"{name} targets must match the non-class dimensions")
    target = targets.to(device=log_probs.device, dtype=torch.long)
    invalid = (target < 0) | (target >= log_probs.shape[-1])
    if bool(invalid.any().item()):
        raise ValueError(f"{name} target is outside the configured class range")
    selected_legal = legal_mask.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    if not bool(selected_legal.all().item()):
        raise ValueError(f"{name} teacher target is illegal under its supplied mask")
    selected = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(selected).all().item()):
        raise ValueError(f"{name} teacher target does not have finite log probability")
    return -selected


def row_balanced_typed_loss(
    *,
    proposal_log_probs: Tensor,
    proposal_targets: Tensor,
    proposal_legal_mask: Tensor,
    action_log_probs: Tensor,
    action_targets: Tensor,
    action_legal_mask: Tensor,
    soft_field_log_probs: Mapping[str, Tensor],
    soft_field_targets: Mapping[str, Tensor],
    soft_field_legal_masks: Mapping[str, Tensor],
    ignore_index: int = -100,
) -> Tensor:
    """Compute ``(proposal + row-mean actions + mean soft fields) / 3``.

    Action tokens are averaged inside each source row before rows are averaged,
    so high-arity formulas do not receive greater optimization weight.  The
    ignore index is allowed only for unsupervised action positions; every row
    must contain at least one supervised legal action.
    """

    if proposal_log_probs.ndim != 2:
        raise ValueError("proposal_log_probs must have shape [batch, strata]")
    proposal_nll = _selected_nll(
        proposal_log_probs,
        proposal_targets,
        proposal_legal_mask,
        name="proposal",
    ).mean()

    if action_log_probs.ndim != 3:
        raise ValueError("action_log_probs must have shape [batch, sequence, actions]")
    _validate_logits_and_mask(action_log_probs, action_legal_mask, name="actions")
    _require_integer_targets(action_targets, name="action targets")
    if action_targets.shape != action_log_probs.shape[:-1]:
        raise ValueError("action targets must have shape [batch, sequence]")
    action_targets = action_targets.to(device=action_log_probs.device, dtype=torch.long)
    supervised = action_targets != int(ignore_index)
    if bool((~supervised.any(dim=1)).any().item()):
        raise ValueError("every row must contain at least one supervised action")
    active_targets = action_targets[supervised]
    invalid = (active_targets < 0) | (active_targets >= action_log_probs.shape[-1])
    if bool(invalid.any().item()):
        raise ValueError("action teacher target is outside the configured action space")
    active_legal = action_legal_mask[supervised]
    selected_legal = active_legal.gather(-1, active_targets.unsqueeze(-1)).squeeze(-1)
    if not bool(selected_legal.all().item()):
        raise ValueError("action teacher target is illegal under its supplied mask")
    active_log_probs = action_log_probs[supervised]
    selected = active_log_probs.gather(-1, active_targets.unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(selected).all().item()):
        raise ValueError("action teacher target does not have finite log probability")
    token_nll = action_log_probs.new_zeros(action_targets.shape)
    token_nll[supervised] = -selected
    per_row_actions = token_nll.sum(dim=1) / supervised.sum(dim=1).to(token_nll.dtype)
    action_nll = per_row_actions.mean()

    expected_fields = set(SOFT_FIELDS)
    if set(soft_field_log_probs) != expected_fields:
        raise ValueError("soft_field_log_probs must contain exactly the three typed fields")
    if set(soft_field_targets) != expected_fields:
        raise ValueError("soft_field_targets must contain exactly the three typed fields")
    if set(soft_field_legal_masks) != expected_fields:
        raise ValueError("soft_field_legal_masks must contain exactly the three typed fields")
    soft_losses = []
    for field in SOFT_FIELDS:
        field_probs = soft_field_log_probs[field]
        if field_probs.ndim != 2:
            raise ValueError(f"{field} log probabilities must have shape [batch, classes]")
        if field_probs.shape[0] != proposal_log_probs.shape[0]:
            raise ValueError(f"{field} batch size does not match proposal batch size")
        soft_losses.append(
            _selected_nll(
                field_probs,
                soft_field_targets[field],
                soft_field_legal_masks[field],
                name=field,
            ).mean()
        )
    soft_nll = torch.stack(soft_losses).mean()
    return (proposal_nll + action_nll + soft_nll) / 3.0


class C3FDLlamaTypedResidualPlanner(nn.Module):
    """Typed input adapter and zero-initialized residual output heads."""

    def __init__(self, config: C3FDLlamaTypedPlannerConfig) -> None:
        super().__init__()
        self.config = config
        typed = int(config.typed_embedding_size)
        hidden = int(config.llama_hidden_size)

        self.stability_goal_embedding = nn.Embedding(
            int(config.num_stability_goals), typed
        )
        self.proposal_state_embedding = nn.Embedding(
            int(config.num_proposal_states), typed
        )
        self.previous_species_embedding = nn.Embedding(
            int(config.num_species) + 2, typed
        )
        self.previous_count_embedding = nn.Embedding(
            int(config.max_count) + 1, typed, padding_idx=0
        )
        self.ledger_projection = nn.Linear(
            int(config.ledger_feature_size), typed, bias=False
        )
        self.typed_norm = nn.LayerNorm(5 * typed)
        self.typed_projector = nn.Linear(5 * typed, hidden)

        self.proposal_head = nn.Linear(hidden, int(config.num_proposal_strata))
        self.action_head = nn.Linear(hidden, int(config.num_joint_actions))
        self.soft_field_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden, classes)
                for name, classes in config.soft_head_dims.items()
            }
        )
        self._zero_residual_outputs()

    def _zero_residual_outputs(self) -> None:
        for head in (self.proposal_head, self.action_head, *self.soft_field_heads.values()):
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    @staticmethod
    def _require_id_range(ids: Tensor, *, minimum: int, maximum: int, name: str) -> None:
        _require_integer_targets(ids, name=name)
        if bool(((ids < minimum) | (ids > maximum)).any().item()):
            raise ValueError(f"{name} is outside {minimum}..{maximum}")

    def typed_inputs_embeds(
        self,
        *,
        stability_goal_ids: Tensor,
        proposal_state_ids: Tensor,
        previous_species_indices: Tensor,
        previous_count_values: Tensor,
        ledger_features: Tensor,
    ) -> Tensor:
        """Build ``[batch, sequence, llama_hidden_size]`` typed inputs embeds."""

        if proposal_state_ids.ndim != 2:
            raise ValueError("proposal_state_ids must have shape [batch, sequence]")
        batch, sequence = proposal_state_ids.shape
        if batch <= 0 or sequence <= 0:
            raise ValueError("typed input batch and sequence dimensions must be non-empty")
        if sequence > int(self.config.max_sequence_length):
            raise ValueError("typed input exceeds max_sequence_length")
        expected_sequence = (batch, sequence)
        if previous_species_indices.shape != expected_sequence:
            raise ValueError("previous_species_indices must match proposal state shape")
        if previous_count_values.shape != expected_sequence:
            raise ValueError("previous_count_values must match proposal state shape")
        if stability_goal_ids.shape != (batch,):
            raise ValueError("stability_goal_ids must have shape [batch]")
        expected_ledger = (batch, sequence, int(self.config.ledger_feature_size))
        if ledger_features.shape != expected_ledger:
            raise ValueError(
                "ledger_features must have shape [batch, sequence, ledger_feature_size]"
            )
        if not ledger_features.is_floating_point():
            raise TypeError("ledger_features must be floating point")
        if not bool(torch.isfinite(ledger_features).all().item()):
            raise ValueError("ledger_features must be finite")
        if bool(
            (ledger_features.abs() > float(self.config.ledger_abs_bound) + 1e-6)
            .any()
            .item()
        ):
            raise ValueError("ledger_features exceed the configured normalized bound")

        self._require_id_range(
            stability_goal_ids,
            minimum=0,
            maximum=int(self.config.num_stability_goals) - 1,
            name="stability_goal_ids",
        )
        self._require_id_range(
            proposal_state_ids,
            minimum=0,
            maximum=int(self.config.num_proposal_states) - 1,
            name="proposal_state_ids",
        )
        self._require_id_range(
            previous_species_indices,
            minimum=-1,
            maximum=int(self.config.eos_species_index),
            name="previous_species_indices",
        )
        self._require_id_range(
            previous_count_values,
            minimum=0,
            maximum=int(self.config.max_count),
            name="previous_count_values",
        )

        species = previous_species_indices.to(dtype=torch.long)
        counts = previous_count_values.to(device=species.device, dtype=torch.long)
        sentinel = species == -1
        eos = species == int(self.config.eos_species_index)
        real = ~(sentinel | eos)
        if bool(((sentinel | eos) & (counts != 0)).any().item()):
            raise ValueError("previous sentinel and EOS actions require count zero")
        if bool((real & (counts == 0)).any().item()):
            raise ValueError("previous real-species actions require a positive count")

        device = self.typed_projector.weight.device
        stability_goal_ids = stability_goal_ids.to(device=device, dtype=torch.long)
        proposal_state_ids = proposal_state_ids.to(device=device, dtype=torch.long)
        species = species.to(device=device)
        counts = counts.to(device=device)
        sentinel = sentinel.to(device=device)
        ledger_features = ledger_features.to(
            device=device, dtype=self.ledger_projection.weight.dtype
        )
        safe_species = species.masked_fill(
            sentinel, int(self.config.sentinel_species_index)
        )
        goal = self.stability_goal_embedding(stability_goal_ids).unsqueeze(1)
        goal = goal.expand(-1, sequence, -1)
        typed_parts = (
            goal,
            self.proposal_state_embedding(proposal_state_ids),
            self.previous_species_embedding(safe_species),
            self.previous_count_embedding(counts),
            self.ledger_projection(ledger_features),
        )
        typed = torch.cat(typed_parts, dim=-1)
        return self.typed_projector(self.typed_norm(typed))

    def forward(
        self,
        llama_hidden_states: Tensor,
        *,
        soft_position_indices: Tensor,
    ) -> TypedResidualLogits:
        """Project caller-owned Llama hidden states into typed residual logits.

        Proposal logits use the initial query state.  Structural soft fields
        must use the terminal composition state supplied by the caller; using
        the initial state would sever their dependence on sampled composition.
        """

        if not isinstance(llama_hidden_states, Tensor) or not llama_hidden_states.is_floating_point():
            raise TypeError("llama_hidden_states must be a floating-point tensor")
        if llama_hidden_states.ndim != 3:
            raise ValueError("llama_hidden_states must have shape [batch, sequence, hidden]")
        if llama_hidden_states.shape[0] <= 0 or llama_hidden_states.shape[1] <= 0:
            raise ValueError("llama_hidden_states batch and sequence must be non-empty")
        if llama_hidden_states.shape[1] > int(self.config.max_sequence_length):
            raise ValueError("llama_hidden_states exceed max_sequence_length")
        if llama_hidden_states.shape[-1] != int(self.config.llama_hidden_size):
            raise ValueError("llama_hidden_states have the wrong hidden dimension")
        if not bool(torch.isfinite(llama_hidden_states).all().item()):
            raise ValueError("llama_hidden_states must be finite")

        _require_integer_targets(
            soft_position_indices, name="soft_position_indices"
        )
        if soft_position_indices.shape != (llama_hidden_states.shape[0],):
            raise ValueError("soft_position_indices must have shape [batch]")
        positions = soft_position_indices.to(
            device=llama_hidden_states.device, dtype=torch.long
        )
        if bool(
            ((positions < 0) | (positions >= llama_hidden_states.shape[1]))
            .any()
            .item()
        ):
            raise ValueError("soft_position_indices are outside the sequence")

        row_state = llama_hidden_states[:, 0, :]
        terminal_state = llama_hidden_states[
            torch.arange(llama_hidden_states.shape[0], device=positions.device),
            positions,
        ]
        return TypedResidualLogits(
            proposal=self.proposal_head(row_state),
            actions=self.action_head(llama_hidden_states),
            soft_fields={
                name: head(terminal_state)
                for name, head in self.soft_field_heads.items()
            },
        )


__all__ = [
    "C3FDLlamaTypedPlannerConfig",
    "C3FDLlamaTypedResidualPlanner",
    "SOFT_FIELDS",
    "TypedResidualLogits",
    "decode_joint_action",
    "joint_action_index",
    "masked_log_softmax",
    "row_balanced_typed_loss",
    "unit_weight_poe_log_probs",
]
