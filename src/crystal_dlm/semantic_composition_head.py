"""Typed semantic composition head for a shared language-model backbone.

The head operates on semantic actions rather than formula text tokens.  A
caller may encode a natural-language prompt with any backbone, then use this
module to predict atom count, species/EOS, count, and optional rich-Plan soft
fields.  Sampling policy, stability objectives, RL, and reranking deliberately
live outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SemanticHeadFlags:
    """Independent ablation switches shared by T0/T1/C3 experiment arms."""

    use_physics: bool = False
    use_pair_prior: bool = False
    use_hard_mask: bool = False


@dataclass
class SemanticCompositionOutput:
    """Typed logits and optional decomposed teacher-forcing losses."""

    n_logits: Tensor
    species_logits: Tensor
    count_logits: Tensor
    rich_logits: dict[str, Tensor]
    loss: Tensor | None = None
    losses: dict[str, Tensor] = field(default_factory=dict)


class SemanticCompositionHead(nn.Module):
    """One-backbone semantic head for composition actions.

    Semantic conventions
    --------------------
    * N targets and count targets use their physical values (1..20 by
      default), not zero-based class IDs.
    * Species IDs are zero-based.  ``num_species`` is the explicit EOS species
      ID and has no count.
    * Teacher-forced inputs are *previous*, already-shifted actions.  ``-1``
      with count ``0`` denotes the position before the first semantic action;
      it is a module sentinel, not a generated token.
    * A joint action index is ``species * max_count + (count - 1)``.  The last
      joint index is EOS.

    ``physics_features`` is a caller-owned, fixed ``[num_species, P]`` matrix.
    It is registered as a non-trainable buffer; only its projection is learned.
    """

    def __init__(
        self,
        hidden_size: int,
        num_species: int,
        *,
        max_atoms: int = 20,
        max_count: int = 20,
        physics_features: Tensor | None = None,
        rich_soft_head_dims: Mapping[str, int] | None = None,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        decoder_dropout: float = 0.0,
        max_sequence_length: int = 16,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if int(num_species) <= 0:
            raise ValueError("num_species must be positive")
        if int(max_atoms) <= 0 or int(max_count) <= 0:
            raise ValueError("max_atoms and max_count must be positive")

        self.hidden_size = int(hidden_size)
        self.num_species = int(num_species)
        self.max_atoms = int(max_atoms)
        self.max_count = int(max_count)
        self.ignore_index = int(ignore_index)
        self.max_sequence_length = int(max_sequence_length)
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if int(decoder_layers) <= 0 or int(decoder_heads) <= 0:
            raise ValueError("decoder_layers and decoder_heads must be positive")
        if self.hidden_size % int(decoder_heads) != 0:
            raise ValueError("hidden_size must be divisible by decoder_heads")
        self.eos_species_index = self.num_species
        self.eos_action_index = self.num_species * self.max_count
        self.num_joint_actions = self.eos_action_index + 1

        # Row zero in count_embedding is the non-generated no-count value used
        # only for EOS and the pre-action sentinel.  Rows 1..max_count are the
        # learned semantic count embeddings.
        self.species_embedding = nn.Embedding(self.num_species + 1, self.hidden_size)
        self.n_embedding = nn.Embedding(
            self.max_atoms + 1,
            self.hidden_size,
            padding_idx=0,
        )
        self.count_embedding = nn.Embedding(
            self.max_count + 1,
            self.hidden_size,
            padding_idx=0,
        )

        if physics_features is None:
            fixed_physics = None
            self.physics_projection: nn.Linear | None = None
        else:
            fixed_physics = torch.as_tensor(
                physics_features,
                dtype=torch.get_default_dtype(),
            ).detach().clone()
            if fixed_physics.ndim != 2:
                raise ValueError("physics_features must have shape [num_species, P]")
            if fixed_physics.shape[0] != self.num_species:
                raise ValueError(
                    "physics_features first dimension must equal num_species"
                )
            if fixed_physics.shape[1] <= 0:
                raise ValueError("physics_features must contain at least one feature")
            self.physics_projection = nn.Linear(
                int(fixed_physics.shape[1]),
                self.hidden_size,
                bias=False,
            )
        self.register_buffer("physics_features", fixed_physics, persistent=True)

        self.context_norm = nn.LayerNorm(self.hidden_size)
        self.position_embedding = nn.Embedding(
            self.max_sequence_length,
            self.hidden_size,
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(decoder_heads),
            dim_feedforward=4 * self.hidden_size,
            dropout=float(decoder_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.semantic_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=int(decoder_layers),
        )
        self.n_head = nn.Linear(self.hidden_size, self.max_atoms)
        self.species_bias = nn.Parameter(torch.zeros(self.num_species + 1))
        self.count_bias = nn.Parameter(torch.zeros(self.max_count))

        rich_dims = dict(rich_soft_head_dims or {})
        for name, size in rich_dims.items():
            if not name or int(size) <= 0:
                raise ValueError("rich soft heads need non-empty names and positive sizes")
        self.rich_soft_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.hidden_size, int(size))
                for name, size in rich_dims.items()
            }
        )

    @staticmethod
    def _flags(flags: SemanticHeadFlags | None) -> SemanticHeadFlags:
        return flags if flags is not None else SemanticHeadFlags()

    def species_representations(self, *, use_physics: bool = False) -> Tensor:
        """Return learned species/EOS prototypes with an optional physics term."""

        representations = self.species_embedding.weight
        if not use_physics:
            return representations
        if self.physics_features is None or self.physics_projection is None:
            raise ValueError("use_physics=True requires a fixed physics feature matrix")
        projected = self.physics_projection(self.physics_features)
        eos_zeros = projected.new_zeros((1, self.hidden_size))
        return representations + torch.cat((projected, eos_zeros), dim=0)

    def embed_semantic_actions(
        self,
        species_indices: Tensor,
        count_values: Tensor,
        *,
        n_values: Tensor | None = None,
        flags: SemanticHeadFlags | None = None,
    ) -> Tensor:
        """Embed shifted semantic actions as species + count + physics.

        Real species require count values in ``1..max_count``.  EOS and the
        pre-action sentinel ``-1`` require count zero.  Missing/sentinel
        positions produce an all-zero embedding.
        """

        if species_indices.shape != count_values.shape:
            raise ValueError("species_indices and count_values must have equal shape")
        species = species_indices.to(dtype=torch.long)
        counts = count_values.to(device=species.device, dtype=torch.long)
        n_actions = (
            torch.zeros_like(species)
            if n_values is None
            else n_values.to(device=species.device, dtype=torch.long)
        )
        if n_actions.shape != species.shape:
            raise ValueError("n_values must match semantic action shape")
        if bool(((n_actions < 0) | (n_actions > self.max_atoms)).any().item()):
            raise ValueError(f"N action outside 0..{self.max_atoms}")

        bad_species = (species < -1) | (species > self.eos_species_index)
        if bool(bad_species.any().item()):
            raise ValueError("species index outside -1..EOS")
        bad_count = (counts < 0) | (counts > self.max_count)
        if bool(bad_count.any().item()):
            raise ValueError(f"count value outside 0..{self.max_count}")

        missing = species == -1
        eos = species == self.eos_species_index
        real = ~(missing | eos)
        if bool(((missing | eos) & (counts != 0)).any().item()):
            raise ValueError("EOS and pre-action sentinel require count zero")
        if bool((real & (counts == 0)).any().item()):
            raise ValueError("real species require a positive count")
        has_n = n_actions > 0
        if bool((has_n & ~missing).any().item()):
            raise ValueError("N action requires the species sentinel")
        if bool((has_n & (counts != 0)).any().item()):
            raise ValueError("N action requires count zero")
        if bool((~missing & (n_actions != 0)).any().item()):
            raise ValueError("species/EOS actions cannot carry N")

        safe_species = species.clamp_min(0)
        active_flags = self._flags(flags)
        species_table = self.species_representations(
            use_physics=active_flags.use_physics
        )
        embedded = F.embedding(safe_species, species_table)
        embedded = embedded + self.count_embedding(counts) + self.n_embedding(n_actions)
        # A species=-1 row is all-zero only when it is the pre-action sentinel;
        # a positive n_values entry is the explicit locked-N semantic action.
        pre_action = missing & ~has_n
        return embedded.masked_fill(pre_action.unsqueeze(-1), 0.0)

    def _contextualize(
        self,
        hidden_states: Tensor,
        previous_species_indices: Tensor | None,
        previous_count_values: Tensor | None,
        previous_n_values: Tensor | None,
        flags: SemanticHeadFlags,
    ) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states last dimension must be {self.hidden_size}"
            )
        if hidden_states.shape[1] == 0:
            raise ValueError("semantic sequence cannot be empty")
        if hidden_states.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"semantic sequence exceeds max_sequence_length={self.max_sequence_length}"
            )
        if (previous_species_indices is None) != (previous_count_values is None):
            raise ValueError(
                "previous_species_indices and previous_count_values must be provided together"
            )
        if previous_species_indices is None and previous_n_values is not None:
            raise ValueError("previous_n_values requires semantic action inputs")

        context = hidden_states
        if previous_species_indices is not None and previous_count_values is not None:
            if previous_species_indices.shape != hidden_states.shape[:-1]:
                raise ValueError("teacher-forced action indices must match [batch, sequence]")
            action_embedding = self.embed_semantic_actions(
                previous_species_indices.to(device=hidden_states.device),
                previous_count_values.to(device=hidden_states.device),
                n_values=previous_n_values,
                flags=flags,
            )
            context = context + action_embedding.to(dtype=context.dtype)
        positions = torch.arange(
            context.shape[1], device=context.device, dtype=torch.long
        )
        context = context + self.position_embedding(positions).to(dtype=context.dtype)
        causal_mask = torch.triu(
            torch.ones(
                context.shape[1],
                context.shape[1],
                device=context.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        context = self.semantic_decoder(context, mask=causal_mask)
        return self.context_norm(context)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        previous_species_indices: Tensor | None = None,
        previous_count_values: Tensor | None = None,
        previous_n_values: Tensor | None = None,
        n_targets: Tensor | None = None,
        species_targets: Tensor | None = None,
        count_targets: Tensor | None = None,
        rich_targets: Mapping[str, Tensor] | None = None,
        flags: SemanticHeadFlags | None = None,
        loss_weights: Mapping[str, float] | None = None,
    ) -> SemanticCompositionOutput:
        """Compute typed logits and optional teacher-forced cross-entropy losses."""

        active_flags = self._flags(flags)
        context = self._contextualize(
            hidden_states,
            previous_species_indices,
            previous_count_values,
            previous_n_values,
            active_flags,
        )
        scale = sqrt(float(self.hidden_size))
        species_table = self.species_representations(
            use_physics=active_flags.use_physics
        )
        count_table = self.count_embedding.weight[1 : self.max_count + 1]

        n_logits = self.n_head(context[:, 0, :])
        species_logits = (
            torch.einsum("bth,sh->bts", context, species_table) / scale
            + self.species_bias
        )
        count_logits = (
            torch.einsum("bth,ch->btc", context, count_table) / scale
            + self.count_bias
        )
        rich_logits = {
            name: head(context) for name, head in self.rich_soft_heads.items()
        }

        component_losses: dict[str, Tensor] = {}
        if n_targets is not None:
            n_classes = self._one_based_targets(
                n_targets,
                maximum=self.max_atoms,
                name="N",
                device=n_logits.device,
            )
            component_losses["n"] = F.cross_entropy(n_logits, n_classes)

        normalized_species: Tensor | None = None
        if species_targets is not None:
            self._require_sequence_shape(species_targets, species_logits, "species_targets")
            normalized_species = species_targets.to(
                device=species_logits.device,
                dtype=torch.long,
            )
            invalid = (normalized_species != self.ignore_index) & (
                (normalized_species < 0)
                | (normalized_species > self.eos_species_index)
            )
            if bool(invalid.any().item()):
                raise ValueError("species target outside 0..EOS")
            species_loss = self._sequence_cross_entropy(
                species_logits,
                normalized_species,
            )
            if species_loss is not None:
                component_losses["species"] = species_loss

        if count_targets is not None:
            self._require_sequence_shape(count_targets, count_logits, "count_targets")
            raw_counts = count_targets.to(device=count_logits.device, dtype=torch.long)
            valid_count = (raw_counts != self.ignore_index) & (raw_counts != 0)
            invalid_count = valid_count & (
                (raw_counts < 1) | (raw_counts > self.max_count)
            )
            if bool(invalid_count.any().item()):
                raise ValueError(f"count target outside 1..{self.max_count}")
            if normalized_species is not None:
                valid_count &= (
                    (normalized_species != self.ignore_index)
                    & (normalized_species != self.eos_species_index)
                )
            count_classes = torch.full_like(raw_counts, self.ignore_index)
            count_classes[valid_count] = raw_counts[valid_count] - 1
            count_loss = self._sequence_cross_entropy(count_logits, count_classes)
            if count_loss is not None:
                component_losses["count"] = count_loss

        for name, target in dict(rich_targets or {}).items():
            if name not in rich_logits:
                raise KeyError(f"unknown rich soft head {name!r}")
            logits = rich_logits[name]
            self._require_sequence_shape(target, logits, f"rich_targets[{name!r}]")
            classes = target.to(device=logits.device, dtype=torch.long)
            invalid = (classes != self.ignore_index) & (
                (classes < 0) | (classes >= logits.shape[-1])
            )
            if bool(invalid.any().item()):
                raise ValueError(f"target outside rich head {name!r} class range")
            rich_loss = self._sequence_cross_entropy(logits, classes)
            if rich_loss is not None:
                component_losses[f"rich:{name}"] = rich_loss

        total_loss = self._weighted_total(component_losses, loss_weights)
        losses = dict(component_losses)
        if total_loss is not None:
            losses["total"] = total_loss
        return SemanticCompositionOutput(
            n_logits=n_logits,
            species_logits=species_logits,
            count_logits=count_logits,
            rich_logits=rich_logits,
            loss=total_loss,
            losses=losses,
        )

    def joint_action_scores(
        self,
        species_logits: Tensor,
        count_logits: Tensor,
        *,
        pair_prior_scores: Tensor | None = None,
        legal_action_mask: Tensor | None = None,
        flags: SemanticHeadFlags | None = None,
    ) -> Tensor:
        """Combine typed logits into flattened semantic action scores.

        Pair-prior scores apply only to real species/count actions and must be
        broadcastable to ``[..., num_species, max_count]``.  A trailing
        ``num_species`` vector is interpreted as a species-only prior and is
        shared across counts.  EOS receives no pair prior.

        When ``use_hard_mask`` is enabled, ``legal_action_mask`` must be
        broadcastable to the returned ``[..., num_joint_actions]`` tensor.
        Illegal actions are exactly ``-inf``; no fallback or repair is applied.
        """

        if species_logits.shape[:-1] != count_logits.shape[:-1]:
            raise ValueError("species_logits and count_logits prefixes must match")
        if species_logits.shape[-1] != self.num_species + 1:
            raise ValueError("species_logits last dimension must include species + EOS")
        if count_logits.shape[-1] != self.max_count:
            raise ValueError("count_logits last dimension must equal max_count")

        active_flags = self._flags(flags)
        real_scores = (
            species_logits[..., : self.num_species].unsqueeze(-1)
            + count_logits.unsqueeze(-2)
        )
        if active_flags.use_pair_prior:
            if pair_prior_scores is None:
                raise ValueError("use_pair_prior=True requires pair_prior_scores")
            prior = torch.as_tensor(
                pair_prior_scores,
                device=real_scores.device,
                dtype=real_scores.dtype,
            )
            if prior.ndim >= 2 and tuple(prior.shape[-2:]) == (
                self.num_species,
                self.max_count,
            ):
                pass
            elif prior.ndim >= 1 and prior.shape[-1] == self.num_species:
                prior = prior.unsqueeze(-1)
            try:
                prior = torch.broadcast_to(prior, real_scores.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "pair_prior_scores must broadcast to [..., num_species, max_count]"
                ) from exc
            real_scores = real_scores + prior

        flat_real_scores = real_scores.reshape(
            *real_scores.shape[:-2],
            self.num_species * self.max_count,
        )
        eos_scores = species_logits[..., self.eos_species_index].unsqueeze(-1)
        joint_scores = torch.cat((flat_real_scores, eos_scores), dim=-1)

        if active_flags.use_hard_mask:
            if legal_action_mask is None:
                raise ValueError("use_hard_mask=True requires legal_action_mask")
            legal = torch.as_tensor(
                legal_action_mask,
                device=joint_scores.device,
                dtype=torch.bool,
            )
            try:
                legal = torch.broadcast_to(legal, joint_scores.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "legal_action_mask must broadcast to joint action scores"
                ) from exc
            joint_scores = joint_scores.masked_fill(~legal, float("-inf"))
        return joint_scores

    def _one_based_targets(
        self,
        targets: Tensor,
        *,
        maximum: int,
        name: str,
        device: torch.device,
    ) -> Tensor:
        classes = targets.to(device=device, dtype=torch.long)
        if classes.ndim != 1:
            raise ValueError(f"{name} targets must have shape [batch]")
        if bool(((classes < 1) | (classes > int(maximum))).any().item()):
            raise ValueError(f"{name} target outside 1..{int(maximum)}")
        return classes - 1

    @staticmethod
    def _require_sequence_shape(target: Tensor, logits: Tensor, name: str) -> None:
        if tuple(target.shape) != tuple(logits.shape[:-1]):
            raise ValueError(f"{name} must match logits [batch, sequence]")

    def _sequence_cross_entropy(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor | None:
        valid = targets != self.ignore_index
        if not bool(valid.any().item()):
            return None
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
        )

    @staticmethod
    def _weighted_total(
        losses: Mapping[str, Tensor],
        weights: Mapping[str, float] | None,
    ) -> Tensor | None:
        if not losses:
            return None
        configured = dict(weights or {})
        weighted: list[Tensor] = []
        for name, loss in losses.items():
            weight = float(configured.get(name, 1.0))
            if not isfinite(weight) or weight < 0.0:
                raise ValueError(f"loss weight for {name!r} must be finite and nonnegative")
            weighted.append(loss * weight)
        total = weighted[0]
        for value in weighted[1:]:
            total = total + value
        return total


__all__ = [
    "SemanticCompositionHead",
    "SemanticCompositionOutput",
    "SemanticHeadFlags",
]
