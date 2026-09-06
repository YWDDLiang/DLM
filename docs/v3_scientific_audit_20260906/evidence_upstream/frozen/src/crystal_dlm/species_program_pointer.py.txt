"""Geometry-supervised Llama pointer for SPAD species programs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.periodic_geometry_ops import element_radius


def maximum_contact_tree_order(
    species: Sequence[str],
    distance_matrix: Sequence[Sequence[float]],
) -> tuple[str, ...]:
    """Build an MP20-only species order from the periodic contact graph.

    The root maximizes average weighted contact degree.  Each later species
    maximizes affinity to the already selected scaffold.  Ties use atomic
    number, making the teacher deterministic without energy or hull labels.
    """

    symbols = tuple(str(value) for value in species)
    if not symbols:
        raise ValueError("contact teacher requires at least one site")
    if any(symbol not in SYMBOL_TO_Z for symbol in symbols):
        raise ValueError("contact teacher received an unsupported element")
    size = len(symbols)
    if len(distance_matrix) != size or any(
        len(row) != size for row in distance_matrix
    ):
        raise ValueError("distance matrix must have shape [N, N]")
    elements = tuple(sorted(set(symbols), key=lambda value: SYMBOL_TO_Z[value]))
    counts = {symbol: symbols.count(symbol) for symbol in elements}
    affinity = {
        left: {right: 0.0 for right in elements} for left in elements
    }
    degree = {symbol: 0.0 for symbol in elements}
    for left in range(size):
        for right in range(left + 1, size):
            distance = float(distance_matrix[left][right])
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError("distance matrix must be finite and nonnegative")
            left_symbol = symbols[left]
            right_symbol = symbols[right]
            radius = element_radius(SYMBOL_TO_Z[left_symbol]) + element_radius(
                SYMBOL_TO_Z[right_symbol]
            )
            weight = math.exp(-((distance / max(radius, 1.0e-6)) ** 2))
            degree[left_symbol] += weight
            degree[right_symbol] += weight
            affinity[left_symbol][right_symbol] += weight
            affinity[right_symbol][left_symbol] += weight

    def tie_key(symbol: str) -> tuple[float, int]:
        return -degree[symbol] / float(counts[symbol]), int(SYMBOL_TO_Z[symbol])

    selected = [min(elements, key=tie_key)]
    while len(selected) < len(elements):
        remaining = [symbol for symbol in elements if symbol not in selected]

        def expansion_key(symbol: str) -> tuple[float, float, int]:
            contact = sum(affinity[symbol][prior] for prior in selected)
            normalized = contact / math.sqrt(
                float(counts[symbol] * sum(counts[prior] for prior in selected))
            )
            return (
                -normalized,
                -degree[symbol] / float(counts[symbol]),
                int(SYMBOL_TO_Z[symbol]),
            )

        selected.append(min(remaining, key=expansion_key))
    return tuple(selected)


try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover - CPU-only documentation hosts
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SpeciesPointerConfig:
    llama_hidden_size: int
    pointer_size: int = 256
    max_elements: int = 7
    max_count: int = 20
    num_lattice_systems: int = 8
    num_spacegroup_buckets: int = 8
    num_volume_per_atom_bins: int = 16

    def __post_init__(self) -> None:
        if int(self.llama_hidden_size) <= 0 or int(self.pointer_size) <= 0:
            raise ValueError("pointer hidden sizes must be positive")
        if not 1 <= int(self.max_elements) <= 20:
            raise ValueError("max_elements must be in 1..20")
        if not 1 <= int(self.max_count) <= 20:
            raise ValueError("max_count must be in 1..20")
        for value in (
            self.num_lattice_systems,
            self.num_spacegroup_buckets,
            self.num_volume_per_atom_bins,
        ):
            if int(value) <= 0:
                raise ValueError("soft-field vocabulary sizes must be positive")


if nn is not None:

    class PlanConditionedSpeciesPointer(nn.Module):
        """Decode a masked permutation of final Plan elements.

        The C3FD certificate fixes the candidate set.  This head can only
        permute those elements, so it cannot alter composition validity.
        """

        def __init__(self, config: SpeciesPointerConfig) -> None:
            super().__init__()
            self.config = config
            width = int(config.pointer_size)
            self.terminal_projection = nn.Linear(
                int(config.llama_hidden_size), width
            )
            self.element_embedding = nn.Embedding(119, width, padding_idx=0)
            self.count_embedding = nn.Embedding(
                int(config.max_count) + 1, width, padding_idx=0
            )
            self.soft_embeddings = nn.ModuleList(
                [
                    nn.Embedding(int(config.num_lattice_systems), width),
                    nn.Embedding(int(config.num_spacegroup_buckets), width),
                    nn.Embedding(int(config.num_volume_per_atom_bins), width),
                ]
            )
            self.step_embedding = nn.Embedding(int(config.max_elements), width)
            self.selected_projection = nn.Linear(width, width, bias=False)
            self.candidate_projection = nn.Linear(width, width, bias=False)
            self.query_projection = nn.Linear(width, width, bias=False)
            self.score = nn.Linear(width, 1, bias=False)

        def _validate(
            self,
            terminal_hidden: Tensor,
            atomic_numbers: Tensor,
            counts: Tensor,
            valid_mask: Tensor,
            soft_field_ids: Tensor,
        ) -> None:
            if terminal_hidden.ndim != 2 or terminal_hidden.shape[-1] != int(
                self.config.llama_hidden_size
            ):
                raise ValueError("terminal_hidden has the wrong shape")
            expected = atomic_numbers.shape
            if len(expected) != 2 or expected[0] != terminal_hidden.shape[0]:
                raise ValueError("atomic_numbers must have shape [batch, elements]")
            if counts.shape != expected or valid_mask.shape != expected:
                raise ValueError("count/mask shapes must match atomic_numbers")
            if valid_mask.dtype is not torch.bool:
                raise TypeError("valid_mask must be boolean")
            if soft_field_ids.shape != (terminal_hidden.shape[0], 3):
                raise ValueError("soft_field_ids must have shape [batch, 3]")
            if bool(((atomic_numbers[valid_mask] < 1) | (atomic_numbers[valid_mask] > 118)).any()):
                raise ValueError("active atomic numbers must be in 1..118")
            if bool(((counts[valid_mask] < 1) | (counts[valid_mask] > int(self.config.max_count))).any()):
                raise ValueError("active counts are outside the configured range")
            if bool((valid_mask.sum(dim=1) < 1).any()):
                raise ValueError("every Plan must contain at least one element")
            if atomic_numbers.shape[1] > int(self.config.max_elements):
                raise ValueError("Plan exceeds max_elements")

        def permutation_logits(
            self,
            terminal_hidden: Tensor,
            atomic_numbers: Tensor,
            counts: Tensor,
            valid_mask: Tensor,
            soft_field_ids: Tensor,
            *,
            teacher_order: Tensor | None = None,
        ) -> Tensor:
            self._validate(
                terminal_hidden,
                atomic_numbers,
                counts,
                valid_mask,
                soft_field_ids,
            )
            device = terminal_hidden.device
            atomic_numbers = atomic_numbers.to(device=device, dtype=torch.long)
            counts = counts.to(device=device, dtype=torch.long)
            valid_mask = valid_mask.to(device=device)
            soft_field_ids = soft_field_ids.to(device=device, dtype=torch.long)
            if teacher_order is not None:
                teacher_order = teacher_order.to(device=device, dtype=torch.long)
                if teacher_order.shape != atomic_numbers.shape:
                    raise ValueError("teacher_order must match the padded element shape")

            candidate = self.element_embedding(atomic_numbers) + self.count_embedding(counts)
            context = self.terminal_projection(terminal_hidden)
            for column, embedding in enumerate(self.soft_embeddings):
                context = context + embedding(soft_field_ids[:, column])
            selected = torch.zeros_like(valid_mask)
            selected_sum = torch.zeros_like(context)
            outputs: list[Tensor] = []
            active_counts = valid_mask.sum(dim=1)
            for step in range(atomic_numbers.shape[1]):
                query = (
                    context
                    + self.selected_projection(selected_sum / max(1, step))
                    + self.step_embedding(
                        torch.full(
                            (atomic_numbers.shape[0],),
                            step,
                            dtype=torch.long,
                            device=device,
                        )
                    )
                )
                logits = self.score(
                    torch.tanh(
                        self.candidate_projection(candidate)
                        + self.query_projection(query).unsqueeze(1)
                    )
                ).squeeze(-1)
                available = valid_mask & ~selected
                active_rows = active_counts > step
                logits = logits.masked_fill(~available, -torch.inf)
                # Padded rows are excluded from loss; keep one finite value so
                # their log-softmax remains numerically defined.
                if bool((~active_rows).any()):
                    logits = logits.clone()
                    logits[~active_rows] = -torch.inf
                    logits[~active_rows, 0] = 0.0
                outputs.append(logits)
                choice = (
                    teacher_order[:, step]
                    if teacher_order is not None
                    else torch.argmax(logits, dim=-1)
                )
                if bool(
                    (active_rows & ~available.gather(1, choice.unsqueeze(1)).squeeze(1)).any()
                ):
                    raise ValueError("permutation prefix selects an unavailable element")
                safe_choice = torch.where(active_rows, choice, torch.zeros_like(choice))
                chosen_state = candidate.gather(
                    1,
                    safe_choice.reshape(-1, 1, 1).expand(-1, 1, candidate.shape[-1]),
                ).squeeze(1)
                selected_sum = selected_sum + chosen_state * active_rows.unsqueeze(1)
                selected.scatter_(1, safe_choice.unsqueeze(1), active_rows.unsqueeze(1))
            return torch.stack(outputs, dim=1)

        def decode(
            self,
            terminal_hidden: Tensor,
            atomic_numbers: Tensor,
            counts: Tensor,
            valid_mask: Tensor,
            soft_field_ids: Tensor,
        ) -> Tensor:
            logits = self.permutation_logits(
                terminal_hidden,
                atomic_numbers,
                counts,
                valid_mask,
                soft_field_ids,
            )
            return torch.argmax(logits, dim=-1)


    def species_pointer_loss(
        logits: Tensor,
        teacher_order: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if logits.ndim != 3 or logits.shape[:2] != teacher_order.shape:
            raise ValueError("pointer logits/targets have incompatible shapes")
        if valid_mask.shape != teacher_order.shape or valid_mask.dtype is not torch.bool:
            raise ValueError("pointer valid mask has the wrong shape or dtype")
        active_steps = (
            torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
            < valid_mask.sum(dim=1, keepdim=True)
        )
        selected = logits.gather(
            -1, teacher_order.to(logits.device).unsqueeze(-1)
        ).squeeze(-1)
        nll = -torch.log_softmax(logits, dim=-1).gather(
            -1, teacher_order.to(logits.device).unsqueeze(-1)
        ).squeeze(-1)
        if not bool(torch.isfinite(selected[active_steps]).all()):
            raise ValueError("teacher order is unavailable under pointer masking")
        return nll[active_steps].mean()


else:  # pragma: no cover
    PlanConditionedSpeciesPointer = None  # type: ignore[assignment]
    species_pointer_loss = None  # type: ignore[assignment]


__all__ = [
    "PlanConditionedSpeciesPointer",
    "SpeciesPointerConfig",
    "maximum_contact_tree_order",
    "species_pointer_loss",
]
