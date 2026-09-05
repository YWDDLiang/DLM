"""Inference-time PMTR transform for native SPAD transactions.

The runtime contains no potential, force, stress, reward, relaxation, or
candidate-selection dependency.  It converts the current committed 7+4N token
state into periodic geometry, asks a small learned head for one transaction
repair, and transports logits only within the active numeric token family.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.manifold_geometry import (
    cartesian_to_fractional,
    lattice_to_metric,
    metric_to_lattice,
    spd_congruence_update,
    wrap_fractional,
)
from crystal_dlm.manifold_repair_head import ManifoldRepairOutput
from crystal_dlm.manifold_token_transport import (
    render_bracketed_token_residual,
    render_periodic_coordinate_token_residual,
)
from crystal_dlm.periodic_geometry_objective import build_geometry_token_support
from crystal_dlm.periodic_geometry_ops import minimum_image_vectors
from crystal_dlm.transaction_logits import TransactionContext, TransactionModelStep


_ELEMENT_RE = re.compile(r"^<E_([A-Z][a-z]?)>$")


@dataclass(frozen=True)
class PMTRRuntimeConfig:
    transport_gain: float = 6.0
    image_radius: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.transport_gain)) or float(self.transport_gain) < 0.0:
            raise ValueError("transport_gain must be finite and non-negative")
        if int(self.image_radius) not in (1, 2):
            raise ValueError("image_radius must be one or two")


@dataclass(frozen=True)
class PMTRTransactionProposal:
    kind: str
    old_values: Tensor
    target_values: Tensor
    head_output: ManifoldRepairOutput

    def __post_init__(self) -> None:
        if self.kind not in ("cell", "site_xyz"):
            raise ValueError("unknown PMTR transaction kind")
        expected = 6 if self.kind == "cell" else 3
        if self.old_values.shape != (expected,) or self.target_values.shape != (expected,):
            raise ValueError("PMTR proposal values do not match transaction width")
        if not bool(torch.isfinite(self.old_values).all().item()) or not bool(
            torch.isfinite(self.target_values).all().item()
        ):
            raise ValueError("PMTR proposal contains non-finite values")


def _sorted_table(table: Mapping[str, Sequence[Any]]) -> tuple[list[float], list[int]]:
    pairs = sorted(
        ((float(value), int(token_id)) for token_id, value in zip(table["ids"], table["values"])),
        key=lambda item: item[0],
    )
    if len(pairs) < 2:
        raise ValueError("numeric token family requires at least two bins")
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _token_value_map(table: Mapping[str, Sequence[Any]]) -> dict[int, float]:
    return {int(token_id): float(value) for token_id, value in zip(table["ids"], table["values"])}


def _lattice_from_values(values: Tensor) -> Tensor:
    if values.shape != (6,):
        raise ValueError("lattice values must contain a,b,c,alpha,beta,gamma")
    lengths = values[:3]
    angles = torch.deg2rad(values[3:])
    a, b, c = lengths.unbind()
    alpha, beta, gamma = angles.unbind()
    sin_gamma = torch.sin(gamma)
    if bool((lengths <= 0).any().item()) or abs(float(sin_gamma.detach())) <= 1.0e-6:
        raise ValueError("token lattice is singular")
    zero = values.new_zeros(())
    cx = c * torch.cos(beta)
    cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / sin_gamma
    cz2 = c.square() - cx.square() - cy.square()
    if float(cz2.detach()) <= 0.0:
        raise ValueError("token lattice has non-positive volume")
    return torch.stack(
        (
            torch.stack((a, zero, zero)),
            torch.stack((b * torch.cos(gamma), b * sin_gamma, zero)),
            torch.stack((cx, cy, torch.sqrt(cz2))),
        )
    )


def _lattice_values(lattice: Tensor) -> Tensor:
    if lattice.shape != (3, 3):
        raise ValueError("lattice must have shape [3,3]")
    lengths = torch.linalg.vector_norm(lattice, dim=-1)

    def angle(left: int, right: int) -> Tensor:
        cosine = torch.dot(lattice[left], lattice[right]) / (
            lengths[left] * lengths[right]
        )
        return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))

    return torch.cat(
        (lengths, torch.stack((angle(1, 2), angle(0, 2), angle(0, 1))))
    )


class PMTRLogitTransform(nn.Module):
    """Compile a learned periodic repair vector into legal SPAD token logits."""

    def __init__(
        self,
        repair_head: nn.Module,
        tokenizer: Any,
        config: PMTRRuntimeConfig = PMTRRuntimeConfig(),
    ) -> None:
        super().__init__()
        self.repair_head = repair_head
        self.runtime_config = config
        self.geometry_support = build_geometry_token_support(tokenizer)
        self.element_by_token: dict[int, tuple[str, int]] = {}
        for token, token_id in tokenizer.get_vocab().items():
            match = _ELEMENT_RE.fullmatch(str(token))
            if match is not None and match.group(1) in SYMBOL_TO_Z:
                symbol = match.group(1)
                self.element_by_token[int(token_id)] = (symbol, int(SYMBOL_TO_Z[symbol]))
        if not self.element_by_token:
            raise ValueError("tokenizer contains no dynamic element tokens")

    def _family(self, kind: str, component: int) -> tuple[str, str]:
        if kind == "site_xyz":
            return "coord", "XYZ"[int(component)]
        if int(component) < 3:
            return "length", "ABC"[int(component)]
        return "angle", "ABG"[int(component) - 3]

    def _merge_complete_state(
        self, context: TransactionContext, model_step: TransactionModelStep
    ) -> Tensor:
        old = context.complete_pre_remask_tokens.to(device=model_step.token_ids.device).clone()
        current = model_step.token_ids
        if current.shape != old.shape:
            raise ValueError("model step and transaction snapshot lengths differ")
        numeric_ids: set[int] = set(self.element_by_token)
        for family in self.geometry_support.values():
            for table in family.values():
                numeric_ids.update(int(value) for value in table["ids"])
        known = torch.zeros_like(current, dtype=torch.bool)
        for token_id in numeric_ids:
            known |= current == int(token_id)
        old[known] = current[known]
        return old

    def _decode_geometry(
        self, context: TransactionContext, model_step: TransactionModelStep
    ) -> tuple[Tensor, Tensor, Tensor, list[str], Tensor]:
        merged = self._merge_complete_state(context, model_step)
        prompt = int(context.prompt_length)
        if (int(context.gen_length) - 7) % 4 != 0:
            raise ValueError("PMTR requires exact 7+4N generation length")
        sites = (int(context.gen_length) - 7) // 4
        if not 1 <= sites <= 20:
            raise ValueError("PMTR site count lies outside 1..20")
        lattice_values: list[float] = []
        for component in range(6):
            family, axis = self._family("cell", component)
            mapping = _token_value_map(self.geometry_support[family][axis])
            token_id = int(merged[prompt + 1 + component].detach().item())
            if token_id not in mapping:
                raise ValueError("lattice token lies outside registered family")
            lattice_values.append(mapping[token_id])
        coordinates: list[list[float]] = []
        atomic_numbers: list[int] = []
        symbols: list[str] = []
        for site in range(sites):
            base = prompt + 7 + 4 * site
            element_id = int(merged[base].detach().item())
            if element_id not in self.element_by_token:
                raise ValueError("site element token is not committed")
            symbol, atomic_number = self.element_by_token[element_id]
            symbols.append(symbol)
            atomic_numbers.append(atomic_number)
            row: list[float] = []
            for component, axis in enumerate("XYZ"):
                mapping = _token_value_map(self.geometry_support["coord"][axis])
                token_id = int(merged[base + 1 + component].detach().item())
                if token_id not in mapping:
                    raise ValueError("coordinate token lies outside registered family")
                row.append(mapping[token_id])
            coordinates.append(row)
        dtype = model_step.hidden_states.dtype if model_step.hidden_states is not None else torch.float32
        device = model_step.logits.device
        lattice_values_tensor = torch.tensor(lattice_values, dtype=dtype, device=device)
        lattice = _lattice_from_values(lattice_values_tensor)
        frac = torch.tensor(coordinates, dtype=dtype, device=device)
        species = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
        return lattice_values_tensor, lattice, frac, symbols, species

    @staticmethod
    def _program_ranks(
        context: TransactionContext, symbols: Sequence[str], device: torch.device
    ) -> Tensor:
        metadata = context.program_metadata
        order = None if metadata is None else metadata.get("species_order")
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            raise ValueError("PMTR requires Llama species_order program metadata")
        normalized = [str(value) for value in order]
        if len(normalized) != len(set(normalized)):
            raise ValueError("species_order contains duplicates")
        rank_by_symbol = {symbol: rank for rank, symbol in enumerate(normalized)}
        if set(rank_by_symbol) != set(symbols):
            raise ValueError("species_order does not match committed crystal species")
        return torch.tensor(
            [rank_by_symbol[symbol] for symbol in symbols],
            dtype=torch.long,
            device=device,
        )

    def prepare(
        self,
        context: TransactionContext,
        model_step: TransactionModelStep,
    ) -> PMTRTransactionProposal:
        hidden = model_step.hidden_states
        if hidden is None or hidden.ndim != 2:
            raise ValueError("PMTR requires final DLM hidden states [sequence,hidden]")
        lattice_values, lattice, frac, symbols, species = self._decode_geometry(
            context, model_step
        )
        prompt = int(context.prompt_length)
        sites = int(frac.shape[0])
        lattice_hidden = torch.stack([hidden[prompt + 1 + i] for i in range(6)])
        site_hidden = torch.stack(
            [
                torch.stack([hidden[prompt + 8 + 4 * site + i] for i in range(3)])
                for site in range(sites)
            ]
        )
        plan_hidden = hidden[:prompt].mean(dim=0) if prompt > 0 else hidden.mean(dim=0)
        ranks = self._program_ranks(context, symbols, hidden.device)
        deltas = frac.unsqueeze(0) - frac.unsqueeze(1)
        mic_vectors, _shifts = minimum_image_vectors(
            deltas, lattice, image_radius=int(self.runtime_config.image_radius)
        )
        diagonal = torch.eye(sites, dtype=torch.bool, device=hidden.device)
        pair_mask = ~diagonal
        output = self.repair_head(
            lattice_hidden=lattice_hidden.unsqueeze(0),
            site_hidden=site_hidden.unsqueeze(0),
            species=species.unsqueeze(0),
            program_rank=ranks.unsqueeze(0),
            site_mask=torch.ones((1, sites), dtype=torch.bool, device=hidden.device),
            mic_vectors=mic_vectors.unsqueeze(0),
            pair_mask=pair_mask.unsqueeze(0),
            plan_hidden=plan_hidden.unsqueeze(0),
        )
        if not isinstance(output, ManifoldRepairOutput):
            # NamedTuple-compatible test doubles are accepted only when all
            # fields are present; formal heads return ManifoldRepairOutput.
            output = ManifoldRepairOutput(
                output.lattice_tangent,
                output.cartesian_site_delta,
                output.site_states,
                output.pair_scalars,
            )
        if context.kind == "cell":
            metric = lattice_to_metric(lattice)
            corrected_metric = spd_congruence_update(metric, output.lattice_tangent[0])
            target = _lattice_values(metric_to_lattice(corrected_metric))
            old = lattice_values
        else:
            site = int(context.site_index)
            if not 0 <= site < sites:
                raise ValueError("active site lies outside committed geometry")
            cartesian_delta = output.cartesian_site_delta[0, site]
            fractional_delta = cartesian_to_fractional(cartesian_delta, lattice)
            old = frac[site]
            target = wrap_fractional(old + fractional_delta)
        return PMTRTransactionProposal(
            kind=context.kind,
            old_values=old,
            target_values=target,
            head_output=output,
        )

    def apply(
        self,
        component: int,
        logits: Tensor,
        proposal: PMTRTransactionProposal,
        context: TransactionContext,
    ) -> Tensor:
        if proposal.kind != context.kind:
            raise ValueError("cached PMTR proposal belongs to another transaction kind")
        family, axis = self._family(context.kind, int(component))
        values, token_ids = _sorted_table(self.geometry_support[family][axis])
        old = proposal.old_values[int(component)].to(device=logits.device, dtype=torch.float32)
        target = proposal.target_values[int(component)].to(
            device=logits.device, dtype=torch.float32
        )
        if family == "coord":
            residual = render_periodic_coordinate_token_residual(
                values,
                token_ids,
                old,
                target,
                vocab_size=int(logits.shape[-1]),
                gain=float(self.runtime_config.transport_gain),
            )
        else:
            residual = render_bracketed_token_residual(
                values,
                token_ids,
                old,
                target,
                vocab_size=int(logits.shape[-1]),
                gain=float(self.runtime_config.transport_gain),
            )
        return logits + residual.to(dtype=logits.dtype)


__all__ = [
    "PMTRLogitTransform",
    "PMTRRuntimeConfig",
    "PMTRTransactionProposal",
]
