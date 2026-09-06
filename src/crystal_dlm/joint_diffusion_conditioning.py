"""Proposal/Planner conditioning inside the existing continuous CSP decoder.

The DLM proposal is an explicit model input, not only the sampler's initial
state. Zero-initialized residuals preserve the frozen model494 decoder exactly
at initialization. No energy, relaxed target, force, or hull value is accepted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class JointDiffusionConditionConfig:
    hidden_size: int = 512
    width: int = 128
    radial_bins: int = 16
    radial_cutoff_A: float = 7.0
    max_sites: int = 100
    schema: str = "dlm_plan_conditioned_continuous_diffusion_v1"

    def __post_init__(self):
        if min(self.hidden_size, self.width, self.radial_bins, self.max_sites) < 1:
            raise ValueError("conditioning dimensions must be positive")
        if self.radial_cutoff_A <= 0:
            raise ValueError("radial cutoff must be positive")


@dataclass(frozen=True)
class JointDiffusionContext:
    proposal_fractional: Tensor
    proposal_lattices: Tensor
    program_rank: Tensor


class ProposalPlanConditioner(nn.Module):
    """Encode proposal-relative periodic geometry and typed program order."""

    def __init__(self, config: JointDiffusionConditionConfig):
        super().__init__()
        self.config = config
        width = config.width
        self.rank_embedding = nn.Embedding(config.max_sites + 1, width)
        site_input = 6 + 3 + config.radial_bins + width + 1
        graph_input = 6 + 6 + 2
        self.site_encoder = nn.Sequential(
            nn.Linear(site_input, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_input, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.site_projection = nn.Linear(width, config.hidden_size, bias=False)
        self.graph_projection = nn.Linear(width, config.hidden_size, bias=False)
        nn.init.zeros_(self.site_projection.weight)
        nn.init.zeros_(self.graph_projection.weight)
        self.register_buffer(
            "radial_centers",
            torch.linspace(0.0, config.radial_cutoff_A, config.radial_bins),
        )

    @staticmethod
    def _gram_features(lattice: Tensor) -> tuple[Tensor, Tensor]:
        gram = lattice @ lattice.transpose(-1, -2)
        scale2 = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-10)
        normalized = torch.stack(
            (gram[:, 0, 0], gram[:, 1, 1], gram[:, 2, 2],
             gram[:, 0, 1], gram[:, 0, 2], gram[:, 1, 2]), -1,
        ) / scale2[:, None]
        volume = torch.linalg.det(lattice).abs().clamp_min(1e-10)
        return normalized, torch.stack((scale2.sqrt().log(), volume.log()), -1)

    def _proposal_radial_environment(
        self, proposal_fractional: Tensor, proposal_lattices: Tensor,
        node2graph: Tensor, num_atoms: Tensor,
    ) -> Tensor:
        result = proposal_fractional.new_zeros(
            proposal_fractional.shape[0], self.config.radial_bins
        )
        start = 0
        spacing = self.config.radial_cutoff_A / max(self.config.radial_bins - 1, 1)
        for graph, count_value in enumerate(num_atoms):
            count = int(count_value)
            stop = start + count
            frac = proposal_fractional[start:stop]
            delta = frac[:, None] - frac[None, :]
            delta = delta - delta.round()
            distance = torch.linalg.vector_norm(delta @ proposal_lattices[graph], dim=-1)
            mask = ~torch.eye(count, dtype=torch.bool, device=distance.device)
            radial = torch.exp(
                -((distance[..., None] - self.radial_centers) / spacing).square()
            )
            envelope = 0.5 * (
                1 + torch.cos(math.pi * (distance / self.config.radial_cutoff_A).clamp(max=1))
            )
            radial = radial * (mask & (distance < self.config.radial_cutoff_A))[..., None] * envelope[..., None]
            result[start:stop] = radial.sum(1)
            start = stop
        if start != proposal_fractional.shape[0]:
            raise ValueError("num_atoms does not cover proposal nodes")
        return result

    def forward(
        self, *, time: Tensor, current_fractional: Tensor, current_lattices: Tensor,
        num_atoms: Tensor, node2graph: Tensor, context: JointDiffusionContext,
    ) -> tuple[Tensor, Tensor]:
        proposal = context.proposal_fractional
        lattices = context.proposal_lattices
        ranks = context.program_rank
        nodes, graphs = current_fractional.shape[0], current_lattices.shape[0]
        if (current_fractional.shape != (nodes, 3) or proposal.shape != (nodes, 3)
                or current_lattices.shape != (graphs, 3, 3)
                or lattices.shape != (graphs, 3, 3)
                or ranks.shape != (nodes,) or node2graph.shape != (nodes,)
                or num_atoms.shape != (graphs,)):
            raise ValueError("joint diffusion context shapes differ from the current batch")
        if int(num_atoms.sum()) != nodes or bool((num_atoms < 1).any()):
            raise ValueError("num_atoms does not partition proposal nodes")
        if bool(((ranks < 0) | (ranks >= self.config.max_sites)).any()):
            raise ValueError("program ranks are outside the registered range")
        if not all(torch.isfinite(value).all() for value in
                   (time, current_fractional, current_lattices, proposal, lattices)):
            raise ValueError("joint diffusion conditioning requires finite complete geometry")
        if bool((torch.linalg.det(current_lattices).abs() <= 1e-10).any()
                or (torch.linalg.det(lattices).abs() <= 1e-10).any()):
            raise ValueError("proposal/current lattices must be nondegenerate")
        delta = current_fractional - proposal
        delta = delta - delta.round()
        phase = 2 * math.pi * delta
        proposal_per_node = lattices[node2graph]
        cell_scale = torch.linalg.vector_norm(proposal_per_node, dim=-1).mean(-1).clamp_min(1e-6)
        cartesian_delta = (delta[:, None] @ proposal_per_node).squeeze(1) / cell_scale[:, None]
        radial = self._proposal_radial_environment(proposal, lattices, node2graph, num_atoms)
        normalized_rank = ranks.float() / max(self.config.max_sites - 1, 1)
        site_features = torch.cat(
            (phase.sin(), phase.cos(), cartesian_delta, radial,
             self.rank_embedding(ranks), normalized_rank[:, None]), -1,
        )
        current_gram, current_scale = self._gram_features(current_lattices)
        proposal_gram, proposal_scale = self._gram_features(lattices)
        graph_features = torch.cat(
            (current_gram, proposal_gram, current_scale - proposal_scale), -1,
        )
        return self.site_projection(self.site_encoder(site_features)), \
            self.graph_projection(self.graph_encoder(graph_features))


class ProposalConditionedCSPNet(nn.Module):
    """Run the existing CSPNet with proposal/Plan residuals before its layers."""

    def __init__(
        self, base_decoder: nn.Module,
        config: JointDiffusionConditionConfig = JointDiffusionConditionConfig(),
    ):
        super().__init__()
        self.base_decoder = base_decoder
        self.conditioner = ProposalPlanConditioner(config)
        self.config = config

    def forward(
        self, time, atom_types, frac_coords, lattices, num_atoms, node2graph,
        *, condition: JointDiffusionContext,
    ):
        decoder = self.base_decoder
        edges, frac_diff = decoder.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        node_features = decoder.node_embedding(
            atom_types if decoder.smooth else atom_types - 1
        )
        node_features = decoder.atom_latent_emb(
            torch.cat((node_features, time.repeat_interleave(num_atoms, dim=0)), dim=1)
        )
        site_delta, graph_delta = self.conditioner(
            time=time[:, 0] if time.ndim == 2 else time,
            current_fractional=frac_coords, current_lattices=lattices,
            num_atoms=num_atoms, node2graph=node2graph, context=condition,
        )
        node_features = node_features + site_delta.to(node_features.dtype)
        node_features = node_features + graph_delta[node2graph].to(node_features.dtype)
        for index in range(decoder.num_layers):
            node_features = decoder._modules[f"csp_layer_{index}"](
                node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff
            )
        if decoder.ln:
            node_features = decoder.final_layer_norm(node_features)
        coord_out = decoder.coord_out(node_features)
        graph_features = node_features.new_zeros((int(num_atoms.shape[0]), node_features.shape[-1]))
        graph_features.index_add_(0, node2graph, node_features)
        graph_features = graph_features / num_atoms[:, None].to(graph_features.dtype)
        lattice_out = decoder.lattice_out(graph_features).view(-1, 3, 3)
        if decoder.ip:
            if decoder.run_type == "sample":
                lattice_out = lattice_out.double()
            lattice_out = torch.einsum("bij,bjk->bik", lattice_out, lattices)
        if decoder.pred_type:
            return lattice_out, coord_out, decoder.type_out(node_features)
        return lattice_out, coord_out

    def save_conditioner(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        torch.save(self.conditioner.state_dict(), root / "joint_diffusion_conditioner.pt")
        (root / "joint_diffusion_conditioner.json").write_text(
            json.dumps(asdict(self.config), indent=2) + "\n", encoding="utf-8"
        )

    def load_conditioner(self, directory):
        root = Path(directory)
        recorded = JointDiffusionConditionConfig(**json.loads(
            (root / "joint_diffusion_conditioner.json").read_text(encoding="utf-8")
        ))
        if recorded != self.config:
            raise ValueError("joint diffusion conditioner config differs")
        self.conditioner.load_state_dict(torch.load(
            root / "joint_diffusion_conditioner.pt", map_location="cpu", weights_only=True
        ), strict=True)


def set_joint_condition_only_trainable(model: ProposalConditionedCSPNet) -> dict[str, int]:
    model.requires_grad_(False)
    model.conditioner.requires_grad_(True)
    return {
        "conditioner": sum(parameter.numel() for parameter in model.conditioner.parameters()),
        "frozen_model494_decoder": sum(
            parameter.numel() for parameter in model.base_decoder.parameters()
        ),
    }
