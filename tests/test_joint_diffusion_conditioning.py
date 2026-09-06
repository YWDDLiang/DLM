from types import SimpleNamespace
import torch
from torch import nn

from crystal_dlm.joint_diffusion_conditioning import (
    JointDiffusionConditionConfig, JointDiffusionContext,
    ProposalConditionedCSPNet, set_joint_condition_only_trainable,
)


class Layer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
    def forward(self, x, *args, **kwargs):
        return x + self.linear(x)


class TinyDecoder(nn.Module):
    def __init__(self, hidden=12):
        super().__init__()
        self.smooth, self.num_layers, self.ln, self.ip = False, 1, False, False
        self.run_type, self.pred_type = "train", False
        self.node_embedding = nn.Embedding(100, 5)
        self.atom_latent_emb = nn.Linear(7, hidden)
        self.add_module("csp_layer_0", Layer(hidden))
        self.coord_out = nn.Linear(hidden, 3)
        self.lattice_out = nn.Linear(hidden, 9)
    def gen_edges(self, counts, frac, lattices, node2graph):
        edge = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        return edge, frac[edge[1]] - frac[edge[0]]
    def forward(self, time, atoms, frac, lattices, counts, batch):
        edges, delta = self.gen_edges(counts, frac, lattices, batch)
        h = self.atom_latent_emb(torch.cat((self.node_embedding(atoms - 1),
                                           time.repeat_interleave(counts, 0)), -1))
        h = self._modules["csp_layer_0"](h, frac, lattices, edges, batch[edges[0]], frac_diff=delta)
        lattice = self.lattice_out(torch.stack([h[batch == i].mean(0) for i in range(len(counts))]))
        return lattice.view(-1, 3, 3), self.coord_out(h)


def sample():
    torch.manual_seed(12)
    base = TinyDecoder()
    wrapper = ProposalConditionedCSPNet(
        base, JointDiffusionConditionConfig(hidden_size=12, width=10, radial_bins=5, max_sites=4)
    )
    counts = torch.tensor([2, 2])
    batch = torch.tensor([0, 0, 1, 1])
    lattices = torch.eye(3).repeat(2, 1, 1) * 4
    frac = torch.tensor([[.99, 0, 0], [.5, 0, 0], [.1, .2, .3], [.6, .2, .3]])
    context = JointDiffusionContext((frac + .07).remainder(1), lattices * 1.03,
                                    torch.tensor([0, 1, 0, 1]))
    time = torch.ones(2, 2)
    atoms = torch.tensor([1, 2, 3, 4])
    return wrapper, time, atoms, frac, lattices, counts, batch, context


def test_zero_conditioning_exactly_reproduces_base_decoder():
    wrapper, time, atoms, frac, lattices, counts, batch, context = sample()
    expected = wrapper.base_decoder(time, atoms, frac, lattices, counts, batch)
    actual = wrapper(time, atoms, frac, lattices, counts, batch, condition=context)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_proposal_and_program_residuals_receive_gradients_then_change_output():
    wrapper, time, atoms, frac, lattices, counts, batch, context = sample()
    counts_report = set_joint_condition_only_trainable(wrapper)
    assert counts_report["conditioner"] > 0 and counts_report["frozen_model494_decoder"] > 0
    optimizer = torch.optim.AdamW(
        [parameter for parameter in wrapper.parameters() if parameter.requires_grad], lr=.02
    )
    for _ in range(2):
        optimizer.zero_grad()
        lattice, coord = wrapper(time, atoms, frac, lattices, counts, batch, condition=context)
        loss = lattice.square().mean() + coord.square().mean()
        loss.backward()
        assert wrapper.conditioner.site_projection.weight.grad.abs().sum() > 0
        assert wrapper.conditioner.graph_projection.weight.grad.abs().sum() > 0
        optimizer.step()
    changed = wrapper(time, atoms, frac, lattices, counts, batch, condition=context)[1]
    baseline = wrapper.base_decoder(time, atoms, frac, lattices, counts, batch)[1]
    assert not torch.equal(changed, baseline)
    assert all(parameter.grad is None for parameter in wrapper.base_decoder.parameters())


def test_periodic_proposal_wrapping_does_not_change_conditioner_output():
    wrapper, time, _, frac, lattices, counts, batch, context = sample()
    first = wrapper.conditioner(
        time=time[:, 0], current_fractional=frac, current_lattices=lattices,
        num_atoms=counts, node2graph=batch, context=context,
    )
    shifted = JointDiffusionContext(
        context.proposal_fractional + torch.tensor([[1., -1., 0.]]),
        context.proposal_lattices, context.program_rank,
    )
    second = wrapper.conditioner(
        time=time[:, 0], current_fractional=frac, current_lattices=lattices,
        num_atoms=counts, node2graph=batch, context=shifted,
    )
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-6)


def test_condition_api_has_no_target_or_physics_fields():
    assert set(JointDiffusionContext.__dataclass_fields__) == {
        "proposal_fractional", "proposal_lattices", "program_rank",
    }
