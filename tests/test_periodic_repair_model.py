from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from crystal_dlm.periodic_repair_model import (
    PeriodicRepairDLM, PeriodicRepairConfig, numeric_features, set_repair_trainable,
    task_features,
)
from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
from crystal_dlm.state_conditioned_model import StateConditionedDLM, context_from_programs
from test_state_programmed_runtime import TinyTokenizer, body, program, TinyBase


class AttentionBase(TinyBase):
    def forward(self, input_ids=None, *, inputs_embeds=None, attention_mask=None,
                attention_bias=None, output_hidden_states=False):
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        def block(value, bias):
            value = value + self.lora_B["default"](self.lora_A["default"](value))
            weights = torch.softmax(value @ value.transpose(-1, -2) / 4 + bias[:, 0], -1)
            return value + weights @ value
        if attention_bias is None:
            attention_bias = x.new_zeros(x.shape[0], 1, x.shape[1], x.shape[1])
        hidden = checkpoint(block, x, attention_bias, use_reentrant=False) if self.recompute else block(x, attention_bias)
        return SimpleNamespace(logits=self.output(hidden),
                               hidden_states=(hidden,) if output_hidden_states else None)


def case():
    torch.manual_seed(903)
    tok = TinyTokenizer()
    tok.mask_token_id = tok.mask_id
    base = AttentionBase(len(tok.vocab), recompute=True)
    config = PeriodicStateConfig(16, width=12, radial_basis_count=4)
    old = StateConditionedDLM(base, tok, config)
    with torch.no_grad():
        old.state_conditioner.cell_projection.weight.normal_(0, .01)
        old.state_conditioner.site_projection.weight.normal_(0, .01)
    model = PeriodicRepairDLM(deepcopy(base), tok, config, PeriodicRepairConfig(16, width=12))
    model.state_conditioner.load_state_dict(old.state_conditioner.state_dict())
    original = torch.tensor([[0] + body(tok, count=2, length=40, x=35)])
    positions = [*range(1, 7), 8, 9, 10, 12, 13, 14]
    context = context_from_programs(original.clone(), prompt_length=1, num_sites=2,
                                   programs=[program()], active_positions={0: positions}, task_id=3)
    current = original.clone()
    current[:, [p + 1 for p in positions]] = tok.mask_id
    return tok, old, model, current, context


def test_zero_initialization_preserves_existing_conditioner_and_logits():
    _, old, model, current, context = case()
    expected = old(current, geometry_context=context).logits
    actual = model(current, geometry_context=context).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert model.state_conditioner.site_projection.weight.abs().sum() > 0


def test_internal_bias_and_numeric_projection_receive_checkpointed_gradients():
    tok, _, model, current, context = case()
    counts = set_repair_trainable(model)
    assert counts["lora"] > 0 and counts["geometry_attention"] > 0
    assert not model.get_output_embeddings().weight.requires_grad
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        logits = model(current, geometry_context=context).logits
        loss = nn.functional.cross_entropy(logits[:, 9], torch.tensor([tok.vocab["<X_030>"]]))
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.geometry_attention.edge[-1].weight.grad.abs().sum() > 0
        assert model.numeric_adapter.projections[6].weight.grad.abs().sum() > 0
        optimizer.step()
    assert model.geometry_attention.edge[0].weight.grad.abs().sum() > 0


def test_periodic_pairs_are_translation_and_wrapping_invariant():
    _, _, model, current, context = case()
    geometry = model.geometry_inputs(context)
    tasks = task_features(context, current, geometry, model.mask_id)
    first, known = model.geometry_attention.pair_features(geometry, tasks)
    shifted = dict(geometry, fractional=(geometry["fractional"] + .137).remainder(1))
    second, known_second = model.geometry_attention.pair_features(shifted, tasks)
    torch.testing.assert_close(first, second, atol=3e-6, rtol=1e-5)
    assert torch.equal(known, known_second)
    assert not known[0, 0, 0]


def test_unknown_geometry_cannot_create_a_pair_and_numeric_aliases_match():
    _, _, model, current, context = case()
    geometry = model.geometry_inputs(context)
    geometry["site_known"][:] = False
    geometry["fractional"][:] = float("nan")
    features, known = model.geometry_attention.pair_features(geometry, torch.zeros(1, 9))
    assert torch.isfinite(features).all() and not features.any() and not known.any()
    periodic = numeric_features([0., 1., .99, .01], "coord")
    assert torch.equal(periodic[0], periodic[1])
    assert torch.linalg.vector_norm(periodic[2] - periodic[3]) < 3
    assert not torch.equal(numeric_features([1., 179.], "angle")[0],
                           numeric_features([1., 179.], "angle")[1])


def test_repair_components_remain_float32_under_parent_conversion():
    _, _, model, _, context = case()
    before = model.geometry_inputs(context)
    model.bfloat16()
    for module in model.repair_modules().values():
        assert all(p.dtype == torch.float32 for p in module.parameters())
    assert model.geometry_values.dtype == torch.float32
    after = model.geometry_inputs(context)
    torch.testing.assert_close(before["fractional"], after["fractional"], atol=0, rtol=0)
    torch.testing.assert_close(before["lattice"], after["lattice"], atol=0, rtol=0)


def test_visible_does_not_imply_clean_and_task_is_explicit():
    _, _, model, _, context = case()
    visible = context.old_token_ids
    geometry = model.geometry_inputs(context)
    unknown_error = task_features(context, visible, geometry, model.mask_id)
    assert unknown_error[0, 0] == 0  # no masked numeric tokens
    assert unknown_error[0, 2] == 0  # unknown error magnitude, not known-clean
    assert unknown_error[0, 7] == 1  # full-cell repair task
    context.numeric_noise_level.fill_(.4)
    known_error = task_features(context, visible, geometry, model.mask_id)
    assert known_error[0, 2] == 1 and abs(float(known_error[0, 3]) - .4) < 1e-6
