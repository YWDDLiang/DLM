from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from crystal_dlm.periodic_repair_model import PeriodicRepairConfig, PeriodicRepairDLM, task_features
from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
from crystal_dlm.periodic_v2_model import (
    PeriodicV2AttentionBias, PeriodicV2Config, numeric_noise_component_features,
)
from crystal_dlm.state_conditioned_model import context_from_programs
from test_state_programmed_runtime import TinyBase, TinyTokenizer, body, program


class _MultiHeadBlock(nn.Module):
    """Real head-separated Q/K/V attention, with one shared external layer bias."""

    def __init__(self, hidden=16, heads=4):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)

    def forward(self, value, bias, mask):
        batch, length, hidden = value.shape
        def split(layer):
            return layer(value).reshape(batch, length, self.heads, hidden // self.heads).transpose(1, 2)
        query, key, values = split(self.query), split(self.key), split(self.value)
        scores = query @ key.transpose(-1, -2) / math.sqrt(hidden // self.heads)
        scores = scores + bias
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None].bool(), -torch.inf)
        attended = (scores.softmax(-1) @ values).transpose(1, 2).reshape(batch, length, hidden)
        return value + self.output(attended)


class _MultiHeadBase(TinyBase):
    def __init__(self, vocabulary, *, recompute=False, heads=4):
        super().__init__(vocabulary, hidden=16, recompute=recompute)
        self.heads = heads
        self.position = nn.Embedding(64, 16)
        self.layers = nn.ModuleList([_MultiHeadBlock(heads=heads) for _ in range(2)])

    def forward(self, input_ids=None, *, inputs_embeds=None, attention_mask=None,
                attention_bias=None, output_hidden_states=False):
        value = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        value = value + self.position(torch.arange(value.shape[1], device=value.device))[None]
        value = value + self.lora_B["default"](self.lora_A["default"](value))
        if attention_bias is None:
            attention_bias = value.new_zeros(value.shape[0], 1, value.shape[1], value.shape[1])
        for layer in self.layers:
            value = (checkpoint(layer, value, attention_bias, attention_mask, use_reentrant=False)
                     if self.recompute else layer(value, attention_bias, attention_mask))
        return SimpleNamespace(logits=self.output(value),
                               hidden_states=(value,) if output_hidden_states else None)


def _noise(context, components):
    values = dict(vars(context))
    values["numeric_noise_components"] = components
    return SimpleNamespace(**values)


def _case(*, count=3, padding=3, recompute=False):
    torch.manual_seed(419)
    tokenizer = TinyTokenizer()
    tokenizer.mask_token_id = tokenizer.mask_id
    base = _MultiHeadBase(len(tokenizer.vocab), recompute=recompute)
    state = PeriodicStateConfig(16, width=12, radial_basis_count=4, max_sites=5)
    legacy_config = PeriodicRepairConfig(16, width=12, radial_bins=6, fourier_modes=2)
    old_model = PeriodicRepairDLM(base, tokenizer, state, legacy_config)
    with torch.no_grad():
        old_model.geometry_attention.edge[-1].weight.normal_(0, .04)
        old_model.state_conditioner.cell_projection.weight.normal_(0, .015)
        old_model.state_conditioner.site_projection.weight.normal_(0, .015)
    prompt = 2
    original = torch.tensor([[0] * prompt + body(tokenizer, count=count, length=40, x=23) + [0] * padding])
    positions = [*range(1, 7), *(7 + 4 * i + axis for i in range(count) for axis in (1, 2, 3))]
    context = context_from_programs(
        original.clone(), prompt_length=prompt, num_sites=count, programs=[program(count)],
        active_positions={0: positions}, task_id=3, max_sites=5,
    )
    context = _noise(context, torch.tensor([[.03, .06, .15]]))
    current = original.clone()
    current[:, [prompt + position for position in positions]] = tokenizer.mask_id
    attention = torch.zeros_like(current)
    attention[:, :prompt + 7 + 4 * count] = 1
    upgraded = deepcopy(old_model)
    upgraded.geometry_attention = PeriodicV2AttentionBias(
        legacy_config,
        PeriodicV2Config(16, heads=4, width=12, radial_bins=6, fourier_modes=2,
                         species_width=4, cell_slot_width=3, max_sites=5),
        legacy=upgraded.geometry_attention,
    )
    return tokenizer, old_model, upgraded, current, context, attention


def _open_residual(module):
    """Nonzero weights for mechanism checks, never a claimed trained result."""
    with torch.no_grad():
        for layer in (module.site_to_site, module.cell_to_site, module.site_to_cell):
            layer.weight.normal_(0, .15)


def _bias(model, current, context):
    geometry = model.geometry_inputs(context)
    tasks = task_features(context, current, geometry, model.mask_id)
    return model.geometry_attention(geometry, context, current.shape[1], tasks), geometry, tasks


def test_zero_upgrade_preserves_nonzero_legacy_bias_and_multihead_active_logits_exactly():
    _, old, upgraded, current, context, attention = _case()
    geometry = old.geometry_inputs(context)
    tasks = task_features(context, current, geometry, old.mask_id)
    legacy = old.geometry_attention(geometry, context, current.shape[1], tasks)
    actual, _, _ = _bias(upgraded, current, context)
    assert legacy.abs().sum() > 0
    assert actual.shape == (1, 4, current.shape[1], current.shape[1])
    torch.testing.assert_close(actual, legacy.expand_as(actual), atol=0, rtol=0)
    expected_logits = old(current, attention_mask=attention, geometry_context=context).logits
    actual_logits = upgraded(current, attention_mask=attention, geometry_context=context).logits
    torch.testing.assert_close(actual_logits, expected_logits, atol=0, rtol=0)
    active = context.active_token_mask
    torch.testing.assert_close(actual_logits[active], expected_logits[active], atol=0, rtol=0)
    metadata = upgraded.geometry_attention.geometry_features(geometry, context, tasks)["metadata"]
    torch.testing.assert_close(2 * upgraded.geometry_attention.noise_gate(metadata).sigmoid(),
                               torch.ones(1, 4), atol=0, rtol=0)


def test_new_head_branches_receive_active_logit_gradients_at_zero_then_encoders_train():
    tokenizer, _, model, current, context, attention = _case(recompute=True)
    model.requires_grad_(False)
    module = model.geometry_attention
    module.requires_grad_(True)
    optimizer = torch.optim.SGD(module.parameters(), lr=.03)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(current, attention_mask=attention, geometry_context=context).logits
        prompt = int(context.prompt_lengths[0])
        loss = (nn.functional.cross_entropy(logits[:, prompt + 1], torch.tensor([tokenizer.vocab["<LA_040>"]]))
                + nn.functional.cross_entropy(logits[:, prompt + 8], torch.tensor([tokenizer.vocab["<X_000>"]])))
        loss.backward()
        for projection in (module.site_to_site, module.cell_to_site, module.site_to_cell):
            assert projection.weight.grad is not None
            assert torch.isfinite(projection.weight.grad).all()
            assert projection.weight.grad.abs().sum() > 0
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in module.parameters())
        if step:
            assert module.pair_encoder[0].weight.grad.abs().sum() > 0
            assert module.cell_site_encoder[0].weight.grad.abs().sum() > 0
            assert module.species.weight.grad.abs().sum() > 0
            assert module.noise_gate.weight.grad.abs().sum() > 0
        optimizer.step()


def test_cell_queries_select_distinct_site_environments_and_six_slots_and_directions_differ():
    _, _, model, current, context, _ = _case()
    _open_residual(model.geometry_attention)
    bias, geometry, tasks = _bias(model, current, context)
    prompt = int(context.prompt_lengths[0])
    cell = torch.arange(prompt + 1, prompt + 7)
    site = torch.tensor([prompt + 7 + 4 * i for i in range(3)])
    cell_rows = bias[0, :, cell][:, :, site]
    assert (cell_rows.amax(-1) - cell_rows.amin(-1)).max() > 1e-5
    assert not torch.allclose(cell_rows[:, 0], cell_rows[:, 1])
    reverse = bias[0, :, site][:, :, cell].transpose(-1, -2)
    assert not torch.allclose(cell_rows, reverse)
    assert not torch.allclose(bias[:, 0], bias[:, 1])

    # Attention with zero Q/K still changes when old periodic environments change;
    # this would be impossible for a row-constant key bias that softmax cancels.
    before = bias[0, :, prompt + 1].softmax(-1)
    changed = dict(geometry, fractional=geometry["fractional"].clone())
    changed["fractional"][0, 2, 0] = .8
    after_bias = model.geometry_attention(changed, context, current.shape[1], tasks)
    after = after_bias[0, :, prompt + 1].softmax(-1)
    assert not torch.allclose(before[:, site], after[:, site])
    values = torch.arange(current.shape[1], dtype=torch.float32)
    assert not torch.allclose(before @ values, after @ values)


def test_absolute_distance_volume_and_log_spd_shape_survive_feature_encoding():
    _, _, model, current, context, _ = _case()
    geometry = model.geometry_inputs(context)
    tasks = task_features(context, current, geometry, model.mask_id)
    features = model.geometry_attention.geometry_features(geometry, context, tasks)
    torch.testing.assert_close(features["cell_features"][0, 0], torch.tensor(math.log(4. ** 3 / 3.)),
                               atol=2e-6, rtol=1e-6)
    assert features["cell_features"][0, 1:7].abs().max() < 1e-6
    scaled = dict(geometry, lattice=geometry["lattice"] * 2.)
    second = model.geometry_attention.geometry_features(scaled, context, tasks)
    torch.testing.assert_close(second["cell_features"][0, 0] - features["cell_features"][0, 0],
                               torch.tensor(math.log(8.)), atol=2e-6, rtol=1e-6)
    radial_bins = model.geometry_attention.config.radial_bins
    pair_a, pair_b = features["pair_features"][0, 0, 1], second["pair_features"][0, 0, 1]
    assert not torch.allclose(pair_a[:radial_bins], pair_b[:radial_bins])
    torch.testing.assert_close(pair_a[radial_bins + 1], pair_b[radial_bins + 1], atol=1e-6, rtol=1e-6)
    anisotropic = dict(geometry, lattice=geometry["lattice"] @ torch.diag(torch.tensor([2., 1., .5])))
    third = model.geometry_attention.geometry_features(anisotropic, context, tasks)
    assert third["cell_features"][0, 1:7].abs().max() > .1


def test_periodic_bias_is_invariant_to_global_translation_and_integer_wrapping():
    _, _, model, current, context, _ = _case()
    _open_residual(model.geometry_attention)
    first, geometry, tasks = _bias(model, current, context)
    translation = torch.tensor([.137, -.281, .493])
    wrapping = torch.tensor([[[1., -2., 3.], [-2., 1., 0.], [3., -1., -2.],
                              [0., 0., 0.], [0., 0., 0.]]])
    translated = dict(geometry, fractional=geometry["fractional"] + translation + wrapping)
    second = model.geometry_attention(translated, context, current.shape[1], tasks)
    torch.testing.assert_close(first, second, atol=1e-5, rtol=3e-5)


def test_unknown_lattice_has_no_geometric_bias_even_after_learning():
    tokenizer, _, model, current, context, _ = _case()
    _open_residual(model.geometry_attention)
    old = context.old_token_ids.clone()
    old[0, int(context.prompt_lengths[0]) + 1] = tokenizer.mask_id
    fields = dict(vars(context), old_token_ids=old)
    missing = SimpleNamespace(**fields)
    bias, geometry, tasks = _bias(model, current, missing)
    assert not geometry["lattice_known"].any()
    features = model.geometry_attention.geometry_features(geometry, missing, tasks)
    assert not features["pair_known"].any()
    assert not features["cell_features"].any()
    assert torch.isfinite(bias).all() and not bias.any()


def test_unknown_target_site_and_padding_never_receive_or_supply_geometric_bias():
    tokenizer, _, model, current, context, _ = _case()
    _open_residual(model.geometry_attention)
    old = context.old_token_ids.clone()
    prompt = int(context.prompt_lengths[0])
    old[0, prompt + 8] = tokenizer.mask_id
    missing = SimpleNamespace(**dict(vars(context), old_token_ids=old))
    bias, geometry, tasks = _bias(model, current, missing)
    site_tokens = torch.arange(prompt + 7, prompt + 11)
    assert not bias[:, :, site_tokens].any()
    assert not bias[:, :, :, site_tokens].any()
    end = prompt + 7 + 4 * int(context.num_sites[0])
    outside = [*range(prompt + 1), *range(end, current.shape[1])]
    assert not bias[:, :, outside].any() and not bias[:, :, :, outside].any()
    poisoned = {key: value.clone() for key, value in geometry.items()}
    poisoned["fractional"][:, 3:] = float("nan")
    poisoned["species"][:, 3:] = 118
    poisoned["program_rank"][:, 3:] = 99
    poisoned["active_sites"][:, 3:] = True
    poisoned_bias = model.geometry_attention(poisoned, missing, current.shape[1], tasks)
    torch.testing.assert_close(poisoned_bias, bias, atol=0, rtol=0)


def test_single_atom_zero_neighbor_pool_keeps_only_real_cell_site_information():
    _, _, model, current, context, _ = _case(count=1)
    _open_residual(model.geometry_attention)
    bias, geometry, tasks = _bias(model, current, context)
    features = model.geometry_attention.geometry_features(geometry, context, tasks)
    assert not features["neighbor_count"].any() and not features["pair_features"].any()
    assert features["site_known"][0, 0]
    prompt = int(context.prompt_lengths[0])
    assert bias[0, :, prompt + 1, prompt + 7:prompt + 11].abs().sum() > 0
    assert not bias[0, :, prompt + 7:prompt + 11, prompt + 7:prompt + 11].any()
    assert torch.isfinite(bias).all()


def test_mixed_counts_and_prompt_lengths_match_separate_bias_calls():
    tokenizer, _, model, current, context, _ = _case()
    _open_residual(model.geometry_attention)
    length = current.shape[1]
    old_second = [0] * 4 + body(tokenizer, count=1, length=30)
    old_second += [0] * (length - len(old_second))
    old = torch.cat((context.old_token_ids, torch.tensor([old_second])), 0)
    active = torch.zeros_like(old, dtype=torch.bool)
    active[0] = context.active_token_mask[0]
    active[1, [4 + position for position in (*range(1, 7), 8, 9, 10)]] = True
    combined = SimpleNamespace(
        old_token_ids=old, prompt_lengths=torch.tensor([2, 4]), num_sites=torch.tensor([3, 1]),
        program_rank=torch.tensor([[0, 0, 0, 5, 5], [0, 5, 5, 5, 5]]),
        active_token_mask=active, task_ids=torch.tensor([3, 3]),
        numeric_noise_level=torch.tensor([-1., -1.]),
        numeric_noise_components=torch.tensor([[.03, .06, .15], [-1., -1., -1.]]),
    )
    current = old.clone()
    current[active] = tokenizer.mask_id
    together, _, _ = _bias(model, current, combined)
    for row in range(2):
        single = SimpleNamespace(**{name: value[row:row + 1] for name, value in vars(combined).items()})
        separate, _, _ = _bias(model, current[row:row + 1], single)
        torch.testing.assert_close(together[row:row + 1], separate, atol=2e-7, rtol=2e-6)
        start = int(single.prompt_lengths[0])
        end = start + 7 + 4 * int(single.num_sites[0])
        invalid = [*range(start + 1), *range(end, length)]
        assert not together[row, :, invalid].any() and not together[row, :, :, invalid].any()


def test_default_32_head_contract_accepts_decoded_geometry_and_broadcasts_legacy():
    _, _, model, current, context, _ = _case()
    geometry = model.geometry_inputs(context)
    tasks = task_features(context, current, geometry, model.mask_id)
    module = PeriodicV2AttentionBias(PeriodicRepairConfig(4096), PeriodicV2Config(4096))
    with torch.no_grad():
        module.legacy.edge[-1].weight.normal_(0, .01)
    actual = module(geometry, context, current.shape[1], tasks)
    assert actual.shape == (1, 32, current.shape[1], current.shape[1])
    expected = module.legacy(geometry, context, current.shape[1], tasks)
    torch.testing.assert_close(actual, expected.expand_as(actual), atol=0, rtol=0)


def test_typed_noise_channels_do_not_reuse_or_mutate_the_legacy_scalar():
    _, _, model, current, context, _ = _case()
    features = numeric_noise_component_features(context)
    torch.testing.assert_close(features, torch.tensor([[.03, .06, .15, 1.]]))
    assert context.numeric_noise_level.item() == -1
    unknown = _noise(context, torch.full((1, 3), -1.))
    assert not numeric_noise_component_features(unknown).any()
    clean = numeric_noise_component_features(_noise(context, torch.zeros(1, 3)))
    torch.testing.assert_close(clean, torch.tensor([[0., 0., 0., 1.]]), atol=0, rtol=0)
    for invalid in (torch.tensor([[-1., .1, -1.]]), torch.tensor([[float("nan"), 0., 0.]]), torch.zeros(1, 2)):
        with pytest.raises(ValueError):
            numeric_noise_component_features(_noise(context, invalid))
    geometry = model.geometry_inputs(context)
    old_task = task_features(context, current, geometry, model.mask_id)
    hidden_task = task_features(unknown, current, geometry, model.mask_id)
    torch.testing.assert_close(old_task, hidden_task, atol=0, rtol=0)


def test_current_mask_gate_is_recomputed_without_caching_or_mutating_old_geometry():
    _, _, model, current, context, _ = _case()
    module = model.geometry_attention
    _open_residual(module)
    with torch.no_grad():
        module.noise_gate.weight[:, 0] = 1.
        module.noise_gate.weight[:, 9] = .7
    first, geometry, tasks = _bias(model, current, context)
    second_tasks = tasks.clone()
    second_tasks[:, 0] = 0.
    second = module(geometry, context, current.shape[1], second_tasks)
    assert not torch.allclose(first, second)
    restored = module(geometry, context, current.shape[1], tasks)
    torch.testing.assert_close(first, restored, atol=0, rtol=0)
    hidden = module(geometry, _noise(context, torch.full((1, 3), -1.)), current.shape[1], tasks)
    assert not torch.allclose(first, hidden)


def test_checkpointed_multihead_attention_matches_outputs_and_parameter_gradients():
    _, _, direct, current, context, attention = _case(recompute=False)
    _open_residual(direct.geometry_attention)
    recomputed = deepcopy(direct)
    recomputed.base_model.recompute = True
    outputs = []
    for model in (direct, recomputed):
        model.zero_grad(set_to_none=True)
        logits = model(current, attention_mask=attention, geometry_context=context).logits
        active = logits[context.active_token_mask]
        loss = active[:, :11].square().mean()
        loss.backward()
        outputs.append(logits.detach())
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    first = dict(direct.geometry_attention.named_parameters())
    second = dict(recomputed.geometry_attention.named_parameters())
    for name in first:
        assert (first[name].grad is None) == (second[name].grad is None)
        if first[name].grad is not None:
            torch.testing.assert_close(first[name].grad, second[name].grad, atol=2e-6, rtol=2e-5)


def test_bias_forward_itself_is_checkpoint_safe_with_explicit_tasks():
    _, _, model, current, context, _ = _case()
    direct = model.geometry_attention
    _open_residual(direct)
    recomputed = deepcopy(direct)
    geometry = model.geometry_inputs(context)
    tasks = task_features(context, current, geometry, model.mask_id)
    for module, use_checkpoint in ((direct, False), (recomputed, True)):
        task_input = tasks.detach().clone().requires_grad_(True)
        call = lambda value: module(geometry, context, current.shape[1], value)
        result = checkpoint(call, task_input, use_reentrant=False) if use_checkpoint else call(task_input)
        result.square().mean().backward()
    for (name, first), (_, second) in zip(direct.named_parameters(), recomputed.named_parameters()):
        assert (first.grad is None) == (second.grad is None), name
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-5)


def test_v2_modules_and_output_remain_float32_after_parent_dtype_conversion():
    _, _, model, current, context, _ = _case()
    module = model.geometry_attention
    _open_residual(module)
    before, geometry, tasks = _bias(model, current, context)
    module.bfloat16()
    assert all(parameter.dtype == torch.float32 for parameter in module.parameters())
    assert all(not buffer.is_floating_point() or buffer.dtype == torch.float32 for buffer in module.buffers())
    after = module(geometry, context, current.shape[1], tasks)
    assert after.dtype == torch.float32
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_config_rejects_an_unroutable_head_count_and_mismatched_periodic_shell():
    with pytest.raises(ValueError):
        PeriodicV2Config(16, heads=3)
    with pytest.raises(ValueError):
        PeriodicV2AttentionBias(PeriodicRepairConfig(16, image_radius=1),
                                PeriodicV2Config(16, heads=4, image_radius=2))
