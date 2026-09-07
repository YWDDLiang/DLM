"""Behavioral audits of the editor's content and decision interfaces.

The fake B0 mixes tokens, returns normalized hidden states, and checkpoints its
encoder. Tied token embeddings isolate numeric conditioning from ordinary token
identity, so the current-state tests cannot pass through the base embedding alone.
"""
from dataclasses import dataclass
import datetime
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from crystal_dlm.expert_edit import (
    EditContext,
    ExpertEditConfig,
    ExpertEditDLM,
    ExpertEditObjective,
    set_editor_trainable,
)
from crystal_dlm.fixed_slot import build_special_tokens


class AuditTokenizer:
    def __init__(self):
        tokens = ["<PAD>", *build_special_tokens(), "<MASK>", "<AUDIT_PROMPT>"]
        assert len(tokens) == len(set(tokens))
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.pad_token_id = self.vocab["<PAD>"]
        self.mask_token_id = self.vocab["<MASK>"]
        self.prompt_token_id = self.vocab["<AUDIT_PROMPT>"]

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


class AuditBase(nn.Module):
    """Small frozen encoder with an already nonzero B0-style LoRA."""

    def __init__(self, vocabulary_size, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocabulary_size, bias=False)
        self.normalization = nn.LayerNorm(hidden_size)
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(hidden_size, 3, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(3, hidden_size, bias=False)}
        )
        nn.init.normal_(self.lora_B["default"].weight, std=0.04)
        self.config = SimpleNamespace(hidden_size=hidden_size, d_model=hidden_size)
        self.recompute = True
        self.encoder_invocations = 0

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output

    def forward(
        self, input_ids=None, *, inputs_embeds=None, attention_mask=None,
        output_hidden_states=False, return_dict=True,
    ):
        del return_dict
        values = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(values.shape[:2], device=values.device)

        def encode(tensor, present):
            self.encoder_invocations += 1
            tensor = tensor + self.lora_B["default"](self.lora_A["default"](tensor))
            scores = tensor @ tensor.transpose(-1, -2) / math.sqrt(tensor.shape[-1])
            scores = scores.masked_fill(
                ~present[:, None, :].bool(), torch.finfo(scores.dtype).min
            )
            mixed = tensor + torch.softmax(scores, -1) @ tensor
            return self.normalization(mixed)

        hidden = (
            checkpoint(encode, values, attention_mask, use_reentrant=False)
            if self.recompute else encode(values, attention_mask)
        )
        return SimpleNamespace(
            logits=self.output(hidden),
            hidden_states=(hidden,) if output_hidden_states else None,
        )


@dataclass
class Case:
    tokenizer: AuditTokenizer
    model: ExpertEditDLM
    current: torch.Tensor
    attention: torch.Tensor
    context: EditContext


def _body(tokenizer, count):
    token = tokenizer.vocab
    result = [token[f"<N_{count:03d}>"]]
    result.extend(token[f"<{axis}_040>"] for axis in ("LA", "LB", "LC"))
    result.extend(token[f"<{axis}_090>"] for axis in ("AA", "AB", "AG"))
    for site in range(count):
        result.append(token["<E_H>" if site % 2 == 0 else "<E_O>"])
        result.extend(
            token[f"<{axis}_{(17 + 31 * site + 11 * offset) % 100:03d}>"]
            for offset, axis in enumerate("XYZ")
        )
    return result


def _numeric_positions(count):
    return [*range(1, 7), *(
        8 + 4 * site + axis for site in range(count) for axis in range(3)
    )]


def _case(*, inspection=False, mixed=False):
    torch.manual_seed(1831)
    tokenizer = AuditTokenizer()
    model = ExpertEditDLM(
        AuditBase(len(tokenizer)), tokenizer,
        ExpertEditConfig(hidden_size=16, width=12, max_sites=20),
    ).eval()
    layouts = [(2, 2), (4, 1)] if mixed else [(2, 2)]
    lengths = [prompt + 7 + 4 * count for prompt, count in layouts]
    old = torch.full(
        (len(layouts), max(lengths)), tokenizer.pad_token_id, dtype=torch.long
    )
    attention = torch.zeros_like(old)
    active = torch.zeros_like(old, dtype=torch.bool)
    for row, ((prompt, count), length) in enumerate(zip(layouts, lengths)):
        old[row, :prompt] = tokenizer.prompt_token_id
        old[row, prompt:length] = torch.tensor(_body(tokenizer, count))
        attention[row, :length] = 1
        if not inspection:
            active[row, [prompt + pos for pos in _numeric_positions(count)]] = True
    current = old.clone().masked_fill(active, tokenizer.mask_token_id)
    context = EditContext(
        old_token_ids=old,
        prompt_lengths=torch.tensor([layout[0] for layout in layouts]),
        num_sites=torch.tensor([layout[1] for layout in layouts]),
        active_token_mask=active,
        task_ids=torch.zeros(len(layouts), dtype=torch.long),
        remaining_steps=torch.full((len(layouts),), 8, dtype=torch.long),
        reveal_fraction=torch.full((len(layouts),), float(inspection)),
    )
    return Case(tokenizer, model, current, attention, context)


def _context(context, **changes):
    values = {
        name: getattr(context, name) for name in (
            "old_token_ids", "prompt_lengths", "num_sites", "active_token_mask",
            "task_ids", "remaining_steps", "reveal_fraction",
        )
    }
    return EditContext(**{**values, **changes})


def _activate_zero_outputs(model):
    """Move zero residual projections off zero without training a fake dataset."""
    generator = torch.Generator().manual_seed(71)
    visited = set()
    with torch.no_grad():
        for module in [model.state_conditioner, *model.extra_modules().values()]:
            for parameter in module.parameters():
                if id(parameter) in visited:
                    continue
                visited.add(id(parameter))
                if not torch.count_nonzero(parameter):
                    if parameter.ndim >= 2:
                        parameter.normal_(std=0.07, generator=generator)
                    else:
                        parameter.fill_(0.03)


def _forward(case, *, current=None, context=None):
    return case.model(
        case.current if current is None else current,
        attention_mask=case.attention,
        edit_context=case.context if context is None else context,
    )


def _relative_logits(case, output, *, body_position=8, family="X"):
    ids = [
        token_id for token, token_id in case.tokenizer.vocab.items()
        if token.startswith(f"<{family}_")
    ]
    assert ids
    position = int(case.context.prompt_lengths[0]) + body_position
    values = output.logits[0, position, ids].float()
    return values - values.mean()


def _assert_content_changed(case, first, second, **query):
    left = _relative_logits(case, first, **query)
    right = _relative_logits(case, second, **query)
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    assert (left - right).abs().max().item() > 1e-7, (
        "The changed condition did not change relative numeric token logits"
    )


def _check_zero_residuals_preserve_b0_logits(inspection):
    case = _case(inspection=inspection, mixed=True)
    with torch.no_grad():
        expected = case.model.base_model(
            case.current, attention_mask=case.attention
        ).logits
        actual = _forward(case)
    torch.testing.assert_close(actual.logits, expected, atol=0, rtol=0)
    assert actual.logits.dtype == expected.dtype


def _check_task_and_budget_affect_content_and_inspection_decisions(field):
    case = _case(inspection=True)
    assert not case.context.active_token_mask.any()
    _activate_zero_outputs(case.model)
    replacement = (
        torch.ones_like(case.context.task_ids) if field == "task_ids"
        else torch.full_like(case.context.remaining_steps, 4)
    )
    with torch.no_grad():
        before = _forward(case)
        after = _forward(case, context=_context(case.context, **{field: replacement}))
    _assert_content_changed(case, before, after)
    decisions_before = torch.cat((before.mode_logits, before.quality_logits), -1)
    decisions_after = torch.cat((after.mode_logits, after.quality_logits), -1)
    assert (decisions_before - decisions_after).abs().max().item() > 1e-7


def _check_masked_cell_retains_distinction_between_invalid_old_angle_values():
    case = _case()
    _activate_zero_outputs(case.model)
    first_old = case.context.old_token_ids.clone()
    second_old = first_old.clone()
    prompt = int(case.context.prompt_lengths[0])
    # Both angle triples violate alpha + beta > gamma. Their raw values,
    # presence flags, and metric-valid flags must not collapse to one state.
    for old, gamma in ((first_old, 100), (second_old, 150)):
        for offset, (axis, value) in enumerate(
            (("AA", 30), ("AB", 30), ("AG", gamma)), start=4
        ):
            old[0, prompt + offset] = case.tokenizer.vocab[f"<{axis}_{value:03d}>"]
    assert (case.current[0, prompt + 1:prompt + 7] == case.tokenizer.mask_token_id).all()
    with torch.no_grad():
        first = _forward(case, context=_context(case.context, old_token_ids=first_old))
        second = _forward(case, context=_context(case.context, old_token_ids=second_old))
    _assert_content_changed(case, first, second, body_position=1, family="LA")


def _check_masked_content_observes_old_fractional_coordinates():
    case = _case()
    _activate_zero_outputs(case.model)
    changed_old = case.context.old_token_ids.clone()
    changed_old[0, int(case.context.prompt_lengths[0]) + 8] = case.tokenizer.vocab["<X_067>"]
    with torch.no_grad():
        first = _forward(case)
        second = _forward(case, context=_context(case.context, old_token_ids=changed_old))
    _assert_content_changed(case, first, second)


def _check_partial_current_values_reach_logits_without_token_embedding_identity(
    body_position, first_token, second_token, query_position, query_family,
):
    case = _case()
    _activate_zero_outputs(case.model)
    first_id, second_id = (case.tokenizer.vocab[name] for name in (first_token, second_token))
    with torch.no_grad():
        table = case.model.get_input_embeddings().weight
        table[second_id].copy_(table[first_id])
    position = int(case.context.prompt_lengths[0]) + body_position
    first_current, second_current = case.current.clone(), case.current.clone()
    first_current[0, position], second_current[0, position] = first_id, second_id
    # The remaining two length or XYZ values stay masked in both canvases.
    assert (first_current[0, position + 1:position + 3] == case.tokenizer.mask_token_id).all()
    context = _context(case.context, reveal_fraction=torch.tensor([1 / 12]))
    with torch.no_grad():
        base_first = case.model.base_model(first_current, attention_mask=case.attention).logits
        base_second = case.model.base_model(second_current, attention_mask=case.attention).logits
        torch.testing.assert_close(base_first, base_second, atol=0, rtol=0)
        first = _forward(case, current=first_current, context=context)
        second = _forward(case, current=second_current, context=context)
    _assert_content_changed(
        case, first, second, body_position=query_position, family=query_family
    )


def _check_content_only_learning_reaches_every_content_module_with_frozen_b0_tables(task_id):
    case = _case()
    case.context = _context(case.context, task_ids=torch.tensor([task_id]))
    prompt = int(case.context.prompt_lengths[0])
    # A deployed partial reveal: a complete current cell and one complete site
    # are visible while the second site's coordinates still need prediction.
    revealed = [*range(1, 7), 8, 9, 10]
    case.current[0, [prompt + pos for pos in revealed]] = (
        case.context.old_token_ids[0, [prompt + pos for pos in revealed]]
    )
    case.context = _context(case.context, reveal_fraction=torch.tensor([9 / 12]))
    set_editor_trainable(case.model)
    case.model.train()
    for module in [case.model.state_conditioner, *case.model.extra_modules().values()]:
        assert all(parameter.requires_grad for parameter in module.parameters())
    frozen = {}
    lora = {}
    for name, parameter in case.model.base_model.named_parameters():
        if ".lora_A." in f".{name}" or ".lora_B." in f".{name}":
            assert parameter.requires_grad, name
            lora[name] = parameter
        else:
            assert not parameter.requires_grad, name
            frozen[name] = parameter.detach().clone()
    assert lora
    optimizer = torch.optim.AdamW(
        [parameter for parameter in case.model.parameters() if parameter.requires_grad],
        lr=0.015, weight_decay=0.0,
    )
    positions = torch.tensor([prompt + pos for pos in _numeric_positions(2)])
    targets = case.context.old_token_ids[0, positions].clone()
    targets[-3] = case.tokenizer.vocab["<X_065>"]
    gradient_totals = {name: 0.0 for name in case.model.content_modules()}
    late_parameter_gradients = {
        f"{module_name}.{parameter_name}": 0.0
        for module_name, module in case.model.content_modules().items()
        for parameter_name, _ in module.named_parameters()
    }
    lora_gradients = {name: 0.0 for name in lora}
    for update in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = _forward(case)
        loss = nn.functional.cross_entropy(output.logits[0, positions].float(), targets)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in case.model.named_parameters():
            assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name
        for name, module in case.model.content_modules().items():
            gradient_totals[name] += sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters() if parameter.grad is not None
            )
            if update >= 1:
                for parameter_name, parameter in module.named_parameters():
                    if parameter.grad is not None:
                        late_parameter_gradients[f"{name}.{parameter_name}"] += (
                            float(parameter.grad.abs().sum())
                        )
        for name, parameter in lora.items():
            if parameter.grad is not None:
                lora_gradients[name] += float(parameter.grad.abs().sum())
        optimizer.step()
    assert all(value > 0 for value in gradient_totals.values()), gradient_totals
    missing_upstream_gradients = [
        name for name, value in late_parameter_gradients.items() if value <= 0
    ]
    assert not missing_upstream_gradients, missing_upstream_gradients
    assert all(value > 0 for value in lora_gradients.values()), lora_gradients
    for name, before in frozen.items():
        parameter = dict(case.model.base_model.named_parameters())[name]
        assert parameter.grad is None, name
        torch.testing.assert_close(parameter, before, atol=0, rtol=0)


def _check_decision_shapes_and_padding_are_isolated_across_mixed_length_rows():
    case = _case(inspection=True, mixed=True)
    _activate_zero_outputs(case.model)
    with torch.no_grad():
        output = _forward(case)
    assert output.logits.shape == (*case.current.shape, len(case.tokenizer))
    for name in ("mode_logits", "count_logits", "quality_logits"):
        value = getattr(output, name)
        assert value.shape == (2, 4)
        assert value.dtype == torch.float32 and torch.isfinite(value).all()
    assert output.site_logits.shape == (2, 20)
    assert output.site_logits.dtype == torch.float32
    for row, count in enumerate(case.context.num_sites.tolist()):
        assert torch.isfinite(output.site_logits[row, :count]).all()
        assert torch.isfinite(output.site_logits[row, count:]).all()
        assert (output.site_logits[row, count:] <= -1e3).all()
        single_context = EditContext(**{
            name: getattr(case.context, name)[row:row + 1]
            for name in case.context.__dataclass_fields__
        })
        with torch.no_grad():
            single = case.model(
                case.current[row:row + 1], attention_mask=case.attention[row:row + 1],
                edit_context=single_context,
            )
        for name in ("logits", "mode_logits", "site_logits", "count_logits", "quality_logits"):
            torch.testing.assert_close(
                getattr(output, name)[row:row + 1], getattr(single, name),
                atol=2e-6, rtol=2e-5,
            )
    changed_current = case.current.clone()
    changed_old = case.context.old_token_ids.clone()
    padding = case.attention == 0
    assert padding.any()
    changed_current[padding] = case.tokenizer.vocab["<X_099>"]
    changed_old[padding] = case.tokenizer.vocab["<LA_001>"]
    with torch.no_grad():
        changed = _forward(
            case, current=changed_current,
            context=_context(case.context, old_token_ids=changed_old),
        )
    torch.testing.assert_close(
        output.logits[case.attention.bool()], changed.logits[case.attention.bool()],
        atol=0, rtol=0,
    )
    for name in ("mode_logits", "site_logits", "count_logits", "quality_logits"):
        torch.testing.assert_close(getattr(output, name), getattr(changed, name), atol=0, rtol=0)


def _ddp_audit_worker(rank, rendezvous):
    """Exercise different label availability on two ranks, without any GPU."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        case = _case(mixed=True)
        set_editor_trainable(case.model)
        learner = DistributedDataParallel(
            case.model, broadcast_buffers=False, static_graph=True
        ).train()
        context = EditContext(**{
            name: getattr(case.context, name)[rank:rank + 1]
            for name in case.context.__dataclass_fields__
        })
        context = _context(context, task_ids=torch.tensor([rank]))
        attention = case.attention[rank:rank + 1]
        parameters = [parameter for parameter in case.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=0.003)
        objective = ExpertEditObjective(case.tokenizer, torch.device("cpu"))
        count = int(context.num_sites[0])
        prompt = int(context.prompt_lengths[0])
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                kind = (step + rank + micro) % 3  # content, inspect, judge
                current = (
                    case.current[rank:rank + 1] if kind == 0
                    else context.old_token_ids
                )
                active = (
                    torch.zeros_like(context.active_token_mask) if kind == 1
                    else context.active_token_mask
                )
                view_context = _context(
                    context, active_token_mask=active,
                    reveal_fraction=torch.tensor([float(kind != 0)]),
                )
                targets = torch.full_like(current, -100)
                mode_targets = torch.tensor([-100])
                count_targets = torch.tensor([-100])
                site_targets = torch.full((1, 20), -1.0)
                quality_targets = torch.zeros(1, 4)
                quality_mask = torch.zeros(1, 4, dtype=torch.bool)
                if kind == 0:
                    positions = [prompt + pos for pos in _numeric_positions(count)]
                    targets[0, positions] = context.old_token_ids[0, positions]
                    targets[0, prompt + 8] = case.tokenizer.vocab["<X_065>"]
                elif kind == 1:
                    mode_targets[0], count_targets[0] = 1, 0
                    site_targets[0, :count] = 0
                    site_targets[0, 0] = 1
                    quality_targets[0, 0], quality_mask[0, 0] = 1, True
                else:
                    quality_targets[0, 3] = float(rank == 0)
                    quality_mask[0, 3] = True
                batch = {
                    "targets": targets, "edit_context": view_context,
                    "mode_targets": mode_targets, "count_targets": count_targets,
                    "site_targets": site_targets, "quality_targets": quality_targets,
                    "quality_mask": quality_mask,
                }
                output = learner(current, attention_mask=attention, edit_context=view_context)
                loss, _ = objective(output, batch)
                assert torch.isfinite(loss)
                (loss / 2).backward()
            assert all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in parameters
            )
            optimizer.step()
        assert case.model.base_model.encoder_invocations > case.model.forward_calls
        # All-reduced gradients must leave the two replicas identical despite
        # different atom counts and different available labels on each rank.
        for parameter in parameters:
            expected = parameter.detach().clone()
            dist.broadcast(expected, src=0)
            torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


class TestExpertEditModel(unittest.TestCase):
    def test_zero_residuals_preserve_masked_b0_logits(self):
        _check_zero_residuals_preserve_b0_logits(inspection=False)

    def test_zero_residuals_preserve_inspection_b0_logits(self):
        _check_zero_residuals_preserve_b0_logits(inspection=True)

    def test_task_affects_content_and_decisions_without_an_active_scope(self):
        _check_task_and_budget_affect_content_and_inspection_decisions("task_ids")

    def test_budget_affects_content_and_decisions_without_an_active_scope(self):
        _check_task_and_budget_affect_content_and_inspection_decisions("remaining_steps")

    def test_invalid_old_raw_cells_remain_distinguishable_after_remasking(self):
        _check_masked_cell_retains_distinction_between_invalid_old_angle_values()

    def test_old_fractional_coordinates_affect_masked_content(self):
        _check_masked_content_observes_old_fractional_coordinates()

    def test_partial_current_lattice_is_numeric_conditioning(self):
        _check_partial_current_values_reach_logits_without_token_embedding_identity(
            1, "<LA_020>", "<LA_060>", 2, "LB",
        )

    def test_partial_current_xyz_is_numeric_conditioning(self):
        _check_partial_current_values_reach_logits_without_token_embedding_identity(
            8, "<X_025>", "<X_075>", 9, "Y",
        )

    def test_g_content_loss_reaches_all_content_modules(self):
        _check_content_only_learning_reaches_every_content_module_with_frozen_b0_tables(0)

    def test_s_content_loss_reaches_all_content_modules(self):
        _check_content_only_learning_reaches_every_content_module_with_frozen_b0_tables(1)

    def test_mixed_batch_shapes_and_padding_isolation(self):
        _check_decision_shapes_and_padding_are_isolated_across_mixed_length_rows()

    def test_bfloat16_base_preserves_numeric_constants_and_float32_decisions(self):
        case = _case(inspection=True)
        _activate_zero_outputs(case.model)
        before = case.model.geometry_values.detach().clone()
        case.model.bfloat16()
        torch.testing.assert_close(
            case.model.geometry_values, before, atol=0, rtol=0, equal_nan=True
        )
        assert case.model.geometry_values.dtype == torch.float32
        with torch.no_grad():
            output = _forward(case)
        assert output.logits.dtype == torch.bfloat16
        assert torch.isfinite(output.logits).all()
        for name in ("mode_logits", "site_logits", "count_logits", "quality_logits"):
            value = getattr(output, name)
            assert value.dtype == torch.float32 and torch.isfinite(value).all()

    def test_static_graph_ddp_with_checkpointing_and_mixed_label_availability(self):
        import torch.distributed as dist
        import torch.multiprocessing as mp

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("CPU Gloo backend is unavailable")
        with tempfile.TemporaryDirectory(prefix="expert_edit_ddp_") as directory:
            rendezvous = (Path(directory) / "rendezvous").resolve().as_uri()
            mp.spawn(_ddp_audit_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
