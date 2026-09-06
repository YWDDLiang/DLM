from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from crystal_dlm.periodic_repair_initialization import (
    INITIALIZATION_MARKER, INITIALIZATION_SCHEMA, FreshPeriodicRepairDLM,
    extend_raw_crystal_vocabulary, initialize_fresh_periodic_repair_model,
    load_fresh_periodic_repair_model, set_fresh_repair_trainable,
)
from crystal_dlm.periodic_repair_model import PeriodicRepairConfig
from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
from crystal_dlm.state_conditioned_model import context_from_programs
from test_state_programmed_runtime import body, program


class RawTokenizer:
    def __init__(self):
        self.vocab = {token: index for index, token in enumerate(
            ("<PAD>", "<EOS>", "<MASK>", "hydrogen", "oxygen", "1", "0", "crystal"),
        )}
        self.pad_token_id, self.eos_token_id, self.mask_token_id = 0, 1, 2
        self.mask_id = self.mask_token_id

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def add_special_tokens(self, spec):
        added = 0
        for token in spec["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added


class RawBase(nn.Module):
    """Small checkpointed backbone; no Transformers or PEFT installation needed."""

    def __init__(self, size, hidden=16):
        super().__init__()
        self.embedding = nn.Embedding(size, hidden)
        self.output = nn.Linear(hidden, size, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(hidden, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, hidden, bias=False)})
        self.lora_dropout = nn.Dropout(.05)
        nn.init.zeros_(self.lora_B["default"].weight)
        self.config = SimpleNamespace(hidden_size=hidden, vocab_size=size)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output

    def set_input_embeddings(self, value):
        self.embedding = value

    def set_output_embeddings(self, value):
        self.output = value

    def forward(self, input_ids=None, *, inputs_embeds=None, attention_mask=None,
                attention_bias=None, output_hidden_states=False):
        value = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        if attention_bias is None:
            attention_bias = value.new_zeros(value.shape[0], 1, value.shape[1], value.shape[1])

        def block(hidden, bias):
            hidden = hidden + self.lora_B["default"](
                self.lora_A["default"](self.lora_dropout(hidden)),
            )
            weights = torch.softmax(hidden @ hidden.transpose(-1, -2) / 4 + bias[:, 0], -1)
            return hidden + weights @ hidden

        hidden = checkpoint(block, value, attention_bias, use_reentrant=False)
        return SimpleNamespace(logits=self.output(hidden),
                               hidden_states=(hidden,) if output_hidden_states else None)

    def save_pretrained(self, path, **kwargs):
        assert kwargs["save_embedding_layers"] is False
        torch.save({key: value for key, value in self.state_dict().items()
                    if "lora_A" in key or "lora_B" in key}, path / "toy_adapter.pt")


def case():
    torch.manual_seed(4001)
    tokenizer = RawTokenizer()
    # Raw embedding padding is deliberately larger than the tokenizer. The
    # first new token IDs reuse these rows and must still be initialized/trained.
    base = RawBase(len(tokenizer) + 5)
    original_input = base.embedding.weight[:len(tokenizer)].detach().clone()
    original_output = base.output.weight[:len(tokenizer)].detach().clone()
    extension = extend_raw_crystal_vocabulary(base, tokenizer)
    state = PeriodicStateConfig(16, width=12, radial_basis_count=4)
    repair = PeriodicRepairConfig(16, width=12)
    initialization = {"schema": INITIALIZATION_SCHEMA, **extension,
                      "state_config": asdict(state), "repair_config": asdict(repair),
                      "legacy_dlm_checkpoint": None, "pretrained_vocabulary_rows_frozen": True}
    original = torch.tensor([[0] + body(tokenizer, length=40, x=35)])
    positions = [*range(1, 7), 8, 9, 10, 12, 13, 14]
    context = context_from_programs(
        original, prompt_length=1, num_sites=2, programs=[program()],
        active_positions={0: positions}, task_id=3,
    )
    current = original.clone()
    current[:, [pos + 1 for pos in positions]] = tokenizer.mask_id
    model = FreshPeriodicRepairDLM(base, tokenizer, state, repair, initialization)
    return tokenizer, model, current, context, original_input, original_output


def test_mean_extension_preserves_pretrained_rows_and_reinitializes_padding_ids():
    _, model, _, _, before_input, before_output = case()
    count = before_input.shape[0]
    torch.testing.assert_close(model.get_input_embeddings().weight[:count], before_input, atol=0, rtol=0)
    torch.testing.assert_close(model.get_output_embeddings().weight[:count], before_output, atol=0, rtol=0)
    for token in model.raw_initialization["new_crystal_token_ids"]:
        torch.testing.assert_close(model.get_input_embeddings().weight[token], before_input.mean(0), atol=0, rtol=0)
        torch.testing.assert_close(model.get_output_embeddings().weight[token], before_output.mean(0), atol=0, rtol=0)


def test_fresh_zero_increment_matches_expanded_base_and_preserves_dropout():
    _, model, current, context, _, _ = case()
    counts = set_fresh_repair_trainable(model)
    assert counts["new_token_rows"] == 2 * len(model.new_token_rows.token_ids) * 16
    assert model.base_model.lora_dropout.p == .05
    model.eval()
    expected = model.base_model(current).logits
    actual = model(current, geometry_context=context).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_new_input_and_output_rows_learn_without_changing_pretrained_tables():
    tokenizer, model, current, context, before_input, before_output = case()
    set_fresh_repair_trainable(model)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(current, geometry_context=context).logits
        loss = nn.functional.cross_entropy(logits[:, 9], torch.tensor([tokenizer.vocab["<X_030>"]]))
        loss.backward()
        assert model.new_token_rows.input_delta.grad.abs().sum() > 0
        assert model.new_token_rows.output_delta.grad.abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    count = before_input.shape[0]
    assert not model.get_input_embeddings().weight.requires_grad
    assert not model.get_output_embeddings().weight.requires_grad
    torch.testing.assert_close(model.get_input_embeddings().weight[:count], before_input, atol=0, rtol=0)
    torch.testing.assert_close(model.get_output_embeddings().weight[:count], before_output, atol=0, rtol=0)
    assert model.new_token_rows.input_delta.abs().sum() > 0
    assert model.new_token_rows.output_delta.abs().sum() > 0
    assert model.base_model.lora_dropout.p == .05


def test_fresh_checkpoint_roundtrip_retains_both_new_rows_and_periodic_modules(tmp_path):
    tokenizer, model, current, context, _, _ = case()
    original_base = deepcopy(model.base_model)
    with torch.no_grad():
        for module in (model.new_token_rows, model.state_conditioner,
                       model.numeric_adapter, model.geometry_attention):
            for parameter in module.parameters():
                parameter.add_(torch.randn_like(parameter) * .001)
        model.base_model.lora_B["default"].weight.normal_(0, .01)
    model.eval()
    expected = model(current, geometry_context=context).logits
    model.save_pretrained(tmp_path)
    recorded = json.loads((tmp_path / INITIALIZATION_MARKER).read_text())
    assert recorded["schema"] == INITIALIZATION_SCHEMA
    partition = torch.load(tmp_path / "periodic_repair.pt", weights_only=True)
    assert "new_token_rows" in partition
    original_base.load_state_dict(torch.load(tmp_path / "toy_adapter.pt", weights_only=True), strict=False)
    restored = FreshPeriodicRepairDLM(original_base, tokenizer, model.state_config,
                                     model.repair_config, recorded)
    restored.load_state_conditioner(tmp_path)
    restored.load_repair(tmp_path)
    restored.eval()
    torch.testing.assert_close(restored(current, geometry_context=context).logits, expected, atol=0, rtol=0)


def test_new_row_parameters_remain_fp32_after_parent_conversion():
    _, model, _, _, _, _ = case()
    model.bfloat16()
    assert model.new_token_rows.input_delta.dtype == torch.float32
    assert model.new_token_rows.output_delta.dtype == torch.float32


def test_legacy_checkpoint_is_rejected_before_loading_transformers(tmp_path):
    with pytest.raises(ValueError, match="legacy DLM is forbidden"):
        load_fresh_periodic_repair_model("unused", tmp_path, "cpu")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="adapted DLM checkpoint"):
        initialize_fresh_periodic_repair_model(tmp_path, "cpu")


def test_already_extended_tokenizer_cannot_be_used_as_raw_source():
    tokenizer, model, _, _, _, _ = case()
    with pytest.raises(ValueError, match="refusing an adapted source"):
        extend_raw_crystal_vocabulary(model.base_model, tokenizer)
