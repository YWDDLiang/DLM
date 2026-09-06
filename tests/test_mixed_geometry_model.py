"""Small CPU modules exercise the actual inspected LLaDA forward contract.

No pretrained weights are loaded.  The forward oracle is the unedited method
recorded in LLADA_ACTUAL_CORE_FORWARD.json, executed on miniature torch blocks;
the production hidden-only implementation is not used as its own reference.

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
    -s tests -p test_mixed_geometry_model.py -v
"""

from copy import deepcopy
from dataclasses import asdict, replace
from enum import Enum
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is not None:
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    from crystal_dlm.fixed_slot import build_special_tokens
    from crystal_dlm.llada_hidden import llada_hidden_forward, resolve_llada_core
    from crystal_dlm.mixed_geometry_diffusion import GeometryState, LatticeNormalizer, geometry_denoising_risk
    from crystal_dlm import mixed_geometry_model as mixed
    from crystal_dlm.periodic_repair_initialization import INITIALIZATION_SCHEMA
    from crystal_dlm.periodic_repair_model import PeriodicRepairConfig
    from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
    from crystal_dlm import periodic_v2_initialization as v2_initialization
    from crystal_dlm.periodic_v2_initialization import PeriodicV2DLM
    from crystal_dlm.periodic_v2_model import PeriodicV2Config
    from crystal_dlm.state_conditioned_model import CrystalStateContext

    class Strategy(str, Enum):
        whole_layer = "whole_layer"
        one_in_two = "one_in_two"
        one_in_three = "one_in_three"
        one_in_four = "one_in_four"

    def reference_ensure_finite(value, *, check_neg_inf, check_pos_inf):
        if check_neg_inf:
            value.masked_fill_(torch.isneginf(value), torch.finfo(value.dtype).min)
        if check_pos_inf:
            value.masked_fill_(torch.isposinf(value), torch.finfo(value.dtype).max)

    def original_forward_oracle():
        evidence = (Path(__file__).resolve().parents[1] / "docs/v3_scientific_audit_20260906"
                    / "evidence/LLADA_ACTUAL_CORE_FORWARD.json")
        record = json.loads(evidence.read_text(encoding="utf-8"))
        source = record["forwards"]["LLaDAModel"]["forward"]["source"]
        namespace = {"torch": torch, "math": math, "F": F,
                     "ActivationCheckpointingStrategy": Strategy,
                     "ensure_finite_": reference_ensure_finite, "LLaDAOutput": SimpleNamespace}
        exec(compile("from __future__ import annotations\n" + source, str(evidence), "exec"), namespace)
        return namespace["forward"]

    class CountingLinear(nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return super().forward(value)

    class SmallLoRALinear(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.base = nn.Linear(width, width, bias=False)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(width, 2, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(2, width, bias=False)})
            nn.init.normal_(self.lora_B["default"].weight, std=0.03)

        def forward(self, value):
            return self.base(value) + self.lora_B["default"](self.lora_A["default"](value))

    class SmallBlock(nn.Module):
        def __init__(self, width, heads):
            super().__init__()
            self.heads = heads
            self.q_proj, self.k_proj, self.v_proj = [SmallLoRALinear(width) for _ in range(3)]
            self.ff_proj = SmallLoRALinear(width)

        def forward(self, value, *, attention_bias=None, layer_past=None, use_cache=False):
            assert layer_past is None and use_cache is False
            batch, length, width = value.shape
            q, k, v = [projection(value).view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                       for projection in (self.q_proj, self.k_proj, self.v_proj)]
            scores = q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)
            if attention_bias is not None:
                scores = scores + attention_bias
            attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, width)
            return value + 0.2 * attended + 0.1 * self.ff_proj(value.tanh()), None

    class SmallBlockGroup(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.blocks = nn.ModuleList(blocks)

        def forward(self, value, *, attention_bias=None, layers_past=None, use_cache=False):
            assert layers_past is None
            for block in self.blocks:
                value, _ = block(value, attention_bias=attention_bias, use_cache=use_cache)
            return value, None

    class SmallCore(nn.Module):
        def __init__(self, vocabulary, width=16, heads=4, group_size=1):
            super().__init__()
            self.config = SimpleNamespace(d_model=width, n_heads=heads, n_layers=2, alibi=False,
                                          rope=True, input_emb_norm=False, block_group_size=group_size,
                                          weight_tying=False, scale_logits=False, use_cache=False)
            self.transformer = nn.ModuleDict({
                "wte": nn.Embedding(vocabulary, width), "emb_drop": nn.Dropout(0.0),
                "ln_f": nn.LayerNorm(width), "ff_out": CountingLinear(width, vocabulary, bias=False)})
            blocks = [SmallBlock(width, heads) for _ in range(2)]
            if group_size == 1:
                self.transformer["blocks"] = nn.ModuleList(blocks)
            else:
                self.transformer["block_groups"] = nn.ModuleList([SmallBlockGroup(blocks)])
            self.activation_checkpointing_strategy = None
            self.checkpoint_calls = 0

        def get_bidirectional_attention_bias(self, length, device):
            return torch.zeros(1, 1, length, length, dtype=torch.float32, device=device)

        def set_activation_checkpointing(self, strategy):
            self.activation_checkpointing_strategy = Strategy(strategy)

        def _activation_checkpoint_fn(self, block, value, **kwargs):
            self.checkpoint_calls += 1
            return checkpoint(block, value, use_reentrant=False, **kwargs)

    # Use the inspected source, not a reimplementation of production hidden code.
    SmallCore.forward = original_forward_oracle()

    class SmallLM(nn.Module):
        def __init__(self, vocabulary, group_size=1):
            super().__init__()
            self.model = SmallCore(vocabulary, group_size=group_size)
            self.config = self.model.config

        def get_input_embeddings(self):
            return self.model.transformer.wte

        def get_output_embeddings(self):
            return self.model.transformer.ff_out

        def forward(self, input_ids=None, *, inputs_embeds=None, **kwargs):
            return self.model(input_ids, input_embeddings=inputs_embeds, **kwargs)

        def save_pretrained(self, output_dir, **kwargs):
            assert kwargs["save_embedding_layers"] is False
            root = Path(output_dir)
            torch.save(self.state_dict(), root / "adapter_model.bin")
            (root / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))

    class SmallTokenizer:
        def __init__(self):
            tokens = ["<PAD>", "<EOS>", "<MASK>", "context", *build_special_tokens()]
            self.vocab = {token: index for index, token in enumerate(tokens)}
            self.mask_token_id = 2

        def get_vocab(self):
            return dict(self.vocab)

    def parent_provenance():
        return {"schema": "mixed_geometry_parent_v1", "parent_kind": "completed_periodic_v2",
                "checkpoint_path": "toy_parent",
                "files_sha256": {name: "0" * 64 for name in
                                  ("CHECKPOINT_FINAL.json", "periodic_v2_config.json",
                                   "raw_periodic_initialization.json", "adapter_config.json")}}

    def case():
        torch.manual_seed(912)
        tokenizer = SmallTokenizer()
        lm = SmallLM(len(tokenizer.vocab))
        state_config = PeriodicStateConfig(16, width=8, max_sites=3, radial_basis_count=4, image_radius=1)
        repair_config = PeriodicRepairConfig(16, width=8, radial_bins=4, image_radius=1)
        v2_config = PeriodicV2Config(16, heads=4, width=8, max_sites=3, radial_bins=4, image_radius=1)
        initialization = {"schema": INITIALIZATION_SCHEMA, "new_crystal_token_ids": list(range(4, len(tokenizer.vocab))),
                          "state_config": asdict(state_config), "repair_config": asdict(repair_config),
                          "legacy_dlm_checkpoint": None, "pretrained_vocabulary_rows_frozen": True}
        token_model = PeriodicV2DLM(lm, tokenizer, state_config, repair_config, initialization, v2_config)
        with torch.no_grad():
            for parameter in token_model.state_conditioner.parameters():
                parameter.add_(0.005 * torch.randn_like(parameter))
            for parameter in token_model.geometry_attention.parameters():
                parameter.add_(0.005 * torch.randn_like(parameter))
            token_model.new_token_rows.input_delta.normal_(std=0.002)
            token_model.new_token_rows.output_delta.normal_(std=0.002)
        vocabulary = tokenizer.vocab
        body = [vocabulary["<N_002>"], *[vocabulary[f"<L{axis}_040>"] for axis in "ABC"],
                *[vocabulary[f"<A{axis}_090>"] for axis in "ABG"],
                vocabulary["<E_Li>"], vocabulary["<X_020>"], vocabulary["<Y_030>"], vocabulary["<Z_040>"],
                vocabulary["<E_O>"], vocabulary["<X_070>"], vocabulary["<Y_060>"], vocabulary["<Z_050>"]]
        # Two left-padded prompt tokens plus one visible context token.
        old = torch.tensor([[0, 0, 3, *body, 0]])
        prompt = torch.tensor([3])
        numeric = torch.zeros_like(old, dtype=torch.bool)
        for position in (*range(1, 7), 8, 9, 10, 12, 13, 14):
            numeric[0, 3 + position] = True
        scaffold = old.clone()
        scaffold[numeric] = tokenizer.mask_token_id
        mask = torch.tensor([[0, 0, *([1] * (len(body) + 1)), 0]], dtype=torch.long)
        context = CrystalStateContext(old, prompt, torch.tensor([2]), torch.tensor([[0, 1, 3]]), numeric,
                                      task_ids=torch.tensor([3]), numeric_noise_level=torch.tensor([-1.0]))
        lattices = torch.stack((torch.eye(3, dtype=torch.float64) * 3, torch.eye(3, dtype=torch.float64) * 5))
        normalizer = LatticeNormalizer.fit(lattices, torch.tensor([2, 2]))
        actual_lattice = torch.eye(3, dtype=torch.float64)[None] * 4
        geometry = GeometryState(normalizer.encode(actual_lattice, 2),
                                 torch.tensor([[[0.2, 0.3, 0.4], [0.7, 0.6, 0.5], [float("nan")]*3]], dtype=torch.float64),
                                 torch.tensor([0.2], dtype=torch.float64), torch.tensor([[True, True, False]]))
        species = torch.tensor([[3, 8, 0]])
        model = mixed.MixedGeometryDLM(token_model, tokenizer, normalizer,
                                       parent_provenance=parent_provenance()).eval()
        return model, tokenizer, scaffold, mask, context, geometry, species


@unittest.skipUnless(torch is not None, "torch unavailable: model tensor tests not executed")
class MixedGeometryModelTests(unittest.TestCase):
    def test_hidden_matches_actual_forward_without_calling_vocabulary_head(self):
        torch.manual_seed(31)
        for grouped in (False, True):
            lm = SmallLM(23, group_size=2 if grouped else 1).eval()
            lm.config.input_emb_norm = True
            embeddings = torch.randn(2, 7, 16)
            mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0]])
            bias = torch.randn(2, 4, 7, 7) * 0.03
            original_bias = bias.clone()
            with torch.no_grad():
                expected = lm(inputs_embeds=embeddings, attention_mask=mask, attention_bias=bias,
                              output_hidden_states=True).hidden_states[-1]
                calls = lm.get_output_embeddings().calls
                actual = llada_hidden_forward(lm, inputs_embeds=embeddings, attention_mask=mask, attention_bias=bias)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(bias, original_bias, atol=0, rtol=0)
            self.assertEqual(lm.get_output_embeddings().calls, calls)
            self.assertIs(resolve_llada_core(lm), lm.model)

    def test_hidden_bool_bias_negative_infinity_and_padding_match_reference(self):
        lm = SmallLM(13).eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.tensor([[0, 1, 1, 1]])
        for bias in (None, torch.ones(1, 1, 4, 4, dtype=torch.bool),
                     torch.full((1, 1, 4, 4), torch.finfo(torch.float32).min)):
            expected = lm(ids, attention_mask=mask, attention_bias=bias, output_hidden_states=True).hidden_states[-1]
            actual = llada_hidden_forward(lm, ids, attention_mask=mask, attention_bias=bias)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_native_checkpoint_path_preserves_input_bias_and_lora_gradients(self):
        lm = SmallLM(17).train()
        lm.model.set_activation_checkpointing("whole_layer")
        embeddings = torch.randn(2, 5, 16, requires_grad=True)
        bias = (torch.randn(2, 4, 5, 5) * 0.01).requires_grad_()
        original = lm(inputs_embeds=embeddings, attention_bias=bias, output_hidden_states=True).hidden_states[-1]
        weights = torch.randn_like(original)
        parameters = [embeddings, bias, *[p for name, p in lm.named_parameters() if "lora_" in name]]
        expected_grads = torch.autograd.grad((original * weights).sum(), parameters)
        before = lm.model.checkpoint_calls
        hidden = llada_hidden_forward(lm, inputs_embeds=embeddings, attention_bias=bias)
        actual_grads = torch.autograd.grad((hidden * weights).sum(), parameters)
        self.assertEqual(lm.model.checkpoint_calls - before, 2)
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            self.assertGreater(float(actual.abs().sum()), 0)

    def test_token_route_is_the_same_v2_object_and_exact_output(self):
        model, _, ids, mask, context, _, _ = case()
        with torch.no_grad():
            expected = model.token_model(ids, attention_mask=mask, geometry_context=context, output_hidden_states=True)
            actual = model(ids, attention_mask=mask, mode="token", geometry_context=context, output_hidden_states=True)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        torch.testing.assert_close(actual.hidden_states[-1], expected.hidden_states[-1], atol=0, rtol=0)
        self.assertIs(model.base_model, model.token_model.base_model)
        ids_of_parameters = [id(parameter) for parameter in model.parameters()]
        self.assertEqual(len(ids_of_parameters), len(set(ids_of_parameters)))
        self.assertFalse(any(name.startswith("base_model.") for name, _ in model.named_parameters()))

    def test_geometry_bypasses_old_decoder_task_numeric_rows_and_vocab_projection(self):
        model, _, ids, mask, context, state, species = case()
        forbidden = AssertionError("a token-only path was called by geometry")
        with patch.object(model.token_model, "geometry_inputs", side_effect=forbidden), \
             patch.object(model.token_model.repair_task_projection, "forward", side_effect=forbidden), \
             patch.object(model.token_model.numeric_adapter, "forward", side_effect=forbidden), \
             patch.object(model.new_token_rows, "add_output_increments", side_effect=forbidden), \
             patch.object(model.get_output_embeddings(), "forward", side_effect=forbidden):
            output = model(ids, mask, mode="geometry", geometry_context=context, geometry_state=state,
                           species=species, output_hidden_states=True)
        self.assertFalse(hasattr(output, "logits"))
        self.assertEqual(output.v_prediction.shape, (1, 6))
        self.assertEqual(output.u_prediction.shape, (1, 3, 3))
        self.assertEqual(float((output.v_prediction.abs().sum() + output.u_prediction.abs().sum()).detach()), 0)
        self.assertTrue(output.hidden_states[-1].requires_grad)

    def test_float_state_changes_hidden_while_old_numeric_tokens_are_ignored(self):
        model, _, ids, mask, context, state, species = case()
        baseline = model(ids, mask, mode="geometry", geometry_context=context, geometry_state=state,
                         species=species, output_hidden_states=True).hidden_states[-1]
        different_old = replace(context, old_token_ids=torch.zeros_like(context.old_token_ids))
        same = model(ids, mask, mode="geometry", geometry_context=different_old, geometry_state=state,
                     species=species, output_hidden_states=True).hidden_states[-1]
        torch.testing.assert_close(same, baseline, atol=0, rtol=0)
        changed_f = state.fractional.clone()
        changed_f[0, 1, 0] += 0.03
        changed = replace(state, fractional=changed_f)
        new_hidden = model(ids, mask, mode="geometry", geometry_context=context, geometry_state=changed,
                           species=species, output_hidden_states=True).hidden_states[-1]
        self.assertGreater(float((new_hidden-baseline).detach().abs().max()), 1e-7)

    def test_time_residual_is_added_only_to_the_actual_body(self):
        model, _, ids, mask, context, state, species = case()
        captured = []
        def capture_hidden(_base, **kwargs):
            captured.append(kwargs["inputs_embeds"].detach().clone())
            return kwargs["inputs_embeds"]
        with patch.object(mixed, "llada_hidden_forward", side_effect=capture_hidden):
            for time in (0.2, 0.8):
                model(ids, mask, mode="geometry", geometry_context=context,
                      geometry_state=replace(state, t=torch.tensor([time])), species=species)
        difference = captured[1] - captured[0]
        self.assertEqual(float(difference[:, :3].abs().sum()), 0)
        self.assertEqual(float(difference[:, -1].abs().sum()), 0)
        self.assertGreater(float(difference[:, 3:-1].abs().sum()), 0)

    def test_peft_style_resolution_keeps_the_original_core_and_parameters(self):
        lm = SmallLM(17)
        class PeftStyle(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base_model = base
            def get_base_model(self):
                return self.base_model
        wrapper = PeftStyle(lm)
        self.assertIs(resolve_llada_core(wrapper), lm.model)
        ids = torch.tensor([[1, 2, 3]])
        expected = llada_hidden_forward(lm, ids)
        actual = llada_hidden_forward(wrapper, ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_two_gradient_steps_open_shared_modules_and_leave_token_only_params_unused(self):
        model, _, ids, mask, context, state, species = case()
        counts = mixed.set_mixed_geometry_trainable(model)
        self.assertEqual(counts["geometry_heads"], 9*(16+1) + 64*128+128 + 128*16+16 + 16)
        model.train()
        model.base_model.model.set_activation_checkpointing("whole_layer")
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.02)
        for index in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(ids, mask, mode="geometry", geometry_context=context, geometry_state=state, species=species)
            risk = geometry_denoising_risk(output.v_prediction, output.u_prediction,
                                           torch.ones_like(output.v_prediction), torch.full_like(output.u_prediction, 0.4), state.atom_mask)
            risk.total.backward()
            self.assertGreater(float(model.geometry_heads.v_head.weight.grad.abs().sum()), 0)
            self.assertIsNone(model.new_token_rows.output_delta.grad)
            self.assertTrue(all(p.grad is None for p in model.token_model.numeric_adapter.parameters()))
            self.assertTrue(all(p.grad is None for p in model.token_model.repair_task_projection.parameters()))
            if index == 1:
                for parameters in ([p for name, p in model.base_model.named_parameters() if "lora_" in name],
                                   list(model.state_conditioner.parameters()), list(model.geometry_attention.parameters()),
                                   list(model.geometry_heads.time_mlp.parameters()), [model.new_token_rows.input_delta]):
                    total = sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)
                    self.assertGreater(total, 0)
            self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
            optimizer.step()

    def test_geometry_layout_rejects_unmasked_numeric_and_wrong_hard_species(self):
        model, _, ids, mask, context, state, species = case()
        with self.assertRaisesRegex(ValueError, "all be MASK"):
            model(context.old_token_ids, mask, mode="geometry", geometry_context=context, geometry_state=state, species=species)
        wrong = species.clone()
        wrong[0, 0] = 4
        with self.assertRaisesRegex(ValueError, "hard E scaffold"):
            model(ids, mask, mode="geometry", geometry_context=context, geometry_state=state, species=wrong)
        with self.assertRaisesRegex(ValueError, "continuous geometry"):
            model(ids, mask, mode="token", geometry_context=context, geometry_state=state)

    def test_new_heads_keep_fp32_when_parent_is_converted(self):
        model, _, _, _, _, _, _ = case()
        model.bfloat16()
        self.assertTrue(all(p.dtype == torch.float32 for p in model.geometry_heads.parameters()))
        self.assertEqual(model.geometry_heads.time_frequencies.dtype, torch.float32)

    def test_mixed_save_load_roundtrip_preserves_t_g_normalizer_and_parent(self):
        model, tokenizer, ids, mask, context, state, species = case()
        with torch.no_grad():
            model.geometry_heads.v_head.weight.normal_(std=0.01)
            model.geometry_heads.u_head.weight.normal_(std=0.01)
            expected_t = model(ids, mask, geometry_context=context).logits
            expected_g = model(ids, mask, mode="geometry", geometry_context=context, geometry_state=state, species=species)
        template = deepcopy(model.token_model)
        def reconstruct(_model_path, root, device, **kwargs):
            rebuilt = deepcopy(template)
            rebuilt.base_model.load_state_dict(torch.load(Path(root)/"adapter_model.bin", map_location="cpu", weights_only=True))
            rebuilt.load_state_conditioner(root)
            rebuilt.load_repair(root)
            return rebuilt.to(device), tokenizer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "arbitrary_epoch_checkpoint"
            model.save_pretrained(root)
            self.assertFalse((root / "CHECKPOINT_FINAL.json").exists())
            with patch.object(mixed, "load_periodic_v2_architecture", side_effect=reconstruct), \
                 patch.object(mixed, "load_periodic_v2_model", side_effect=AssertionError("mixed load must not select the V2 policy")):
                loaded, loaded_tokenizer = mixed.load_mixed_geometry_model("unused", root, "cpu")
            self.assertIs(loaded_tokenizer, tokenizer)
            self.assertEqual(loaded.normalizer.to_dict(), model.normalizer.to_dict())
            self.assertEqual(loaded.parent_provenance, model.parent_provenance)
            torch.testing.assert_close(loaded(ids, mask, geometry_context=context).logits, expected_t, atol=0, rtol=0)
            actual_g = loaded(ids, mask, mode="geometry", geometry_context=context, geometry_state=state, species=species)
            torch.testing.assert_close(actual_g.v_prediction, expected_g.v_prediction, atol=0, rtol=0)
            torch.testing.assert_close(actual_g.u_prediction, expected_g.u_prediction, atol=0, rtol=0)
            with (root / mixed.MIXED_NORMALIZER_FILE).open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "normalizer hash"):
                mixed.load_mixed_geometry_model("unused", root, "cpu")

    def test_public_v2_loader_keeps_final_gate_before_shared_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "periodic_v2_config.json").write_text("{}")
            with patch.object(v2_initialization, "load_periodic_v2_architecture") as rebuild:
                with self.assertRaisesRegex(ValueError, "completed registered final"):
                    v2_initialization.load_periodic_v2_model("unused", root, "cpu")
                rebuild.assert_not_called()
            with patch.object(v2_initialization, "validate_v2_final_checkpoint") as validate, \
                 patch.object(v2_initialization, "load_periodic_v2_architecture", return_value=("model", "tokenizer")) as rebuild:
                result = v2_initialization.load_periodic_v2_model("raw", root, "cpu", trainable=True)
                self.assertEqual(result, ("model", "tokenizer"))
                validate.assert_called_once_with(root)
                rebuild.assert_called_once_with("raw", root, "cpu", trainable=True, torch_dtype=None)

    def test_initializer_uses_strict_parent_loader_and_records_parent_hashes(self):
        model, tokenizer, _, _, _, _, _ = case()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            model.token_model.save_pretrained(parent)
            (parent / "CHECKPOINT_FINAL.json").write_text(json.dumps({"eligible_policy": True}))
            with patch.object(mixed, "load_periodic_v2_model", return_value=(model.token_model, tokenizer)) as strict:
                initialized, _ = mixed.initialize_mixed_geometry_model("raw", parent, model.normalizer, "cpu")
            strict.assert_called_once_with("raw", parent, "cpu", trainable=True, torch_dtype=None)
            self.assertEqual(initialized.parent_provenance["checkpoint_path"], str(parent.resolve()))
            self.assertIn("CHECKPOINT_FINAL.json", initialized.parent_provenance["files_sha256"])
            self.assertIn("adapter_model.bin", initialized.parent_provenance["files_sha256"])
            self.assertTrue(initialized.geometry_heads.v_head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
