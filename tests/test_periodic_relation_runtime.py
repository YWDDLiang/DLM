from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from crystal_dlm.periodic_relation_runtime import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_STATE_NAME,
    _soft_or_committed_with_confidence,
    build_periodic_relation_support,
    soft_geometry_from_q0,
    wrap_with_periodic_relation,
)


class _Tokenizer:
    def __init__(self) -> None:
        tokens = ["<MASK>", "<N_002>", "<E_Li>", "<E_O>"]
        tokens += [f"<L{axis}_{value:03d}>" for axis in "ABC" for value in (39, 40, 41)]
        tokens += [f"<A{axis}_{value:03d}>" for axis in "ABG" for value in (89, 90, 91)]
        tokens += [f"<{axis}_{value:03d}>" for axis in "XYZ" for value in (0, 25, 50, 75)]
        self.vocab = {token: index for index, token in enumerate(tokens)}

    def get_vocab(self):
        return dict(self.vocab)


class _Base(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int = 12) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.head(hidden))

    def get_output_embeddings(self):
        return self.head

    def get_input_embeddings(self):
        return self.embedding

    def save_pretrained(self, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "base.marker").write_text("ok\n")


def _tokens(tokenizer: _Tokenizer) -> torch.Tensor:
    values = [
        "<N_002>", "<LA_040>", "<LB_040>", "<LC_040>",
        "<AA_090>", "<AB_090>", "<AG_090>",
        "<E_Li>", "<X_000>", "<Y_000>", "<Z_000>",
        "<E_O>", "<X_050>", "<Y_050>", "<Z_050>",
    ]
    return torch.tensor([[tokenizer.vocab[token] for token in values]])


class PeriodicRelationRuntimeTest(unittest.TestCase):
    def test_uncertainty_confidence_is_bounded_and_committed_is_one(self) -> None:
        table = {"ids": [0, 1, 2, 3], "values": [0.0, 0.25, 0.5, 0.75]}
        uniform = torch.zeros(4)
        _mean, low = _soft_or_committed_with_confidence(
            uniform, 99, table, periodic=True, floor=0.25
        )
        peaked_logits = torch.tensor([8.0, -8.0, -8.0, -8.0])
        _mean, high = _soft_or_committed_with_confidence(
            peaked_logits, 99, table, periodic=True, floor=0.25
        )
        value, committed = _soft_or_committed_with_confidence(
            uniform, 2, table, periodic=True, floor=0.25
        )
        self.assertGreaterEqual(low.item(), 0.25)
        self.assertLessEqual(high.item(), 1.0)
        self.assertGreater(high.item(), low.item())
        self.assertEqual(value.item(), 0.5)
        self.assertEqual(committed.item(), 1.0)

    def test_adapter_keeps_float32_with_bfloat16_base(self) -> None:
        tokenizer = _Tokenizer()
        base = _Base(len(tokenizer.vocab)).to(dtype=torch.bfloat16)
        wrapped = wrap_with_periodic_relation(base, tokenizer, rank=6)
        self.assertEqual(
            wrapped.periodic_relation_adapter.output_projection.weight.dtype,
            torch.float32,
        )
        ids = _tokens(tokenizer)
        wrapped.set_geometry_context(torch.tensor([0]), torch.tensor([2]))
        loss = wrapped(ids).logits.float().square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in wrapped.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_committed_geometry_decodes_without_target_leakage(self) -> None:
        tokenizer = _Tokenizer()
        ids = _tokens(tokenizer)
        q0 = torch.randn(1, ids.shape[1], len(tokenizer.vocab))
        geometry = soft_geometry_from_q0(
            q0=q0,
            input_ids=ids,
            prompt_lengths=torch.tensor([0]),
            num_sites=torch.tensor([2]),
            support=build_periodic_relation_support(tokenizer),
        )
        self.assertEqual(geometry.lattice.shape, (1, 3, 3))
        self.assertEqual(geometry.fractional_coordinates.shape, (1, 20, 3))
        self.assertEqual(geometry.species[0, :2].tolist(), [3, 8])
        self.assertTrue(torch.isfinite(geometry.lattice).all())

    def test_wrapper_is_exact_at_step_zero_then_changes_logits(self) -> None:
        tokenizer = _Tokenizer()
        base = _Base(len(tokenizer.vocab))
        ids = _tokens(tokenizer)
        with torch.no_grad():
            expected = base(ids).logits
        wrapped = wrap_with_periodic_relation(base, tokenizer, rank=6)
        wrapped.set_geometry_context(torch.tensor([0]), torch.tensor([2]))
        observed = wrapped(ids).logits
        self.assertTrue(torch.equal(expected, observed))
        self.assertTrue(wrapped.step0_checked)
        self.assertEqual(wrapped.step0_max_logit_delta, 0.0)
        with torch.no_grad():
            wrapped.periodic_relation_adapter.output_projection.weight.normal_(0.0, 0.02)
        changed = wrapped(ids).logits
        self.assertGreater((changed - observed).abs().max().item(), 0.0)

    def test_uncertainty_gate_keeps_step_zero_and_emits_confidence(self) -> None:
        tokenizer = _Tokenizer()
        base = _Base(len(tokenizer.vocab))
        ids = _tokens(tokenizer)
        ids[0, 8] = tokenizer.vocab["<MASK>"]
        wrapped = wrap_with_periodic_relation(
            base,
            tokenizer,
            rank=6,
            uncertainty_gate=True,
            uncertainty_gate_floor=0.25,
        )
        wrapped.set_geometry_context(torch.tensor([0]), torch.tensor([2]))
        with torch.no_grad():
            expected = base(ids).logits
            observed = wrapped(ids).logits
            geometry = soft_geometry_from_q0(
                q0=expected,
                input_ids=ids,
                prompt_lengths=torch.tensor([0]),
                num_sites=torch.tensor([2]),
                support=build_periodic_relation_support(tokenizer),
                uncertainty_gate=True,
                uncertainty_gate_floor=0.25,
            )
        self.assertTrue(torch.equal(expected, observed))
        self.assertIsNotNone(geometry.lattice_confidence)
        self.assertIsNotNone(geometry.site_confidence)
        self.assertGreaterEqual(geometry.site_confidence.min().item(), 0.25)
        self.assertLessEqual(geometry.site_confidence.max().item(), 1.0)

    def test_checkpoint_roundtrip(self) -> None:
        tokenizer = _Tokenizer()
        wrapped = wrap_with_periodic_relation(_Base(len(tokenizer.vocab)), tokenizer, rank=6)
        ids = _tokens(tokenizer)
        wrapped.set_geometry_context(torch.tensor([0]), torch.tensor([2]))
        wrapped(ids)
        with torch.no_grad():
            wrapped.periodic_relation_adapter.output_projection.weight.normal_(0.0, 0.02)
        with tempfile.TemporaryDirectory() as tmp:
            wrapped.save_pretrained(tmp)
            self.assertTrue((Path(tmp) / ADAPTER_CONFIG_NAME).is_file())
            self.assertTrue((Path(tmp) / ADAPTER_STATE_NAME).is_file())
            loaded = wrap_with_periodic_relation(
                _Base(len(tokenizer.vocab)), tokenizer, rank=6, checkpoint=tmp
            )
            loaded.set_geometry_context(torch.tensor([0]), torch.tensor([2]))
            self.assertTrue(loaded.step0_checked)
            self.assertEqual(loaded.step0_max_logit_delta, 0.0)
            self.assertTrue(torch.isfinite(loaded(ids).logits).all())
            for left, right in zip(
                wrapped.periodic_relation_adapter.parameters(),
                loaded.periodic_relation_adapter.parameters(),
            ):
                self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
