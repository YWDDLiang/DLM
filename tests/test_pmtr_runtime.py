from types import SimpleNamespace
import unittest

import torch

from crystal_dlm.manifold_repair_head import ManifoldRepairOutput
from crystal_dlm.pmtr_runtime import PMTRLogitTransform, PMTRRuntimeConfig
from crystal_dlm.transaction_logits import TransactionContext, TransactionModelStep


class FakeTokenizer:
    def __init__(self):
        tokens = ["<P>", "<N_2>", "<E_Li>", "<E_O>"]
        for axis in "ABC":
            tokens += [f"<L{axis}_020>", f"<L{axis}_040>", f"<L{axis}_060>"]
        for axis in "ABG":
            tokens += [f"<A{axis}_060>", f"<A{axis}_090>", f"<A{axis}_120>"]
        for axis in "XYZ":
            tokens += [f"<{axis}_{value:03d}>" for value in (0, 25, 50, 75, 100)]
        self.vocab = {token: index for index, token in enumerate(tokens)}

    def get_vocab(self):
        return dict(self.vocab)


class FixedHead(torch.nn.Module):
    def __init__(self, *, zero=False):
        super().__init__()
        self.zero = zero
        self.config = SimpleNamespace(max_sites=20)

    def forward(self, **kwargs):
        batch, sites = kwargs["species"].shape
        dtype = kwargs["site_hidden"].dtype
        device = kwargs["site_hidden"].device
        tangent = torch.zeros(batch, 3, 3, dtype=dtype, device=device)
        delta = torch.zeros(batch, sites, 3, dtype=dtype, device=device)
        if not self.zero:
            tangent[:, 0, 0] = 0.10
            delta[:, 0, 0] = 0.50
            delta[:, 1, 0] = -0.50
        return ManifoldRepairOutput(
            tangent,
            delta,
            torch.zeros(batch, sites, 4, dtype=dtype, device=device),
            torch.zeros(batch, sites, sites, dtype=dtype, device=device),
        )


class PMTRRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        v = self.tokenizer.vocab
        self.prompt = 1
        body = [v["<N_2>"]]
        body += [v["<LA_040>"], v["<LB_040>"], v["<LC_040>"]]
        body += [v["<AA_090>"], v["<AB_090>"], v["<AG_090>"]]
        body += [v["<E_Li>"], v["<X_000>"], v["<Y_000>"], v["<Z_000>"]]
        body += [v["<E_O>"], v["<X_050>"], v["<Y_050>"], v["<Z_050>"]]
        self.tokens = torch.tensor([v["<P>"], *body], dtype=torch.long)
        hidden = torch.randn(len(self.tokens), 8)
        self.step = TransactionModelStep(
            token_ids=self.tokens.clone(),
            logits=torch.zeros(len(self.tokens), len(v)),
            hidden_states=hidden,
        )

    def _context(self, kind):
        if kind == "cell":
            positions = tuple(range(2, 8))
            site = order = None
            components = tuple(range(6))
        else:
            positions = (9, 10, 11)
            site = order = 0
            components = (0, 1, 2)
        return TransactionContext(
            kind=kind,
            active_positions=positions,
            complete_pre_remask_tokens=self.tokens.clone(),
            previous_active_token_ids=tuple(int(self.tokens[p]) for p in positions),
            prompt_length=self.prompt,
            gen_length=15,
            program_metadata={"species_order": ["Li", "O"]},
            site_index=site,
            site_order_index=order,
            component_indices=components,
        )

    def test_zero_head_is_exact_identity_for_cell_and_site(self):
        transform = PMTRLogitTransform(FixedHead(zero=True), self.tokenizer)
        for kind in ("cell", "site_xyz"):
            context = self._context(kind)
            proposal = transform.prepare(context, self.step)
            width = 6 if kind == "cell" else 3
            for component in range(width):
                logits = torch.randn(len(self.tokenizer.vocab))
                self.assertTrue(
                    torch.equal(
                        transform.apply(component, logits, proposal, context), logits
                    )
                )

    def test_nonzero_head_moves_only_numeric_family_logits(self):
        transform = PMTRLogitTransform(
            FixedHead(), self.tokenizer, PMTRRuntimeConfig(transport_gain=5.0)
        )
        context = self._context("site_xyz")
        proposal = transform.prepare(context, self.step)
        logits = torch.zeros(len(self.tokenizer.vocab))
        changed = transform.apply(0, logits, proposal, context)
        touched = set(torch.nonzero(changed, as_tuple=False).flatten().tolist())
        x_ids = {
            token_id
            for token, token_id in self.tokenizer.vocab.items()
            if token.startswith("<X_")
        }
        self.assertTrue(touched)
        self.assertTrue(touched <= x_ids)
        self.assertFalse(torch.equal(proposal.old_values, proposal.target_values))

    def test_program_metadata_is_required_and_exact(self):
        context = self._context("site_xyz")
        bad = TransactionContext(
            **{
                **context.__dict__,
                "program_metadata": {"species_order": ["Li", "Na"]},
            }
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            PMTRLogitTransform(FixedHead(), self.tokenizer).prepare(bad, self.step)


if __name__ == "__main__":
    unittest.main()
