from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest


try:
    import torch
    from crystal_dlm.llada_generation import _model_logits
    from crystal_dlm.transaction_logits import (
        TransactionContext,
        TransactionModelStep,
        apply_transaction_logit_transform,
    )
except ModuleNotFoundError:
    torch = None
    _model_logits = None
    TransactionContext = None
    TransactionModelStep = None
    apply_transaction_logit_transform = None


_TorchModuleBase = torch.nn.Module if torch is not None else object


@unittest.skipIf(torch is None, "torch unavailable")
class TransactionLogitsTest(unittest.TestCase):
    class HeadModel(_TorchModuleBase):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 4)
            self.output = torch.nn.Linear(4, 32, bias=False)
            self.forward_count = 0

        def get_output_embeddings(self):
            return self.output

        def forward(self, token_ids, attention_mask=None):
            del attention_mask
            self.forward_count += 1
            hidden = self.embedding(token_ids)
            return SimpleNamespace(logits=self.output(hidden))

    class IdentityTransform:
        def prepare(self, context, model_step):
            return (context.kind, model_step.hidden_states)

        def apply(self, component, logits, proposal, context):
            del component, proposal, context
            return logits

    class ReopenSupportTransform(IdentityTransform):
        def apply(self, component, logits, proposal, context):
            del component, proposal, context
            output = logits.clone()
            output[2] = 1000.0
            output[3] = 7.0
            return output

    def _site_context(self):
        return TransactionContext(
            kind="site_xyz",
            active_positions=(10, 11, 12),
            complete_pre_remask_tokens=torch.arange(20, dtype=torch.long),
            previous_active_token_ids=(10, 11, 12),
            prompt_length=2,
            gen_length=18,
            plan_metadata={"family": "oxide"},
            program_metadata={"species_order": ["O", "Li"]},
            batch_index=0,
            block_index=1,
            site_index=3,
            site_order_index=0,
            component_indices=(0, 1, 2),
            lattice_version=4,
        )

    def test_context_is_frozen_and_exposes_generation_positions(self):
        context = self._site_context()
        self.assertEqual(context.generation_positions, (8, 9, 10))
        self.assertEqual(context.plan_metadata["family"], "oxide")
        with self.assertRaises(FrozenInstanceError):
            context.lattice_version = 5
        with self.assertRaises(TypeError):
            context.plan_metadata["family"] = "halide"

    def test_transform_cannot_reopen_existing_hard_support(self):
        minimum = torch.finfo(torch.float32).min
        logits = torch.tensor([0.0, 1.0, minimum, -torch.inf], dtype=torch.float32)
        output = apply_transaction_logit_transform(
            self.ReopenSupportTransform(),
            component=0,
            active_logits=logits,
            proposal=None,
            context=self._site_context(),
        )
        self.assertEqual(float(output[2]), minimum)
        self.assertTrue(bool(torch.isneginf(output[3])))
        self.assertTrue(torch.equal(output[:2], logits[:2]))

    def test_model_step_captures_hidden_with_one_forward(self):
        model = self.HeadModel()
        token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        step = _model_logits(
            model,
            token_ids,
            torch.ones_like(token_ids),
            torch.zeros_like(token_ids, dtype=torch.bool),
            0.0,
            31,
            return_model_step=True,
        )
        self.assertIsInstance(step, TransactionModelStep)
        self.assertEqual(model.forward_count, 1)
        self.assertIsNotNone(step.hidden_states)
        self.assertEqual(tuple(step.hidden_states.shape), (1, 3, 4))
        row = step.row(0)
        self.assertEqual(tuple(row.logits.shape), (3, 32))
        self.assertEqual(tuple(row.hidden_states.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
