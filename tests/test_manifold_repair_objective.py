import unittest

import torch

from crystal_dlm.manifold_repair_head import ManifoldRepairOutput
from crystal_dlm.manifold_repair_objective import (
    ManifoldRepairLossConfig,
    manifold_repair_losses,
    masked_transaction_cross_entropy,
    reference_logit_kl,
    trainable_gradient_l2,
)


class ManifoldRepairObjectiveTests(unittest.TestCase):
    def _output(self):
        return ManifoldRepairOutput(
            lattice_tangent=torch.zeros(2, 3, 3, requires_grad=True),
            cartesian_site_delta=torch.zeros(2, 3, 3, requires_grad=True),
            site_states=torch.zeros(2, 3, 4),
            pair_scalars=torch.zeros(2, 3, 3),
        )

    def test_mixed_transaction_batch_is_normalized_and_finite(self):
        output = self._output()
        target_lattice = torch.zeros(2, 3, 3)
        target_lattice[0, 0, 0] = 0.1
        target_sites = torch.zeros(2, 3, 3)
        target_sites[1, 2, 0] = 0.2
        losses = manifold_repair_losses(
            output,
            target_lattice_tangent=target_lattice,
            target_cartesian_site_delta=target_sites,
            site_mask=torch.ones(2, 3, dtype=torch.bool),
            lattice_active=torch.tensor([True, False]),
            active_site_mask=torch.tensor([[False, False, False], [False, False, True]]),
            config=ManifoldRepairLossConfig(0.1, 0.2),
        )
        self.assertTrue(torch.isfinite(losses.repair))
        self.assertAlmostEqual(float(losses.lattice), 1.0 / 6.0, places=6)
        self.assertAlmostEqual(float(losses.coordinate), 1.0 / 3.0, places=6)
        losses.repair.backward()
        self.assertTrue(torch.isfinite(output.lattice_tangent.grad).all())
        self.assertTrue(torch.isfinite(output.cartesian_site_delta.grad).all())

    def test_token_ce_and_reference_identity(self):
        logits = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]], requires_grad=True)
        targets = torch.tensor([[1, 0]])
        mask = torch.tensor([[True, False]])
        loss = masked_transaction_cross_entropy(logits, targets, mask)
        self.assertGreater(float(loss), 0.0)
        kl = reference_logit_kl(logits, logits.detach().clone(), mask)
        self.assertAlmostEqual(float(kl), 0.0, places=7)

    def test_gradient_probe(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        parameter.square().sum().backward()
        self.assertAlmostEqual(
            float(trainable_gradient_l2([parameter])), (20.0) ** 0.5, places=6
        )

    def test_rejects_conflicting_transaction_roles(self):
        with self.assertRaisesRegex(ValueError, "one cell"):
            manifold_repair_losses(
                self._output(),
                target_lattice_tangent=torch.zeros(2, 3, 3),
                target_cartesian_site_delta=torch.zeros(2, 3, 3),
                site_mask=torch.ones(2, 3, dtype=torch.bool),
                lattice_active=torch.tensor([True, False]),
                active_site_mask=torch.tensor([[True, False, False], [False, True, False]]),
                config=ManifoldRepairLossConfig(0.1, 0.2),
            )


if __name__ == "__main__":
    unittest.main()
