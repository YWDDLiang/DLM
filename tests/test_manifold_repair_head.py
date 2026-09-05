import unittest

import torch

from crystal_dlm.manifold_repair_head import (
    ManifoldRepairConfig,
    ManifoldRepairHead,
)


class ManifoldRepairHeadTests(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(7)
        batch, sites, hidden = 2, 4, 12
        vectors = torch.randn(batch, sites, sites, 3)
        vectors = 0.5 * (vectors - vectors.transpose(1, 2))
        mask = torch.ones(batch, sites, dtype=torch.bool)
        diagonal = torch.eye(sites, dtype=torch.bool).unsqueeze(0)
        pair = (mask.unsqueeze(1) & mask.unsqueeze(2) & ~diagonal)
        return {
            "lattice_hidden": torch.randn(batch, 6, hidden),
            "site_hidden": torch.randn(batch, sites, 3, hidden),
            "species": torch.tensor([[8, 14, 8, 14], [3, 8, 13, 16]]),
            "program_rank": torch.tensor([[0, 1, 2, 3], [2, 0, 3, 1]]),
            "site_mask": mask,
            "mic_vectors": vectors,
            "pair_mask": pair,
            "plan_hidden": torch.randn(batch, hidden),
        }

    def test_zero_initialization_is_exact_identity_delta(self):
        head = ManifoldRepairHead(ManifoldRepairConfig(hidden_size=12, width=16))
        output = head(**self._inputs())
        self.assertEqual(float(output.lattice_tangent.abs().max()), 0.0)
        self.assertEqual(float(output.cartesian_site_delta.abs().max()), 0.0)
        self.assertEqual(float(output.pair_scalars.abs().max()), 0.0)

    def test_outputs_are_symmetric_translation_free_and_bounded(self):
        head = ManifoldRepairHead(ManifoldRepairConfig(hidden_size=12, width=16))
        torch.nn.init.normal_(head.pair_output.weight, std=0.1)
        torch.nn.init.normal_(head.metric_output.weight, std=0.1)
        output = head(**self._inputs())
        self.assertTrue(
            torch.allclose(
                output.lattice_tangent,
                output.lattice_tangent.transpose(-1, -2),
            )
        )
        self.assertTrue(
            torch.allclose(
                output.cartesian_site_delta.sum(dim=1),
                torch.zeros_like(output.cartesian_site_delta[:, 0]),
                atol=1.0e-6,
            )
        )
        norms = torch.linalg.vector_norm(output.cartesian_site_delta, dim=-1)
        self.assertLessEqual(float(norms.max()), 0.75 + 1.0e-6)
        self.assertLessEqual(float(output.lattice_tangent.abs().max()), 0.20 + 1.0e-6)

    def test_site_permutation_equivariance(self):
        head = ManifoldRepairHead(ManifoldRepairConfig(hidden_size=12, width=16))
        torch.nn.init.normal_(head.pair_output.weight, std=0.1)
        values = self._inputs()
        expected = head(**values).cartesian_site_delta
        order = torch.tensor([2, 0, 3, 1])
        permuted = dict(values)
        for name in ("site_hidden", "species", "program_rank", "site_mask"):
            permuted[name] = values[name].index_select(1, order)
        permuted["mic_vectors"] = values["mic_vectors"].index_select(1, order).index_select(2, order)
        permuted["pair_mask"] = values["pair_mask"].index_select(1, order).index_select(2, order)
        observed = head(**permuted).cartesian_site_delta
        self.assertTrue(torch.allclose(observed, expected.index_select(1, order), atol=1.0e-6))

    def test_gradients_are_finite(self):
        head = ManifoldRepairHead(ManifoldRepairConfig(hidden_size=12, width=16))
        torch.nn.init.normal_(head.pair_output.weight, std=0.1)
        torch.nn.init.normal_(head.metric_output.weight, std=0.1)
        values = self._inputs()
        values["site_hidden"].requires_grad_(True)
        output = head(**values)
        loss = output.cartesian_site_delta.square().mean() + output.lattice_tangent.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(values["site_hidden"].grad).all())
        for parameter in head.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_bfloat16_backbone_hidden_is_cast_before_float32_layer_norm(self):
        head = ManifoldRepairHead(ManifoldRepairConfig(hidden_size=12, width=16))
        values = self._inputs()
        values["lattice_hidden"] = values["lattice_hidden"].to(torch.bfloat16)
        values["site_hidden"] = values["site_hidden"].to(torch.bfloat16)
        values["plan_hidden"] = values["plan_hidden"].to(torch.bfloat16)
        output = head(**values)
        self.assertEqual(output.lattice_tangent.dtype, torch.float32)
        self.assertEqual(output.cartesian_site_delta.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
