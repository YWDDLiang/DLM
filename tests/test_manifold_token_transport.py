from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.manifold_token_transport import (
    render_bracketed_token_residual,
    render_periodic_coordinate_token_residual,
)


class ManifoldTokenTransportTest(unittest.TestCase):
    def test_brackets_predicted_target_far_from_old_value(self) -> None:
        values = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        token_ids = torch.tensor([2, 5, 8, 11])
        old = torch.tensor(0.1, dtype=torch.float64)
        target = torch.tensor(2.4, dtype=torch.float64, requires_grad=True)
        residual = render_bracketed_token_residual(
            values, token_ids, old, target, vocab_size=16, gain=2.0
        )
        touched = torch.nonzero(residual, as_tuple=False).flatten().tolist()
        self.assertEqual(touched, [8, 11])
        self.assertEqual(residual[:2].abs().sum().item(), 0.0)
        residual.sum().backward()
        self.assertTrue(torch.isfinite(target.grad))
        self.assertNotEqual(target.grad.item(), 0.0)

    def test_only_legal_target_bins_and_active_positions_are_touched(self) -> None:
        old = torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float64)
        target = torch.tensor([[1.25, 1.25], [1.25, 1.25]], dtype=torch.float64)
        active = torch.tensor([[True, False], [False, True]])
        residual = render_bracketed_token_residual(
            [0.0, 1.0, 2.0],
            [1, 4, 7],
            old,
            target,
            vocab_size=10,
            active_mask=active,
        )
        self.assertEqual(residual.shape, (2, 2, 10))
        self.assertEqual(residual[0, 1].abs().sum().item(), 0.0)
        self.assertEqual(residual[1, 0].abs().sum().item(), 0.0)
        non_family = [0, 2, 3, 5, 6, 8, 9]
        self.assertEqual(residual[..., non_family].abs().sum().item(), 0.0)
        self.assertGreater(residual[0, 0, 4].item(), 0.0)
        self.assertGreater(residual[0, 0, 7].item(), 0.0)

    def test_zero_correction_or_gain_is_exactly_zero(self) -> None:
        old = torch.tensor([0.2, 0.8], dtype=torch.float64)
        unchanged = render_bracketed_token_residual(
            [0.0, 0.5, 1.0], [1, 2, 3], old, old, vocab_size=6, gain=3.0
        )
        no_gain = render_bracketed_token_residual(
            [0.0, 0.5, 1.0], [1, 2, 3], old, 1.0 - old, vocab_size=6, gain=0.0
        )
        self.assertTrue(torch.equal(unchanged, torch.zeros_like(unchanged)))
        self.assertTrue(torch.equal(no_gain, torch.zeros_like(no_gain)))

    def test_periodic_boundary_uses_one_continuous_zero_alias(self) -> None:
        values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        ids = torch.tensor([1, 3, 5, 7, 9])
        old = torch.tensor(0.4, dtype=torch.float64)
        near_upper = render_periodic_coordinate_token_residual(
            values, ids, old, torch.tensor(0.999999, dtype=torch.float64), vocab_size=12
        )
        equivalent = render_periodic_coordinate_token_residual(
            values, ids, old, torch.tensor(-0.000001, dtype=torch.float64), vocab_size=12
        )
        self.assertTrue(torch.allclose(near_upper, equivalent, atol=1.0e-12))
        self.assertGreater(near_upper[1].item(), 0.0)
        self.assertGreater(near_upper[7].item(), 0.0)
        self.assertEqual(near_upper[9].item(), 0.0)
        self.assertEqual(near_upper[[0, 2, 4, 6, 8, 10, 11]].abs().sum().item(), 0.0)

    def test_periodic_transport_has_finite_gradient_across_wrap(self) -> None:
        target = torch.tensor([-0.03, 1.03], dtype=torch.float64, requires_grad=True)
        residual = render_periodic_coordinate_token_residual(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [1, 3, 5, 7, 9],
            torch.tensor([0.2, 0.8], dtype=torch.float64),
            target,
            vocab_size=12,
            gain=torch.tensor([1.0, 0.5], dtype=torch.float64),
        )
        weighted = residual * torch.arange(12, dtype=torch.float64)
        weighted.sum().backward()
        self.assertTrue(torch.isfinite(target.grad).all())


if __name__ == "__main__":
    unittest.main()
