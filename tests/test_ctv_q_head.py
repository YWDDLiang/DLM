from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ctv_q_head import (
    advantage_is_supported,
    centered_prediction_pairs,
    disjoint_plan_group,
    pairwise_order_accuracy,
    plan_bootstrap_spearman,
    robust_scale,
    supported_token_ids,
    token_support_counts,
)


class CTVQHeadBookkeepingTest(unittest.TestCase):
    def test_disjoint_plan_group(self):
        self.assertEqual(disjoint_plan_group(0), 0)
        self.assertEqual(disjoint_plan_group(7), 1)
        with self.assertRaises(ValueError):
            disjoint_plan_group(-1)

    def test_support_counts_and_intersection(self):
        rows = [
            {"action_token": 4, "plan_ordinal": plan, "energy": -1.0}
            for plan in range(4)
            for _ in range(2)
        ]
        counts = token_support_counts(rows)
        self.assertEqual(counts[4], {"known_returns": 8, "unique_plans": 4})
        self.assertEqual(supported_token_ids(counts, counts), {4})

    def test_robust_scale_has_floor(self):
        center, scale = robust_scale([2.0, 2.0])
        self.assertEqual(center, 2.0)
        self.assertEqual(scale, 1e-3)

    def test_advantage_support_contract(self):
        self.assertTrue(advantage_is_supported(0.003, -0.002))
        self.assertTrue(advantage_is_supported(0.010, 0.012))
        self.assertFalse(advantage_is_supported(0.010, -0.012))
        self.assertFalse(advantage_is_supported(0.0, 0.010))

    def test_pairwise_accuracy_and_plan_bootstrap(self):
        accuracy, comparisons, correct = pairwise_order_accuracy(
            [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]
        )
        self.assertEqual((accuracy, comparisons, correct), (1.0, 3, 3.0))
        result = plan_bootstrap_spearman(
            {0: [(0.0, 0.0), (1.0, 1.0)], 1: [(2.0, 2.0), (3.0, 3.0)]},
            draws=20,
            seed=7,
        )
        self.assertAlmostEqual(result["point"], 1.0)
        self.assertGreaterEqual(result["lcb_95"], 0.0)

    def test_state_centering_removes_absolute_offset(self):
        pairs = centered_prediction_pairs([100.0, 101.0], [-5.0, -3.0])
        self.assertEqual(pairs, [(-0.5, -1.0), (0.5, 1.0)])


try:
    import torch
except ImportError:  # pragma: no cover - optional local torch environment.
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CTVQHeadTorchTest(unittest.TestCase):
    def test_head_shapes_and_centered_advantage(self):
        from crystal_dlm.ctv_q_head import build_q_head

        model = build_q_head(256)
        q_value, advantage = model(
            torch.randn(3, 256),
            torch.randn(3, 8, 256),
            torch.randn(3, 8, 1),
            torch.randn(3, 8, 1),
            torch.nn.functional.one_hot(
                torch.zeros(3, 8, dtype=torch.long), num_classes=9
            ).float(),
        )
        self.assertEqual(tuple(q_value.shape), (3, 8))
        self.assertEqual(tuple(advantage.shape), (3, 8))
        self.assertTrue(torch.allclose(advantage.mean(dim=1), torch.zeros(3), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
