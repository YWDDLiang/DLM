from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ctv_q_head import (
    disjoint_plan_group,
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
