import math
from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:  # Lightweight local contract environment.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if torch is not None:
    from crystal_dlm.listwise_alignment import (
        listwise_lair_loss,
        robust_centered_rewards,
        shared_noise_listwise_alignment_loss,
        softmax_advantages,
    )


@unittest.skipIf(torch is None, "torch is not installed")
class ListwiseTargetTest(unittest.TestCase):
    def test_robust_reward_prefers_lower_energy_and_is_median_centered(self):
        rewards = robust_centered_rewards(
            torch.tensor([-2.0, -1.9, -1.8]),
            mad_scale=1.0,
            scale_floor=0.01,
            reward_clip=10.0,
        )
        torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0, -1.0]))
        self.assertEqual(rewards.dtype, torch.float32)

    def test_zero_mad_uses_explicit_floor_without_nan(self):
        rewards = robust_centered_rewards(
            torch.tensor([-2.0, -2.0, -2.0]),
            mad_scale=1.4826,
            scale_floor=0.005,
            reward_clip=5.0,
        )
        torch.testing.assert_close(rewards, torch.zeros(3))

    def test_softmax_advantages_sum_zero_and_preserve_order(self):
        targets = softmax_advantages(
            torch.tensor([2.0, 0.0, -1.0]),
            temperature=0.5,
        )
        torch.testing.assert_close(targets.probabilities.sum(), torch.tensor(1.0))
        self.assertLess(abs(float(targets.advantages.sum())), 1e-7)
        self.assertGreater(float(targets.advantages[0]), 0.0)
        self.assertLess(float(targets.advantages[-1]), 0.0)

    def test_all_scientific_constants_are_required_and_validated(self):
        with self.assertRaisesRegex(ValueError, "scale_floor"):
            robust_centered_rewards(
                [-2.0, -1.0],
                mad_scale=1.0,
                scale_floor=0.0,
                reward_clip=4.0,
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            softmax_advantages([1.0, -1.0], temperature=0.0)


@unittest.skipIf(torch is None, "torch is not installed")
class ListwiseLossTest(unittest.TestCase):
    def test_linear_quadratic_and_best_anchor_components(self):
        scores = torch.tensor([1.0, 0.0, -1.0])
        advantages = torch.tensor([0.5, 0.0, -0.5])
        anchors = torch.tensor([0.4, 0.6, 0.8])
        total, linear, quadratic, anchor = listwise_lair_loss(
            scores,
            advantages,
            anchors,
            best_index=0,
            linear_weight=1.0,
            quadratic_weight=0.3,
            best_anchor_weight=0.2,
            group_weight=1.0,
            advantage_sum_tolerance=1e-6,
        )
        torch.testing.assert_close(linear, torch.tensor(-1.0))
        torch.testing.assert_close(quadratic, torch.tensor(0.2))
        torch.testing.assert_close(anchor, torch.tensor(0.4))
        torch.testing.assert_close(total, torch.tensor(-0.72))

    def test_gradient_raises_better_candidate_score(self):
        scores = torch.zeros(3, requires_grad=True)
        advantages = torch.tensor([0.4, 0.0, -0.4])
        total, _, _, _ = listwise_lair_loss(
            scores,
            advantages,
            torch.tensor([0.2, 0.3, 0.4]),
            best_index=0,
            linear_weight=1.0,
            quadratic_weight=0.1,
            best_anchor_weight=0.0,
            group_weight=1.0,
            advantage_sum_tolerance=1e-6,
        )
        total.backward()
        self.assertLess(float(scores.grad[0]), 0.0)
        self.assertGreater(float(scores.grad[-1]), 0.0)

    def test_noncentered_advantages_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to zero"):
            listwise_lair_loss(
                [0.0, 0.0],
                [0.5, 0.1],
                [0.2, 0.3],
                best_index=0,
                linear_weight=1.0,
                quadratic_weight=0.1,
                best_anchor_weight=0.2,
                group_weight=1.0,
                advantage_sum_tolerance=1e-6,
            )

    def test_linear_term_is_shift_invariant_but_bound_is_not(self):
        advantages = torch.tensor([0.5, -0.5])
        anchors = torch.tensor([0.1, 0.2])
        base = listwise_lair_loss(
            torch.tensor([1.0, -1.0]),
            advantages,
            anchors,
            best_index=0,
            linear_weight=1.0,
            quadratic_weight=0.25,
            best_anchor_weight=0.0,
            group_weight=1.0,
            advantage_sum_tolerance=1e-6,
        )
        shifted = listwise_lair_loss(
            torch.tensor([3.0, 1.0]),
            advantages,
            anchors,
            best_index=0,
            linear_weight=1.0,
            quadratic_weight=0.25,
            best_anchor_weight=0.0,
            group_weight=1.0,
            advantage_sum_tolerance=1e-6,
        )
        torch.testing.assert_close(base[1], shifted[1])
        self.assertGreater(float(shifted[2]), float(base[2]))

    def test_end_to_end_wrapper_uses_lowest_energy_anchor(self):
        output = shared_noise_listwise_alignment_loss(
            torch.tensor([0.2, -0.1, 0.0], dtype=torch.bfloat16),
            torch.tensor([-1.8, -2.0, -1.9]),
            torch.tensor([0.7, 0.3, 0.5]),
            mad_scale=1.0,
            scale_floor=0.01,
            reward_clip=5.0,
            reward_temperature=0.5,
            linear_weight=1.0,
            quadratic_weight=0.1,
            best_anchor_weight=0.2,
            group_weight=1.0,
            advantage_sum_tolerance=1e-6,
        )
        self.assertEqual(output.best_index, 1)
        torch.testing.assert_close(output.best_anchor_loss, torch.tensor(0.3))
        self.assertEqual(output.loss.dtype, torch.float32)
        self.assertTrue(math.isfinite(float(output.loss)))
        self.assertLess(abs(float(output.advantages.sum())), 1e-6)

    def test_shape_and_group_weight_guards(self):
        with self.assertRaisesRegex(ValueError, "identical shape"):
            shared_noise_listwise_alignment_loss(
                [0.0, 1.0],
                [-2.0, -1.9, -1.8],
                [0.2, 0.3],
                mad_scale=1.0,
                scale_floor=0.01,
                reward_clip=5.0,
                reward_temperature=0.5,
                linear_weight=1.0,
                quadratic_weight=0.1,
                best_anchor_weight=0.2,
                group_weight=1.0,
                advantage_sum_tolerance=1e-6,
            )
        with self.assertRaisesRegex(ValueError, "group_weight"):
            listwise_lair_loss(
                [0.0, 0.0],
                [0.5, -0.5],
                [0.2, 0.3],
                best_index=0,
                linear_weight=1.0,
                quadratic_weight=0.1,
                best_anchor_weight=0.2,
                group_weight=0.0,
                advantage_sum_tolerance=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
