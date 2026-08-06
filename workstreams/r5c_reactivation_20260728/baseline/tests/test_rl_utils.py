import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from crystal_dlm.rl_utils import (
    base_reward,
    base_reward_for_reason,
    duplicate_penalty,
    normalize_rewards,
    score_rollout_records,
    trace_items_from_sequence,
)


class RlRewardUtilsTest(unittest.TestCase):
    def test_base_reward_failure_and_smact_cases(self):
        self.assertEqual(base_reward(False, False, False), -1.0)
        self.assertEqual(base_reward(True, False, False), -0.8)
        self.assertEqual(base_reward(True, True, False), -0.4)
        self.assertEqual(base_reward(True, True, True), 1.0)

    def test_shortcut_aware_base_reward(self):
        self.assertEqual(
            base_reward_for_reason(
                True,
                True,
                True,
                "charge_neutral_pauling_valid",
                reward_mode="shortcut_aware",
            ),
            1.0,
        )
        self.assertEqual(
            base_reward_for_reason(
                True,
                True,
                True,
                "all_metal_shortcut",
                reward_mode="shortcut_aware",
            ),
            0.0,
        )
        self.assertEqual(
            base_reward_for_reason(
                True,
                True,
                True,
                "single_element_shortcut",
                reward_mode="shortcut_aware",
            ),
            -0.2,
        )
        self.assertEqual(
            base_reward_for_reason(
                True,
                True,
                True,
                "all_metal_shortcut",
                reward_mode="shortcut_strict",
            ),
            -0.4,
        )
        self.assertEqual(
            base_reward_for_reason(
                True,
                True,
                True,
                "single_element_shortcut",
                reward_mode="shortcut_strict",
            ),
            -0.6,
        )
        self.assertEqual(
            base_reward_for_reason(
                True,
                True,
                False,
                "charge_neutrality_fail",
                reward_mode="reason_aware",
            ),
            -0.55,
        )

    def test_duplicate_penalty_clamps(self):
        self.assertEqual(duplicate_penalty(1, per_extra=0.1, floor=-0.3), 0.0)
        self.assertEqual(duplicate_penalty(3, per_extra=0.1, floor=-0.3), -0.2)
        self.assertEqual(duplicate_penalty(99, per_extra=0.1, floor=-0.3), -0.3)

    def test_score_rollout_records_adds_duplicate_penalties(self):
        rows = [
            {
                "diagnostics": {
                    "parse_ok": True,
                    "graph_ok": True,
                    "smact_valid": True,
                    "formula": "Li2O",
                    "symbols": ["Li", "O"],
                    "smact_reason": "charge_neutral_pauling_valid",
                }
            },
            {
                "diagnostics": {
                    "parse_ok": True,
                    "graph_ok": True,
                    "smact_valid": True,
                    "formula": "Li2O",
                    "symbols": ["O", "Li"],
                    "smact_reason": "charge_neutral_pauling_valid",
                }
            },
            {"diagnostics": {"parse_ok": True, "graph_ok": True, "smact_valid": False}},
        ]
        scored, summary = score_rollout_records(rows)
        self.assertEqual(len(scored), 3)
        self.assertLess(scored[0]["reward"], 1.0)
        self.assertEqual(summary["smact_valid"], 2)
        self.assertEqual(summary["smact_valid_rate"], 2 / 3)

    def test_shortcut_aware_score_penalizes_shortcuts(self):
        rows = [
            {
                "diagnostics": {
                    "parse_ok": True,
                    "graph_ok": True,
                    "smact_valid": True,
                    "formula": "Li2O",
                    "symbols": ["Li", "O"],
                    "smact_reason": "charge_neutral_pauling_valid",
                }
            },
            {
                "diagnostics": {
                    "parse_ok": True,
                    "graph_ok": True,
                    "smact_valid": True,
                    "formula": "Li",
                    "symbols": ["Li"],
                    "smact_reason": "single_element_shortcut",
                }
            },
            {
                "diagnostics": {
                    "parse_ok": True,
                    "graph_ok": True,
                    "smact_valid": True,
                    "formula": "FeNi",
                    "symbols": ["Fe", "Ni"],
                    "smact_reason": "all_metal_shortcut",
                }
            },
        ]
        scored, summary = score_rollout_records(rows, reward_mode="shortcut_aware")
        self.assertEqual(summary["reward_mode"], "shortcut_aware")
        self.assertEqual(summary["shortcut_rate"], 2 / 3)
        self.assertGreater(scored[0]["reward"], scored[1]["reward"])
        self.assertGreater(scored[0]["reward"], scored[2]["reward"])

    def test_reason_aware_score_penalizes_pbc_duplicates(self):
        rows = [
            {
                "diagnostics": {
                    "parse_ok": True,
                    "graph_ok": True,
                    "smact_valid": True,
                    "formula": "Li2O",
                    "symbols": ["Li", "O"],
                    "smact_reason": "charge_neutral_pauling_valid",
                    "has_pbc_equivalent_duplicate": True,
                }
            }
        ]
        scored, summary = score_rollout_records(rows, reward_mode="reason_aware")
        self.assertEqual(summary["pbc_equivalent_duplicate_rate"], 1.0)
        self.assertLessEqual(scored[0]["reward"], -0.8)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TraceRlUtilsTest(unittest.TestCase):
    def test_trace_items_do_not_mask_prompt_tokens(self):
        input_ids = torch.arange(10, dtype=torch.long)
        items = trace_items_from_sequence(
            input_ids,
            prompt_length=3,
            step_map=[-1, 0, 1, 1, 2, 2, 3],
            reward=1.0,
            mask_id=999,
            shrink=1,
        )
        self.assertTrue(items)
        for item in items:
            self.assertFalse(item.p_mask[:3].any().item())
            self.assertFalse((item.noisy_ids[:3] == 999).any().item())
            self.assertTrue(torch.all(item.token_weights == 1.0).item())

    def test_trace_items_carry_token_weights(self):
        input_ids = torch.arange(8, dtype=torch.long)
        items = trace_items_from_sequence(
            input_ids,
            prompt_length=2,
            step_map=[0, 1, 1, 2, 2, 3],
            reward=1.0,
            mask_id=999,
            shrink=1,
            token_weights=[1.0, 1.2, 0.8, 1.0, 1.0, 1.0],
        )
        self.assertTrue(items)
        self.assertAlmostEqual(items[0].token_weights[3].item(), 1.2)

    def test_normalize_rewards_constant_is_zero(self):
        rewards = torch.tensor([1.0, 1.0, 1.0])
        normalized = normalize_rewards(rewards)
        self.assertTrue(torch.equal(normalized, torch.zeros_like(rewards)))


if __name__ == "__main__":
    unittest.main()
