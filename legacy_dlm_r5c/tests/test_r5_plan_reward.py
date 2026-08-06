import unittest

from scripts.reward_r5_plan_rollouts import build_rewarded_rollouts, reward_for_record


class R5PlanRewardTests(unittest.TestCase):
    def test_invalid_charge_gets_negative_reward_and_kept(self):
        record = {
            "sample_idx": 7,
            "parsed": True,
            "conditioning_prompt": "Generate plan",
            "text": "P01=Z008:C007;P02=Z011:C002;P03=Z020:C001",
            "plan_validation": {"valid": True},
            "plan_state": {
                "formula": "O7Na2Ca",
                "N": 10,
                "elements": ["O", "Na", "Ca"],
                "counts": [7, 2, 1],
                "charge_bucket": "charge_fail",
                "validator": {"valid": False, "reason": "charge_neutrality_fail"},
            },
        }
        reward, diagnostics = reward_for_record(record)
        self.assertLess(reward, 0.0)
        self.assertEqual(diagnostics["validator_reason"], "charge_neutrality_fail")
        rows, summary = build_rewarded_rollouts([record])
        self.assertEqual(len(rows), 1)
        self.assertTrue(summary["all_rollouts_retained"])
        self.assertFalse(summary["uses_candidate_selection"])

    def test_neutral_plan_gets_positive_reward(self):
        record = {
            "parsed": True,
            "conditioning_prompt": "Generate plan",
            "text": "P01=Z008:C002;P02=Z031:C002;P03=Z056:C001",
            "plan_validation": {"valid": True},
            "plan_state": {
                "formula": "O2Ga2Ba",
                "N": 5,
                "elements": ["O", "Ga", "Ba"],
                "counts": [2, 2, 1],
                "charge_bucket": "neutral_plausible",
                "validator": {"valid": True, "reason": "charge_neutral_pauling_valid"},
            },
        }
        reward, diagnostics = reward_for_record(record)
        self.assertGreater(reward, 0.0)
        self.assertEqual(diagnostics["formula"], "O2Ga2Ba")

    def test_strict_v2_treats_all_metal_as_negative_training_signal(self):
        record = {
            "parsed": True,
            "conditioning_prompt": "Generate plan",
            "text": "P01=Z027:C001;P02=Z028:C001;P03=Z051:C001",
            "plan_validation": {"valid": True},
            "plan_state": {
                "formula": "CoNiSb",
                "N": 3,
                "elements": ["Co", "Ni", "Sb"],
                "counts": [1, 1, 1],
                "charge_bucket": "all_metal",
                "validator": {"valid": True, "reason": "all_metal_shortcut"},
            },
        }
        v1_reward, _ = reward_for_record(record)
        strict_reward, diagnostics = reward_for_record(record, reward_mode="strict_v2")
        self.assertGreater(v1_reward, 0.0)
        self.assertLess(strict_reward, 0.0)
        self.assertEqual(diagnostics["reward_mode"], "strict_v2")

        rows, summary = build_rewarded_rollouts([record], reward_mode="strict_v2")
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["reward_mode"], "strict_v2")
        self.assertTrue(summary["all_rollouts_retained"])
        self.assertFalse(summary["uses_sampling_prior"])


if __name__ == "__main__":
    unittest.main()
