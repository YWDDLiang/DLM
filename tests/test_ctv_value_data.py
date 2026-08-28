from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ctv_value_data import (
    cross_continuation_pair_agreement,
    linear_quantile,
    spearman,
    validate_branch_split,
)


class CTVValueDataTest(unittest.TestCase):
    def test_quantile_and_spearman(self):
        self.assertEqual(linear_quantile([0.0, 10.0], 0.25), 2.5)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)

    def test_cross_continuation_pair_agreement(self):
        values = {
            1: {10: 0.0, 20: 0.1},
            2: {10: 1.0, 20: 1.1},
            3: {10: 2.0, 20: -1.0},
        }
        agreements, comparisons = cross_continuation_pair_agreement(values)
        self.assertEqual((agreements, comparisons), (1, 3))

    def test_validate_branch_split_aligns_complete_state(self):
        state = {
            "state_id": "s",
            "composition_id": "8:1|26:1",
            "sample_idx": 4,
            "milestone": 0.6,
            "action_token_ids": list(range(8)),
        }
        second = {**state, "state_id": "t", "milestone": 0.8}
        branches = []
        labels = []
        ordinal = 0
        for state_index, current in enumerate((state, second)):
            for action in range(8):
                branch = {
                    "branch_ordinal": ordinal,
                    "state_id": current["state_id"],
                    "composition_id": current["composition_id"],
                    "sample_idx": current["sample_idx"],
                    "canary_plan_idx": 0,
                    "milestone": current["milestone"],
                    "intervention_position": 1,
                    "action_token": action,
                    "action_rank": action,
                    "base_action_probability": 0.125,
                    "continuation_seed": 99,
                    "continuation_rank": 0,
                    "num_atoms": 2,
                }
                branches.append(branch)
                labels.append(
                    {
                        "global_branch_ordinal": ordinal,
                        "state_id": current["state_id"],
                        "action_token": action,
                        "continuation_seed": 99,
                        "direct_valid": True,
                        "reconstructed": True,
                        "chgnet_relaxation_known": True,
                        "chgnet_energy_per_atom": float(action + state_index),
                    }
                )
                ordinal += 1
        result = validate_branch_split(
            split="train",
            branches=branches,
            states=[state, second],
            labels=labels,
            expected_branches=16,
            expected_states=2,
            continuations_per_action=1,
        )
        self.assertEqual(result["report"]["branches"], 16)
        self.assertEqual(result["report"]["known_returns"], 16)
        self.assertEqual(len(result["state_summaries"]), 2)


if __name__ == "__main__":
    unittest.main()
