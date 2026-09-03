import math
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.spad_energy_posterior import (  # noqa: E402
    SPAD_E_ACTION_COUNT,
    SPAD_E_KL_BUDGET_NATS,
    build_spad_energy_posterior,
    spad_energy_posterior_loss,
    validate_common_state_group,
)


NO_OP = (10, 20, 30)
ACTIONS = (NO_OP, (11, 20, 30), (10, 21, 30), (10, 20, 31))


def posterior(
    *,
    energies=(0.0, 1.0, 2.0, 3.0),
    legal=(True, True, True, True),
    actions=ACTIONS,
    reference_scores=(0.0, 0.0, 0.0, 0.0),
):
    return build_spad_energy_posterior(
        reference_scores,
        energies,
        legal,
        action_triplets=actions,
        no_op_triplet=NO_OP,
    )


class SPADEnergyPosteriorTest(unittest.TestCase):
    def test_validity_dominates_arbitrarily_favorable_invalid_energy(self):
        result = posterior(
            energies=(0.0, 1.0, -1.0e30, 2.0),
            legal=(True, True, False, True),
        )
        self.assertEqual(float(result.target_probabilities[2]), 0.0)
        self.assertEqual(float(result.reference_probabilities[2]), 0.0)
        self.assertFalse(bool(result.energy_preference_mask[2]))
        self.assertGreater(
            float(result.target_probabilities[0]),
            float(result.target_probabilities[1]),
        )

    def test_no_op_only_is_exact_fallback(self):
        result = posterior(
            energies=(None, -100.0, -200.0, -300.0),
            legal=(True, False, False, False),
        )
        torch.testing.assert_close(
            result.target_probabilities,
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )
        self.assertEqual(result.tilt, 0.0)
        self.assertEqual(result.kl_nats, 0.0)
        self.assertEqual(result.action_count, SPAD_E_ACTION_COUNT)

    def test_deterministic_bisection_reaches_but_does_not_exceed_kl_bound(self):
        result = posterior(energies=(0.0, 1.0, 2.0, 3.0))
        self.assertFalse(result.infinite_tilt_limit)
        self.assertGreater(result.tilt, 0.0)
        self.assertLessEqual(result.kl_nats, SPAD_E_KL_BUDGET_NATS + 1e-12)
        self.assertAlmostEqual(result.kl_nats, SPAD_E_KL_BUDGET_NATS, places=10)

    def test_lower_known_energy_receives_larger_probability(self):
        result = posterior(energies=(3.0, 2.0, 1.0, 0.0))
        probabilities = result.target_probabilities.tolist()
        self.assertLess(probabilities[0], probabilities[1])
        self.assertLess(probabilities[1], probabilities[2])
        self.assertLess(probabilities[2], probabilities[3])

    def test_unknown_energy_is_neutral_and_stays_in_accounting(self):
        result = posterior(energies=(None, 0.0, 2.0, math.nan))
        probabilities = result.target_probabilities
        self.assertEqual(result.known_energy_count, 2)
        self.assertEqual(result.unknown_energy_count, 2)
        self.assertEqual(result.action_count, 4)
        self.assertEqual(float(result.normalized_terminal_energies[0]), 0.0)
        self.assertEqual(float(result.normalized_terminal_energies[3]), 0.0)
        torch.testing.assert_close(probabilities[0], probabilities[3])
        self.assertGreater(float(probabilities[1]), float(probabilities[0]))
        self.assertGreater(float(probabilities[0]), float(probabilities[2]))

    def test_duplicate_triplets_remain_separate_draws(self):
        actions = (NO_OP, NO_OP, (11, 20, 30), (10, 21, 30))
        result = posterior(
            energies=(0.0, 0.0, 1.0, 2.0),
            actions=actions,
        )
        self.assertEqual(result.action_count, 4)
        self.assertEqual(result.duplicate_draw_count, 1)
        self.assertEqual(result.target_probabilities.numel(), 4)
        torch.testing.assert_close(
            result.target_probabilities[0], result.target_probabilities[1]
        )
        torch.testing.assert_close(
            result.target_probabilities.sum(), torch.tensor(1.0, dtype=torch.float64)
        )

    def test_k_no_op_and_legal_mask_contracts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "K exactly 4"):
            build_spad_energy_posterior(
                [0.0] * 3,
                [0.0] * 3,
                [True] * 3,
                action_triplets=ACTIONS[:3],
                no_op_triplet=NO_OP,
            )
        with self.assertRaisesRegex(ValueError, "action 0"):
            posterior(actions=((11, 20, 30), NO_OP, ACTIONS[2], ACTIONS[3]))
        with self.assertRaisesRegex(ValueError, "mandatory no-op"):
            posterior(legal=(False, True, True, True))


class CommonStateValidationTest(unittest.TestCase):
    def test_common_state_and_differing_positions_are_validated(self):
        metadata = {
            "sample_idx": 7,
            "plan_id": "plan-7",
            "program": ["Na", "Cl"],
            "site_index": 1,
        }
        result = validate_common_state_group(
            action_triplets=ACTIONS,
            no_op_triplet=NO_OP,
            state_metadata_by_action=[dict(metadata) for _ in range(4)],
            active_positions=(40, 41, 42),
            differing_positions_by_action=((), (40,), (41,), (42,)),
        )
        self.assertEqual(result.action_count, 4)
        self.assertEqual(result.differing_positions[2], (41,))

        mismatched = [dict(metadata) for _ in range(4)]
        mismatched[3]["site_index"] = 2
        with self.assertRaisesRegex(ValueError, "bitwise-identical"):
            validate_common_state_group(
                action_triplets=ACTIONS,
                no_op_triplet=NO_OP,
                state_metadata_by_action=mismatched,
                active_positions=(40, 41, 42),
                differing_positions_by_action=((), (40,), (41,), (42,)),
            )

        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_common_state_group(
                action_triplets=ACTIONS,
                no_op_triplet=NO_OP,
                state_metadata_by_action=[dict(metadata) for _ in range(4)],
                active_positions=(40, 41, 42),
                differing_positions_by_action=((), (40,), (41,), (40, 42)),
            )


class SPADEnergyPosteriorLossTest(unittest.TestCase):
    def test_loss_has_finite_gradients_and_invalid_score_has_zero_gradient(self):
        target = posterior(
            energies=(0.0, 1.0, -1.0e30, 2.0),
            legal=(True, True, False, True),
        )
        policy_scores = torch.tensor(
            [0.2, -0.1, 1000.0, 0.3],
            dtype=torch.float32,
            requires_grad=True,
        )
        output = spad_energy_posterior_loss(policy_scores, target)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(torch.isfinite(output.cross_entropy))
        torch.testing.assert_close(
            output.kl,
            output.cross_entropy - output.target_entropy,
            rtol=1e-5,
            atol=1e-6,
        )
        output.loss.backward()
        self.assertTrue(bool(torch.isfinite(policy_scores.grad).all()))
        self.assertEqual(float(policy_scores.grad[2]), 0.0)
        self.assertLess(abs(float(policy_scores.grad.sum())), 1e-6)


if __name__ == "__main__":
    unittest.main()
