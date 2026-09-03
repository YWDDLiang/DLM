from pathlib import Path
import math
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
    from crystal_dlm.potential_closure import (
        MAX_KL_BUDGET_NATS,
        build_potential_closure_posterior,
        potential_closure_loss,
        validate_potential_closure_group,
    )


def action_group(k: int, transaction_length: int):
    no_op = tuple(range(10, 10 + transaction_length))
    actions = [no_op]
    differences = [()]
    active_positions = tuple(range(40, 40 + transaction_length))
    for index in range(1, k):
        candidate = list(no_op)
        local_position = (index - 1) % transaction_length
        candidate[local_position] += 100 + index
        actions.append(tuple(candidate))
        differences.append((active_positions[local_position],))
    metadata = {
        "composition_key": "Na1Cl1",
        "sample_idx": 17,
        "masked_state": [1, 2, 3],
    }
    return {
        "action_tokens": tuple(actions),
        "no_op_tokens": no_op,
        "state_metadata_by_action": tuple(dict(metadata) for _ in range(k)),
        "active_positions": active_positions,
        "differing_positions_by_action": tuple(differences),
    }


def posterior(
    *,
    k: int,
    transaction_length: int = 3,
    scores=None,
    energies=None,
    legal=None,
):
    kwargs = action_group(k, transaction_length)
    return build_potential_closure_posterior(
        [0.0] * k if scores is None else scores,
        [0.0, 0.002, 0.004, 0.006][:k] if energies is None else energies,
        [True] * k if legal is None else legal,
        **kwargs,
    )


@unittest.skipIf(torch is None, "torch is not installed")
class VariableKPosteriorTest(unittest.TestCase):
    def test_k2_lower_raw_energy_gains_probability(self):
        result = posterior(k=2, energies=(-2.0, -1.998))
        self.assertEqual(result.action_count, 2)
        self.assertTrue(result.informative)
        self.assertGreater(
            float(result.target_probabilities[0]),
            float(result.reference_probabilities[0]),
        )
        self.assertLessEqual(result.kl_nats, MAX_KL_BUDGET_NATS + 1.0e-12)

    def test_k3_unknown_action_keeps_exact_reference_mass(self):
        result = posterior(
            k=3,
            scores=(0.2, -0.1, 0.4),
            energies=(-2.0, -1.990, None),
        )
        self.assertTrue(result.informative)
        torch.testing.assert_close(
            result.target_probabilities[2],
            result.reference_probabilities[2],
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(
            float(result.target_probabilities[0]),
            float(result.reference_probabilities[0]),
        )

    def test_k4_illegal_action_has_zero_support(self):
        result = posterior(
            k=4,
            energies=(-2.0, -1.99, -100.0, -1.98),
            legal=(True, True, False, True),
        )
        self.assertEqual(result.action_count, 4)
        self.assertEqual(float(result.target_probabilities[2]), 0.0)
        self.assertEqual(float(result.reference_probabilities[2]), 0.0)
        self.assertFalse(bool(result.energy_preference_mask[2]))

    def test_kl_budget_is_reached_or_respected_and_cannot_exceed_cap(self):
        result = posterior(k=4, energies=(-2.0, -1.9, -1.8, -1.7))
        self.assertLessEqual(result.kl_nats, MAX_KL_BUDGET_NATS + 1.0e-12)
        self.assertAlmostEqual(result.kl_nats, MAX_KL_BUDGET_NATS, places=10)

        kwargs = action_group(2, 3)
        with self.assertRaisesRegex(ValueError, "cannot exceed 0.05"):
            build_potential_closure_posterior(
                [0.0, 0.0],
                [-2.0, -1.9],
                [True, True],
                kl_budget_nats=0.050001,
                **kwargs,
            )


@unittest.skipIf(torch is None, "torch is not installed")
class TransactionValidationTest(unittest.TestCase):
    def test_three_and_six_token_transactions_are_supported(self):
        xyz = posterior(k=2, transaction_length=3)
        lattice = posterior(k=2, transaction_length=6)
        self.assertEqual(xyz.transaction_length, 3)
        self.assertEqual(lattice.transaction_length, 6)

    def test_duplicate_complete_actions_are_rejected(self):
        kwargs = action_group(3, 3)
        actions = list(kwargs["action_tokens"])
        actions[2] = actions[1]
        kwargs["action_tokens"] = actions
        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_potential_closure_posterior(
                [0.0, 0.0, 0.0],
                [-2.0, -1.9, -1.8],
                [True, True, True],
                **kwargs,
            )

    def test_no_op_must_be_first_and_legal(self):
        kwargs = action_group(2, 3)
        kwargs["action_tokens"] = tuple(reversed(kwargs["action_tokens"]))
        with self.assertRaisesRegex(ValueError, "action 0"):
            build_potential_closure_posterior(
                [0.0, 0.0],
                [-2.0, -1.9],
                [True, True],
                **kwargs,
            )

        kwargs = action_group(2, 3)
        with self.assertRaisesRegex(ValueError, "must be legal"):
            build_potential_closure_posterior(
                [0.0, 0.0],
                [-2.0, -1.9],
                [False, True],
                **kwargs,
            )

    def test_metadata_and_active_block_are_exact(self):
        kwargs = action_group(2, 3)
        mismatched = list(kwargs["state_metadata_by_action"])
        mismatched[1] = {**mismatched[1], "composition_key": "K1Cl1"}
        with self.assertRaisesRegex(ValueError, "identical state metadata"):
            validate_potential_closure_group(
                legal_mask=[True, True],
                **{**kwargs, "state_metadata_by_action": mismatched},
            )

        wrong_differences = list(kwargs["differing_positions_by_action"])
        wrong_differences[1] = (999,)
        with self.assertRaisesRegex(ValueError, "active block"):
            validate_potential_closure_group(
                legal_mask=[True, True],
                **{
                    **kwargs,
                    "differing_positions_by_action": wrong_differences,
                },
            )


@unittest.skipIf(torch is None, "torch is not installed")
class InformativeAndLossTest(unittest.TestCase):
    def test_fewer_than_two_known_or_sub_mev_spread_is_uninformative(self):
        one_known = posterior(k=3, energies=(-2.0, None, math.nan))
        tiny_spread = posterior(k=2, energies=(-2.0, -1.9995))
        for result in (one_known, tiny_spread):
            self.assertFalse(result.informative)
            self.assertEqual(result.tilt, 0.0)
            torch.testing.assert_close(
                result.target_probabilities,
                result.reference_probabilities,
            )

    def test_illegal_policy_score_has_exactly_zero_gradient(self):
        target = posterior(
            k=4,
            energies=(-2.0, -1.99, -100.0, -1.98),
            legal=(True, True, False, True),
        )
        scores = torch.tensor(
            [0.2, -0.1, 1000.0, 0.3],
            dtype=torch.float32,
            requires_grad=True,
        )
        output = potential_closure_loss(scores, target)
        output.loss.backward()
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertEqual(float(scores.grad[2]), 0.0)
        self.assertLess(abs(float(scores.grad.sum())), 1.0e-6)

    def test_group_kl_and_ce_are_divided_by_transaction_length(self):
        xyz_target = posterior(k=3, transaction_length=3)
        lattice_target = posterior(k=3, transaction_length=6)
        policy_scores = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
        xyz = potential_closure_loss(policy_scores, xyz_target)
        lattice = potential_closure_loss(policy_scores, lattice_target)

        torch.testing.assert_close(xyz.group_kl, lattice.group_kl)
        torch.testing.assert_close(
            xyz.group_cross_entropy,
            lattice.group_cross_entropy,
        )
        torch.testing.assert_close(xyz.kl, xyz.group_kl / 3.0)
        torch.testing.assert_close(lattice.kl, lattice.group_kl / 6.0)
        torch.testing.assert_close(
            xyz.cross_entropy,
            xyz.group_cross_entropy / 3.0,
        )
        torch.testing.assert_close(
            lattice.cross_entropy,
            lattice.group_cross_entropy / 6.0,
        )
        torch.testing.assert_close(xyz.loss, 2.0 * lattice.loss)


if __name__ == "__main__":
    unittest.main()
