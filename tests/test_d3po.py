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
    from crystal_dlm.d3po import (
        composition_normalized_pair_weights,
        d3po_pair_loss,
        legal_target_log_probs,
        masked_sequence_log_ratio,
        shared_geometry_corruption,
        winner_denoising_anchor,
    )
    from crystal_dlm.stable_geometry_curriculum import (
        dynamic_geometry_relative_positions,
    )


@unittest.skipIf(torch is None, "torch is not installed")
class SharedGeometryCorruptionTest(unittest.TestCase):
    def test_mask_and_probability_are_shared_and_only_geometry_is_masked(self):
        prompt_length = 2
        num_atoms = 2
        sequence_length = prompt_length + 7 + 4 * num_atoms
        winner = torch.arange(100, 100 + sequence_length, dtype=torch.long).unsqueeze(0)
        loser = winner + 200
        mask_token_id = 999

        result = shared_geometry_corruption(
            winner,
            loser,
            prompt_lengths=torch.tensor([prompt_length]),
            num_atoms=torch.tensor([num_atoms]),
            mask_token_id=mask_token_id,
            p_mask=torch.tensor([1.0]),
        )

        expected_mask = torch.zeros_like(winner, dtype=torch.bool)
        for relative_position in dynamic_geometry_relative_positions(num_atoms):
            expected_mask[0, prompt_length + relative_position] = True
        self.assertTrue(torch.equal(result.geometry_mask, expected_mask))
        self.assertTrue(torch.equal(result.masked_positions, expected_mask))
        self.assertTrue(
            torch.equal(result.winner_noisy_ids == mask_token_id, result.loser_noisy_ids == mask_token_id)
        )
        torch.testing.assert_close(result.p_mask, torch.tensor([1.0]))

        # Dynamic N and the two element tokens (relative positions 7 and 11)
        # are visible even when p=1 masks every geometry candidate.
        for relative_position in (0, 7, 11):
            absolute_position = prompt_length + relative_position
            self.assertFalse(bool(result.masked_positions[0, absolute_position]))
            self.assertEqual(
                int(result.winner_noisy_ids[0, absolute_position]),
                int(winner[0, absolute_position]),
            )
            self.assertEqual(
                int(result.loser_noisy_ids[0, absolute_position]),
                int(loser[0, absolute_position]),
            )

    def test_empty_shared_mask_is_not_forced_nonempty(self):
        num_atoms = 1
        sequence_length = 7 + 4 * num_atoms
        winner = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
        loser = winner + 50
        empty_mask = torch.zeros_like(winner, dtype=torch.bool)
        result = shared_geometry_corruption(
            winner,
            loser,
            prompt_lengths=0,
            num_atoms=num_atoms,
            mask_token_id=999,
            p_mask=0.5,
            shared_mask=empty_mask,
        )
        self.assertFalse(bool(result.masked_positions.any()))
        self.assertTrue(torch.equal(result.winner_noisy_ids, winner))
        self.assertTrue(torch.equal(result.loser_noisy_ids, loser))

        policy = torch.zeros_like(winner, dtype=torch.float32)
        reference = torch.zeros_like(winner, dtype=torch.float32)
        score = masked_sequence_log_ratio(
            policy,
            reference,
            result.masked_positions,
            result.p_mask,
            geometry_mask=result.geometry_mask,
        )
        anchor = winner_denoising_anchor(
            policy,
            result.masked_positions,
            result.p_mask,
            result.geometry_mask,
        )
        torch.testing.assert_close(score, torch.zeros(1))
        torch.testing.assert_close(anchor, torch.zeros(1))

    def test_explicit_non_geometry_mask_is_rejected(self):
        ids = torch.arange(11, dtype=torch.long).unsqueeze(0)
        invalid_mask = torch.zeros_like(ids, dtype=torch.bool)
        invalid_mask[0, 0] = True  # N token
        with self.assertRaisesRegex(ValueError, "non-geometry"):
            shared_geometry_corruption(
                ids,
                ids + 20,
                prompt_lengths=0,
                num_atoms=1,
                p_mask=0.5,
                shared_mask=invalid_mask,
            )


@unittest.skipIf(torch is None, "torch is not installed")
class LegalLogProbabilityTest(unittest.TestCase):
    def test_position_specific_normalization_excludes_illegal_and_mask_tokens(self):
        # Token 0 is illegal and token 5 is MASK; both have deliberately huge
        # logits and therefore expose any accidental full-vocabulary softmax.
        logits = torch.tensor(
            [
                [100.0, 2.0, 0.0, -4.0, -4.0, 200.0],
                [100.0, -4.0, -4.0, 0.0, 0.0, 200.0],
            ],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        targets = torch.tensor([1, 4], dtype=torch.long)
        observed = legal_target_log_probs(
            logits,
            targets,
            legal_token_ids_by_position=((1, 2, 5), (3, 4, 5)),
            mask_token_id=5,
        )
        expected = torch.tensor(
            [
                2.0 - torch.logsumexp(torch.tensor([2.0, 0.0]), dim=0),
                -math.log(2.0),
            ],
            dtype=torch.float32,
        )
        self.assertEqual(observed.dtype, torch.float32)
        torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)

        observed.sum().backward()
        self.assertEqual(float(logits.grad[0, 0]), 0.0)
        self.assertEqual(float(logits.grad[0, 5]), 0.0)
        self.assertNotEqual(float(logits.grad[0, 1]), 0.0)
        self.assertNotEqual(float(logits.grad[0, 2]), 0.0)

    def test_target_outside_legal_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside its legal support"):
            legal_target_log_probs(
                torch.zeros((1, 4)),
                torch.tensor([3]),
                legal_token_ids_by_position=((0, 1),),
                mask_token_id=2,
            )


@unittest.skipIf(torch is None, "torch is not installed")
class D3POMathTest(unittest.TestCase):
    def test_policy_equal_reference_has_zero_margin_and_log2_hard_loss(self):
        masked = torch.tensor([[True, False, True], [False, True, False]])
        selected_log_probs = torch.tensor([-0.2, -0.3, -0.4])
        scores = masked_sequence_log_ratio(
            selected_log_probs,
            selected_log_probs.clone(),
            masked,
            p_mask=torch.tensor([0.5, 0.25]),
        )
        torch.testing.assert_close(scores, torch.zeros(2))

        output = d3po_pair_loss(scores, scores)
        torch.testing.assert_close(output.margin, torch.zeros(2))
        torch.testing.assert_close(
            output.preference_loss,
            torch.tensor(math.log(2.0), dtype=torch.float32),
        )
        torch.testing.assert_close(output.loss, output.preference_loss)

    def test_swapping_winner_and_loser_reverses_margin(self):
        winner = torch.tensor([2.0, -0.5])
        loser = torch.tensor([-1.0, 0.25])
        forward = d3po_pair_loss(winner, loser)
        swapped = d3po_pair_loss(loser, winner)
        torch.testing.assert_close(swapped.margin, -forward.margin)

    def test_sequence_score_uses_one_over_p_in_fp32(self):
        masked = torch.tensor([[True, False, True]])
        policy = torch.tensor([0.5, 1.0], dtype=torch.bfloat16)
        reference = torch.tensor([0.0, 0.0], dtype=torch.bfloat16)
        score = masked_sequence_log_ratio(policy, reference, masked, p_mask=0.25)
        self.assertEqual(score.dtype, torch.float32)
        torch.testing.assert_close(score, torch.tensor([6.0]))

    def test_sequence_score_normalizes_dynamic_geometry_length(self):
        masked = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, True],
            ]
        )
        geometry = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, True],
            ]
        )
        policy = torch.ones(6)
        reference = torch.zeros(6)
        score = masked_sequence_log_ratio(
            policy,
            reference,
            masked,
            p_mask=torch.ones(2),
            geometry_mask=geometry,
        )
        # Both sequences average to one despite different masked/geometry counts.
        torch.testing.assert_close(score, torch.ones(2))

    def test_winner_anchor_is_added_with_frozen_weight(self):
        masked = torch.tensor([[True, False, False, False]])
        geometry = torch.tensor([[True, True, False, False]])
        winner_log_probs = torch.tensor([math.log(0.5)])
        anchor = winner_denoising_anchor(
            winner_log_probs,
            masked,
            p_mask=0.5,
            geometry_mask=geometry,
        )
        torch.testing.assert_close(anchor, torch.tensor([math.log(2.0)]))

        output = d3po_pair_loss(
            torch.zeros(1),
            torch.zeros(1),
            winner_denoising_losses=anchor,
        )
        expected = math.log(2.0) + 0.2 * math.log(2.0)
        torch.testing.assert_close(output.loss, torch.tensor(expected))

    def test_soft_target_uses_frozen_energy_temperature(self):
        output = d3po_pair_loss(
            torch.zeros(1),
            torch.zeros(1),
            energy_gaps=torch.tensor([0.03]),
        )
        torch.testing.assert_close(output.target_probability, torch.sigmoid(torch.ones(1)))
        # At zero margin all soft labels still produce log(2).
        torch.testing.assert_close(output.preference_loss, torch.tensor(math.log(2.0)))

    def test_pair_weights_sum_to_one_within_each_composition(self):
        composition_ids = ("A", "A", "B", "C", "C", "C")
        weights = composition_normalized_pair_weights(
            composition_ids,
            raw_weights=torch.tensor([1.0, 3.0, 2.0, 1.0, 1.0, 2.0]),
        )
        expected = torch.tensor([0.25, 0.75, 1.0, 0.25, 0.25, 0.5])
        torch.testing.assert_close(weights, expected)
        for composition in set(composition_ids):
            group_total = sum(
                float(weights[index])
                for index, value in enumerate(composition_ids)
                if value == composition
            )
            self.assertAlmostEqual(group_total, 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
