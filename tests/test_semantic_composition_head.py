from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - optional local CPU runtime.
    raise unittest.SkipTest("PyTorch is required for semantic-head tests") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.semantic_composition_head import (
    SemanticCompositionHead,
    SemanticHeadFlags,
)


class SemanticCompositionHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.physics = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.head = SemanticCompositionHead(
            hidden_size=8,
            num_species=3,
            physics_features=self.physics,
            rich_soft_head_dims={"anion": 4, "lattice": 3},
        )

    def test_typed_head_shapes_and_fixed_physics_buffer(self):
        hidden = torch.randn(2, 4, 8)
        previous_species = torch.tensor(
            [
                [-1, 0, 1, self.head.eos_species_index],
                [-1, 2, 0, 1],
            ]
        )
        previous_counts = torch.tensor(
            [
                [0, 1, 2, 0],
                [0, 3, 1, 2],
            ]
        )
        output = self.head(
            hidden,
            previous_species_indices=previous_species,
            previous_count_values=previous_counts,
            flags=SemanticHeadFlags(use_physics=True),
        )

        self.assertEqual(output.n_logits.shape, (2, 20))
        self.assertEqual(output.species_logits.shape, (2, 4, 4))
        self.assertEqual(output.count_logits.shape, (2, 4, 20))
        self.assertEqual(output.rich_logits["anion"].shape, (2, 4, 4))
        self.assertEqual(output.rich_logits["lattice"].shape, (2, 4, 3))
        self.assertEqual(self.head.eos_action_index, 60)
        self.assertEqual(self.head.num_joint_actions, 61)
        self.assertFalse(self.head.physics_features.requires_grad)
        self.assertNotIn(
            "physics_features",
            dict(self.head.named_parameters()),
        )

    def test_teacher_forced_loss_is_finite_and_decomposed(self):
        hidden = torch.randn(2, 4, 8, requires_grad=True)
        previous_species = torch.tensor(
            [
                [-1, 0, 1, 2],
                [-1, 2, 1, 0],
            ]
        )
        previous_counts = torch.tensor(
            [
                [0, 2, 1, 3],
                [0, 1, 2, 1],
            ]
        )
        species_targets = torch.tensor(
            [
                [0, 1, 2, self.head.eos_species_index],
                [2, 1, self.head.eos_species_index, -100],
            ]
        )
        # Count zero at EOS is a semantic no-count and is ignored by the loss.
        count_targets = torch.tensor(
            [
                [2, 1, 3, 0],
                [1, 2, 0, -100],
            ]
        )
        output = self.head(
            hidden,
            previous_species_indices=previous_species,
            previous_count_values=previous_counts,
            n_targets=torch.tensor([6, 4]),
            species_targets=species_targets,
            count_targets=count_targets,
            rich_targets={
                "anion": torch.tensor(
                    [[0, 1, 2, 3], [1, 2, 3, -100]]
                )
            },
            flags=SemanticHeadFlags(use_physics=True),
        )

        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss).item())
        self.assertEqual(
            set(output.losses),
            {"n", "species", "count", "rich:anion", "total"},
        )
        for loss in output.losses.values():
            self.assertTrue(torch.isfinite(loss).item())
        output.loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all().item())

    def test_physics_ablation_changes_real_species_but_not_eos(self):
        head = SemanticCompositionHead(
            hidden_size=2,
            num_species=2,
            physics_features=torch.tensor([[1.0], [2.0]]),
            decoder_heads=1,
        )
        with torch.no_grad():
            head.species_embedding.weight.zero_()
            head.physics_projection.weight.fill_(1.0)

        without_physics = head.species_representations(use_physics=False)
        with_physics = head.species_representations(use_physics=True)
        self.assertTrue(torch.equal(without_physics, torch.zeros_like(without_physics)))
        self.assertTrue(torch.equal(with_physics[0], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.equal(with_physics[1], torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.equal(with_physics[head.eos_species_index], torch.zeros(2)))

        actions_off = head.embed_semantic_actions(
            torch.tensor([[0, 1]]),
            torch.tensor([[1, 1]]),
            flags=SemanticHeadFlags(use_physics=False),
        )
        actions_on = head.embed_semantic_actions(
            torch.tensor([[0, 1]]),
            torch.tensor([[1, 1]]),
            flags=SemanticHeadFlags(use_physics=True),
        )
        self.assertFalse(torch.allclose(actions_off, actions_on))

    def test_joint_scores_add_pair_prior_and_leave_eos_separate(self):
        head = SemanticCompositionHead(hidden_size=4, num_species=2, max_count=3)
        species_logits = torch.tensor([[[1.0, 2.0, 7.0]]])
        count_logits = torch.tensor([[[10.0, 20.0, 30.0]]])
        pair_prior = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [1.0, 2.0, 3.0],
            ]
        )
        scores = head.joint_action_scores(
            species_logits,
            count_logits,
            pair_prior_scores=pair_prior,
            flags=SemanticHeadFlags(use_pair_prior=True),
        )
        expected = torch.tensor(
            [[[11.1, 21.2, 31.3, 13.0, 24.0, 35.0, 7.0]]]
        )
        self.assertTrue(torch.allclose(scores, expected))

        no_prior = head.joint_action_scores(
            species_logits,
            count_logits,
            pair_prior_scores=pair_prior,
            flags=SemanticHeadFlags(use_pair_prior=False),
        )
        self.assertTrue(
            torch.allclose(
                no_prior,
                torch.tensor([[[11.0, 21.0, 31.0, 12.0, 22.0, 32.0, 7.0]]]),
            )
        )

    def test_hard_mask_sets_every_illegal_action_to_negative_infinity(self):
        head = SemanticCompositionHead(hidden_size=4, num_species=2, max_count=3)
        species_logits = torch.zeros(1, 1, 3)
        count_logits = torch.zeros(1, 1, 3)
        legal = torch.tensor([True, False, True, False, False, True, False])
        scores = head.joint_action_scores(
            species_logits,
            count_logits,
            legal_action_mask=legal,
            flags=SemanticHeadFlags(use_hard_mask=True),
        )
        self.assertTrue(torch.isneginf(scores[..., ~legal]).all().item())
        self.assertTrue(torch.isfinite(scores[..., legal]).all().item())

        unmasked = head.joint_action_scores(
            species_logits,
            count_logits,
            legal_action_mask=legal,
            flags=SemanticHeadFlags(use_hard_mask=False),
        )
        self.assertTrue(torch.isfinite(unmasked).all().item())
        with self.assertRaisesRegex(ValueError, "requires legal_action_mask"):
            head.joint_action_scores(
                species_logits,
                count_logits,
                flags=SemanticHeadFlags(use_hard_mask=True),
            )

    def test_semantic_action_validation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "positive count"):
            self.head.embed_semantic_actions(
                torch.tensor([[0]]),
                torch.tensor([[0]]),
            )
        with self.assertRaisesRegex(ValueError, "require count zero"):
            self.head.embed_semantic_actions(
                torch.tensor([[self.head.eos_species_index]]),
                torch.tensor([[1]]),
            )
        no_physics_head = SemanticCompositionHead(hidden_size=4, num_species=2)
        with self.assertRaisesRegex(ValueError, "requires a fixed physics"):
            no_physics_head.species_representations(use_physics=True)

    def test_explicit_N_action_has_its_own_embedding_and_contract(self):
        with torch.no_grad():
            self.head.species_embedding.weight.zero_()
            self.head.count_embedding.weight.zero_()
            self.head.n_embedding.weight.zero_()
            self.head.n_embedding.weight[6].fill_(2.0)
        embedded = self.head.embed_semantic_actions(
            torch.tensor([[-1, -1]]),
            torch.tensor([[0, 0]]),
            n_values=torch.tensor([[0, 6]]),
        )
        self.assertTrue(torch.equal(embedded[0, 0], torch.zeros(8)))
        self.assertTrue(torch.equal(embedded[0, 1], torch.full((8,), 2.0)))
        with self.assertRaisesRegex(ValueError, "requires the species sentinel"):
            self.head.embed_semantic_actions(
                torch.tensor([[0]]),
                torch.tensor([[1]]),
                n_values=torch.tensor([[6]]),
            )

    def test_semantic_decoder_is_causal(self):
        self.head.eval()
        hidden_a = torch.zeros(1, 3, 8)
        hidden_b = hidden_a.clone()
        hidden_b[:, 2, :] = 100.0
        previous_species = torch.tensor([[-1, -1, 0]])
        previous_counts = torch.tensor([[0, 0, 1]])
        previous_n = torch.tensor([[0, 6, 0]])
        out_a = self.head(
            hidden_a,
            previous_species_indices=previous_species,
            previous_count_values=previous_counts,
            previous_n_values=previous_n,
        )
        out_b = self.head(
            hidden_b,
            previous_species_indices=previous_species,
            previous_count_values=previous_counts,
            previous_n_values=previous_n,
        )
        self.assertTrue(torch.allclose(out_a.species_logits[:, :2], out_b.species_logits[:, :2]))

    def test_proposal_heads_and_ledger_features_are_explicit(self):
        head = SemanticCompositionHead(
            hidden_size=8,
            num_species=3,
            num_families=7,
            max_arity=7,
            ledger_feature_size=6,
            decoder_layers=1,
            decoder_heads=2,
        )
        head.eval()
        hidden = torch.zeros(2, 3, 8)
        previous_species = torch.tensor([[-1, -1, 0], [-1, -1, 0]])
        previous_counts = torch.tensor([[0, 0, 1], [0, 0, 1]])
        previous_n = torch.tensor([[0, 2, 0], [0, 2, 0]])
        ledger = torch.zeros(2, 3, 6)
        ledger[1, :, 0] = 1.0
        output = head(
            hidden,
            previous_species_indices=previous_species,
            previous_count_values=previous_counts,
            previous_n_values=previous_n,
            ledger_features=ledger,
            family_targets=torch.tensor([0, 1]),
            arity_targets=torch.tensor([2, 2]),
        )
        self.assertEqual(output.family_logits.shape, (2, 7))
        self.assertEqual(output.arity_logits.shape, (2, 7))
        self.assertIn("family", output.losses)
        self.assertIn("arity", output.losses)
        self.assertFalse(
            torch.allclose(output.species_logits[0], output.species_logits[1])
        )


if __name__ == "__main__":
    unittest.main()
