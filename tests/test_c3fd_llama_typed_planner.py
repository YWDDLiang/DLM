from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required for typed Planner tests") from exc


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_llama_typed_planner import (
    C3FDLlamaTypedPlannerConfig,
    C3FDLlamaTypedResidualPlanner,
    SOFT_FIELDS,
    joint_action_index,
    masked_log_softmax,
    row_balanced_typed_loss,
    unit_weight_poe_log_probs,
)


def config() -> C3FDLlamaTypedPlannerConfig:
    return C3FDLlamaTypedPlannerConfig(
        llama_hidden_size=8,
        typed_embedding_size=4,
        num_stability_goals=3,
        num_proposal_states=5,
        num_proposal_strata=4,
        num_species=3,
        max_count=2,
        ledger_feature_size=3,
        num_lattice_systems=3,
        num_spacegroup_buckets=4,
        num_volume_per_atom_bins=2,
        max_sequence_length=5,
    )


def all_legal(shape):
    return torch.ones(shape, dtype=torch.bool)


class TypedPoETest(unittest.TestCase):
    def test_step0_exactly_preserves_masked_c3fd_probabilities(self):
        torch.manual_seed(1)
        module = C3FDLlamaTypedResidualPlanner(config())
        hidden = torch.randn(2, 3, 8)
        residual = module(hidden, soft_position_indices=torch.tensor([2, 2])).actions
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

        c3fd = torch.tensor(
            [
                [[2.0, -1.0, 0.4, 8.0, -3.0, 0.0, 1.0]] * 3,
                [[-0.2, 0.7, 1.1, -4.0, 2.0, 3.0, -1.0]] * 3,
            ]
        )
        legal = torch.ones_like(c3fd, dtype=torch.bool)
        legal[..., 3] = False
        expected = masked_log_softmax(c3fd, legal)
        observed = unit_weight_poe_log_probs(c3fd, residual, legal)
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.isneginf(observed[..., 3]).all().item())

    def test_illegal_entries_stay_negative_infinity_and_bad_masks_fail(self):
        c3fd = torch.tensor([[0.0, 1.0, -torch.inf]])
        residual = torch.tensor([[2.0, -1.0, -torch.inf]])
        legal = torch.tensor([[True, True, False]])
        fused = unit_weight_poe_log_probs(c3fd, residual, legal)
        self.assertTrue(torch.isneginf(fused[0, 2]).item())
        self.assertAlmostEqual(float(torch.exp(fused[0, :2]).sum()), 1.0, places=6)
        with self.assertRaises(ValueError):
            unit_weight_poe_log_probs(c3fd, residual[:, :2], legal)
        with self.assertRaises(TypeError):
            unit_weight_poe_log_probs(c3fd, residual, legal.float())
        with self.assertRaises(ValueError):
            unit_weight_poe_log_probs(c3fd, residual, torch.zeros_like(legal))

    def test_nonzero_residual_allows_llama_to_change_one_step(self):
        module = C3FDLlamaTypedResidualPlanner(config())
        with torch.no_grad():
            module.action_head.weight[0, 0] = 2.0
        hidden = torch.zeros(1, 1, 8)
        hidden[..., 0] = 1.0
        residual = module(hidden, soft_position_indices=torch.tensor([0])).actions
        c3fd = torch.zeros_like(residual)
        legal = torch.ones_like(residual, dtype=torch.bool)
        fused = unit_weight_poe_log_probs(c3fd, residual, legal)
        detached = fused.detach()
        self.assertGreater(float(detached[0, 0, 0]), float(detached[0, 0, 1]))
        self.assertEqual(int(fused[0, 0].argmax()), 0)


class TypedInputAndHeadTest(unittest.TestCase):
    def test_typed_input_shape_and_action_validation(self):
        module = C3FDLlamaTypedResidualPlanner(config())
        embedded = module.typed_inputs_embeds(
            stability_goal_ids=torch.tensor([1, 2]),
            proposal_state_ids=torch.tensor([[0, 1, 2], [2, 3, 4]]),
            previous_species_indices=torch.tensor([[-1, 0, 3], [-1, 1, 2]]),
            previous_count_values=torch.tensor([[0, 2, 0], [0, 1, 2]]),
            ledger_features=torch.zeros(2, 3, 3),
        )
        self.assertEqual(tuple(embedded.shape), (2, 3, 8))
        with self.assertRaises(ValueError):
            module.typed_inputs_embeds(
                stability_goal_ids=torch.tensor([1, 2]),
                proposal_state_ids=torch.tensor([[0, 1], [2, 3]]),
                previous_species_indices=torch.tensor([[-1, 0], [-1, 1]]),
                previous_count_values=torch.tensor([[1, 2], [0, 1]]),
                ledger_features=torch.zeros(2, 2, 3),
            )
        with self.assertRaises(ValueError):
            module.typed_inputs_embeds(
                stability_goal_ids=torch.tensor([1, 2]),
                proposal_state_ids=torch.tensor([[0, 1], [2, 3]]),
                previous_species_indices=torch.tensor([[-1, 0], [-1, 1]]),
                previous_count_values=torch.tensor([[0, 2], [0, 1]]),
                ledger_features=torch.full((2, 2, 3), 1.01),
            )
        with self.assertRaises(ValueError):
            module(
                torch.zeros(2, 8),
                soft_position_indices=torch.tensor([0, 0]),
            )

    def test_all_residual_output_heads_are_zero_initialized(self):
        module = C3FDLlamaTypedResidualPlanner(config())
        heads = [module.proposal_head, module.action_head, *module.soft_field_heads.values()]
        for head in heads:
            self.assertTrue(torch.equal(head.weight, torch.zeros_like(head.weight)))
            self.assertTrue(torch.equal(head.bias, torch.zeros_like(head.bias)))
        output = module(
            torch.randn(2, 3, 8),
            soft_position_indices=torch.tensor([1, 2]),
        )
        self.assertEqual(tuple(output.proposal.shape), (2, 4))
        self.assertEqual(tuple(output.actions.shape), (2, 3, 7))
        self.assertEqual(set(output.soft_fields), set(SOFT_FIELDS))

    def test_soft_heads_read_terminal_composition_state(self):
        module = C3FDLlamaTypedResidualPlanner(config())
        with torch.no_grad():
            module.soft_field_heads["lattice_system"].weight[0, 0] = 1.0
        hidden = torch.zeros(2, 3, 8)
        hidden[0, 1, 0] = 2.0
        hidden[0, 2, 0] = 7.0
        hidden[1, 1, 0] = 3.0
        hidden[1, 2, 0] = 11.0
        output = module(
            hidden,
            soft_position_indices=torch.tensor([1, 2]),
        )
        observed = output.soft_fields["lattice_system"][:, 0]
        self.assertTrue(torch.equal(observed, torch.tensor([2.0, 11.0])))
        with self.assertRaises(ValueError):
            module(hidden, soft_position_indices=torch.tensor([3, 2]))

    def test_joint_action_teacher_encoding_fails_closed(self):
        cfg = config()
        self.assertEqual(
            joint_action_index(2, 2, num_species=cfg.num_species, max_count=cfg.max_count),
            5,
        )
        self.assertEqual(
            joint_action_index(3, 0, num_species=cfg.num_species, max_count=cfg.max_count),
            6,
        )
        with self.assertRaises(ValueError):
            joint_action_index(3, 1, num_species=cfg.num_species, max_count=cfg.max_count)
        with self.assertRaises(ValueError):
            joint_action_index(1, 0, num_species=cfg.num_species, max_count=cfg.max_count)

    def test_no_pair_by_language_vocabulary_parameter(self):
        module = C3FDLlamaTypedResidualPlanner(config())
        self.assertFalse(hasattr(module.config, "vocab_size"))
        self.assertEqual(module.action_head.out_features, 3 * 2 + 1)
        self.assertTrue(all(parameter.ndim <= 2 for parameter in module.parameters()))


class RowBalancedLossTest(unittest.TestCase):
    @staticmethod
    def loss_for_action_targets(action_targets: torch.Tensor) -> torch.Tensor:
        batch, sequence = action_targets.shape
        proposal_logits = torch.zeros(batch, 2)
        proposal_mask = all_legal(proposal_logits.shape)
        proposal_probs = masked_log_softmax(proposal_logits, proposal_mask)

        action_logits = torch.zeros(batch, sequence, 3)
        action_logits[..., 0] = 1.0
        action_mask = all_legal(action_logits.shape)
        action_probs = masked_log_softmax(action_logits, action_mask)

        soft_probs = {}
        soft_masks = {}
        soft_targets = {}
        for field in SOFT_FIELDS:
            logits = torch.zeros(batch, 2)
            mask = all_legal(logits.shape)
            soft_probs[field] = masked_log_softmax(logits, mask)
            soft_masks[field] = mask
            soft_targets[field] = torch.zeros(batch, dtype=torch.long)
        return row_balanced_typed_loss(
            proposal_log_probs=proposal_probs,
            proposal_targets=torch.zeros(batch, dtype=torch.long),
            proposal_legal_mask=proposal_mask,
            action_log_probs=action_probs,
            action_targets=action_targets,
            action_legal_mask=action_mask,
            soft_field_log_probs=soft_probs,
            soft_field_targets=soft_targets,
            soft_field_legal_masks=soft_masks,
        )

    def test_different_arities_have_equal_row_weight(self):
        short = torch.tensor([[0, -100, -100], [1, -100, -100]])
        long = torch.tensor([[0, -100, -100], [1, 1, 1]])
        self.assertTrue(torch.allclose(self.loss_for_action_targets(short), self.loss_for_action_targets(long)))

    def test_illegal_teacher_action_fails_closed(self):
        targets = torch.tensor([[0], [1]])
        batch = 2
        proposal_logits = torch.zeros(batch, 2)
        proposal_mask = all_legal(proposal_logits.shape)
        action_logits = torch.zeros(batch, 1, 3)
        action_mask = all_legal(action_logits.shape)
        action_mask[1, 0, 1] = False
        soft_logits = {field: torch.zeros(batch, 2) for field in SOFT_FIELDS}
        soft_masks = {field: all_legal((batch, 2)) for field in SOFT_FIELDS}
        with self.assertRaises(ValueError):
            row_balanced_typed_loss(
                proposal_log_probs=masked_log_softmax(proposal_logits, proposal_mask),
                proposal_targets=torch.zeros(batch, dtype=torch.long),
                proposal_legal_mask=proposal_mask,
                action_log_probs=masked_log_softmax(action_logits, action_mask),
                action_targets=targets,
                action_legal_mask=action_mask,
                soft_field_log_probs={
                    field: masked_log_softmax(logits, soft_masks[field])
                    for field, logits in soft_logits.items()
                },
                soft_field_targets={field: torch.zeros(batch, dtype=torch.long) for field in SOFT_FIELDS},
                soft_field_legal_masks=soft_masks,
            )

    def test_gradients_reach_output_head_and_typed_projector(self):
        torch.manual_seed(3)
        module = C3FDLlamaTypedResidualPlanner(config())
        with torch.no_grad():
            module.proposal_head.weight.normal_(std=0.05)
            module.action_head.weight.normal_(std=0.05)
            for head in module.soft_field_heads.values():
                head.weight.normal_(std=0.05)

        hidden = module.typed_inputs_embeds(
            stability_goal_ids=torch.tensor([0, 1]),
            proposal_state_ids=torch.tensor([[0, 1], [1, 2]]),
            previous_species_indices=torch.tensor([[-1, 0], [-1, 1]]),
            previous_count_values=torch.tensor([[0, 1], [0, 2]]),
            ledger_features=torch.zeros(2, 2, 3),
        )
        residual = module(
            hidden,
            soft_position_indices=torch.tensor([1, 1]),
        )
        proposal_mask = all_legal(residual.proposal.shape)
        action_mask = all_legal(residual.actions.shape)
        proposal_probs = unit_weight_poe_log_probs(
            torch.zeros_like(residual.proposal), residual.proposal, proposal_mask
        )
        action_probs = unit_weight_poe_log_probs(
            torch.zeros_like(residual.actions), residual.actions, action_mask
        )
        soft_probs = {}
        soft_masks = {}
        soft_targets = {}
        for field, logits in residual.soft_fields.items():
            mask = all_legal(logits.shape)
            soft_masks[field] = mask
            soft_probs[field] = unit_weight_poe_log_probs(
                torch.zeros_like(logits), logits, mask
            )
            soft_targets[field] = torch.zeros(2, dtype=torch.long)
        loss = row_balanced_typed_loss(
            proposal_log_probs=proposal_probs,
            proposal_targets=torch.tensor([0, 1]),
            proposal_legal_mask=proposal_mask,
            action_log_probs=action_probs,
            action_targets=torch.tensor([[0, 1], [1, 6]]),
            action_legal_mask=action_mask,
            soft_field_log_probs=soft_probs,
            soft_field_targets=soft_targets,
            soft_field_legal_masks=soft_masks,
        )
        loss.backward()
        self.assertGreater(float(module.action_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.typed_projector.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
