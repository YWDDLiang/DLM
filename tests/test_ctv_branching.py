import unittest

from crystal_dlm.ctv_branching import (
    free_geometry_positions,
    make_branch_layout,
    newly_crossed_milestones,
    validate_canary_layout,
    visible_free_geometry_fraction,
)


class CTVBranchingProtocolTest(unittest.TestCase):
    def test_free_geometry_excludes_count_and_elements(self):
        positions = free_geometry_positions(2)
        self.assertEqual(positions, (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14))
        self.assertNotIn(0, positions)
        self.assertNotIn(7, positions)
        self.assertNotIn(11, positions)

    def test_visible_fraction_and_milestone_crossing(self):
        mask = 999
        suffix = [1] + [mask] * 12 + [mask, mask]
        for position in free_geometry_positions(2)[:8]:
            suffix[position] = 3
        fraction = visible_free_geometry_fraction(suffix, mask_id=mask, num_atoms=2)
        self.assertAlmostEqual(fraction, 8 / 12)
        self.assertEqual(newly_crossed_milestones(7 / 12, fraction), (0.60,))
        self.assertEqual(newly_crossed_milestones(0.79, 0.81), (0.80,))

    def test_canary_layout_uses_common_noise_across_actions(self):
        rows = []
        for plan in range(8):
            for milestone in (0.60, 0.80):
                rows.extend(
                    make_branch_layout(
                        composition_id=f"plan-{plan}",
                        sample_idx=plan,
                        milestone=milestone,
                        intervention_position=10,
                        action_token_ids=range(100, 108),
                        continuation_seeds=(7001, 7002),
                    )
                )
        report = validate_canary_layout(rows)
        self.assertEqual(report["rows"], 256)
        self.assertEqual(report["states"], 16)
        self.assertEqual(report["common_noise_groups"], 32)
        groups = {
            row["noise_group"]
            for row in rows
            if row["composition_id"] == "plan-0"
            and row["milestone"] == 0.60
            and row["continuation_seed"] == 7001
        }
        self.assertEqual(len(groups), 1)


try:
    from types import SimpleNamespace

    import torch

    from crystal_dlm.ctv_branching import (
        require_gamma_zero_identity,
        select_intervention_from_masked_logits,
        stateless_gumbel_scores,
    )
    from crystal_dlm.ctv_rollout import (
        collect_ctv_branch_states,
        complete_ctv_forced_branches,
    )
except Exception:  # pragma: no cover - torch is optional in lightweight CI.
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CTVBranchingTensorTest(unittest.TestCase):
    def test_intervention_position_uses_confidence_then_position_tie(self):
        mask = 99
        suffix = torch.full((15,), mask, dtype=torch.long)
        allowed = [[0] for _ in range(15)]
        for position in free_geometry_positions(2):
            allowed[position] = list(range(100))
        logits = torch.zeros((15, 128), dtype=torch.float32)
        # Keep quantile actions distinct while making position 9 more certain.
        logits[8, 99] = 0.2
        logits[9, 99] = 0.4
        result = select_intervention_from_masked_logits(
            logits=logits,
            suffix_token_ids=suffix,
            allowed_token_ids_by_generation_pos=allowed,
            num_atoms=2,
            mask_id=mask,
        )
        self.assertEqual(result["position"], 9)
        self.assertEqual(len(set(result["action_token_ids"])), 8)

    def test_common_noise_is_action_independent_and_continuation_specific(self):
        logits = torch.zeros((3, 2, 9), dtype=torch.float32)
        scores = stateless_gumbel_scores(
            logits,
            temperature=0.7,
            noise_groups=("same", "same", "different"),
            denoise_step=4,
        )
        self.assertTrue(torch.equal(scores[0], scores[1]))
        self.assertFalse(torch.equal(scores[0], scores[2]))

    def test_gamma_zero_is_bit_identical(self):
        values = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        report = require_gamma_zero_identity(values, values.clone())
        self.assertTrue(report["passed"])

    def test_tiny_exact_schedule_collects_and_completes_branches(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.output = torch.nn.Embedding(128, 1)

            def get_output_embeddings(self):
                return self.output

            def forward(self, token_ids, attention_mask=None):
                del attention_mask
                batch, length = token_ids.shape
                logits = torch.zeros((batch, length, 128), dtype=torch.float32)
                logits[..., 99] = 0.4
                return SimpleNamespace(logits=logits)

        model = TinyModel()
        prompt = torch.tensor([[1, 2]], dtype=torch.long)
        attention = torch.ones_like(prompt)
        mask = 127
        allowed = [[10]] + [list(range(100)) for _ in range(6)]
        allowed += [[11], list(range(100)), list(range(100)), list(range(100))]
        schedule = [[0], [7], [1, 2, 3, 4, 5, 6], [8], [9], [10]]
        _base, snapshots = collect_ctv_branch_states(
            model,
            prompt,
            attention_mask=attention,
            num_atoms=1,
            gen_length=11,
            temperature=0.0,
            mask_id=mask,
            allowed_token_ids_by_generation_pos=allowed,
            prefill_token_ids_by_generation_pos={0: [10], 7: [11]},
            generation_position_groups=schedule,
            lightweight_decoding_constraints=None,
            base_noise_group="base",
        )
        self.assertEqual([row["milestone"] for row in snapshots], [0.60, 0.80])
        self.assertEqual(snapshots[0]["position"], 8)
        self.assertEqual(snapshots[1]["position"], 10)
        self.assertEqual(len(set(snapshots[0]["action_token_ids"])), 8)
        completed, layout = complete_ctv_forced_branches(
            model,
            snapshots[0],
            composition_id="8:1|26:1",
            sample_idx=3,
            continuation_seeds=(7001, 7002),
            gen_length=11,
            temperature=0.0,
            mask_id=mask,
            allowed_token_ids_by_generation_pos=allowed,
            generation_position_groups=schedule,
            lightweight_decoding_constraints=None,
        )
        self.assertEqual(tuple(completed.shape), (16, 13))
        self.assertFalse(bool((completed[:, 2:] == mask).any()))
        position = 2 + int(snapshots[0]["position"])
        self.assertEqual(
            completed[:, position].tolist(),
            [int(row["action_token"]) for row in layout],
        )


if __name__ == "__main__":
    unittest.main()
