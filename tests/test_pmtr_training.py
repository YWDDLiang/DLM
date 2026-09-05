import unittest

import torch

from crystal_dlm.pmtr_training import (
    freeze_spad_model,
    materialize_transaction_start,
    subset_pmtr_batch,
)


class PMTRTrainingTest(unittest.TestCase):
    def _batch(self):
        prompt = 1
        length = 20
        target = torch.arange(2 * length, dtype=torch.long).reshape(2, length)
        source = target.clone()
        source[0, 2:8] += 50
        source[1, 9:12] += 70
        attention = torch.ones_like(target)
        forced = torch.zeros_like(target, dtype=torch.bool)
        losses = torch.zeros_like(target, dtype=torch.bool)
        # Current PMTR rows supervise the complete transaction from its start.
        forced[0, 2:8] = True
        losses[0, 2:8] = True
        # Future sites in the active species block remain masked but unsupervised.
        forced[1, [9, 10, 11, 13, 14, 15]] = True
        losses[1, 9:12] = True
        return {
            "input_ids": target,
            "source_input_ids": source,
            "attention_mask": attention,
            "prompt_lengths": torch.tensor([prompt, prompt]),
            "num_atoms": torch.tensor([3, 3]),
            "forced_mask_indices": forced,
            "forced_loss_indices": losses,
            "pmtr_repair_targets": [
                {
                    "kind": "cell",
                    "lattice_tangent": torch.eye(3).tolist(),
                    "site_slot_index": None,
                    "cartesian_site_delta_A": None,
                },
                {
                    "kind": "site",
                    "lattice_tangent": None,
                    "site_slot_index": 0,
                    "cartesian_site_delta_A": [0.2, -0.1, 0.0],
                },
            ],
            "pmtr_closure": [
                {"cell_component_index": 3},
                {
                    "reverse_block_index": 1,
                    "site_index_within_block": 0,
                    "site_slot_index": 0,
                },
            ],
            "pmtr_plan_metadata": [{}, {}],
            "pmtr_program_metadata": [
                {"species_order": ["Li", "O"]},
                {"species_order": ["Li", "O"]},
            ],
        }

    def test_rows_materialize_one_full_transaction_start(self):
        batch = self._batch()
        observed = materialize_transaction_start(
            batch, mode="corrupt_repair", mask_id=127
        )
        self.assertEqual(observed.specs[0].active_positions, tuple(range(2, 8)))
        self.assertEqual(observed.specs[1].active_positions, (9, 10, 11))
        self.assertEqual(int(observed.transaction_loss_mask[0].sum()), 6)
        self.assertEqual(int(observed.transaction_loss_mask[1].sum()), 3)
        self.assertTrue(bool(observed.model_mask[0, 2:8].all()))
        self.assertTrue(bool(observed.model_mask[1, 9:12].all()))
        # Existing future-block masks remain visible to the reconstruction.
        self.assertTrue(bool(observed.model_mask[1, 13:16].all()))
        self.assertTrue(
            bool((observed.noisy_tokens[0, 2:8] == 127).all())
        )
        self.assertTrue(
            torch.equal(observed.complete_tokens, batch["source_input_ids"])
        )

    def test_clean_identity_uses_clean_snapshot_with_same_transaction_mask(self):
        batch = self._batch()
        observed = materialize_transaction_start(
            batch, mode="clean_identity", mask_id=127
        )
        self.assertTrue(torch.equal(observed.complete_tokens, batch["input_ids"]))
        self.assertEqual(int(observed.transaction_loss_mask[0].sum()), 6)
        self.assertEqual(int(observed.transaction_loss_mask[1].sum()), 3)

    def test_freeze_spad_model_disables_every_parameter(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Linear(5, 2))
        expected = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(freeze_spad_model(model), expected)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_fallback_row_remains_available_for_clean_identity_and_can_be_subset(self):
        batch = self._batch()
        batch["pmtr_repair_targets"][1] = None
        clean = materialize_transaction_start(
            batch, mode="clean_identity", mask_id=127
        )
        self.assertEqual(len(clean.specs), 2)
        certified = subset_pmtr_batch(batch, [0])
        self.assertEqual(tuple(certified["input_ids"].shape), (1, 20))
        self.assertEqual(len(certified["pmtr_repair_targets"]), 1)
        repair = materialize_transaction_start(
            certified, mode="corrupt_repair", mask_id=127
        )
        self.assertEqual(len(repair.specs), 1)


if __name__ == "__main__":
    unittest.main()
