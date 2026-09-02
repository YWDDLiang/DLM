import unittest


try:
    import json
    from pathlib import Path
    import tempfile
    import torch
    from scripts.train_spad_species_pointer import (
        DATASET_SCHEMA,
        PointerDataset,
        order_metrics,
    )
except ModuleNotFoundError:
    torch = None
    order_metrics = None


@unittest.skipIf(torch is None, "torch unavailable")
class SPADPointerTrainerTest(unittest.TestCase):
    def test_validation_can_disclose_and_skip_unscorable_frozen_strata(self):
        rows = [
            {
                "schema": DATASET_SCHEMA,
                "source_row_idx": 1,
                "proposal_target": {"family_id": 2, "N": 4, "arity": 2},
            },
            {
                "schema": DATASET_SCHEMA,
                "source_row_idx": 2,
                "proposal_target": {"family_id": 9, "N": 20, "arity": 7},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            dataset = PointerDataset(
                path,
                allowed_strata={(2, 4, 2)},
                skip_unsupported=True,
            )
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.skipped_unsupported[0]["source_row_idx"], 2)

    def test_order_metrics_handle_variable_arity_without_changing_set(self):
        predicted = torch.tensor([[1, 0, 2], [0, 1, 0]])
        teacher = torch.tensor([[1, 2, 0], [0, 1, 0]])
        valid = torch.tensor([[True, True, True], [True, True, False]])
        metrics = order_metrics(predicted, teacher, valid)
        self.assertAlmostEqual(metrics["root_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["exact_permutation_accuracy"], 0.5)
        self.assertGreater(metrics["pairwise_order_accuracy"], 0.5)

    def test_order_metrics_reject_candidate_set_mutation(self):
        with self.assertRaisesRegex(RuntimeError, "candidate set"):
            order_metrics(
                torch.tensor([[0, 0]]),
                torch.tensor([[0, 1]]),
                torch.tensor([[True, True]]),
            )


if __name__ == "__main__":
    unittest.main()
