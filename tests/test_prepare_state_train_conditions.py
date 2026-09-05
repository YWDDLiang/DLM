from types import SimpleNamespace
import unittest

from scripts.prepare_state_train_conditions import select_conditions


class TrainConditionSelectionTest(unittest.TestCase):
    def fixture(self):
        rows = [{"source_row_idx": i, "proposal_target": {"family_id": 0, "N": 1, "arity": 1},
                 "canonical_atomic_numbers": [i + 1], "canonical_element_counts": [1]}
                for i in range(8)]
        return rows, [{"source_row_idx": i} for i in range(8)], SimpleNamespace(stratum_to_index={(0, 1, 1): 0})

    def test_fixed_seed_and_unique_composition(self):
        rows, sft, bundle = self.fixture()
        rows[1]["canonical_atomic_numbers"] = rows[0]["canonical_atomic_numbers"]
        first, _, report = select_conditions(rows, sft, bundle, count=5, seed=17)
        second, _, _ = select_conditions(rows, sft, bundle, count=5, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len({tuple(r["canonical_atomic_numbers"]) for r in first}), 5)
        self.assertEqual(report["eligible_metadata_rows"], 8)

    def test_metadata_exclusions_are_reported_not_energy_selected(self):
        rows, sft, bundle = self.fixture()
        rows[0]["proposal_target"]["N"] = 20
        for row in rows:
            row["energy"] = object()
        selected, _, report = select_conditions(rows, sft, bundle, count=5, seed=17)
        self.assertEqual(report["metadata_unavailable_or_unsupported"], 1)
        self.assertNotIn(0, [r["source_row_idx"] for r in selected])


if __name__ == "__main__":
    unittest.main()
