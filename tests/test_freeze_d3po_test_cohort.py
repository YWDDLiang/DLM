import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_d3po_test_cohort", ROOT / "scripts/freeze_d3po_test_cohort.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(sample_idx, elements, counts):
    return {
        "sample_idx": sample_idx,
        "plan_state": {
            "N": sum(counts),
            "elements": elements,
            "counts": counts,
            "anion_framework": "oxide",
        },
    }


class FreezeD3POTestCohortTest(unittest.TestCase):
    def test_reduced_identity_blocks_supercell_leakage(self):
        source = [
            row(0, ["Ca", "O"], [2, 2]),
            row(1, ["Li", "O"], [2, 1]),
            row(2, ["Na", "Cl"], [1, 1]),
        ]
        blocked = [row(9, ["Ca", "O"], [1, 1])]
        selected, rejected = MODULE.freeze_rows(source, blocked, count=2)
        self.assertEqual([item["sample_idx"] for item in selected], [1, 2])
        self.assertEqual(rejected["blocked_reduced_identity"], 1)
        self.assertEqual(selected[0]["d3po_test_ordinal"], 0)

    def test_duplicate_source_composition_is_skipped(self):
        source = [
            row(0, ["Li", "O"], [2, 1]),
            row(1, ["Li", "O"], [4, 2]),
            row(2, ["Na", "Cl"], [1, 1]),
        ]
        selected, rejected = MODULE.freeze_rows(source, [], count=2)
        self.assertEqual([item["sample_idx"] for item in selected], [0, 2])
        self.assertEqual(rejected["duplicate_source_reduced_identity"], 1)

    def test_exact_identity_preserves_n(self):
        primitive = row(0, ["Ca", "O"], [1, 1])
        supercell = row(1, ["Ca", "O"], [2, 2])
        self.assertNotEqual(
            MODULE.exact_identity(primitive), MODULE.exact_identity(supercell)
        )
        self.assertEqual(
            MODULE.reduced_identity(primitive), MODULE.reduced_identity(supercell)
        )


if __name__ == "__main__":
    unittest.main()
