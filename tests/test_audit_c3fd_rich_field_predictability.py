import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_c3fd_rich_field_predictability",
    ROOT / "scripts" / "audit_c3fd_rich_field_predictability.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RichFieldPredictabilityAuditTest(unittest.TestCase):
    def test_ece_is_zero_for_perfect_binary_predictions(self):
        self.assertAlmostEqual(
            MODULE.expected_calibration_error([1.0, 1.0], [True, True]),
            0.0,
        )

    def test_summary_reports_majority_and_ordinal_error(self):
        result = MODULE.summarize_predictions(
            [0, 0, 1, 2],
            [0, 1, 1, 1],
            [0.9, 0.6, 0.8, 0.7],
            nll_sum=2.0,
            labels=["a", "b", "c"],
            ordered=True,
        )
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["majority_accuracy"], 0.5)
        self.assertEqual(result["ordinal_mae_bins"], 0.5)

    def test_spacegroup_relation_exposes_compiler_redundancy(self):
        lattice = list(MODULE.LATTICE_TO_SPACEGROUP)
        spacegroups = [MODULE.LATTICE_TO_SPACEGROUP[value] for value in lattice]
        result = MODULE.summarize_spacegroup_relation(
            lattice_targets=[0, 1, 2],
            lattice_predictions=[0, 2, 2],
            sg_targets=[0, 1, 2],
            sg_predictions=[0, 1, 1],
            lattice_labels=lattice,
            sg_labels=spacegroups,
        )
        self.assertEqual(result["target_lattice_sg_compatible"], 1.0)
        self.assertAlmostEqual(result["lattice_derived_sg_accuracy"], 2 / 3)
        self.assertTrue(result["deployed_sg_is_deterministic_compiler_output"])
        self.assertEqual(result["deployed_sg_incremental_entropy_given_lattice_nats"], 0.0)


if __name__ == "__main__":
    unittest.main()
