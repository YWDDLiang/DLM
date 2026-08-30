import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_rich_recovery_cohort",
    ROOT / "scripts" / "freeze_rich_recovery_cohort.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(index: int):
    return {
        "sample_idx": index,
        "plan_state": {
            "N": 2,
            "elements": ["Na", "Cl" if index % 2 == 0 else "F"],
            "counts": [1, 1],
            "formula": "NaCl" if index % 2 == 0 else "NaF",
            "anion_framework": "halide",
            "charge_bucket": "neutral_plausible",
            "oxidation_candidates": [-1, 1],
            "lattice_system": ["cubic", "tetragonal", "triclinic"][index % 3],
            "spacegroup_bucket": ["sg_195_230", "sg_075_142", "sg_001_002"][index % 3],
            "volume_per_atom_bin": ["volpa_010_014", "volpa_015_019", "volpa_020_024"][index % 3],
            "validator": {"valid": True, "reason": "charge_neutral_pauling_valid"},
            "ccfd_v2_certificate": {"benchmark_valid": True},
        },
    }


class FreezeRichRecoveryCohortTest(unittest.TestCase):
    def test_views_pair_hard_fields_and_derange_joint_tuple(self):
        rows = [fixture(index) for index in range(6)]
        # Make exact compositions unique while keeping valid symbols.
        for index, row in enumerate(rows):
            row["plan_state"]["N"] = index + 2
            row["plan_state"]["counts"] = [index + 1, 1]
            row["plan_state"]["formula"] = f"Na{index + 1}{row['plan_state']['elements'][1]}"
        ledger, views, report = MODULE.freeze(
            rows,
            count=6,
            blocked_exact=set(),
            rcf_shift=1,
        )
        self.assertEqual(len(ledger), 6)
        self.assertTrue(report["rcf_donor_derangement"])
        self.assertTrue(report["rcf_joint_tuple_multiset_preserved"])
        original = [MODULE.soft_tuple(row["plan_state"]) for row in views["R0"]]
        counterfactual = [MODULE.soft_tuple(row["plan_state"]) for row in views["RCF"]]
        self.assertCountEqual(original, counterfactual)
        for m0, r0, rcf in zip(views["M0"], views["R0"], views["RCF"]):
            for field in MODULE.HARD_FIELDS:
                self.assertEqual(r0["plan_state"].get(field), rcf["plan_state"].get(field))
            self.assertIn("dynamic_crystal_body:", m0["prompt"])
            self.assertNotIn("plan_state:", m0["prompt"])
            self.assertIn("plan_state:", r0["prompt"])
            self.assertIn("plan_state:", rcf["prompt"])

    def test_blocked_and_duplicate_exact_compositions_are_removed(self):
        rows = [fixture(0), fixture(0), fixture(1)]
        blocked = {MODULE.exact_identity(rows[0]["plan_state"])}
        with self.assertRaises(RuntimeError):
            MODULE.freeze(rows, count=2, blocked_exact=blocked, rcf_shift=1)

    def test_nested_plan_state_is_discovered(self):
        row = {"source_row": {"source_plan_state": fixture(0)["plan_state"]}}
        self.assertIsNotNone(MODULE.find_plan_state(row))


if __name__ == "__main__":
    unittest.main()
