from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_c3fd_native_prospective_cohort.py"
SPEC = importlib.util.spec_from_file_location("freeze_native_prospective", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def plan(element, index):
    return {
        "sample_idx": index,
        "plan_state": {
            "N": 3,
            "elements": [element, "O"],
            "counts": [2, 1],
            "anion_framework": "oxide",
            "lattice_system": "cubic",
            "spacegroup_bucket": "sg_195_230",
            "volume_per_atom_bin": "volpa_015_019",
        },
    }


class FreezeC3FDNativeProspectiveCohortTest(unittest.TestCase):
    def test_sampling_seed_is_configurable(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--planner-sampling-seed", source)
        self.assertIn("int(args.planner_sampling_seed)", source)

    def test_exact_blocking_and_contiguous_execution_indices(self):
        rows = [plan("Li", 0), plan("Na", 1), plan("K", 2)]
        blocked = {MODULE.exact_identity(rows[0]["plan_state"])}
        plans, ledger, report = MODULE.freeze(
            source_rows=rows,
            blocked_exact=blocked,
            count=2,
        )
        self.assertEqual([row["sample_idx"] for row in plans], [0, 1])
        self.assertEqual([row["source_sample_idx"] for row in ledger], [1, 2])
        self.assertEqual(report["exclusions"]["blocked_exact"], 1)
        self.assertTrue(all(row["prompt"].endswith("dynamic_crystal_body:") for row in plans))

    def test_exact_identity_preserves_nonreduced_counts(self):
        value = plan("Li", 0)["plan_state"]
        self.assertEqual(MODULE.exact_identity(value), "Li:2|O:1")
        self.assertEqual(MODULE.reduced_identity(value), "Li:2|O:1")


if __name__ == "__main__":
    unittest.main()
