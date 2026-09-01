import importlib.util
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("H1_ACTIVE_DENOMINATOR", "1200")
sys.path.insert(0, str(ROOT / "eval_runtime"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLLECT = load("collect_plan_official_inputs", "scripts/collect_plan_official_inputs.py")
FREEZE = load(
    "freeze_official_available_plan_splits",
    "scripts/freeze_official_available_plan_splits.py",
)


def plan(index, elements=("Na", "Cl")):
    return {
        "sample_idx": index,
        "plan_state": {"N": 2, "elements": list(elements), "counts": [1, 1]},
    }


class OfficialAvailabilityTest(unittest.TestCase):
    def test_collects_unique_chemsys_without_energy(self) -> None:
        rows = [
            {**plan(0), "trajectory_attempts": 1, "comp_valid": True},
            {**plan(1), "trajectory_attempts": 1, "comp_valid": True},
            {"sample_idx": 2, "trajectory_attempts": 1, "comp_valid": False},
        ]
        wanted, failures = COLLECT.collect(rows, expected_requested=3)
        self.assertEqual(wanted, {"Cl-Na"})
        self.assertEqual(failures, 1)

    def test_first_known_primary_and_remainder_are_reindexed(self) -> None:
        records = [
            {**plan(index), "trajectory_attempts": 1, "comp_valid": True}
            for index in range(4)
        ]
        plans = [plan(index) for index in range(4)]
        accounting, primary, remainder = FREEZE.freeze(
            records,
            plans,
            resolved={"Cl-Na"},
            unresolved=set(),
            expected_requested=4,
            primary_count=3,
        )
        self.assertEqual(len(primary), 3)
        self.assertEqual(len(remainder), 1)
        self.assertEqual([row["sample_idx"] for row in primary], [0, 1, 2])
        self.assertEqual(remainder[0]["sample_idx"], 0)
        self.assertEqual(remainder[0]["source_sample_idx"], 3)
        self.assertTrue(all(row["eligible"] for row in accounting))

    def test_unknowns_are_preserved_but_excluded_from_execution_splits(self) -> None:
        records = [
            {**plan(0), "trajectory_attempts": 1, "comp_valid": True},
            {**plan(1, ("K", "Br")), "trajectory_attempts": 1, "comp_valid": True},
        ]
        plans = [plan(0), plan(1, ("K", "Br"))]
        accounting, primary, remainder = FREEZE.freeze(
            records,
            plans,
            resolved={"Cl-Na"},
            unresolved={"Br-K"},
            expected_requested=2,
            primary_count=1,
        )
        self.assertEqual(len(primary), 1)
        self.assertEqual(remainder, [])
        self.assertEqual(accounting[1]["reason"], "official_reference_unknown")
        self.assertFalse(accounting[1]["eligible"])


if __name__ == "__main__":
    unittest.main()
