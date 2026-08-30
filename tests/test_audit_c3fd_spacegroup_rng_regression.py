import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_c3fd_spacegroup_rng_regression",
    ROOT / "scripts" / "audit_c3fd_spacegroup_rng_regression.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_rows(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def plan(index, sg):
    return {
        "sample_idx": index,
        "plan_state": {
            "N": 2,
            "elements": ["Na", "Cl"],
            "counts": [1, 1],
            "formula": "NaCl",
            "anion_framework": "halide",
            "charge_bucket": "neutral_plausible",
            "lattice_system": "cubic",
            "spacegroup_bucket": sg,
            "volume_per_atom_bin": "volpa_020_024",
        },
    }


def raw(index):
    return {
        "sample_idx": index,
        "semantic_trace": [{"action": "proposal"}, {"action": "EOS"}],
        "target_proposal": {"N": 2},
        "certificate": {"benchmark_valid": True},
        "failure": None,
    }


class SpacegroupRNGRegressionTest(unittest.TestCase):
    def test_only_spacegroup_may_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy"
            corrected = root / "corrected"
            legacy.mkdir()
            corrected.mkdir()
            write_rows(legacy / "plans_for_dlm.jsonl", [plan(0, "sg_195_230")])
            write_rows(corrected / "plans_for_dlm.jsonl", [plan(0, "sg_016_074")])
            write_rows(legacy / "raw_generations.jsonl", [raw(0)])
            write_rows(corrected / "raw_generations.jsonl", [raw(0)])
            result = MODULE.compare_pair(name="seed17", legacy_dir=legacy, corrected_dir=corrected)
            self.assertTrue(result["gate"]["pass"])
            self.assertEqual(result["spacegroup_changed"], 1)

    def test_prefix_change_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy"
            corrected = root / "corrected"
            legacy.mkdir()
            corrected.mkdir()
            left = plan(0, "sg_195_230")
            right = plan(0, "sg_016_074")
            right["plan_state"]["volume_per_atom_bin"] = "volpa_025_029"
            write_rows(legacy / "plans_for_dlm.jsonl", [left])
            write_rows(corrected / "plans_for_dlm.jsonl", [right])
            write_rows(legacy / "raw_generations.jsonl", [raw(0)])
            write_rows(corrected / "raw_generations.jsonl", [raw(0)])
            result = MODULE.compare_pair(name="seed17", legacy_dir=legacy, corrected_dir=corrected)
            self.assertFalse(result["gate"]["pass"])
            self.assertEqual(result["prefix_mismatch"], [0])


if __name__ == "__main__":
    unittest.main()
