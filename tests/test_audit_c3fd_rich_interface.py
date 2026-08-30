import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_c3fd_rich_interface", ROOT / "scripts/audit_c3fd_rich_interface.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plan():
    return {
        "N": 3,
        "elements": ["Li", "O"],
        "counts": [2, 1],
        "formula": "Li2O",
        "anion_framework": "oxide",
        "charge_bucket": "neutral_plausible",
        "lattice_system": "cubic",
        "spacegroup_bucket": "sg_195_230",
        "volume_per_atom_bin": "volpa_015_019",
    }


class C3FDRichInterfaceAuditTest(unittest.TestCase):
    def test_minimal_and_rich_prompts_preserve_composition(self):
        state = plan()
        self.assertEqual(MODULE.exact_identity(state), "Li:2|O:1")
        self.assertIn('"N":3', MODULE.minimal_prompt(state))
        rich = MODULE.expected_rich_text(state)
        self.assertIn("lattice: cubic", rich)
        self.assertIn("spacegroup: sg_195_230", rich)

    def test_lattice_spacegroup_map_is_complete(self):
        self.assertEqual(len(MODULE.LATTICE_TO_SPACEGROUP), 7)
        self.assertEqual(MODULE.LATTICE_TO_SPACEGROUP["hexagonal"], "sg_168_194")

    def test_distribution_tvd_is_zero_for_equal_counts(self):
        from collections import Counter

        left = Counter({"a": 2, "b": 1})
        self.assertEqual(MODULE.tvd(left, left), 0.0)


if __name__ == "__main__":
    unittest.main()
