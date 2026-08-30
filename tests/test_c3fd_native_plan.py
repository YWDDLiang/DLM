from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_native_plan import (  # noqa: E402
    C3FD_NATIVE_PLAN_VERSION,
    build_native_body_prompt,
    mask_native_soft_fields,
    parse_native_plan_line,
    serialize_native_plan,
)


def plan():
    return {
        "N": 3,
        "elements": ["Li", "O"],
        "counts": [2, 1],
        "formula": "Li2O",
        "reduced_formula": "Li2O",
        "anion_framework": "oxide",
        "charge_bucket": "neutral_plausible",
        "valence_species": [
            {"element": "Li", "count": 2, "oxidation_state": 1},
            {"element": "O", "count": 1, "oxidation_state": -2},
        ],
        "lattice_system": "cubic",
        "spacegroup_bucket": "sg_195_230",
        "volume_per_atom_bin": "volpa_015_019",
        "prototype_key": "legacy-must-not-leak",
        "oxidation_candidates": "unknown",
    }


class C3FDNativePlanTest(unittest.TestCase):
    def test_native_roundtrip_preserves_hard_and_soft_fields(self):
        source = plan()
        line = serialize_native_plan(source)
        self.assertTrue(line.startswith(C3FD_NATIVE_PLAN_VERSION + ";"))
        self.assertNotIn("prototype", line)
        self.assertNotIn("oxidation_candidates", line)
        parsed = parse_native_plan_line(line)
        self.assertEqual(parsed["N"], source["N"])
        self.assertEqual(parsed["elements"], source["elements"])
        self.assertEqual(parsed["counts"], source["counts"])
        self.assertEqual(parsed["anion_framework"], "oxide")
        self.assertEqual(parsed["lattice_system"], "cubic")
        self.assertEqual(parsed["spacegroup_bucket"], "sg_195_230")
        self.assertEqual(parsed["volume_per_atom_bin"], "volpa_015_019")

    def test_soft_masking_changes_only_deployment_hints(self):
        line = serialize_native_plan(plan())
        masked = mask_native_soft_fields(line)
        self.assertIn("LS=<SOFT_MASK>", masked)
        self.assertIn("SG=<SOFT_MASK>", masked)
        self.assertIn("VP=<SOFT_MASK>", masked)
        for key in ("N=N003", "AF=oxide", "P01=", "P02=", "CB=B_NEU"):
            self.assertIn(key, masked)

    def test_body_prompt_marks_soft_fields_as_hints(self):
        prompt = build_native_body_prompt(plan())
        self.assertIn("soft structural hints", prompt)
        self.assertIn("c3fd_native_plan:", prompt)
        self.assertTrue(prompt.endswith("dynamic_crystal_body:"))

    def test_valence_composition_mismatch_fails_closed(self):
        invalid = plan()
        invalid["valence_species"][0]["count"] = 1
        with self.assertRaises(ValueError):
            serialize_native_plan(invalid)


if __name__ == "__main__":
    unittest.main()
