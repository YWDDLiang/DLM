from pathlib import Path
import json
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_native_plan import (  # noqa: E402
    C3FD_NATIVE_PLAN_VERSION,
    build_native_body_prompt,
    build_native_inference_prompt,
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
        self.assertEqual(json.loads(line)["schema"], C3FD_NATIVE_PLAN_VERSION)
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
        payload = json.loads(masked)
        self.assertEqual(payload["lattice_system"], "<SOFT_MASK>")
        self.assertEqual(payload["spacegroup_bucket"], "<SOFT_MASK>")
        self.assertEqual(payload["volume_per_atom_bin"], "<SOFT_MASK>")
        self.assertEqual(payload["N"], 3)
        self.assertEqual(payload["elements"], ["Li", "O"])
        self.assertEqual(payload["counts"], [2, 1])
        self.assertNotIn("charge_bucket", payload)
        self.assertNotIn("valence_species", payload)

    def test_body_prompt_marks_soft_fields_as_hints(self):
        prompt = build_native_body_prompt(plan())
        self.assertIn("soft structural hints", prompt)
        self.assertIn("c3fd_native_plan:", prompt)
        self.assertTrue(prompt.endswith("dynamic_crystal_body:"))

    def test_legacy_certificate_fields_do_not_change_native_interface(self):
        clean = plan()
        legacy = dict(clean)
        legacy["charge_bucket"] = "charge_fail"
        legacy["valence_species"] = [
            {"element": "Li", "count": 1, "oxidation_state": 7}
        ]
        self.assertEqual(serialize_native_plan(clean), serialize_native_plan(legacy))

    def test_train_and_inference_renderers_are_byte_identical(self):
        source = plan()
        teacher_prompt = build_native_body_prompt(source)
        inference_prompt = build_native_inference_prompt(source, source)
        self.assertEqual(teacher_prompt, inference_prompt)

    def test_predicted_prompt_changes_only_three_soft_values(self):
        source = plan()
        predicted = {
            "lattice_system": "tetragonal",
            "spacegroup_bucket": "sg_075_142",
            "volume_per_atom_bin": "volpa_025_029",
        }
        teacher = json.loads(serialize_native_plan(source))
        rendered = build_native_inference_prompt(source, predicted)
        line = rendered.split("c3fd_native_plan: ", 1)[1].splitlines()[0]
        candidate = json.loads(line)
        changed = {key for key in teacher if teacher[key] != candidate[key]}
        self.assertEqual(
            changed,
            {"lattice_system", "spacegroup_bucket", "volume_per_atom_bin"},
        )
        self.assertEqual(
            rendered.split("c3fd_native_plan: ", 1)[0],
            build_native_body_prompt(source).split("c3fd_native_plan: ", 1)[0],
        )
        self.assertTrue(rendered.endswith("\ndynamic_crystal_body:"))


if __name__ == "__main__":
    unittest.main()
