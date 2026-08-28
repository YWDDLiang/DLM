import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_ctv_minimal_spec_data",
    ROOT / "scripts" / "build_ctv_minimal_spec_data.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import build_ctv_minimal_spec_data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildCTVMinimalSpecDataTest(unittest.TestCase):
    def test_required_vocab_asset_is_copied_with_identical_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (source / "vocab_tokens.txt").write_text(
                "<N_001>\n<X_000>\n", encoding="utf-8"
            )
            report = MODULE.copy_required_static_assets(source, output)
            self.assertEqual(
                report["vocab_tokens.txt"]["sha256"],
                report["vocab_tokens.txt"]["source_sha256"],
            )
            self.assertEqual(
                (output / "vocab_tokens.txt").read_text(encoding="utf-8"),
                "<N_001>\n<X_000>\n",
            )

    def plan(self):
        return {
            "N": 5,
            "elements": ["Fe", "O"],
            "counts": [2, 3],
            "formula": "Fe2O3",
            "anion_framework": "oxide",
            "charge_bucket": "neutral_plausible",
            "validator": {
                "valid": True,
                "reason": "charge_neutral_pauling_valid",
            },
        }

    def test_minimal_spec_is_canonical_and_contains_no_soft_fields(self):
        spec = MODULE.minimal_spec_from_plan(self.plan())
        self.assertEqual(
            spec,
            {
                "N": 5,
                "charge": "certified_neutral",
                "counts": [3, 2],
                "elements": ["O", "Fe"],
                "family": "oxide",
                "formula": "O3Fe2",
            },
        )
        prompt = MODULE.minimal_prompt(spec)
        self.assertEqual(
            prompt,
            '{"N":5,"charge":"certified_neutral","counts":[3,2],'
            '"elements":["O","Fe"],"family":"oxide","formula":"O3Fe2"}'
            "\ndynamic_crystal_body:",
        )
        self.assertNotIn("lattice", prompt)
        self.assertNotIn("volume", prompt)
        self.assertNotIn("stability", prompt)

    def test_convert_row_removes_counterfactual_prompt(self):
        row = {
            "prompt": "old rich prompt",
            "counterfactual_prompt": "old counterfactual",
            "counterfactual_grounding_eligible": True,
            "answer": "<N_005>",
            "plan_state": self.plan(),
        }
        certificate = {
            "composition_supervision": True,
            "source_row_idx": 7,
            "plan_state": dict(row["plan_state"]),
            "species_labels": [1, 2],
        }
        converted, reason = MODULE.convert_row(row, certificate)
        self.assertEqual(reason, "kept")
        self.assertNotIn("counterfactual_prompt", converted)
        self.assertFalse(converted["counterfactual_grounding_eligible"])
        self.assertEqual(converted["reduced_composition_identity"], "8:3|26:2")
        self.assertEqual(converted["c3fd_certificate_source_row_idx"], 7)

    def test_invalid_composition_is_excluded(self):
        plan = self.plan()
        plan["validator"] = {"valid": False, "reason": "charge_neutrality_fail"}
        converted, reason = MODULE.convert_row({"plan_state": plan, "answer": "x"})
        self.assertIsNone(converted)
        self.assertIn("validator is not positive", reason)

    def test_certificate_sidecar_overrides_stale_validator(self):
        plan = self.plan()
        plan["validator"] = {"valid": False, "reason": "stale"}
        certificate = {
            "composition_supervision": True,
            "source_row_idx": 3,
            "plan_state": dict(plan),
            "species_labels": [1, 2],
        }
        converted, reason = MODULE.convert_row(
            {"plan_state": plan, "answer": "x", "prompt": "old"},
            certificate,
        )
        self.assertEqual(reason, "kept")
        self.assertEqual(converted["minimal_spec"]["charge"], "certified_neutral")

    def test_compiler_node_witness_is_accepted_without_species_ids(self):
        plan = self.plan()
        certificate = {
            "composition_supervision": True,
            "source_row_idx": 5,
            "plan_state": dict(plan),
            "nodes": [
                {"atomic_number": 26, "oxidation_state": 3},
                {"atomic_number": 8, "oxidation_state": -2},
            ],
        }
        converted, reason = MODULE.convert_row(
            {"plan_state": plan, "answer": "x", "prompt": "old"}, certificate
        )
        self.assertEqual(reason, "kept")
        self.assertEqual(converted["minimal_spec"]["charge"], "certified_neutral")


if __name__ == "__main__":
    unittest.main()
