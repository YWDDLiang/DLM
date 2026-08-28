import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_ctv_minimal_spec_data",
    ROOT / "scripts" / "audit_ctv_minimal_spec_data.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import audit_ctv_minimal_spec_data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(range(len(str(text).split())))}


class AuditCTVMinimalSpecDataTest(unittest.TestCase):
    def row(self):
        plan = {
            "N": 5,
            "elements": ["O", "Fe"],
            "counts": [3, 2],
            "formula": "O3Fe2",
            "anion_framework": "oxide",
            "charge_bucket": "neutral_plausible",
            "validator": {
                "valid": True,
                "reason": "charge_neutral_pauling_valid",
            },
        }
        spec = MODULE.minimal_spec_from_plan(plan)
        return {
            "plan_state": plan,
            "minimal_spec": spec,
            "minimal_spec_schema": "h1a2_ctv_minimal_spec_v1",
            "reduced_composition_identity": "8:3|26:2",
            "prompt": MODULE.minimal_prompt(spec),
            "answer": "<N_005>",
        }

    def test_valid_row_passes_and_reports_prompt_length(self):
        result = MODULE.validate_row(self.row(), FakeTokenizer())
        self.assertEqual(result["identity"], "8:3|26:2")
        self.assertGreater(result["prompt_tokens"], 0)

    def test_soft_field_leak_fails(self):
        row = self.row()
        row["minimal_spec"]["lattice_system"] = "cubic"
        with self.assertRaisesRegex(ValueError, "deterministic Plan projection"):
            MODULE.validate_row(row, FakeTokenizer())


if __name__ == "__main__":
    unittest.main()
