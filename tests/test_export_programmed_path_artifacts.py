import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("export_programmed_path_artifacts", Path(__file__).resolve().parents[1] / "scripts/export_programmed_path_artifacts.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Structure:
    num_sites = 1
    composition = "H"

    def to(self, **kwargs):
        return "CIF"

    def as_dict(self):
        return {"fixture": "structure"}

    @classmethod
    def from_str(cls, *args, **kwargs):
        return cls()


class ExportAccountingTest(unittest.TestCase):
    def run_export(self, *, refined=False, changed_composition=False):
        row = {"sample_idx": 0, "source_split": "evaluation", "success": False, "body": "complete",
               "structure": {"stale": "native"}, "cif_path": "stale.cif"}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "artifacts"
            argv = ["export", "--paths-jsonl", "unused.jsonl", "--output-dir", str(output)]
            if refined:
                argv += ["--refined-pt", "unused.pt"]
            parsed = Structure()
            if changed_composition:
                parsed.composition = "He"
            with patch.object(sys, "argv", argv), \
                 patch.dict(sys.modules, {"pymatgen.core": SimpleNamespace(Structure=Structure)}), \
                 patch.object(MODULE, "read_jsonl", return_value=[row]), \
                 patch.object(MODULE, "parse_dynamic_answer", return_value={}), \
                 patch.object(MODULE, "arrays_to_structure", return_value=Structure()), \
                 patch.object(MODULE, "load_refined_payload", return_value=({}, {})), \
                 patch.object(Structure, "from_str", return_value=parsed):
                MODULE.main()
            return json.loads((output / "paths.jsonl").read_text())

    def test_available_native_cif_is_evaluated_even_after_runtime_failure(self):
        row = self.run_export()
        self.assertTrue(row["parseable"])
        self.assertTrue(row["success"])
        self.assertFalse(row["native_execution_success"])

    def test_missing_refined_output_drops_stale_native_structure(self):
        row = self.run_export(refined=True)
        self.assertFalse(row["success"])
        self.assertFalse(row["parseable"])
        self.assertNotIn("structure", row)
        self.assertNotIn("cif_path", row)

    def test_cif_roundtrip_must_preserve_composition(self):
        row = self.run_export(changed_composition=True)
        self.assertFalse(row["success"])
        self.assertFalse(row["parseable"])
        self.assertIn("composition", row["artifact_error"])


if __name__ == "__main__":
    unittest.main()
