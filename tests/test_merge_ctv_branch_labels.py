import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "merge_ctv_branch_labels",
    ROOT / "scripts" / "merge_ctv_branch_labels.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import CTV label merger")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MergeCTVBranchLabelsTest(unittest.TestCase):
    def test_row_ordinal_prefers_explicit_value(self):
        self.assertEqual(MODULE.row_ordinal({"ordinal": 7, "attempt_id": "x-99"}), 7)
        self.assertEqual(MODULE.row_ordinal({"attempt_id": "ctv-train-00042"}), 42)

    def test_row_ordinal_requires_identity(self):
        with self.assertRaises(ValueError):
            MODULE.row_ordinal({})


if __name__ == "__main__":
    unittest.main()
