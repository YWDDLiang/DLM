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

    def test_localize_chunk_rows_accepts_local_ordinals(self):
        rows = [{"ordinal": ordinal} for ordinal in range(256)]
        localized = MODULE.localize_chunk_rows(rows, chunk_index=3)
        self.assertEqual(set(localized), set(range(256)))
        self.assertIs(localized[17], rows[17])

    def test_localize_chunk_rows_maps_global_attempt_ids(self):
        rows = [
            {"attempt_id": f"ctv-validation-{256 + ordinal:05d}"}
            for ordinal in range(256)
        ]
        localized = MODULE.localize_chunk_rows(rows, chunk_index=1)
        self.assertEqual(set(localized), set(range(256)))
        self.assertIs(localized[17], rows[17])

    def test_localize_chunk_rows_rejects_partial_or_mixed_scope(self):
        with self.assertRaisesRegex(ValueError, "duplicate or missing"):
            MODULE.localize_chunk_rows(
                [{"ordinal": ordinal} for ordinal in range(255)], chunk_index=0
            )
        rows = [
            {"attempt_id": f"ctv-validation-{256 + ordinal:05d}"}
            for ordinal in range(256)
        ]
        rows[-1] = {"attempt_id": "ctv-validation-00999"}
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            MODULE.localize_chunk_rows(rows, chunk_index=1)


if __name__ == "__main__":
    unittest.main()
