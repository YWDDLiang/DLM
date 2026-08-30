import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_live_assets", ROOT / "scripts" / "audit_live_assets.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LiveAssetAuditTest(unittest.TestCase):
    def test_directory_tree_hash_is_order_invariant(self):
        rows = [
            {"relative_path": "b", "size_bytes": 1, "sha256": "2" * 64},
            {"relative_path": "a", "size_bytes": 1, "sha256": "1" * 64},
        ]
        self.assertEqual(MODULE.tree_digest(rows), MODULE.tree_digest(list(reversed(rows))))

    def test_asset_hashes_explicit_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.txt").write_text("beta", encoding="utf-8")
            result = MODULE.audit_asset("fixture", root, workers=2)
            self.assertTrue(result["exists"])
            self.assertEqual(result["file_count"], 2)
            self.assertEqual({row["relative_path"] for row in result["files"]}, {"a.txt", "b.txt"})

    def test_missing_asset_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = MODULE.audit_asset("missing", Path(temporary) / "none", workers=1)
            self.assertFalse(result["exists"])
            self.assertEqual(result["kind"], "missing")


if __name__ == "__main__":
    unittest.main()
