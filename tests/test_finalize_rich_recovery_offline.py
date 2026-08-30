import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_rich_recovery_offline",
    ROOT / "scripts" / "finalize_rich_recovery_offline.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalizeRichRecoveryOfflineTest(unittest.TestCase):
    def test_cell_directory_mapping(self):
        root = Path("/eval")
        self.assertEqual(MODULE.cell_root(root, "refined", 17, "R0"), root / "stream17" / "R0")
        self.assertEqual(MODULE.cell_root(root, "raw", 18, "RCF"), root / "stream18" / "raw_RCF")

    def test_comparison_contract_is_frozen(self):
        self.assertEqual(
            [value[0] for value in MODULE.COMPARISONS],
            ["R0_minus_RCF", "R0_minus_M0", "RCF_minus_M0"],
        )
        self.assertEqual(MODULE.STAGES, ("raw", "refined"))


if __name__ == "__main__":
    unittest.main()
