import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_spad_raw_direct",
    ROOT / "scripts/finalize_spad_raw_direct.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import SPAD raw finalizer")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SPADRawFinalizerTest(unittest.TestCase):
    def test_exact_mcnemar_counts_both_discordant_directions(self):
        result = MODULE.exact_mcnemar(
            [True, True, False, False],
            [True, False, True, False],
        )
        self.assertEqual(result["right_only"], 1)
        self.assertEqual(result["left_only"], 1)
        self.assertEqual(result["two_sided_exact_p"], 1.0)

    def test_attempt_ordinal_is_strict(self):
        self.assertEqual(MODULE.ordinal_from_attempt_id("cell-s17-0042"), 42)
        with self.assertRaises(ValueError):
            MODULE.ordinal_from_attempt_id("missing")


if __name__ == "__main__":
    unittest.main()
