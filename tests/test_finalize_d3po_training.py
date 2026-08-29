import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_d3po_training", ROOT / "scripts/finalize_d3po_training.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalizeD3POTrainingStaticTest(unittest.TestCase):
    def test_two_seeds_and_no_selection_are_frozen(self):
        self.assertEqual(MODULE.SEEDS, (81017, 81018))
        self.assertEqual(MODULE.UPDATES, 348)
        text = (ROOT / "scripts/finalize_d3po_training.py").read_text(encoding="utf-8")
        self.assertIn('"checkpoint_or_seed_selection": False', text)
        self.assertIn('"scientific_result_available": False', text)
        self.assertIn("scheduler_kill_ceiling_gpu_hours", text)


if __name__ == "__main__":
    unittest.main()
