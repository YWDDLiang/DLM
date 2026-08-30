import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_d3po_generation", ROOT / "scripts/finalize_d3po_generation.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalizeD3POGenerationStaticTest(unittest.TestCase):
    def test_frozen_cells_and_resource_accounting(self):
        self.assertEqual(MODULE.STREAMS, (17, 18))
        self.assertEqual(
            MODULE.ARMS, ("base", "d3po_seed81017", "d3po_seed81018")
        )
        text = (ROOT / "scripts/finalize_d3po_generation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"scientific_stability_result_available": False', text)
        self.assertIn('"scheduler_kill_ceiling_gpu_hours": 72.0', text)
        self.assertIn("No failed ordinal was replaced", text)


if __name__ == "__main__":
    unittest.main()
