import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "assemble_ctv_branch_generation",
    ROOT / "scripts" / "assemble_ctv_branch_generation.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import CTV branch assembler")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AssembleCTVBranchGenerationTest(unittest.TestCase):
    def test_chunk_coordinates_preserve_global_order(self):
        self.assertEqual(MODULE.chunk_coordinates(0), (0, 0))
        self.assertEqual(MODULE.chunk_coordinates(255), (0, 255))
        self.assertEqual(MODULE.chunk_coordinates(256), (1, 0))
        self.assertEqual(MODULE.chunk_coordinates(2047), (7, 255))

    def test_invalid_chunk_coordinates_fail(self):
        with self.assertRaises(ValueError):
            MODULE.chunk_coordinates(-1)
        with self.assertRaises(ValueError):
            MODULE.chunk_coordinates(1, chunk_size=0)


if __name__ == "__main__":
    unittest.main()
