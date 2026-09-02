from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "assemble_grounding_repeat.py"
SPEC = importlib.util.spec_from_file_location("assemble_grounding_tested", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class InvalidRefinedGeometryTests(unittest.TestCase):
    def payload(self) -> dict:
        return {
            "sample_indices": torch.tensor([0, 1]),
            "num_atoms": torch.tensor([[1, 1]]),
            "atom_types": torch.tensor([[3, 3]]),
            "frac_coords": torch.tensor([[[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]]),
            "lengths": torch.tensor([[[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]]),
            "angles": torch.tensor([[[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]]]),
        }

    def test_tolerant_mode_converts_one_bad_geometry_to_failure(self) -> None:
        structures, failures = MODULE._refined_structures(
            self.payload(), invalid_as_failure=True
        )
        self.assertEqual(set(structures), {0})
        self.assertEqual(set(failures), {1})

    def test_historical_mode_remains_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample 1"):
            MODULE.refined_structures(self.payload())


if __name__ == "__main__":
    unittest.main()
