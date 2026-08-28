from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.stable_geometry_curriculum import (
    dynamic_geometry_relative_positions,
    forbidden_training_paths,
    source_ehull,
    strip_training_outcomes,
)


class StableGeometryCurriculumTest(unittest.TestCase):
    def test_dynamic_geometry_positions_exclude_n_and_elements(self):
        self.assertEqual(
            dynamic_geometry_relative_positions(2),
            (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14),
        )

    def test_strip_outcomes_is_recursive(self):
        row = {
            "prompt": "composition only",
            "metadata": {"e_above_hull": 0.0, "material_id": "mp-1"},
            "nested": [{"formation_energy_per_atom": -1.0, "keep": 2}],
        }
        clean = strip_training_outcomes(row)
        self.assertEqual(forbidden_training_paths(clean), [])
        self.assertEqual(clean["metadata"], {"material_id": "mp-1"})
        self.assertEqual(clean["nested"], [{"keep": 2}])

    def test_source_ehull_requires_finite_metadata(self):
        self.assertEqual(source_ehull({"metadata": {"e_above_hull": 0.01}}), 0.01)
        with self.assertRaises(ValueError):
            source_ehull({"metadata": {}})


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class SGTCLLaDAMaskTest(unittest.TestCase):
    def test_forward_process_candidate_mask_is_geometry_only(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "llada_sft_sgtc", ROOT / "src/scripts/llada_sft.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import llada_sft")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        input_ids = torch.arange(15, dtype=torch.long).reshape(1, -1)
        result = module.forward_process(
            input_ids,
            torch.ones_like(input_ids),
            torch.tensor([0]),
            mask_policy_ids=torch.tensor([module.MASK_POLICY_TO_ID["normal"]]),
            empty_token_id=999,
            dynamic_geometry_only=True,
        )
        observed = tuple(
            int(value)
            for value in torch.nonzero(
                result["candidate_mask"][0], as_tuple=False
            ).reshape(-1)
        )
        self.assertEqual(observed, dynamic_geometry_relative_positions(2))


if __name__ == "__main__":
    unittest.main()
