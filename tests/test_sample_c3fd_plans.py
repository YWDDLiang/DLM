import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required for C3FD sampling tests") from exc

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sample_c3fd_plans", ROOT / "scripts" / "sample_c3fd_plans.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import sample_c3fd_plans.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SampleC3FDPlansTest(unittest.TestCase):
    def test_semantic_history_starts_with_locked_N(self):
        species, counts, n_values, target = MODULE.semantic_inputs(5, [3, 4], [2, 3])
        self.assertEqual(target, 3)
        self.assertTrue(torch.equal(species, torch.tensor([[-1, -1, 3, 4]])))
        self.assertTrue(torch.equal(counts, torch.tensor([[0, 0, 2, 3]])))
        self.assertTrue(torch.equal(n_values, torch.tensor([[0, 5, 0, 0]])))

    def test_sampler_never_selects_negative_infinity(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)
        logits = torch.tensor([float("-inf"), 0.0, float("-inf")])
        for _ in range(10):
            self.assertEqual(
                MODULE.sample_index(
                    logits,
                    rng=generator,
                    temperature=0.9,
                    top_p=0.95,
                    top_k=50,
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
