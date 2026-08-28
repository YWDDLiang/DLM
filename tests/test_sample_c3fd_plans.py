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

from crystal_dlm.ccfd import FormulaToken
from crystal_dlm.ccfd_v2 import CCFDv2State, SetAtomCount


class SampleC3FDPlansTest(unittest.TestCase):
    def test_semantic_history_starts_with_locked_N(self):
        start = CCFDv2State.start().apply(SetAtomCount(5))
        after_first = start.apply(FormulaToken.from_symbol("O", -2, 2))
        after_second = after_first.apply(FormulaToken.from_symbol("Fe", 2, 2))
        species, counts, n_values, ledger, target = MODULE.semantic_inputs(
            5,
            [3, 4],
            [2, 2],
            state_history=[start, after_first, after_second],
            target_arity=3,
        )
        self.assertEqual(target, 3)
        self.assertTrue(torch.equal(species, torch.tensor([[-1, -1, 3, 4]])))
        self.assertTrue(torch.equal(counts, torch.tensor([[0, 0, 2, 2]])))
        self.assertTrue(torch.equal(n_values, torch.tensor([[0, 5, 0, 0]])))
        self.assertEqual(tuple(ledger.shape), (1, 4, 6))
        self.assertTrue(torch.equal(ledger[0, 0], torch.zeros(6)))
        self.assertAlmostEqual(float(ledger[0, 1, 0]), 5 / 20)
        self.assertAlmostEqual(float(ledger[0, -1, 2]), 1 / 7)

    def test_family_prefix_policy_rejects_higher_priority_anion(self):
        self.assertFalse(MODULE.element_allowed_for_family("O", "sulfide"))
        self.assertTrue(MODULE.element_allowed_for_family("S", "sulfide"))
        self.assertFalse(MODULE.element_allowed_for_family("F", "nitride"))
        self.assertTrue(MODULE.element_allowed_for_family("N", "nitride"))

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
