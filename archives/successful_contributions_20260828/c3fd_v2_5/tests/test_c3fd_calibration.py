from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required for C3FD calibration tests") from exc

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_calibration import (
    StratumInteraction,
    calibrated_top_p_probabilities,
    fit_temperature,
)


class C3FDCalibrationTest(unittest.TestCase):
    def test_temperature_fit_reduces_validation_nll(self):
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]])
        targets = torch.tensor([1, 0, 0, 1])
        result = fit_temperature(logits, targets)
        self.assertGreater(result.temperature, 1.0)
        self.assertLess(result.nll_after, result.nll_before)

    def test_train_only_stratum_interaction_preserves_supported_tuples(self):
        interaction = StratumInteraction.fit(
            [(0, 2, 2), (0, 2, 2), (1, 4, 3)], alpha=1.0
        )
        self.assertEqual(interaction.strata, ((0, 2, 2), (1, 4, 3)))
        scores = interaction.joint_scores(
            torch.zeros(2), torch.zeros(20), torch.zeros(7)
        )
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.isfinite(scores).all().item())

    def test_top_p_policy_has_no_hidden_top_k(self):
        probabilities = calibrated_top_p_probabilities(
            torch.tensor([2.0, 1.0, 0.0, -1.0]),
            temperature=1.0,
            top_p=1.0,
        )
        self.assertTrue((probabilities > 0).all().item())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
