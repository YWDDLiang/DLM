from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required for C3FD model tests") from exc

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.c3fd_planner_model import C3FDPlannerConfig, C3FDPlannerModel
from crystal_dlm.semantic_composition_head import SemanticHeadFlags


class C3FDPlannerModelTest(unittest.TestCase):
    def test_forward_uses_explicit_N_and_typed_actions(self):
        config = C3FDPlannerConfig(
            context_size=12,
            semantic_size=8,
            num_species=3,
            physics_feature_size=2,
            rich_soft_head_dims={"lattice": 4},
            decoder_layers=1,
            decoder_heads=2,
            decoder_dropout=0.0,
        )
        model = C3FDPlannerModel(config, physics_features=torch.ones(3, 2))
        previous_species = torch.tensor([[-1, -1, 0, 1]])
        previous_counts = torch.tensor([[0, 0, 1, 1]])
        previous_n = torch.tensor([[0, 2, 0, 0]])
        output = model(
            torch.randn(1, 12),
            previous_species_indices=previous_species,
            previous_count_values=previous_counts,
            previous_n_values=previous_n,
            n_targets=torch.tensor([2]),
            species_targets=torch.tensor([[-100, 0, 1, 3]]),
            count_targets=torch.tensor([[-100, 1, 1, 0]]),
            rich_targets={"lattice": torch.tensor([[-100, -100, -100, 2]])},
            flags=SemanticHeadFlags(use_physics=True),
        )
        self.assertEqual(output.n_logits.shape, (1, 20))
        self.assertEqual(output.species_logits.shape, (1, 4, 4))
        self.assertTrue(torch.isfinite(output.loss).item())


if __name__ == "__main__":
    unittest.main()
