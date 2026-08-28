import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required for C3FD training tests") from exc

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_c3fd_planner", ROOT / "scripts" / "train_c3fd_planner.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import train_c3fd_planner.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainC3FDPlannerTest(unittest.TestCase):
    def test_collate_places_N_then_teacher_actions_then_EOS(self):
        row = {
            "N_target": 3,
            "species_labels": [2, 3],
            "count_targets": [1, 2],
            "soft_labels": {"lattice": 4},
        }
        batch = MODULE.collate([row], eos_species_id=10, soft_fields=("lattice",))
        self.assertTrue(
            torch.equal(batch["previous_species_indices"], torch.tensor([[-1, -1, 2, 3]]))
        )
        self.assertTrue(
            torch.equal(batch["previous_n_values"], torch.tensor([[0, 3, 0, 0]]))
        )
        self.assertTrue(
            torch.equal(batch["species_targets"], torch.tensor([[-100, 2, 3, 10]]))
        )
        self.assertTrue(
            torch.equal(batch["count_targets"], torch.tensor([[-100, 1, 2, 0]]))
        )
        self.assertEqual(int(batch["rich:lattice"][0, 3]), 4)


if __name__ == "__main__":
    unittest.main()
