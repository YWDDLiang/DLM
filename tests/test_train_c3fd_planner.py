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
            "composition_supervision": True,
            "N_target": 3,
            "proposal_targets": {"family": 2, "N": 3, "arity": 2},
            "species_labels": [2, 3],
            "count_targets": [1, 2],
            "ledger_steps": [
                {"remaining_atoms": 3, "net_charge": 0, "remaining_species": 2, "branch": "unset"},
                {"remaining_atoms": 3, "net_charge": 0, "remaining_species": 2, "branch": "unset"},
                {"remaining_atoms": 2, "net_charge": -2, "remaining_species": 1, "branch": "ionic"},
                {"remaining_atoms": 0, "net_charge": 0, "remaining_species": 0, "branch": "ionic"},
            ],
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
        self.assertEqual(int(batch["family_targets"][0]), 2)
        self.assertEqual(int(batch["arity_targets"][0]), 2)
        self.assertEqual(tuple(batch["ledger_features"].shape), (1, 4, 6))
        self.assertTrue(torch.equal(batch["ledger_features"][0, 0], torch.zeros(6)))
        self.assertTrue(torch.equal(batch["ledger_features"][0, -1, :3], torch.zeros(3)))
        self.assertTrue(
            torch.equal(
                batch["ledger_features"][0, -1, 3:],
                torch.tensor([0.0, 1.0, 0.0]),
            )
        )

    def test_proposal_only_row_ignores_partial_oov_semantic_sequence(self):
        row = {
            "composition_supervision": False,
            "proposal_targets": {"family": 1, "N": 4, "arity": 2},
            "species_labels": [3],
            "count_targets": [1, 3],
            "ledger_steps": [],
            "soft_labels": {"lattice": 2},
        }
        batch = MODULE.collate(
            [row], eos_species_id=10, soft_fields=("lattice",)
        )
        self.assertEqual(tuple(batch["species_targets"].shape), (1, 2))
        self.assertTrue((batch["species_targets"] == -100).all().item())
        self.assertEqual(int(batch["family_targets"][0]), 1)
        self.assertEqual(int(batch["arity_targets"][0]), 2)


if __name__ == "__main__":
    unittest.main()
