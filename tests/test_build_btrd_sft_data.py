import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required") from exc


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_btrd_sft_data", ROOT / "scripts" / "build_btrd_sft_data.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import BTRD data builder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BTRDDataTest(unittest.TestCase):
    def test_refined_payload_maps_ragged_atoms_by_sample_idx(self) -> None:
        payload = {
            "sample_indices": torch.tensor([2, 0]),
            "num_atoms": torch.tensor([[1, 2]]),
            "lengths": torch.tensor([[[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]]),
            "angles": torch.tensor([[[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]]]),
            "atom_types": torch.tensor([[3, 11, 17]]),
            "frac_coords": torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),
        }
        mapped = MODULE.refined_geometry_by_index(payload)
        self.assertEqual(mapped[2]["species"], ["Li"])
        self.assertEqual(mapped[0]["species"], ["Na", "Cl"])

    def test_missing_teacher_falls_back_without_row_deletion(self) -> None:
        selected = [
            {
                "btrd_index": 0,
                "btrd_target_mode": "model494_tau200",
                "answer": "original",
                "answer_sha256": "old",
                "plan_state": {"N": 2, "elements": ["Na", "Cl"], "counts": [1, 1]},
            },
            {
                "btrd_index": 1,
                "btrd_target_mode": "mp20_anchor",
                "answer": "anchor",
                "answer_sha256": "anchor-sha",
                "plan_state": {"N": 1, "elements": ["Li"], "counts": [1]},
            },
        ]
        rows, audit = MODULE.build_rows(selected, {})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["answer"], "original")
        self.assertEqual(rows[0]["btrd_effective_target_mode"], "mp20_anchor")
        self.assertEqual(audit["tau200_missing_fallback_anchor"], 1)

    def test_tau800_endpoint_mode_is_explicit(self) -> None:
        selected = [
            {
                "btrd_index": 0,
                "btrd_target_mode": "model494_tau200",
                "answer": "original",
                "answer_sha256": "old",
                "plan_state": {"N": 1, "elements": ["Li"], "counts": [1]},
            }
        ]
        refined = {
            0: {
                "lengths": [4.0, 4.0, 4.0],
                "angles": [90.0, 90.0, 90.0],
                "species": ["Li"],
                "frac_coords": [[0.0, 0.0, 0.0]],
            }
        }
        rows, audit = MODULE.build_rows(selected, refined, teacher_steps=800)
        self.assertEqual(rows[0]["btrd_effective_target_mode"], "model494_tau800")
        self.assertEqual(rows[0]["btrd_teacher_steps"], 800)
        self.assertEqual(audit["tau800_teacher"], 1)


if __name__ == "__main__":
    unittest.main()
