import json
from pathlib import Path
import tempfile
import unittest


try:
    import torch
    from scripts.apply_model494_response_backfill import (
        _pair_task,
        load_model494_endpoints,
    )
except ModuleNotFoundError:
    torch = None
    _pair_task = None
    load_model494_endpoints = None


@unittest.skipIf(torch is None, "torch unavailable")
class Model494ResponseBackfillTest(unittest.TestCase):
    def test_load_endpoint_splits_flat_atom_tensors_by_sample_index(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            payload_path = root / "refined.pt"
            torch.save(
                {
                    "sample_indices": torch.tensor([7, 3]),
                    "num_atoms": torch.tensor([[2, 1]]),
                    "frac_coords": torch.tensor(
                        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]
                    ),
                    "atom_types": torch.tensor([[8, 8, 11]]),
                },
                payload_path,
            )
            metrics = root / "metrics.json"
            metrics.write_text(
                json.dumps({"output_file": str(payload_path)}), encoding="utf-8"
            )
            endpoints = load_model494_endpoints(metrics)
            self.assertEqual(set(endpoints), {3, 7})
            self.assertEqual(endpoints[7]["num_atoms"], 2)
            self.assertEqual(endpoints[7]["atom_types"], [8, 8])
            self.assertEqual(len(endpoints[3]["frac_coords"]), 1)

    def test_pairing_requires_matching_model494_atom_order(self):
        plan = {"N": 2, "elements": ["Na", "Cl"], "counts": [1, 1]}
        row = {
            "sample_idx": 4,
            "parsed": True,
            "plan_state": plan,
            "conditioning_prompt": "plan",
            "text": (
                "<N_002><LA_040><LB_040><LC_040><AA_090><AB_090><AG_090>"
                "<E_Cl><X_000><Y_000><Z_000>"
                "<E_Na><X_050><Y_050><Z_050>"
            ),
            "prompt_record": {
                "species_program": ["Na", "Cl"],
                "species_program_source": "test",
            },
        }
        endpoint = {
            "num_atoms": 2,
            "atom_types": [11, 17],
            "frac_coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        }
        task, reason = _pair_task(row, endpoint)
        self.assertIsNone(task)
        self.assertEqual(reason, "model494_atom_order_mismatch")

    def test_pairing_maps_process_graph_order_back_to_body_slots(self):
        plan = {"N": 2, "elements": ["Na", "Cl"], "counts": [1, 1]}
        row = {
            "sample_idx": 4,
            "parsed": True,
            "plan_state": plan,
            "conditioning_prompt": "plan",
            "text": (
                "<N_002><LA_040><LB_040><LC_040><AA_090><AB_090><AG_090>"
                "<E_Cl><X_000><Y_000><Z_000>"
                "<E_Na><X_050><Y_050><Z_050>"
            ),
            "prompt_record": {
                "species_program": ["Na", "Cl"],
                "species_program_source": "test",
            },
        }
        endpoint = {
            "num_atoms": 2,
            "atom_types": [11, 17],
            "frac_coords": [[0.51, 0.52, 0.53], [0.01, 0.02, 0.03]],
        }
        graph = {
            "a_type": [11, 17],
            "x_coord": torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]),
        }
        task, reason = _pair_task(row, endpoint, graph)
        self.assertIsNone(reason)
        self.assertEqual(task["model494_graph_to_body_permutation"], [1, 0])
        self.assertEqual(
            task["target_frac_coords"],
            [[0.01, 0.02, 0.03], [0.51, 0.52, 0.53]],
        )


if __name__ == "__main__":
    unittest.main()
