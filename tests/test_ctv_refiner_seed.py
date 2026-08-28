import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch

    SPEC = importlib.util.spec_from_file_location(
        "refine_dlm_with_crysllmgen",
        ROOT / "src" / "scripts" / "refine_dlm_with_crysllmgen.py",
    )
    if SPEC is None or SPEC.loader is None:
        raise RuntimeError("cannot import refiner")
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
except Exception:  # pragma: no cover - torch is optional in lightweight CI.
    torch = None
    MODULE = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CTVRefinerSeedTest(unittest.TestCase):
    def test_proposal_dataset_preserves_common_refiner_seed(self):
        class FakeData:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        graph = {
            "n_atom": 1,
            "edge_indices": torch.empty((0, 2), dtype=torch.long),
            "length": [3.0, 3.0, 3.0],
            "angle": [90.0, 90.0, 90.0],
            "x_coord": [[0.0, 0.0, 0.0]],
            "a_type": [8],
            "to_jimages": torch.empty((0, 3), dtype=torch.long),
            "sample_idx": 17,
            "ctv_refiner_seed": 12345,
        }
        dataset = MODULE.ProposalDataset(
            [graph], FakeData, seed_from_graph_field="ctv_refiner_seed"
        )
        item = dataset[0]
        self.assertEqual(int(item.sample_idx.item()), 17)
        self.assertEqual(int(item.refiner_seed.item()), 12345)


if __name__ == "__main__":
    unittest.main()
