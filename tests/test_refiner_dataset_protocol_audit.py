"""Compare the actual frozen dataset adapter and current seed-before-loader path.

This fixture uses only CPU tensors and the real torch DataLoader iterator. It
does not exercise CUDA scatter reductions or claim GPU sampling repeatability.
"""
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.refine_dlm_with_crysllmgen import ProposalDataset, frozen_seeded_batches


ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATH = ROOT / "tests/fixtures/r03_frozen_refiner_runtime.py"
FROZEN_SHA256 = "a0cb2c54c149aee0a4f36c147d93eca6799e3b17879f76c108721c934f087b2a"


def cpu_loader(dataset, **kwargs):
    return DataLoader(dataset, collate_fn=lambda rows: rows[0], **kwargs)


def fixture_graph(count, index, seed):
    source = np.random.default_rng(1403 + count)
    edge_indices = np.asarray(
        [(i, j) for i in range(count) for j in range(count)], dtype=np.int32
    )
    return {
        "sample_idx": index, "refiner_noise_seed": seed,
        "n_atom": np.asarray([count], dtype=np.int64),
        "length": np.asarray([4.123456789, 5.987654321, 3.004567891], dtype=np.float64),
        "angle": np.asarray([89.123456789, 111.765432198, 72.123456789], dtype=np.float64),
        "x_coord": source.random((count, 3), dtype=np.float64),
        "a_type": np.asarray([11 if i % 2 == 0 else 17 for i in range(count)], dtype=np.int32),
        "edge_indices": edge_indices,
        "to_jimages": source.integers(-1, 2, size=(len(edge_indices), 3), dtype=np.int64),
    }


def reseed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def after_loader_state():
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state().clone(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        "python": random.getstate(),
        "torch_draw": torch.randn(20, 3),
        "numpy_draw": np.random.randn(9),
        "python_draw": [random.random() for _ in range(3)],
    }


class RefinerDatasetProtocolAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FROZEN_PATH.is_file():
            raise AssertionError("The audited frozen dataset source fixture is missing")
        if hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest() != FROZEN_SHA256:
            raise AssertionError("The audited frozen dataset source fixture changed")
        name = "_audited_frozen_refiner_dataset"
        specification = importlib.util.spec_from_file_location(name, FROZEN_PATH)
        cls.frozen = importlib.util.module_from_spec(specification)
        previous_path = sys.path[:]
        try:
            specification.loader.exec_module(cls.frozen)
        finally:
            # The old CLI adds its relative src directory on import. Keep that
            # side effect out of other current-repository tests.
            sys.path[:] = previous_path

    def test_every_model_input_value_dtype_shape_and_stride_matches_frozen_adapter(self):
        for count, index, seed in ((1, 0, 8127182874598280710), (20, 1, 36468167202603192)):
            with self.subTest(count=count):
                graph = fixture_graph(count, index, seed)
                before = self.frozen.ProposalDataset([graph], SimpleNamespace)[0]
                after = ProposalDataset(
                    [graph], SimpleNamespace, seed_from_graph_field="refiner_noise_seed"
                )[0]
                self.assertEqual(set(vars(after)) - set(vars(before)), {"sample_idx", "refiner_seed"})
                for name, expected in vars(before).items():
                    actual = getattr(after, name)
                    if isinstance(expected, torch.Tensor):
                        self.assertEqual(actual.dtype, expected.dtype, name)
                        self.assertEqual(actual.device, expected.device, name)
                        self.assertEqual(actual.shape, expected.shape, name)
                        self.assertEqual(actual.stride(), expected.stride(), name)
                        self.assertEqual(actual.is_contiguous(), expected.is_contiguous(), name)
                        self.assertTrue(torch.equal(actual, expected), name)
                    else:
                        self.assertEqual(type(actual), type(expected), name)
                        self.assertEqual(actual, expected, name)
                self.assertEqual(after.sample_idx.item(), index)
                self.assertEqual(after.refiner_seed.item(), seed)

    def test_loader_random_states_and_draws_match_for_one_and_twenty_atoms(self):
        graphs = [
            fixture_graph(1, 0, 8127182874598280710),
            fixture_graph(20, 1, 36468167202603192),
        ]
        expected = {}
        for graph in graphs:
            reseed_cpu(graph["refiner_noise_seed"])
            dataset = self.frozen.ProposalDataset([graph], SimpleNamespace)
            next(iter(cpu_loader(dataset, batch_size=1, shuffle=False)))
            expected[graph["sample_idx"]] = after_loader_state()
        # Opposite order plus unrelated preceding draws checks that each request
        # starts from its own seed, independently of the previous request.
        torch.rand(31)
        np.random.rand(7)
        random.random()
        with patch.object(torch.cuda, "is_available", return_value=False):
            for batch in frozen_seeded_batches(
                list(reversed(graphs)), SimpleNamespace, cpu_loader, "refiner_noise_seed"
            ):
                actual = after_loader_state()
                want = expected[batch.sample_idx.item()]
                self.assertTrue(torch.equal(actual["torch"], want["torch"]))
                self.assertEqual(actual["python"], want["python"])
                self.assertEqual(actual["numpy"][0], want["numpy"][0])
                np.testing.assert_array_equal(actual["numpy"][1], want["numpy"][1])
                self.assertEqual(actual["numpy"][2:], want["numpy"][2:])
                self.assertTrue(torch.equal(actual["torch_draw"], want["torch_draw"]))
                np.testing.assert_array_equal(actual["numpy_draw"], want["numpy_draw"])
                self.assertEqual(actual["python_draw"], want["python_draw"])


if __name__ == "__main__":
    unittest.main()
