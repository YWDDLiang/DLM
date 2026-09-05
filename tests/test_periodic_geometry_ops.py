from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.periodic_geometry_ops import (
    minimum_image_distances_125,
    minimum_image_vectors_125,
    minimum_image_vectors_27,
)


class PeriodicGeometryOpsTest(unittest.TestCase):
    def test_skew_cell_vector_norm_matches_existing_125_image_distance(self) -> None:
        lattice = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.344034729614033, 0.31420764349108027, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        delta = torch.tensor(
            [-0.3227580642396569, 0.32245827732342236, 0.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        vector, shift = minimum_image_vectors_125(delta, lattice)
        distance = minimum_image_distances_125(delta, lattice)
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(vector), distance, atol=1.0e-12))
        self.assertTrue(torch.allclose(vector, (delta + shift) @ lattice, atol=1.0e-12))
        distance.backward()
        self.assertTrue(torch.isfinite(delta.grad).all())

    def test_integer_translation_changes_shift_not_vector(self) -> None:
        lattice = torch.tensor(
            [[3.2, 0.0, 0.0], [0.7, 3.5, 0.0], [0.2, 0.4, 4.1]],
            dtype=torch.float64,
        )
        delta = torch.tensor([0.42, -0.37, 0.61], dtype=torch.float64)
        translation = torch.tensor([3.0, -2.0, 5.0], dtype=torch.float64)
        first_vector, first_shift = minimum_image_vectors_125(delta, lattice)
        second_vector, second_shift = minimum_image_vectors_125(delta + translation, lattice)
        self.assertTrue(torch.allclose(first_vector, second_vector, atol=1.0e-12))
        self.assertTrue(torch.equal(second_shift, first_shift - translation.to(torch.long)))

    def test_batched_and_unbatched_vector_results_agree(self) -> None:
        lattices = torch.tensor(
            [
                [[3.0, 0.0, 0.0], [0.4, 3.4, 0.0], [0.2, 0.1, 4.0]],
                [[4.0, 0.0, 0.0], [-0.3, 3.1, 0.0], [0.5, 0.2, 3.7]],
            ],
            dtype=torch.float64,
        )
        deltas = torch.tensor(
            [
                [[0.4, 0.4, 0.4], [-0.45, 0.49, 0.1]],
                [[0.8, -0.6, 0.2], [0.05, 0.51, -0.7]],
            ],
            dtype=torch.float64,
        )
        batch_vectors, batch_shifts = minimum_image_vectors_27(deltas, lattices)
        for batch in range(2):
            vectors, shifts = minimum_image_vectors_27(deltas[batch], lattices[batch])
            self.assertTrue(torch.allclose(batch_vectors[batch], vectors, atol=1.0e-12))
            self.assertTrue(torch.equal(batch_shifts[batch], shifts))


if __name__ == "__main__":
    unittest.main()
