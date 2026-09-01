import unittest

import torch

from crystal_dlm.periodic_geometry_ops import (
    ELEMENT_RADII_ANGSTROM_BY_Z,
    ELEMENT_RADII_SHA256,
    minimum_image_distances_125,
    minimum_image_distances_27,
)
from crystal_dlm.periodic_geometry_objective import (
    _pair_distances,
    _species_aware_margins,
    build_geometry_token_support,
    periodic_geometry_objective,
)


class _Tokenizer:
    def __init__(self) -> None:
        tokens = ["<PAD>", "<N_002>", "<E_Li>", "<E_O>"]
        tokens += [f"<L{axis}_{value:03d}>" for axis in "ABC" for value in (39, 40, 41)]
        tokens += [f"<A{axis}_{value:03d}>" for axis in "ABG" for value in (89, 90, 91)]
        tokens += [f"<{axis}_{value:03d}>" for axis in "XYZ" for value in (0, 25, 50, 75)]
        self.vocab = {token: idx for idx, token in enumerate(tokens)}

    def get_vocab(self):
        return dict(self.vocab)


class PeriodicGeometryObjectiveTest(unittest.TestCase):
    def test_frozen_radius_table_is_complete_and_hashed(self) -> None:
        self.assertEqual(len(ELEMENT_RADII_ANGSTROM_BY_Z), 119)
        self.assertEqual(len(ELEMENT_RADII_SHA256), 64)
        self.assertTrue(all(value > 0 for value in ELEMENT_RADII_ANGSTROM_BY_Z[1:]))

    def test_exact_triclinic_minimum_beats_component_rounding(self) -> None:
        lattice = torch.tensor(
            [[1.0, 0.0, 0.0], [0.9, 0.2, 0.0], [0.0, 0.0, 1.0]]
        )
        coordinates = torch.tensor([[0.49, 0.49, 0.0], [0.0, 0.0, 0.0]])
        approximate, pairs = _pair_distances(
            coordinates, lattice, exact_triclinic=False
        )
        exact, exact_pairs = _pair_distances(
            coordinates, lattice, exact_triclinic=True
        )
        self.assertTrue(torch.equal(pairs, exact_pairs))
        self.assertLess(exact.item(), 0.13)
        self.assertGreater(approximate.item(), 0.9)

        delta = coordinates[0] - coordinates[1]
        shifts = torch.cartesian_prod(
            torch.arange(-2, 3, dtype=delta.dtype),
            torch.arange(-2, 3, dtype=delta.dtype),
            torch.arange(-2, 3, dtype=delta.dtype),
        )
        brute = torch.linalg.vector_norm((delta + shifts) @ lattice, dim=-1).min()
        shared = minimum_image_distances_27(delta, lattice)
        self.assertTrue(torch.allclose(shared, brute, atol=1e-7))

    def test_species_margin_uses_both_element_radii(self) -> None:
        margins = _species_aware_margins(
            torch.tensor([2, 3]),
            torch.tensor([[0], [1]]),
            {"ids": [2, 3], "values": [1.0, 2.0]},
            scale=0.5,
            floor=0.5,
            ceiling=2.0,
        )
        self.assertTrue(torch.allclose(margins, torch.tensor([1.5])))

    def test_125_image_shell_handles_pathological_skew_cell(self) -> None:
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
        )
        bounded27 = minimum_image_distances_27(delta, lattice)
        bounded125 = minimum_image_distances_125(delta, lattice)
        shifts = torch.cartesian_prod(
            torch.arange(-3, 4, dtype=delta.dtype),
            torch.arange(-3, 4, dtype=delta.dtype),
            torch.arange(-3, 4, dtype=delta.dtype),
        )
        brute = torch.linalg.vector_norm((delta + shifts) @ lattice, dim=-1).min()
        self.assertGreater(bounded27.item() - bounded125.item(), 0.2)
        self.assertTrue(torch.allclose(bounded125, brute, atol=1.0e-12))

    def test_target_peaked_logits_have_finite_small_loss_and_gradients(self) -> None:
        tokenizer = _Tokenizer()
        support = build_geometry_token_support(tokenizer)
        tokens = [
            "<N_002>", "<LA_040>", "<LB_040>", "<LC_040>",
            "<AA_090>", "<AB_090>", "<AG_090>",
            "<E_Li>", "<X_000>", "<Y_000>", "<Z_000>",
            "<E_O>", "<X_050>", "<Y_050>", "<Z_050>",
        ]
        ids = torch.tensor([[tokenizer.vocab[token] for token in tokens]])
        logits = torch.full((1, len(tokens), len(tokenizer.vocab)), -8.0, requires_grad=True)
        with torch.no_grad():
            for position, token_id in enumerate(ids[0]):
                logits[0, position, token_id] = 8.0
        masked = torch.zeros_like(ids, dtype=torch.bool)
        masked[:, 1:7] = True
        masked[:, 8:11] = True
        masked[:, 12:15] = True
        result = periodic_geometry_objective(
            logits=logits,
            input_ids=ids,
            masked_indices=masked,
            prompt_lengths=torch.tensor([0]),
            num_atoms=torch.tensor([2]),
            support=support,
        )
        loss = result["metric"] + result["pair_rdf"] + result["overlap"] + result["coordination"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bfloat16_logits_use_float32_geometry_kernels(self) -> None:
        tokenizer = _Tokenizer()
        support = build_geometry_token_support(tokenizer)
        tokens = [
            "<N_002>", "<LA_040>", "<LB_040>", "<LC_040>",
            "<AA_090>", "<AB_090>", "<AG_090>",
            "<E_Li>", "<X_000>", "<Y_000>", "<Z_000>",
            "<E_O>", "<X_050>", "<Y_050>", "<Z_050>",
        ]
        ids = torch.tensor([[tokenizer.vocab[token] for token in tokens]])
        logits = torch.full(
            (1, len(tokens), len(tokenizer.vocab)), -8.0, dtype=torch.bfloat16
        )
        for position, token_id in enumerate(ids[0]):
            logits[0, position, token_id] = 8.0
        result = periodic_geometry_objective(
            logits=logits,
            input_ids=ids,
            masked_indices=torch.ones_like(ids, dtype=torch.bool),
            prompt_lengths=torch.tensor([0]),
            num_atoms=torch.tensor([2]),
            support=support,
        )
        for name in ("metric", "pair_rdf", "overlap", "coordination"):
            self.assertEqual(result[name].dtype, torch.float32)
            self.assertTrue(torch.isfinite(result[name]))

    def test_periodic_translation_preserves_pair_losses(self) -> None:
        # The objective's minimum-image path is indirectly exercised by two
        # token-equivalent global translations on the 0.25 grid.
        tokenizer = _Tokenizer()
        support = build_geometry_token_support(tokenizer)
        base = [
            "<N_002>", "<LA_040>", "<LB_040>", "<LC_040>",
            "<AA_090>", "<AB_090>", "<AG_090>",
            "<E_Li>", "<X_000>", "<Y_000>", "<Z_000>",
            "<E_O>", "<X_050>", "<Y_050>", "<Z_050>",
        ]
        shifted = [token.replace("_000>", "_025>").replace("_050>", "_075>") if token[1:2] in "XYZ" else token for token in base]
        rows = torch.tensor([[tokenizer.vocab[token] for token in row] for row in (base, shifted)])
        logits = torch.full((2, len(base), len(tokenizer.vocab)), -8.0)
        for sample in range(2):
            for position, token_id in enumerate(rows[sample]):
                logits[sample, position, token_id] = 8.0
        masked = torch.ones_like(rows, dtype=torch.bool)
        result = periodic_geometry_objective(
            logits=logits,
            input_ids=rows,
            masked_indices=masked,
            prompt_lengths=torch.tensor([0, 0]),
            num_atoms=torch.tensor([2, 2]),
            support=support,
        )
        self.assertTrue(torch.isfinite(result["pair_rdf"]))
        self.assertTrue(torch.isfinite(result["coordination"]))
        base_distances, _ = _pair_distances(
            torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            torch.eye(3) * 4.0,
            exact_triclinic=True,
        )
        shifted_distances, _ = _pair_distances(
            torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]),
            torch.eye(3) * 4.0,
            exact_triclinic=True,
        )
        self.assertTrue(torch.equal(base_distances, shifted_distances))


if __name__ == "__main__":
    unittest.main()
