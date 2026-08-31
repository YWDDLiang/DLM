from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("PyTorch is required for periodic relation adapter tests") from exc

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.periodic_relation_adapter import (
    PeriodicRelationAdapter,
    PeriodicRelationConfig,
    SoftCrystalGeometry,
    acyclic_periodic_residual_forward,
)


class PeriodicRelationAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(731)

    def _case(
        self,
        *,
        num_sites: int = 3,
        hidden_size: int = 12,
        prompt_length: int = 2,
        same_species: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        sequence_length = prompt_length + 7 + 4 * num_sites
        hidden = torch.randn(1, sequence_length, hidden_size, dtype=dtype)
        lattice = torch.tensor(
            [[[3.2, 0.0, 0.0], [0.7, 3.5, 0.0], [0.2, 0.4, 4.1]]],
            dtype=dtype,
        )
        base_coordinates = torch.tensor(
            [[[0.04, 0.11, 0.19], [0.42, 0.63, 0.87], [0.91, 0.28, 0.54]]],
            dtype=dtype,
        )
        if num_sites <= 3:
            coordinates = base_coordinates[:, :num_sites].clone()
        else:
            coordinates = torch.rand(1, num_sites, 3, dtype=dtype)
        if same_species:
            species = torch.full((1, num_sites), 8, dtype=torch.long)
        else:
            species = (torch.arange(num_sites, dtype=torch.long) % 10 + 1).unsqueeze(0)
        geometry = SoftCrystalGeometry(
            lattice=lattice,
            fractional_coordinates=coordinates,
            species=species,
            prompt_lengths=torch.tensor([prompt_length]),
            num_sites=torch.tensor([num_sites]),
        )
        adapter = PeriodicRelationAdapter(
            PeriodicRelationConfig(hidden_size=hidden_size, rank=6, num_rbf=8)
        ).to(dtype=dtype)
        return adapter, hidden, geometry

    @staticmethod
    def _randomize_output(adapter: PeriodicRelationAdapter) -> None:
        with torch.no_grad():
            adapter.output_projection.weight.normal_(mean=0.0, std=0.03)

    def test_step_zero_is_exact_and_helper_is_acyclic(self) -> None:
        adapter, hidden, geometry = self._case()
        head = torch.nn.Linear(hidden.shape[-1], 17, bias=False, dtype=hidden.dtype)
        calls = []

        def build_geometry(q0):
            calls.append(q0)
            return geometry

        result = acyclic_periodic_residual_forward(hidden, head, build_geometry, adapter)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], result.q0)
        self.assertTrue(torch.equal(result.relation.residual, torch.zeros_like(hidden)))
        self.assertTrue(torch.equal(result.relation.hidden_states, hidden))
        self.assertTrue(torch.equal(result.q1, result.q0))
        self.assertGreater(result.relation.internal_activation.abs().sum().item(), 0.0)

    def test_initial_gradient_reaches_only_output_projection(self) -> None:
        adapter, hidden, geometry = self._case()
        output = adapter(hidden, geometry)
        first_x = int(geometry.prompt_lengths[0]) + 8
        output.hidden_states[0, first_x, 0].backward()

        gradient = adapter.output_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        for name, parameter in adapter.named_parameters():
            if name == "output_projection.weight" or parameter.grad is None:
                continue
            self.assertTrue(
                torch.equal(parameter.grad, torch.zeros_like(parameter.grad)),
                msg=f"{name} received a nonzero step-zero gradient",
            )

    def test_global_fractional_translation_invariance(self) -> None:
        adapter, hidden, geometry = self._case()
        self._randomize_output(adapter)
        shift = torch.tensor([0.37, -0.22, 0.61], dtype=hidden.dtype)
        translated = SoftCrystalGeometry(
            lattice=geometry.lattice,
            fractional_coordinates=torch.remainder(geometry.fractional_coordinates + shift, 1.0),
            species=geometry.species,
            prompt_lengths=geometry.prompt_lengths,
            num_sites=geometry.num_sites,
        )
        original = adapter(hidden, geometry)
        shifted = adapter(hidden, translated)
        self.assertTrue(torch.allclose(original.pair_distances, shifted.pair_distances, atol=1e-12))
        self.assertTrue(torch.allclose(original.hidden_states, shifted.hidden_states, atol=1e-12))

    def test_same_species_site_permutation_equivariance(self) -> None:
        adapter, hidden, geometry = self._case(same_species=True)
        self._randomize_output(adapter)
        permutation = torch.tensor([2, 0, 1])
        prompt = int(geometry.prompt_lengths[0])

        permuted_hidden = hidden.clone()
        for new_site, old_site in enumerate(permutation.tolist()):
            new_start = prompt + 7 + 4 * new_site
            old_start = prompt + 7 + 4 * old_site
            permuted_hidden[:, new_start : new_start + 4] = hidden[:, old_start : old_start + 4]
        permuted_geometry = SoftCrystalGeometry(
            lattice=geometry.lattice,
            fractional_coordinates=geometry.fractional_coordinates[:, permutation],
            species=geometry.species[:, permutation],
            prompt_lengths=geometry.prompt_lengths,
            num_sites=geometry.num_sites,
        )

        original = adapter(hidden, geometry)
        permuted = adapter(permuted_hidden, permuted_geometry)
        lattice_positions = torch.arange(prompt + 1, prompt + 7)
        self.assertTrue(
            torch.allclose(
                original.residual[:, lattice_positions],
                permuted.residual[:, lattice_positions],
                atol=1e-12,
            )
        )
        for new_site, old_site in enumerate(permutation.tolist()):
            new_xyz = torch.arange(prompt + 8 + 4 * new_site, prompt + 11 + 4 * new_site)
            old_xyz = torch.arange(prompt + 8 + 4 * old_site, prompt + 11 + 4 * old_site)
            self.assertTrue(
                torch.allclose(
                    permuted.residual[:, new_xyz],
                    original.residual[:, old_xyz],
                    atol=1e-12,
                )
            )

    def test_triclinic_minimum_image_enumerates_neighbor_cells(self) -> None:
        adapter, hidden, geometry = self._case(num_sites=2)
        lattice = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]]],
            dtype=hidden.dtype,
        )
        coordinates = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.49, 0.49, 0.0]]],
            dtype=hidden.dtype,
        )
        triclinic = SoftCrystalGeometry(
            lattice=lattice,
            fractional_coordinates=coordinates,
            species=geometry.species,
            prompt_lengths=geometry.prompt_lengths,
            num_sites=geometry.num_sites,
        )
        output = adapter(hidden, triclinic)

        displacement = coordinates[0, 1] - coordinates[0, 0]
        naive = (displacement - torch.round(displacement)) @ lattice[0]
        shifts = torch.cartesian_prod(
            torch.arange(-1, 2, dtype=hidden.dtype),
            torch.arange(-1, 2, dtype=hidden.dtype),
            torch.arange(-1, 2, dtype=hidden.dtype),
        )
        brute_force = torch.linalg.vector_norm((displacement + shifts) @ lattice[0], dim=-1).min()
        measured = output.pair_distances[0, 0, 1]
        self.assertTrue(torch.allclose(measured, brute_force, atol=1e-12))
        self.assertLess(measured.item(), torch.linalg.vector_norm(naive).item() / 10.0)

    def test_twenty_site_memory_bound_finite_output_and_scatter_scope(self) -> None:
        adapter, hidden, geometry = self._case(num_sites=20, prompt_length=2)
        self._randomize_output(adapter)
        output = adapter(hidden, geometry)

        self.assertEqual(len(adapter.message_layers), 2)
        self.assertEqual(output.pair_mask.shape, (1, 20, 20))
        self.assertLessEqual(output.allocated_directed_pair_slots, 400)
        self.assertEqual(output.active_directed_pairs.tolist(), [380])
        self.assertTrue(torch.isfinite(output.hidden_states).all().item())
        self.assertTrue(torch.isfinite(output.pair_distances).all().item())

        prompt = int(geometry.prompt_lengths[0])
        forbidden = list(range(prompt)) + [prompt] + [prompt + 7 + 4 * site for site in range(20)]
        self.assertTrue(torch.equal(output.residual[:, forbidden], torch.zeros_like(output.residual[:, forbidden])))
        allowed = list(range(prompt + 1, prompt + 7))
        for site in range(20):
            allowed.extend(range(prompt + 8 + 4 * site, prompt + 11 + 4 * site))
        self.assertGreater(output.residual[:, allowed].abs().sum().item(), 0.0)

        too_wide_geometry = SoftCrystalGeometry(
            lattice=geometry.lattice,
            fractional_coordinates=torch.rand(1, 21, 3, dtype=hidden.dtype),
            species=torch.ones(1, 21, dtype=torch.long),
            prompt_lengths=torch.tensor([prompt]),
            num_sites=torch.tensor([21]),
        )
        too_wide_hidden = torch.randn(1, prompt + 7 + 4 * 21, hidden.shape[-1], dtype=hidden.dtype)
        with self.assertRaisesRegex(ValueError, "max_sites"):
            adapter(too_wide_hidden, too_wide_geometry)


if __name__ == "__main__":
    unittest.main()
