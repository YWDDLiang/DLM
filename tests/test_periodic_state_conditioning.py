from dataclasses import asdict
import io
import itertools
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.periodic_state_conditioning import (
    PeriodicStateConditioner,
    PeriodicStateConfig,
)


class PeriodicStateConditioningTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(731)

    def _case(self, *, n=3, batch=2, image_radius=2, radial_basis_count=8):
        config = PeriodicStateConfig(
            hidden_size=12, width=16, max_sites=20,
            image_radius=image_radius, radial_basis_count=radial_basis_count,
        )
        module = PeriodicStateConditioner(config)
        inputs = {
            "lattice": torch.tensor([
                [3.2, 0.0, 0.0], [0.7, 3.5, 0.0], [0.2, 0.4, 4.1],
            ]).expand(batch, -1, -1).clone(),
            "fractional": torch.rand(batch, n, 3),
            "species": (torch.arange(n) % 10 + 1).expand(batch, -1).clone(),
            "site_known": torch.ones(batch, n, dtype=torch.bool),
            "lattice_known": torch.ones(batch, dtype=torch.bool),
            "program_rank": torch.arange(n).expand(batch, -1).clone(),
            "active_sites": (torch.arange(n) % 2 == 0).expand(batch, -1).clone(),
        }
        return module, inputs

    @staticmethod
    def _open_projections(module):
        with torch.no_grad():
            module.cell_projection.weight.normal_(std=0.1)
            module.site_projection.weight.normal_(std=0.1)

    def _capture(self, module, inputs, *names):
        captured = {}

        def hook(name):
            def record(_module, args):
                captured[name] = args[0].detach().clone()
            return record

        handles = [getattr(module, name).register_forward_pre_hook(hook(name)) for name in names]
        try:
            output = module(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        return output, captured

    def assertFinite(self, value):
        self.assertTrue(torch.isfinite(value).all().item())

    def assertOutputsEqual(self, first, second, **kwargs):
        for key in ("cell_embedding", "site_embeddings"):
            torch.testing.assert_close(first[key], second[key], **kwargs)

    def test_config_defaults_asdict_and_state_dict_reload(self):
        self.assertEqual(asdict(PeriodicStateConfig(hidden_size=12)), {
            "hidden_size": 12, "width": 128, "max_sites": 20,
            "radial_basis_count": 16, "radial_cutoff_A": 6.0, "image_radius": 2,
        })
        module, inputs = self._case()
        self._open_projections(module)
        checkpoint = io.BytesIO()
        torch.save({"config": asdict(module.config), "state": module.state_dict()}, checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, weights_only=True, map_location="cpu")
        copy = PeriodicStateConditioner(PeriodicStateConfig(**restored["config"]))
        copy.load_state_dict(restored["state"])
        self.assertOutputsEqual(module(**inputs), copy(**inputs), rtol=0, atol=0)

    def test_shapes_finite_exact_zero_and_nonzero_encoders(self):
        for radius, n, basis in itertools.product((1, 2), (1, 3, 20), (1, 8)):
            with self.subTest(radius=radius, n=n, basis=basis):
                module, inputs = self._case(n=n, image_radius=radius, radial_basis_count=basis)
                output, features = self._capture(module, inputs, "cell_projection", "site_projection")
                self.assertEqual(set(output), {"cell_embedding", "site_embeddings"})
                self.assertEqual(output["cell_embedding"].shape, (2, 12))
                self.assertEqual(output["site_embeddings"].shape, (2, n, 12))
                self.assertEqual(module.image_shifts.shape, ((2 * radius + 1) ** 3, 3))
                for value in output.values():
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertFinite(value)
                    self.assertEqual(torch.count_nonzero(value).item(), 0)
                for value in features.values():
                    self.assertFinite(value)
                    self.assertGreater(value.abs().sum().item(), 0)
                for name, parameter in module.named_parameters():
                    if "projection" not in name:
                        self.assertGreater(parameter.abs().sum().item(), 0, name)

    def test_two_steps_unlock_all_encoder_gradients(self):
        module, inputs = self._case()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
        targets = {"cell_embedding": torch.randn(2, 12),
                   "site_embeddings": torch.randn(2, 3, 12)}
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = module(**inputs)
            loss = sum((outputs[key] - target).square().mean() for key, target in targets.items())
            loss.backward()
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertFinite(parameter.grad)
                nonzero = parameter.grad.abs().sum().item()
                if step == 0 and "projection" not in name:
                    self.assertEqual(nonzero, 0, name)
                else:
                    self.assertGreater(nonzero, 0, name)
            optimizer.step()
        for output in module(**inputs).values():
            self.assertFinite(output)
            self.assertGreater(output.abs().sum().item(), 0)

    def test_same_canvas_different_old_metric_is_identifiable(self):
        module, inputs = self._case(batch=1)
        changed = {**inputs, "lattice": 1.35 * inputs["lattice"]}
        # Species, rank and masks (the canvas-side metadata) stay identical.
        _, first = self._capture(module, inputs, "cell_projection", "site_projection")
        _, second = self._capture(module, changed, "cell_projection", "site_projection")
        for key in first:
            self.assertFalse(torch.allclose(first[key], second[key]), key)
        self._open_projections(module)
        first_output, second_output = module(**inputs), module(**changed)
        for key in first_output:
            self.assertFalse(torch.allclose(first_output[key], second_output[key]), key)

    def test_cell_receives_packing_through_explicit_neighbor_pool(self):
        module, inputs = self._case(batch=1)
        changed = {**inputs, "fractional": inputs["fractional"].clone()}
        changed["fractional"][0, 1] = changed["fractional"][0, 0] + 0.08
        # Remove all per-site coordinate information. Only the RBF/relative-pair
        # route can communicate this packing change to the cell branch.
        def constant_sites(_module, _args, output):
            return torch.ones_like(output)

        handle = module.site_encoder.register_forward_hook(constant_sites)
        try:
            _, first = self._capture(module, inputs, "cell_encoder", "cell_projection", "pair_mlp")
            _, second = self._capture(module, changed, "cell_encoder", "cell_projection")
        finally:
            handle.remove()
        width = module.config.width
        torch.testing.assert_close(first["cell_encoder"][:, :12 + width],
                                   second["cell_encoder"][:, :12 + width], rtol=0, atol=0)
        self.assertFalse(torch.allclose(first["cell_encoder"][:, -width:],
                                        second["cell_encoder"][:, -width:]))
        self.assertFalse(torch.allclose(first["cell_projection"], second["cell_projection"]))
        self.assertEqual(first["pair_mlp"].shape, (1, 3, 3, 8 + 7 + 2 * width))

    def test_cell_metric_uses_row_gram_scale_and_actual_vpa(self):
        module, inputs = self._case(batch=1)
        _, features = self._capture(module, inputs, "cell_encoder")
        lattice = inputs["lattice"][0]
        gram = lattice @ lattice.T
        scale_squared = torch.trace(gram) / 3
        expected = torch.stack([
            gram[0, 0], gram[1, 1], gram[2, 2], gram[0, 1], gram[0, 2], gram[1, 2],
        ]) / scale_squared
        torch.testing.assert_close(features["cell_encoder"][0, :6], expected)
        self.assertAlmostEqual(features["cell_encoder"][0, 6].item(),
                               0.5 * math.log(scale_squared.item()), places=6)
        self.assertAlmostEqual(features["cell_encoder"][0, 7].item(),
                               math.log(abs(torch.linalg.det(lattice).item()) / 3), places=6)

    def test_padding_ignores_arbitrary_unknown_values_and_has_zero_increment(self):
        module, inputs = self._case(batch=1, n=2)
        self._open_projections(module)
        baseline = module(**inputs)
        padded = {**inputs,
            "fractional": torch.cat([inputs["fractional"], torch.full((1, 2, 3), float("nan"))], dim=1),
            "species": torch.cat([inputs["species"], torch.zeros(1, 2, dtype=torch.long)], dim=1),
            "site_known": torch.ones(1, 4, dtype=torch.bool),
            "program_rank": torch.cat([inputs["program_rank"], torch.tensor([[-999, 999]])], dim=1),
            "active_sites": torch.cat([inputs["active_sites"], torch.ones(1, 2, dtype=torch.bool)], dim=1),
        }
        output = module(**padded)
        torch.testing.assert_close(output["cell_embedding"], baseline["cell_embedding"], atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(output["site_embeddings"][:, :2], baseline["site_embeddings"], atol=1e-7, rtol=1e-6)
        self.assertEqual(torch.count_nonzero(output["site_embeddings"][:, 2:]).item(), 0)
        empty = {**padded, "species": torch.zeros_like(padded["species"])}
        for value in module(**empty).values():
            self.assertFinite(value)
            self.assertEqual(torch.count_nonzero(value).item(), 0)

    def test_unknown_coordinates_are_not_fake_neighbors_and_inactive_sites_remain(self):
        module, inputs = self._case(batch=1)
        self._open_projections(module)
        inputs["site_known"][0, 1] = False
        inputs["fractional"][0, 1] = float("nan")
        baseline, captured = self._capture(module, inputs, "pair_mlp")
        radial = captured["pair_mlp"][..., :module.config.radial_basis_count]
        self.assertEqual(torch.count_nonzero(radial[:, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(radial[:, :, 1]).item(), 0)
        arbitrary = {**inputs, "fractional": inputs["fractional"].clone()}
        arbitrary["fractional"][0, 1] = torch.tensor([999.0, -42.0, 0.5])
        self.assertOutputsEqual(baseline, module(**arbitrary), rtol=0, atol=0)
        self.assertGreater(baseline["site_embeddings"][:, 1].abs().sum().item(), 0)
        # Site 2 remains context even if it is not selected for modification.
        inputs["active_sites"].zero_()
        _, inactive = self._capture(module, inputs, "pair_mlp")
        self.assertGreater(inactive["pair_mlp"][:, 2, 2, :8].sum().item(), 0)

    def test_unknown_lattice_masks_metric_and_environment_not_known_fractional(self):
        module, inputs = self._case(batch=1)
        self._open_projections(module)
        inputs["lattice_known"].zero_()
        inputs["lattice"].fill_(float("nan"))
        baseline, features = self._capture(module, inputs, "cell_encoder", "pair_mlp")
        self.assertEqual(torch.count_nonzero(features["cell_encoder"][:, :8]).item(), 0)
        self.assertEqual(torch.count_nonzero(features["cell_encoder"][:, -module.config.width:]).item(), 0)
        self.assertEqual(torch.count_nonzero(features["pair_mlp"][..., :8]).item(), 0)
        arbitrary = {**inputs, "lattice": torch.eye(3).unsqueeze(0) * 100}
        self.assertOutputsEqual(baseline, module(**arbitrary), rtol=0, atol=0)
        for value in baseline.values():
            self.assertFinite(value)

    def test_n1_self_images_match_explicit_shell_sum_and_cutoff(self):
        for radius in (1, 2):
            with self.subTest(radius=radius):
                module, inputs = self._case(batch=1, n=1, image_radius=radius)
                inputs["lattice"] = torch.eye(3).unsqueeze(0)
                _, features = self._capture(module, inputs, "pair_mlp", "cell_encoder")
                radial = features["pair_mlp"][0, 0, 0, :8]
                expected = torch.zeros_like(radial)
                count = 0
                for shift in itertools.product(range(-radius, radius + 1), repeat=3):
                    if shift == (0, 0, 0):
                        continue
                    distance = math.sqrt(sum(value ** 2 for value in shift))
                    if distance < module.config.radial_cutoff_A:
                        count += 1
                        envelope = 0.5 * (1 + math.cos(math.pi * distance / module.config.radial_cutoff_A))
                        expected += torch.exp(-module.radial_gamma * (distance - module.radial_centers).square()) * envelope
                self.assertEqual(count, (2 * radius + 1) ** 3 - 1)
                torch.testing.assert_close(radial, expected)
                self.assertGreater(radial.sum().item(), 0)
                self.assertGreater(features["cell_encoder"][:, -module.config.width:].abs().sum().item(), 0)
                inputs["lattice"] *= 10
                _, remote = self._capture(module, inputs, "pair_mlp", "cell_encoder")
                self.assertEqual(torch.count_nonzero(remote["pair_mlp"][..., :8]).item(), 0)
                self.assertEqual(torch.count_nonzero(remote["cell_encoder"][:, -module.config.width:]).item(), 0)

    def test_skew_cell_pair_rbf_matches_independent_finite_image_enumeration(self):
        for radius in (1, 2):
            with self.subTest(radius=radius):
                module, inputs = self._case(batch=1, n=2, image_radius=radius)
                inputs["lattice"] = torch.tensor([
                    [[1.0, 0.0, 0.0], [2.34, 0.31, 0.0], [0.2, 0.1, 2.7]],
                ])
                inputs["fractional"] = torch.tensor([
                    [[0.9, 0.12, 0.2], [0.07, 0.91, 0.6]],
                ])
                _, features = self._capture(module, inputs, "pair_mlp")
                actual = features["pair_mlp"][0, :, :, :8].double()
                lattice = inputs["lattice"][0].double()
                fractional = inputs["fractional"][0].double()
                expected = torch.zeros(2, 2, 8, dtype=torch.float64)
                for i, j in itertools.product(range(2), repeat=2):
                    delta = fractional[j] - fractional[i]
                    delta = delta - delta.round()
                    for shift in itertools.product(range(-radius, radius + 1), repeat=3):
                        if i == j and shift == (0, 0, 0):
                            continue
                        vector = (delta + torch.tensor(shift, dtype=torch.float64)) @ lattice
                        distance = torch.linalg.vector_norm(vector).item()
                        if distance < module.config.radial_cutoff_A:
                            envelope = 0.5 * (1 + math.cos(math.pi * distance / module.config.radial_cutoff_A))
                            expected[i, j] += torch.exp(
                                -module.radial_gamma.double()
                                * (distance - module.radial_centers.double()).square()
                            ) * envelope
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_mixed_batch_masks_match_individual_calls(self):
        module, inputs = self._case(batch=3)
        self._open_projections(module)
        inputs["species"][0, 1:] = 0
        inputs["fractional"][0, 1:] = float("nan")
        inputs["site_known"][1, 1] = False
        inputs["fractional"][1, 1] = float("inf")
        inputs["lattice_known"][2] = False
        inputs["lattice"][2] = float("nan")
        batched = module(**inputs)
        for row in range(3):
            individual = module(**{key: value[row:row + 1] for key, value in inputs.items()})
            self.assertOutputsEqual(
                {key: value[row:row + 1] for key, value in batched.items()},
                individual, rtol=1e-5, atol=1e-7,
            )

    def test_unknown_nan_inputs_have_zero_finite_gradients(self):
        module, inputs = self._case(batch=2)
        self._open_projections(module)
        inputs["lattice_known"][0] = False
        inputs["lattice"][0] = float("nan")
        inputs["site_known"][:, 1] = False
        inputs["species"][:, 2] = 0
        inputs["fractional"][:, 1:] = float("nan")
        inputs["lattice"].requires_grad_()
        inputs["fractional"].requires_grad_()
        loss = sum(value.square().mean() for value in module(**inputs).values())
        loss.backward()
        self.assertFinite(inputs["lattice"].grad)
        self.assertFinite(inputs["fractional"].grad)
        self.assertEqual(torch.count_nonzero(inputs["lattice"].grad[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(inputs["fractional"].grad[:, 1:]).item(), 0)
        for parameter in module.parameters():
            self.assertFinite(parameter.grad)

    def test_integer_coordinate_aliases_preserve_features(self):
        module, inputs = self._case(batch=1)
        self._open_projections(module)
        # Integer aliases only; do not impose arbitrary origin invariance on
        # the intentionally absolute old-coordinate sin/cos encoding.
        translated = {**inputs, "fractional": inputs["fractional"] + torch.tensor([
            [[1.0, -2.0, 3.0], [0.0, 1.0, -1.0], [-2.0, 0.0, 1.0]],
        ])}
        self.assertOutputsEqual(module(**inputs), module(**translated), atol=1e-7, rtol=1e-5)

    def test_species_rank_known_and_active_flags_reach_encoder(self):
        module, inputs = self._case(batch=1)
        _, baseline = self._capture(module, inputs, "site_projection")
        for name in ("species", "program_rank", "site_known", "active_sites"):
            with self.subTest(name=name):
                changed = {**inputs, name: inputs[name].clone()}
                if changed[name].dtype == torch.bool:
                    changed[name][0, 0] = ~changed[name][0, 0]
                else:
                    changed[name][0, 0] += 1
                _, features = self._capture(module, changed, "site_projection")
                self.assertFalse(torch.allclose(features["site_projection"], baseline["site_projection"]))

    def test_fp32_under_default_dtype_parent_casts_and_cpu_autocast(self):
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            module, inputs = self._case()
        finally:
            torch.set_default_dtype(default_dtype)
        self._open_projections(module)
        original = {key: value.clone() for key, value in module.state_dict().items()}
        parent = nn.ModuleDict({"state": module})
        for dtype in (torch.bfloat16, torch.float16, torch.float64):
            parent.to(dtype=dtype)
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
                if value.is_floating_point():
                    self.assertEqual(value.dtype, torch.float32)
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                low_precision = {**inputs, "lattice": inputs["lattice"].to(dtype),
                                 "fractional": inputs["fractional"].to(dtype)}
                expected = module(**low_precision)
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    outputs, features = self._capture(module, low_precision, "cell_encoder", "pair_mlp")
                    loss = sum(value.square().mean() for value in outputs.values())
                self.assertOutputsEqual(outputs, expected, rtol=0, atol=0)
                for value in list(outputs.values()) + list(features.values()):
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertFinite(value)
                module.zero_grad(set_to_none=True)
                loss.backward()
                for parameter in module.parameters():
                    self.assertEqual(parameter.grad.dtype, torch.float32)
                    self.assertFinite(parameter.grad)

    def test_repeated_metric_and_coincident_sites_have_finite_input_gradients(self):
        for n in (1, 2):
            with self.subTest(n=n):
                module, inputs = self._case(batch=1, n=n)
                self._open_projections(module)
                inputs["lattice"] = (torch.eye(3).unsqueeze(0) * 3).requires_grad_()
                inputs["fractional"] = torch.zeros(1, n, 3, requires_grad=True)
                loss = sum(value.square().mean() for value in module(**inputs).values())
                loss.backward()
                for key in ("lattice", "fractional"):
                    self.assertFinite(inputs[key].grad)
                    self.assertGreater(inputs[key].grad.abs().sum().item(), 0)

    def test_config_and_shape_errors(self):
        for field, value in (("hidden_size", 0), ("width", 0), ("max_sites", 0),
                             ("radial_basis_count", 0), ("radial_cutoff_A", 0),
                             ("radial_cutoff_A", float("nan")), ("image_radius", 3)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                PeriodicStateConfig(**{"hidden_size": 12, field: value})
        module, inputs = self._case()
        with self.assertRaisesRegex(ValueError, "species"):
            module(**{**inputs, "species": inputs["species"][:, :1]})
        with self.assertRaisesRegex(TypeError, "bool"):
            module(**{**inputs, "site_known": inputs["site_known"].long()})
        oversized, inputs = self._case(n=21)
        with self.assertRaisesRegex(ValueError, "max_sites"):
            oversized(**inputs)


if __name__ == "__main__":
    unittest.main()
