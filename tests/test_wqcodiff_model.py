from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is exercised in local/Slurm model smoke")
class ModelContractTests(unittest.TestCase):
    def _record(self):
        from crystal_dlm.wqcodiff.dataset import tolerance_tag

        primary = {
            "state": {
                "space_group": 1,
                "lattice_system": "triclinic",
                "lattice_chart": [1.6, 1.6, 1.6, 0.0, 0.0, 0.0],
            },
            "primitive_lattice_transform": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "orbits": [
                {
                    "orbit": {
                        "orbit_id": "o0",
                        "wyckoff_type": 0,
                        "species": 6,
                        "multiplicity": 1,
                        "primitive_multiplicity": 1,
                        "chart_dimension": 3,
                        "free_coordinate": [0.2, 0.3, 0.4],
                    },
                    "primitive_fractional_coordinates": [[0.2, 0.3, 0.4]],
                    "primitive_chart_jacobians": [
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    ],
                }
            ],
        }
        return {
            "material_id": "toy-p1-c",
            "selected": True,
            "decompositions": {tolerance_tag(1.0e-2): primary},
        }

    def _batch(self):
        import torch

        from crystal_dlm.wqcodiff.model import WQTensorBatch

        return WQTensorBatch(
            atom_species=torch.tensor([1, 8, 8]),
            frac_coords=torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.6, 0.5, 0.4]],
                dtype=torch.float32,
            ),
            lattices=torch.eye(3)[None] * 5.0,
            atom_batch=torch.zeros(3, dtype=torch.long),
            atom_to_orbit=torch.tensor([0, 1, 1]),
            orbit_species=torch.tensor([1, 8]),
            orbit_wyckoff=torch.tensor([1, 2]),
            orbit_batch=torch.zeros(2, dtype=torch.long),
            space_group=torch.tensor([1]),
            time=torch.tensor([0.7]),
            geometry_evidence=torch.zeros((2, 6)),
        )

    def test_empty_set_prior_and_all_losses_are_finite(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.losses import (
            WQLossTargets,
            WQPriorTargets,
            compute_wq_loss_terms,
            compute_wq_losses,
        )
        from crystal_dlm.wqcodiff.model import WQCoDenoiser, WQVariant

        torch.manual_seed(7)
        model = WQCoDenoiser().eval()
        batch = self._batch()
        output = model(batch, variant=WQVariant.STRAT_GEO)
        masked = model.forward_prior(torch.ones(1), torch.zeros(1, dtype=torch.long))
        conditioned = model.forward_prior(torch.ones(1), torch.ones(1, dtype=torch.long))
        targets = WQLossTargets(
            space_group=torch.tensor([0]),
            species=torch.tensor([0, 7]),
            wyckoff=torch.tensor([0, 1]),
            event=torch.tensor([0]),
            event_orbit=torch.tensor([0.0, 1.0]),
            event_orbit_mask=torch.tensor([True, True]),
            birth_species=torch.tensor([-100]),
            birth_wyckoff=torch.tensor([-100]),
            birth_coordinate=torch.zeros((1, 3)),
            birth_coordinate_mask=torch.zeros((1, 3), dtype=torch.bool),
            revision=torch.zeros((2, 3)),
            revision_mask=torch.ones((2, 3), dtype=torch.bool),
            coordinate_score=torch.zeros((3, 3)),
            coordinate_mask=torch.ones(3, dtype=torch.bool),
            coordinate_weight=torch.ones(3),
            lattice_score=torch.zeros((1, 6)),
            lattice_mask=torch.ones((1, 6), dtype=torch.bool),
            bridge_coordinate=torch.zeros((2, 3)),
            bridge_mask=torch.zeros((2, 3), dtype=torch.bool),
        )
        prior_targets = WQPriorTargets(
            space_group=torch.tensor([0]),
            first_species=torch.tensor([0]),
            first_wyckoff=torch.tensor([0]),
            first_coordinate=torch.zeros((1, 3)),
            first_coordinate_mask=torch.ones((1, 3), dtype=torch.bool),
            lattice_chart=torch.zeros((1, 6)),
            lattice_chart_mask=torch.ones((1, 6), dtype=torch.bool),
        )
        losses = compute_wq_losses(
            output,
            targets,
            masked_prior=masked,
            conditioned_prior=conditioned,
            prior_target=prior_targets,
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses))
        self.assertGreater(float(losses.prior.detach()), 0.0)
        terms = compute_wq_loss_terms(
            output,
            targets,
            masked_prior=masked,
            conditioned_prior=conditioned,
            prior_target=prior_targets,
        )
        self.assertTrue(torch.equal(losses.total, sum(terms)))

    def test_sigma_squared_coordinate_dsm_is_scale_invariant(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.losses import _coordinate_score_loss

        sigma = torch.tensor([0.005, 0.05, 0.5])
        prediction = torch.zeros((3, 3))
        target = torch.zeros((3, 3))
        target[:, 0] = sigma.reciprocal()
        loss = _coordinate_score_loss(
            prediction,
            target,
            torch.ones(3, dtype=torch.bool),
            sigma.square(),
        )
        self.assertAlmostEqual(float(loss), 1.0 / 3.0, places=6)

    def test_wrapped_coordinate_nll_is_integer_shift_invariant(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.losses import _wrapped_gaussian_nll

        mean = torch.tensor([[0.99, 0.02, 0.50]])
        value = torch.tensor([[0.01, 0.98, 0.55]])
        log_scale = torch.full_like(mean, 0.08).log()
        reference = _wrapped_gaussian_nll(mean, log_scale, value)
        shifted = _wrapped_gaussian_nll(mean + 2.0, log_scale, value - 3.0)
        self.assertTrue(torch.allclose(reference, shifted, atol=2.0e-6, rtol=0.0))
        # A two-percent displacement across the unit boundary must have the
        # same likelihood as the same displacement away from the boundary.
        boundary = _wrapped_gaussian_nll(
            torch.tensor([[0.99]]), torch.tensor([[0.08]]).log(), torch.tensor([[0.01]])
        )
        interior = _wrapped_gaussian_nll(
            torch.tensor([[0.49]]), torch.tensor([[0.08]]).log(), torch.tensor([[0.51]])
        )
        self.assertTrue(torch.allclose(boundary, interior, atol=1.0e-6, rtol=0.0))

    def test_periodic_coordinate_scales_are_smoothly_bounded(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.model import (
            PERIODIC_COORDINATE_SCALE_MAX,
            PERIODIC_COORDINATE_SCALE_MIN,
            _bounded_periodic_log_scale,
        )

        raw = torch.tensor([-100.0, 0.0, 100.0], requires_grad=True)
        scales = _bounded_periodic_log_scale(raw).exp()
        self.assertGreaterEqual(
            float(scales.detach().min()), PERIODIC_COORDINATE_SCALE_MIN - 1.0e-7
        )
        self.assertLessEqual(
            float(scales.detach().max()), PERIODIC_COORDINATE_SCALE_MAX + 1.0e-7
        )
        scales[1].backward()
        self.assertGreater(float(raw.grad[1]), 0.0)

    def test_set_model_is_equivariant_to_orbit_storage_order(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.model import WQCoDenoiser, WQTensorBatch, WQVariant

        torch.manual_seed(13)
        model = WQCoDenoiser().eval()
        batch = self._batch()
        with torch.no_grad():
            first = model(batch, variant=WQVariant.STRAT_GEO)
        permutation = torch.tensor([1, 0])
        inverse = torch.tensor([1, 0])
        permuted = WQTensorBatch(
            atom_species=batch.atom_species,
            frac_coords=batch.frac_coords,
            lattices=batch.lattices,
            atom_batch=batch.atom_batch,
            atom_to_orbit=inverse[batch.atom_to_orbit],
            orbit_species=batch.orbit_species[permutation],
            orbit_wyckoff=batch.orbit_wyckoff[permutation],
            orbit_batch=batch.orbit_batch[permutation],
            space_group=batch.space_group,
            time=batch.time,
            geometry_evidence=batch.geometry_evidence[permutation],
        )
        with torch.no_grad():
            second = model(permuted, variant=WQVariant.STRAT_GEO)
        difference = torch.max(
            torch.abs(first.species_logits - second.species_logits[inverse])
        )
        self.assertLess(float(difference), 1.0e-5)
        graph_difference = torch.max(
            torch.abs(first.space_group_logits - second.space_group_logits)
        )
        self.assertLess(float(graph_difference), 1.0e-5)

    def test_transient_attention_padding_cannot_change_real_outputs(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.model import WQCoDenoiser, WQTensorBatch, WQVariant
        from crystal_dlm.wqcodiff.runtime import (
            concatenate_tensor_batches,
            split_model_output,
        )

        torch.manual_seed(17)
        model = WQCoDenoiser().eval()
        large = self._batch()
        small = WQTensorBatch(
            atom_species=large.atom_species[:1],
            frac_coords=large.frac_coords[:1],
            lattices=large.lattices,
            atom_batch=torch.zeros(1, dtype=torch.long),
            atom_to_orbit=torch.zeros(1, dtype=torch.long),
            orbit_species=large.orbit_species[:1],
            orbit_wyckoff=large.orbit_wyckoff[:1],
            orbit_batch=torch.zeros(1, dtype=torch.long),
            space_group=large.space_group,
            time=large.time,
            geometry_evidence=large.geometry_evidence[:1],
        )
        with torch.no_grad():
            small_alone = model(small, variant=WQVariant.STRAT_GEO)
            large_alone = model(large, variant=WQVariant.STRAT_GEO)
            combined_inputs = concatenate_tensor_batches((small, large))
            combined = model(combined_inputs, variant=WQVariant.STRAT_GEO)
            small_batched, large_batched = split_model_output(
                combined, (small, large)
            )
        for first, second in (
            (small_alone.species_logits, small_batched.species_logits),
            (small_alone.space_group_logits, small_batched.space_group_logits),
            (small_alone.atom_coordinate_score, small_batched.atom_coordinate_score),
            (large_alone.species_logits, large_batched.species_logits),
            (large_alone.space_group_logits, large_batched.space_group_logits),
            (large_alone.atom_coordinate_score, large_batched.atom_coordinate_score),
        ):
            self.assertLess(float(torch.max(torch.abs(first - second))), 1.0e-5)

    def test_transient_attention_scatter_preserves_gradients(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.model import RaggedSetBlock, WQModelConfig

        torch.manual_seed(19)
        block = RaggedSetBlock(WQModelConfig())
        features = torch.randn(5, 256, requires_grad=True)
        orbit_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        output = block(features, orbit_batch, causal=False)
        loss = output.square().mean()
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_periodic_messages_support_bfloat16_autocast(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.model import PeriodicMessageLayer, WQModelConfig

        torch.manual_seed(21)
        layer = PeriodicMessageLayer(WQModelConfig()).eval()
        # Reproduce the CUDA training dtype boundary: learned node features
        # are BF16 while fractional coordinates, lattices, distances, and
        # cutoff weights remain FP32 under autocast.
        features = torch.randn(3, 256, dtype=torch.bfloat16)
        frac_coords = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.8, 0.2, 0.7]],
            dtype=torch.float32,
        )
        lattices = torch.eye(3, dtype=torch.float32)[None] * 5.0
        atom_batch = torch.zeros(3, dtype=torch.long)
        with torch.no_grad(), torch.autocast(
            device_type="cpu", dtype=torch.bfloat16
        ):
            output = layer(features, frac_coords, lattices, atom_batch)
        self.assertEqual(output.dtype, features.dtype)
        self.assertTrue(torch.isfinite(output.float()).all())

    def test_geometry_score_norm_is_current_model_signal_not_teacher_input(self) -> None:
        import dataclasses

        import torch

        from crystal_dlm.wqcodiff.model import WQCoDenoiser, WQVariant

        torch.manual_seed(23)
        model = WQCoDenoiser().eval()
        batch = self._batch()
        poisoned = batch.geometry_evidence.clone()
        poisoned[:, 4] = torch.tensor([1.0e6, -1.0e6])
        with torch.no_grad():
            clean_output = model(batch, variant=WQVariant.STRAT_GEO)
            poisoned_output = model(
                dataclasses.replace(batch, geometry_evidence=poisoned),
                variant=WQVariant.STRAT_GEO,
            )
        for clean, changed in (
            (clean_output.revision_logits, poisoned_output.revision_logits),
            (clean_output.species_logits, poisoned_output.species_logits),
            (clean_output.atom_coordinate_score, poisoned_output.atom_coordinate_score),
        ):
            self.assertLess(float(torch.max(torch.abs(clean - changed))), 1.0e-7)

    def test_registered_corruptions_tensorize_for_every_matched_variant(self) -> None:
        import torch

        from crystal_dlm.wqcodiff.losses import compute_wq_losses
        from crystal_dlm.wqcodiff.model import WQCoDenoiser, WQVariant
        from crystal_dlm.wqcodiff.training_data import build_corrupted_batch

        model = WQCoDenoiser().eval()
        for index, variant in enumerate(WQVariant):
            corrupted = build_corrupted_batch(
                [self._record()],
                seed=100 + index,
                variant=variant,
                enable_revision_training=True,
            )
            output = model(corrupted.inputs, variant=variant)
            masked = model.forward_prior(
                torch.ones(1), torch.zeros(1, dtype=torch.long)
            )
            conditioned = model.forward_prior(
                torch.ones(1), corrupted.prior_targets.space_group + 1
            )
            losses = compute_wq_losses(
                output,
                corrupted.targets,
                masked_prior=masked,
                conditioned_prior=conditioned,
                prior_target=corrupted.prior_targets,
            )
            self.assertTrue(
                torch.isfinite(losses.total),
                f"non-finite loss for {variant.value}: {losses}",
            )

    def test_atom_baseline_uses_unpadded_p1_atom_set(self) -> None:
        from pymatgen.core import Lattice, Structure

        from crystal_dlm.wqcodiff.training_data import _atom_level_primary

        primary = self._record()["decompositions"]["symprec_1e-02"]
        primary["primitive_structure"] = Structure(
            Lattice.cubic(5.0),
            [6, 8],
            [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]],
        ).as_dict()
        converted = _atom_level_primary(primary)
        self.assertEqual(converted["state"]["space_group"], 1)
        self.assertEqual(len(converted["orbits"]), 2)
        self.assertTrue(
            all(item["orbit"]["primitive_multiplicity"] == 1 for item in converted["orbits"])
        )
        self.assertTrue(
            all(item["orbit"]["chart_dimension"] == 3 for item in converted["orbits"])
        )

    def test_screening_stop_uses_the_100k_schedule_boundary(self) -> None:
        from crystal_dlm.wqcodiff.model import WQVariant
        from crystal_dlm.wqcodiff.training import TrainingConfig

        config = TrainingConfig(
            dataset_paths=("train.jsonl",),
            output_dir="screen",
            variant=WQVariant.STRAT_GEO,
            training_seed=11,
            source_bundle_sha256="a" * 64,
            stop_after_update=85_000,
        )
        self.assertEqual(config.shared_updates, 60_000)
        self.assertEqual(config.end_update, 85_000)
        self.assertFalse(config.paper_eligible)

    def test_training_rejects_unbound_source_tree(self) -> None:
        from crystal_dlm.wqcodiff.model import WQVariant
        from crystal_dlm.wqcodiff.training import TrainingConfig

        with self.assertRaisesRegex(ValueError, "source_bundle_sha256"):
            TrainingConfig(
                dataset_paths=("train.jsonl",),
                output_dir="unbound",
                variant=WQVariant.STRAT_GEO,
                training_seed=11,
                source_bundle_sha256="not-a-sha",
            )


if __name__ == "__main__":
    unittest.main()
