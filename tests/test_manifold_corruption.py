import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.dynamic_crystal import parse_dynamic_answer
from crystal_dlm.manifold_corruption import (
    canonical_lattice_from_metric,
    CorruptionCertificate,
    CorruptionConfig,
    CrystalGeometry,
    minimum_image_cartesian_retraction,
    relative_spd_tangent,
    select_first_certified_corruption,
    spd_congruence_update,
)

try:
    import torch

    from crystal_dlm.manifold_geometry import (
        cartesian_to_fractional as torch_cartesian_to_fractional,
        fractional_to_cartesian as torch_fractional_to_cartesian,
        metric_to_lattice as torch_metric_to_lattice,
        relative_spd_tangent as torch_relative_spd_tangent,
        spd_congruence_update as torch_spd_congruence_update,
        wrap_fractional as torch_wrap_fractional,
        wrapped_fractional_delta as torch_wrapped_fractional_delta,
    )
except ImportError:
    torch = None


def clean_geometry():
    return CrystalGeometry.from_mapping(
        {
            "lengths": [4.2, 5.1, 6.3],
            "angles": [78.0, 91.0, 103.0],
            "species": ["O", "O", "Na", "Cl"],
            "frac_coords": [
                [0.03, 0.17, 0.29],
                [0.41, 0.53, 0.67],
                [0.72, 0.11, 0.84],
                [0.26, 0.78, 0.45],
            ],
        }
    )


def accepting_certificate(_proposal):
    return CorruptionCertificate(
        post_quantization_valid=True,
        delta_energy=0.25,
        coordinate_force_dot_clean_retraction=0.8,
        lattice_descent_dot_spd_retraction=0.4,
    )


class ManifoldCorruptionTest(unittest.TestCase):
    config = CorruptionConfig(
        lattice_log_std=0.10,
        coordinate_cartesian_std_A=0.28,
        max_delta_energy=1.0,
    )

    def test_coordinate_retraction_uses_periodic_minimum_image(self):
        vectors = minimum_image_cartesian_retraction(
            target_fractional=np.asarray([[0.01, 0.20, 0.30]]),
            source_fractional=np.asarray([[0.99, 0.20, 0.30]]),
            lattice=np.diag([10.0, 8.0, 7.0]),
        )
        self.assertTrue(np.allclose(vectors, [[0.2, 0.0, 0.0]], atol=1.0e-12))

    @unittest.skipIf(torch is None, "Torch is unavailable in the lightweight local runtime")
    def test_numpy_spd_builder_matches_training_torch_primitive(self):
        metric = np.asarray(
            [[12.0, 1.2, -0.4], [1.2, 15.0, 0.8], [-0.4, 0.8, 18.0]],
            dtype=float,
        )
        tangent = np.asarray(
            [[0.03, -0.02, 0.01], [-0.02, -0.01, 0.015], [0.01, 0.015, 0.02]],
            dtype=float,
        )
        numpy_updated = spd_congruence_update(metric, tangent)
        torch_metric = torch.tensor(metric, dtype=torch.float64)
        torch_tangent = torch.tensor(tangent, dtype=torch.float64)
        torch_updated = torch_spd_congruence_update(torch_metric, torch_tangent)
        self.assertTrue(
            np.allclose(numpy_updated, torch_updated.detach().cpu().numpy(), atol=1.0e-10)
        )
        self.assertTrue(
            np.allclose(
                canonical_lattice_from_metric(numpy_updated),
                torch_metric_to_lattice(torch_updated).detach().cpu().numpy(),
                atol=1.0e-10,
            )
        )
        numpy_reverse = relative_spd_tangent(numpy_updated, metric)
        torch_reverse = torch_relative_spd_tangent(torch_updated, torch_metric)
        self.assertTrue(
            np.allclose(numpy_reverse, torch_reverse.detach().cpu().numpy(), atol=1.0e-10)
        )
        lattice = canonical_lattice_from_metric(numpy_updated)
        fractional = np.asarray([[0.99, -0.02, 1.04], [0.21, 0.35, 0.48]])
        numpy_cartesian = fractional @ lattice
        torch_lattice = torch.tensor(lattice, dtype=torch.float64)
        torch_fractional = torch.tensor(fractional, dtype=torch.float64)
        torch_cartesian = torch_fractional_to_cartesian(
            torch_fractional, torch_lattice
        )
        self.assertTrue(
            np.allclose(
                numpy_cartesian, torch_cartesian.detach().cpu().numpy(), atol=1.0e-10
            )
        )
        torch_roundtrip = torch_cartesian_to_fractional(
            torch_cartesian, torch_lattice
        )
        self.assertTrue(
            np.allclose(
                np.mod(fractional, 1.0),
                torch_wrap_fractional(torch_roundtrip).detach().cpu().numpy(),
                atol=1.0e-10,
            )
        )
        delta = np.asarray([[0.98, -0.98, 1.02]])
        self.assertTrue(
            np.allclose(
                delta - np.round(delta),
                torch_wrapped_fractional_delta(
                    torch.tensor(delta, dtype=torch.float64)
                )
                .detach()
                .cpu()
                .numpy(),
                atol=1.0e-10,
            )
        )

    def test_spd_pbc_native_roundtrip_and_request_determinism(self):
        first = select_first_certified_corruption(
            clean_geometry(),
            request_key={"split": "train", "row": 7},
            certify=accepting_certificate,
            seed=19,
            config=self.config,
        )
        second = select_first_certified_corruption(
            clean_geometry(),
            request_key={"row": 7, "split": "train"},
            certify=accepting_certificate,
            seed=19,
            config=self.config,
        )
        self.assertFalse(first.fallback)
        self.assertIsNotNone(first.proposal)
        proposal = first.proposal
        assert proposal is not None
        self.assertEqual(proposal.body, second.proposal.body)
        self.assertTrue(np.all(np.linalg.eigvalsh(proposal.geometry.metric) > 0.0))
        self.assertTrue(np.all(proposal.geometry.frac_coords >= 0.0))
        self.assertTrue(np.all(proposal.geometry.frac_coords < 1.0))
        self.assertTrue(
            np.allclose(
                proposal.sampled_zero_com_cartesian_displacement.mean(axis=0),
                0.0,
                atol=1.0e-12,
            )
        )
        parsed = parse_dynamic_answer(proposal.body, strict=True)
        self.assertEqual(len(parsed["tokens"]), 7 + 4 * 4)
        self.assertEqual(parsed["species"], ["O", "O", "Na", "Cl"])
        self.assertTrue(proposal.lattice_changed_positions)
        self.assertTrue(proposal.coordinate_changed_positions)

        different = select_first_certified_corruption(
            clean_geometry(),
            request_key={"split": "train", "row": 8},
            certify=accepting_certificate,
            seed=19,
            config=self.config,
        )
        self.assertNotEqual(proposal.body, different.proposal.body)

    def test_first_certified_is_selected_without_best_energy_search(self):
        calls = []

        def provider(proposal):
            calls.append(proposal.proposal_index)
            return {
                "post_quantization_valid": proposal.proposal_index >= 1,
                "delta_energy": 0.8 if proposal.proposal_index == 1 else 0.01,
                "coordinate_force_dot_clean_retraction": 1.0,
                "lattice_descent_dot_spd_retraction": 1.0,
            }

        selection = select_first_certified_corruption(
            clean_geometry(),
            request_key="first-certified",
            certify=provider,
            seed=4,
            config=self.config,
        )
        self.assertEqual(selection.proposal.proposal_index, 1)
        self.assertEqual(selection.attempted_proposals, 2)
        self.assertEqual(calls, [0, 1])
        self.assertAlmostEqual(selection.certificate.delta_energy, 0.8)

    def test_certificate_requires_bounded_energy_and_active_descent_dots(self):
        clean = clean_geometry()
        body_selection = select_first_certified_corruption(
            clean,
            request_key="proposal-fixture",
            certify=accepting_certificate,
            seed=11,
            config=self.config,
        )
        proposal = body_selection.proposal
        assert proposal is not None
        bad_values = [
            CorruptionCertificate(True, 0.0, 1.0, 1.0),
            CorruptionCertificate(True, 2.0, 1.0, 1.0),
            CorruptionCertificate(True, 0.2, -0.1, 1.0),
            CorruptionCertificate(True, 0.2, 1.0, 0.0),
            CorruptionCertificate(False, 0.2, 1.0, 1.0),
        ]
        self.assertTrue(accepting_certificate(proposal).accepts(proposal, self.config))
        self.assertTrue(all(not item.accepts(proposal, self.config) for item in bad_values))

    def test_no_certificate_emits_clean_ce_fallback_after_four_proposals(self):
        selection = select_first_certified_corruption(
            clean_geometry(),
            request_key="fallback",
            certify=lambda _proposal: None,
            seed=9,
            config=self.config,
        )
        self.assertTrue(selection.fallback)
        self.assertEqual(selection.attempted_proposals, 4)
        self.assertEqual(selection.source_body, selection.clean_body)
        self.assertIsNone(selection.certificate)


if __name__ == "__main__":
    unittest.main()
