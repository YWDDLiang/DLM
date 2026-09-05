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
    CorruptionCertificate,
    CorruptionConfig,
    CrystalGeometry,
    minimum_image_cartesian_retraction,
    select_first_certified_corruption,
)


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
