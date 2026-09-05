from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer
from crystal_dlm.manifold_corruption import (
    CorruptionConfig,
    CrystalGeometry,
    generate_corruption_proposal,
    minimum_image_cartesian_retraction,
    relative_spd_tangent,
)
from crystal_dlm.offline_pmtr_certification import (
    CertificationConfig,
    certify_corruption_proposals,
    joint_toward_clean_probe,
)


def clean_case():
    body, _ = arrays_to_dynamic_answer(
        lengths=[4.2, 5.1, 6.3],
        angles=[78.0, 91.0, 103.0],
        species=["O", "O", "Na", "Cl"],
        frac_coords=[
            [0.03, 0.17, 0.29],
            [0.41, 0.53, 0.67],
            [0.72, 0.11, 0.84],
            [0.26, 0.78, 0.45],
        ],
        separator="",
    )
    parsed = parse_dynamic_answer(body, strict=True)
    return body, tuple(parsed["tokens"]), CrystalGeometry.from_mapping(parsed)


def proposal(index=0):
    body, tokens, clean = clean_case()
    config = CorruptionConfig(
        max_proposals=4,
        lattice_log_std=0.10,
        coordinate_cartesian_std_A=0.28,
        max_delta_energy=10.0,
    )
    return generate_corruption_proposal(
        clean,
        clean_body=body,
        clean_tokens=tokens,
        request_key="train:7",
        proposal_index=index,
        seed=17,
        config=config,
    ), config


class QuadraticFakeEvaluator:
    def __init__(self, clean: CrystalGeometry) -> None:
        self.clean = clean
        self.calls = []

    def evaluate(self, geometries, *, batch_size):
        self.calls.append((len(geometries), batch_size))
        output = []
        for geometry in geometries:
            vectors = minimum_image_cartesian_retraction(
                self.clean.frac_coords,
                geometry.frac_coords,
                geometry.lattice,
                image_radius=2,
            )
            tangent = relative_spd_tangent(geometry.metric, self.clean.metric)
            energy = float(
                np.mean(np.sum(vectors * vectors, axis=1))
                + 0.05 * np.sum(tangent * tangent)
            )
            output.append(
                {
                    "e": energy,
                    "f": 2.0 * vectors,
                    "s": np.full((3, 3), 1.0e6),
                    "m": np.zeros(len(geometry.species)),
                }
            )
        return output


class SequenceFakeEvaluator:
    def __init__(self, energies, force) -> None:
        self.energies = list(energies)
        self.force = np.asarray(force, dtype=float)

    def evaluate(self, geometries, *, batch_size):
        self.batch_size = batch_size
        return [
            {
                "e": energy,
                "f": self.force,
                "s": np.eye(3),
                "m": np.zeros(len(geometry.species)),
            }
            for geometry, energy in zip(geometries, self.energies, strict=True)
        ]


class OfflinePMTRCertificationTest(unittest.TestCase):
    def test_joint_probe_is_closer_and_certifies_without_stress_contraction(self):
        item, corruption_config = proposal()
        evaluator = QuadraticFakeEvaluator(item.clean_geometry)
        probe = joint_toward_clean_probe(item, fraction=0.10)
        before_metric = np.linalg.norm(
            relative_spd_tangent(item.geometry.metric, item.clean_geometry.metric)
        )
        after_metric = np.linalg.norm(
            relative_spd_tangent(probe.metric, item.clean_geometry.metric)
        )
        self.assertLess(after_metric, before_metric)
        endpoint = joint_toward_clean_probe(item, fraction=1.0 - 1.0e-9)
        endpoint_vectors = minimum_image_cartesian_retraction(
            item.clean_geometry.frac_coords,
            endpoint.frac_coords,
            endpoint.lattice,
            image_radius=2,
        )
        self.assertLess(np.linalg.norm(endpoint_vectors), 1.0e-7)

        records = certify_corruption_proposals(
            [item],
            evaluator=evaluator,
            corruption_config=corruption_config,
            certification_config=CertificationConfig(probe_fraction=0.10),
            batch_size=8,
        )
        self.assertEqual(evaluator.calls, [(3, 8)])
        self.assertTrue(records[0]["certified"])
        certificate = records[0]["certificate"]
        self.assertGreater(certificate["delta_energy"], 0.0)
        self.assertGreater(
            certificate["coordinate_force_dot_clean_retraction"], 0.0
        )
        self.assertGreater(
            certificate["lattice_descent_dot_spd_retraction"], 0.0
        )
        self.assertIn(
            "no_stress_tangent_dot", records[0]["evidence"]["method"]
        )

    def test_joint_probe_must_actually_lower_energy(self):
        item, corruption_config = proposal()
        positive_force = item.clean_coordinate_retraction_cartesian
        evaluator = SequenceFakeEvaluator(
            energies=[0.0, 0.5, 0.6], force=positive_force
        )
        record = certify_corruption_proposals(
            [item],
            evaluator=evaluator,
            corruption_config=corruption_config,
            batch_size=16,
        )[0]
        self.assertFalse(record["certified"])
        self.assertFalse(record["evidence"]["joint_probe_downhill"])
        self.assertLess(
            record["certificate"]["lattice_descent_dot_spd_retraction"], 0.0
        )

    def test_coordinate_force_must_point_toward_clean(self):
        item, corruption_config = proposal()
        opposite_force = -item.clean_coordinate_retraction_cartesian
        evaluator = SequenceFakeEvaluator(
            energies=[0.0, 0.5, 0.4], force=opposite_force
        )
        record = certify_corruption_proposals(
            [item],
            evaluator=evaluator,
            corruption_config=corruption_config,
            batch_size=16,
        )[0]
        self.assertFalse(record["certified"])
        self.assertFalse(record["evidence"]["coordinate_force_aligned"])

    def test_clean_evaluation_is_deduplicated_across_proposals(self):
        first, corruption_config = proposal(0)
        second, _ = proposal(1)
        evaluator = QuadraticFakeEvaluator(first.clean_geometry)
        records = certify_corruption_proposals(
            [first, second],
            evaluator=evaluator,
            corruption_config=corruption_config,
            batch_size=8,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(evaluator.calls, [(5, 8)])
        self.assertEqual([row["proposal_index"] for row in records], [0, 1])

    def test_batch_size_outside_eight_to_sixteen_is_rejected(self):
        item, corruption_config = proposal()
        evaluator = QuadraticFakeEvaluator(item.clean_geometry)
        with self.assertRaisesRegex(ValueError, "8..16"):
            certify_corruption_proposals(
                [item],
                evaluator=evaluator,
                corruption_config=corruption_config,
                batch_size=4,
            )


if __name__ == "__main__":
    unittest.main()
