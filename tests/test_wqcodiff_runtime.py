from __future__ import annotations

import importlib.util
import math
import unittest
from unittest import mock

import numpy as np

from crystal_dlm.wqcodiff.bridge import ChartCatalog, ChartSpec
from crystal_dlm.wqcodiff.charts import LatticeChartCodec
from crystal_dlm.wqcodiff.state import OrbitState, StratifiedState


SPGLIB_AVAILABLE = importlib.util.find_spec("spglib") is not None
PYXTAL_AVAILABLE = importlib.util.find_spec("pyxtal") is not None


class GeometrySignalTests(unittest.TestCase):
    def test_vectorized_signals_match_registered_pair_loop(self) -> None:
        from crystal_dlm.wqcodiff.charts import periodic_cartesian_distance
        from crystal_dlm.wqcodiff.runtime import geometry_signals_from_graph

        coordinates = np.asarray(
            [
                [0.02, 0.07, 0.11],
                [0.96, 0.09, 0.13],
                [0.42, 0.51, 0.63],
                [0.44, 0.49, 0.60],
                [0.77, 0.24, 0.85],
            ],
            dtype=np.float64,
        )
        lattice = np.asarray(
            [[4.7, 0.0, 0.0], [0.6, 5.3, 0.0], [-0.4, 0.8, 6.1]],
            dtype=np.float64,
        )
        mapping = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)

        minimum = np.full(len(coordinates), np.inf, dtype=np.float64)
        neighbors = np.zeros(len(coordinates), dtype=np.int64)
        for first in range(len(coordinates)):
            for second in range(first + 1, len(coordinates)):
                distance = periodic_cartesian_distance(
                    coordinates[first], coordinates[second], lattice
                )
                minimum[first] = min(minimum[first], distance)
                minimum[second] = min(minimum[second], distance)
                if distance <= 3.0:
                    neighbors[first] += 1
                    neighbors[second] += 1
        expected_collision = np.zeros(3, dtype=np.float64)
        expected_coordination = np.zeros(3, dtype=np.float64)
        for orbit_index in range(3):
            selected = mapping == orbit_index
            min_distance = float(np.min(minimum[selected]))
            expected_collision[orbit_index] = max(
                0.0, min(1.0, (0.8 - min_distance) / 0.8)
            )
            expected_coordination[orbit_index] = float(
                np.mean((neighbors[selected] < 1) | (neighbors[selected] > 12))
            )
        volume_per_atom = abs(float(np.linalg.det(lattice))) / len(coordinates)
        condition = float(np.linalg.cond(lattice))
        expected_strain = min(
            1.0,
            abs(math.log(max(volume_per_atom, 1.0e-12) / 20.0)) / 4.0
            + max(0.0, math.log(max(condition, 1.0)) - math.log(8.0)) / 4.0,
        )

        collision, coordination, strain = geometry_signals_from_graph(
            coordinates, lattice, mapping, 3
        )
        np.testing.assert_allclose(collision, expected_collision, atol=1.0e-14)
        np.testing.assert_allclose(
            coordination, expected_coordination, atol=0.0
        )
        self.assertAlmostEqual(strain, expected_strain, places=14)


class RuntimeRedetectionControlTests(unittest.TestCase):
    def test_absent_geometry_can_skip_unused_space_group_redetection(self) -> None:
        from crystal_dlm.wqcodiff.runtime import expand_state

        state = StratifiedState(
            space_group=1,
            lattice_system="triclinic",
            lattice_chart=LatticeChartCodec.encode_matrix(
                np.eye(3) * 5.0, "triclinic"
            ),
            orbits=(OrbitState("o0", 0, 6, 1, 3, (0.2, 0.3, 0.4)),),
        )
        with mock.patch(
            "crystal_dlm.wqcodiff.runtime._redetect_space_group",
            side_effect=AssertionError("redetection must not run"),
        ):
            expanded = expand_state(
                state, _P1Catalog(), redetect_space_group=False
            )
        self.assertEqual(expanded.redetected_space_group, 0)


class _P1Catalog(ChartCatalog):
    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        if (space_group, wyckoff_type) != (1, 0):
            raise KeyError((space_group, wyckoff_type))
        return ChartSpec(1, 0, "a", 1, 3)

    def types(self, space_group: int):
        return (0,)

    def expand(self, space_group: int, wyckoff_type: int, free_coordinate):
        return (tuple(float(value) % 1.0 for value in free_coordinate),)


@unittest.skipUnless(SPGLIB_AVAILABLE, "spglib runtime path is tested locally/on server")
class RuntimeExpansionTests(unittest.TestCase):
    def test_p1_expansion_and_tangent_projection(self) -> None:
        from crystal_dlm.wqcodiff.runtime import expand_state, project_atom_scores

        state = StratifiedState(
            space_group=1,
            lattice_system="triclinic",
            lattice_chart=LatticeChartCodec.encode_matrix(np.eye(3) * 5.0, "triclinic"),
            orbits=(OrbitState("o0", 0, 6, 1, 3, (0.2, 0.3, 0.4)),),
        )
        expanded = expand_state(state, _P1Catalog())
        self.assertEqual(expanded.atom_count, 1)
        self.assertTrue(1 <= expanded.redetected_space_group <= 230)
        target = np.asarray([1.5, -2.0, 0.25])
        score = np.asarray([expanded.orbit_jacobians[0][0] @ target])
        projected = project_atom_scores(expanded, score)
        np.testing.assert_allclose(projected[0], target, atol=1.0e-6)

    @unittest.skipUnless(PYXTAL_AVAILABLE, "PyXtal 1.1.4 runtime path is locked on server")
    def test_p_lattice_is_not_reduced_by_accidental_orbit_translation(self) -> None:
        from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog
        from crystal_dlm.wqcodiff.runtime import (
            _PRIMITIVE_TRANSFORM_CACHE,
            expand_state,
        )

        catalog = PyXtalChartCatalog()
        space_group = 77
        wyckoff_type = 1
        spec = catalog.get(space_group, wyckoff_type)
        self.assertEqual(spec.multiplicity, spec.primitive_multiplicity)
        matrix = np.diag([5.0, 5.0, 7.0])
        state = StratifiedState(
            space_group=space_group,
            lattice_system="tetragonal",
            lattice_chart=LatticeChartCodec.encode_matrix(matrix, "tetragonal"),
            orbits=(
                OrbitState(
                    "o0",
                    wyckoff_type,
                    14,
                    spec.multiplicity,
                    spec.dimension,
                    (0.37,),
                    primitive_multiplicity=spec.primitive_multiplicity,
                ),
            ),
        )
        _PRIMITIVE_TRANSFORM_CACHE.pop(space_group, None)
        expanded = expand_state(state, catalog)
        self.assertEqual(expanded.atom_count, spec.primitive_multiplicity)
        self.assertAlmostEqual(
            abs(np.linalg.det(expanded.primitive_lattice))
            / abs(np.linalg.det(expanded.conventional_lattice)),
            1.0,
            places=10,
        )

    @unittest.skipUnless(PYXTAL_AVAILABLE, "PyXtal 1.1.4 runtime path is locked on server")
    def test_centered_groups_use_exact_pyxtal_jacobians(self) -> None:
        from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog
        from crystal_dlm.wqcodiff.formal import regularized_projector_error
        from crystal_dlm.wqcodiff.runtime import _PRIMITIVE_TRANSFORM_CACHE, expand_state
        from crystal_dlm.wqcodiff.vocabulary import crystal_system_from_space_group

        catalog = PyXtalChartCatalog()
        for space_group in (166, 225):
            with self.subTest(space_group=space_group):
                _PRIMITIVE_TRANSFORM_CACHE.pop(space_group, None)
                wyckoff_type = max(
                    value
                    for value in catalog.types(space_group)
                    if catalog.get(space_group, value).primitive_multiplicity <= 20
                )
                spec = catalog.get(space_group, wyckoff_type)
                system = crystal_system_from_space_group(space_group)
                if system in {"trigonal", "hexagonal"}:
                    matrix = np.array(
                        [[5.0, 0.0, 0.0], [-2.5, 4.3301270189, 0.0], [0.0, 0.0, 8.0]]
                    )
                else:
                    matrix = np.eye(3) * 5.0
                free = tuple(0.17 + 0.13 * axis for axis in range(spec.dimension))
                state = StratifiedState(
                    space_group=space_group,
                    lattice_system=system,
                    lattice_chart=LatticeChartCodec.encode_matrix(matrix, system),
                    orbits=(
                        OrbitState(
                            "o0",
                            wyckoff_type,
                            14,
                            spec.multiplicity,
                            spec.dimension,
                            free,
                            primitive_multiplicity=spec.primitive_multiplicity,
                        ),
                    ),
                )
                expanded = expand_state(state, catalog)
                self.assertEqual(expanded.atom_count, spec.primitive_multiplicity)
                self.assertLess(
                    regularized_projector_error(expanded.orbit_jacobians[0]),
                    1.0e-6,
                )


if __name__ == "__main__":
    unittest.main()
