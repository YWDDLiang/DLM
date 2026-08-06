from __future__ import annotations

import math
import importlib.util
import unittest

import numpy as np

from crystal_dlm.wqcodiff.charts import (
    AffineOrbitChart,
    CRYSTAL_SYSTEM_DIMENSIONS,
    LatticeChartCodec,
    derive_affine_orbit_chart,
    periodic_cartesian_distance,
    wyckoff_letter_to_type,
    wyckoff_type_to_letter,
)


PYXTAL_AVAILABLE = importlib.util.find_spec("pyxtal") is not None


class FakeOperation:
    def __init__(self, rotation, translation=(0.0, 0.0, 0.0)):
        self.rotation_matrix = np.asarray(rotation, dtype=float)
        self.translation_vector = np.asarray(translation, dtype=float)

    def operate(self, point):
        return self.rotation_matrix @ np.asarray(point) + self.translation_vector


class LatticeChartTests(unittest.TestCase):
    def test_every_registered_chart_is_spd_and_has_registered_dimension(self) -> None:
        matrices = {
            "triclinic": np.array([[4.0, 0.0, 0.0], [0.4, 5.0, 0.0], [0.3, -0.2, 6.0]]),
            "monoclinic": np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-1.5, 0.0, 5.8]]),
            "orthorhombic": np.diag([4.0, 5.0, 6.0]),
            "tetragonal": np.diag([4.0, 4.0, 6.0]),
            "hexagonal": np.array([[4.0, 0.0, 0.0], [-2.0, 2.0 * math.sqrt(3.0), 0.0], [0.0, 0.0, 6.0]]),
            "trigonal": np.array([[4.0, 0.0, 0.0], [-2.0, 2.0 * math.sqrt(3.0), 0.0], [0.0, 0.0, 6.0]]),
            "cubic": np.diag([4.0, 4.0, 4.0]),
        }
        for system, matrix in matrices.items():
            with self.subTest(system=system):
                chart = LatticeChartCodec.encode_matrix(matrix, system)
                decoded = LatticeChartCodec.decode_matrix(chart, system)
                self.assertEqual(len(chart), CRYSTAL_SYSTEM_DIMENSIONS[system])
                self.assertTrue(np.all(np.linalg.eigvalsh(decoded @ decoded.T) > 0.0))
                self.assertAlmostEqual(abs(np.linalg.det(decoded)), abs(np.linalg.det(matrix)), places=8)


class OrbitChartTests(unittest.TestCase):
    def test_sg47_uppercase_a_does_not_alias_lowercase_a(self) -> None:
        self.assertEqual(wyckoff_letter_to_type("a"), 0)
        self.assertEqual(wyckoff_letter_to_type("z"), 25)
        self.assertEqual(wyckoff_letter_to_type("A"), 26)
        self.assertEqual(wyckoff_type_to_letter(26), "A")

    def test_nonorthonormal_affine_basis_uses_pseudoinverse(self) -> None:
        chart = AffineOrbitChart(
            origin=(0.0, 0.25, 0.0),
            basis=((1.0,), (1.0,), (0.0,)),
            dimension=1,
            fit_residual=0.0,
        )
        coordinate = chart.decode((0.17,))
        self.assertAlmostEqual(chart.encode(coordinate)[0] % 1.0, 0.17, places=12)

    def test_general_position_has_three_degrees_of_freedom(self) -> None:
        identity = FakeOperation(np.eye(3))
        lattice = np.diag([4.0, 4.0, 4.0])
        point = (0.13, 0.27, 0.41)
        chart = derive_affine_orbit_chart(point, [identity], lattice, symprec=1.0e-3)
        self.assertEqual(chart.dimension, 3)
        self.assertLess(periodic_cartesian_distance(chart.decode(chart.encode(point)), point, lattice), 1.0e-10)

    @unittest.skipUnless(PYXTAL_AVAILABLE, "PyXtal 1.1.4 is verified in the locked server env")
    def test_pyxtal_official_free_coordinate_roundtrip(self) -> None:
        from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog

        catalog = PyXtalChartCatalog()
        lattice = np.eye(3) * 5.0
        for space_group in (1, 62, 225):
            for wyckoff_type in catalog.types(space_group):
                with self.subTest(space_group=space_group, wyckoff_type=wyckoff_type):
                    spec = catalog.get(space_group, wyckoff_type)
                    free = tuple(0.17 + 0.11 * axis for axis in range(spec.dimension))
                    point = catalog.decode_generator(space_group, wyckoff_type, free)
                    recovered = catalog.encode_free(space_group, wyckoff_type, point)
                    self.assertEqual(len(catalog.expand(space_group, wyckoff_type, free)), spec.multiplicity)
                    self.assertLess(
                        max(
                            (
                                abs(((left - right + 0.5) % 1.0) - 0.5)
                                for left, right in zip(free, recovered)
                            ),
                            default=0.0,
                        ),
                        1.0e-6,
                    )
                    chart = catalog.affine_chart(space_group, wyckoff_type, lattice)
                    self.assertLess(
                        periodic_cartesian_distance(chart.decode(free), point, lattice),
                        1.0e-6,
                    )

    @unittest.skipUnless(PYXTAL_AVAILABLE, "PyXtal 1.1.4 is verified in the locked server env")
    def test_sg47_catalog_contains_all_27_positions(self) -> None:
        from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog

        catalog = PyXtalChartCatalog()
        self.assertEqual(catalog.types(47), tuple(range(27)))
        self.assertEqual(catalog.get(47, 26).letter, "A")

    @unittest.skipUnless(PYXTAL_AVAILABLE, "PyXtal 1.1.4 is verified in the locked server env")
    def test_pyxtal_immutable_metadata_is_cached_without_changing_values(self) -> None:
        from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog

        catalog = PyXtalChartCatalog()
        positions = catalog._positions(62)
        types = catalog.types(62)
        wyckoff_type = types[0]
        spec = catalog.get(62, wyckoff_type)
        free = tuple(0.17 + 0.03 * axis for axis in range(spec.dimension))
        first = catalog.expand_with_jacobians(62, wyckoff_type, free)
        second = catalog.expand_with_jacobians(62, wyckoff_type, free)

        self.assertIs(catalog._positions(62), positions)
        self.assertIs(catalog.get(62, wyckoff_type), spec)
        self.assertEqual(first, second)
        self.assertEqual(len(catalog._unit_affine_charts), 1)

    def test_inversion_center_has_zero_degrees_of_freedom(self) -> None:
        identity = FakeOperation(np.eye(3))
        inversion = FakeOperation(-np.eye(3))
        lattice = np.diag([5.0, 5.0, 5.0])
        chart = derive_affine_orbit_chart((0.0, 0.0, 0.0), [identity, inversion], lattice, symprec=1.0e-3)
        self.assertEqual(chart.dimension, 0)
        self.assertEqual(chart.encode((0.0, 0.0, 0.0)), ())
        self.assertLess(periodic_cartesian_distance(chart.decode(()), (0.0, 0.0, 0.0), lattice), 1.0e-10)

    def test_mirror_plane_has_two_degrees_of_freedom(self) -> None:
        identity = FakeOperation(np.eye(3))
        mirror = FakeOperation(np.diag([1.0, 1.0, -1.0]))
        lattice = np.diag([5.0, 5.0, 5.0])
        point = (0.17, 0.29, 0.0)
        chart = derive_affine_orbit_chart(point, [identity, mirror], lattice, symprec=1.0e-3)
        self.assertEqual(chart.dimension, 2)
        self.assertLess(periodic_cartesian_distance(chart.decode(chart.encode(point)), point, lattice), 1.0e-10)


if __name__ == "__main__":
    unittest.main()
