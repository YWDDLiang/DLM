from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.composition_pair_prior import ValenceNode
from crystal_dlm.species_physics import (
    aufbau_occupancy,
    feature_names,
    species_physics_matrix,
    species_physics_vector,
)


class SpeciesPhysicsTest(unittest.TestCase):
    def test_isoelectronic_ions_share_occupancy_features(self):
        sodium_plus = species_physics_vector(ValenceNode(11, 1))
        neon = species_physics_vector(ValenceNode(10, 0))
        # The first four scalars include identity/oxidation; the remaining
        # orbital and shell occupancies are isoelectronic.
        self.assertEqual(sodium_plus[4:], neon[4:])

    def test_anion_adds_electrons(self):
        chlorine_minus = aufbau_occupancy(18)
        chlorine_neutral = aufbau_occupancy(17)
        self.assertEqual(sum(chlorine_minus), 18)
        self.assertEqual(sum(chlorine_neutral), 17)

    def test_matrix_is_finite_fixed_width(self):
        nodes = (ValenceNode(8, -2), ValenceNode(26, 2), ValenceNode(26, 3))
        matrix = species_physics_matrix(nodes)
        self.assertEqual(len(matrix), 3)
        self.assertTrue(all(len(row) == len(feature_names()) for row in matrix))
        self.assertEqual(matrix, species_physics_matrix(nodes))

    def test_invalid_electron_count_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "electron count"):
            species_physics_vector(ValenceNode(1, 2))


if __name__ == "__main__":
    unittest.main()
