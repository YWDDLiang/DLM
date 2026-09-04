import unittest

import numpy as np

from crystal_dlm.transaction_physics import (
    _finish_direction,
    lattice_matrix_from_dynamic_arrays,
    propose_force_site_transactions,
    propose_stress_lattice_transactions,
)


def crystal(*, lengths=(10.0, 10.0, 10.0), angles=(90.0, 90.0, 90.0), coords=None):
    return {
        "lengths": list(lengths),
        "angles": list(angles),
        "species": ["Li", "O"],
        "frac_coords": coords or [[0.10, 0.20, 0.30], [0.60, 0.70, 0.80]],
    }


class TransactionPhysicsTest(unittest.TestCase):
    def test_row_vector_force_conversion_and_sign_in_orthogonal_cell(self):
        source = crystal()
        positive, negative = propose_force_site_transactions(
            source,
            0,
            [2.0, 0.0, 0.0],
            step_sizes_A=(0.10,),
        )
        self.assertEqual(positive.status, "accepted")
        self.assertEqual(negative.status, "accepted")
        self.assertEqual(positive.transaction_tokens, ("<X_011>", "<Y_020>", "<Z_030>"))
        self.assertEqual(negative.transaction_tokens, ("<X_009>", "<Y_020>", "<Z_030>"))

    def test_row_vector_conversion_in_triclinic_cell(self):
        source = crystal(
            lengths=(5.0, 6.0, 7.0),
            angles=(70.0, 80.0, 75.0),
        )
        lattice = lattice_matrix_from_dynamic_arrays(source)
        force = np.array([0.4, -0.7, 0.2])
        step = 0.20
        expected = (
            np.asarray(source["frac_coords"][0])
            + step * force / np.linalg.norm(force) @ np.linalg.inv(lattice)
        ) % 1.0
        positive, _negative = propose_force_site_transactions(
            source, 0, force, step_sizes_A=(step,)
        )
        self.assertEqual(positive.status, "accepted")
        np.testing.assert_allclose(
            positive.frac_coords[0],
            np.floor(expected * 100.0 + 0.5) / 100.0,
            atol=1.0e-12,
        )

    def test_fixed_step_sequence_escapes_quantized_noop_without_energy(self):
        source = crystal()
        positive, _negative = propose_force_site_transactions(
            source,
            0,
            [1.0, 0.0, 0.0],
            step_sizes_A=(0.001, 0.10),
        )
        self.assertEqual(positive.status, "accepted")
        self.assertEqual(positive.step, 0.10)

    def test_all_quantized_noops_are_explicit(self):
        source = crystal()
        positive, negative = propose_force_site_transactions(
            source,
            0,
            [1.0, 0.0, 0.0],
            step_sizes_A=(0.001, 0.002),
        )
        self.assertEqual(positive.status, "noop")
        self.assertEqual(negative.status, "noop")

    def test_force_candidate_with_triclinic_pbc_collision_is_rejected(self):
        source = crystal(
            lengths=(5.0, 6.0, 7.0),
            angles=(70.0, 80.0, 75.0),
            coords=[[0.10, 0.20, 0.30], [0.22, 0.20, 0.30]],
        )
        lattice = lattice_matrix_from_dynamic_arrays(source)
        # Move site 1 toward site 0 by 0.2 A in the exact Cartesian direction.
        cart_delta = (np.asarray(source["frac_coords"][0]) - np.asarray(source["frac_coords"][1])) @ lattice
        positive, negative = propose_force_site_transactions(
            source,
            1,
            cart_delta,
            step_sizes_A=(0.20,),
            minimum_distance_A=0.50,
        )
        self.assertEqual(positive.status, "invalid")
        self.assertEqual(positive.reason, "pbc_minimum_distance")
        self.assertEqual(negative.status, "accepted")

    def test_stress_sign_contracts_and_expands_orthogonal_cell(self):
        source = crystal()
        downhill, reverse = propose_stress_lattice_transactions(
            source,
            np.eye(3),
            strain_steps=(0.02,),
        )
        self.assertEqual(downhill.status, "accepted")
        self.assertEqual(reverse.status, "accepted")
        self.assertLess(downhill.lengths[0], 10.0)
        self.assertGreater(reverse.lengths[0], 10.0)
        self.assertEqual(downhill.species, ("Li", "O"))
        self.assertEqual(downhill.frac_coords, tuple(tuple(x) for x in source["frac_coords"]))

    def test_stress_proposals_support_triclinic_cell_and_symmetric_part(self):
        source = crystal(
            lengths=(5.0, 6.0, 7.0),
            angles=(70.0, 80.0, 75.0),
        )
        stress = np.array([[1.0, 0.3, 0.0], [-0.1, -0.5, 0.2], [0.0, 0.2, 0.1]])
        downhill, reverse = propose_stress_lattice_transactions(
            source, stress, strain_steps=(0.02,)
        )
        self.assertEqual(downhill.status, "accepted")
        self.assertEqual(reverse.status, "accepted")
        self.assertNotEqual(downhill.transaction_tokens, reverse.transaction_tokens)
        self.assertGreater(np.prod(downhill.lengths), 0.0)

    def test_duplicate_transaction_status_is_explicit(self):
        source = crystal()
        accepted, _ = propose_force_site_transactions(
            source, 0, [1.0, 0.0, 0.0], step_sizes_A=(0.10,)
        )
        duplicate = _finish_direction(
            kind="site_xyz",
            direction="duplicate_test",
            attempts=((0.10, accepted.full_tokens, {
                "lengths": accepted.lengths,
                "angles": accepted.angles,
                "species": accepted.species,
                "frac_coords": accepted.frac_coords,
            }),),
            no_op_transaction=("<X_010>", "<Y_020>", "<Z_030>"),
            seen_transactions={accepted.transaction_tokens},
            transaction_slice=slice(8, 11),
            expected_species=("Li", "O"),
            minimum_distance_A=0.50,
            image_radius=2,
        )
        self.assertEqual(duplicate.status, "duplicate")


if __name__ == "__main__":
    unittest.main()
