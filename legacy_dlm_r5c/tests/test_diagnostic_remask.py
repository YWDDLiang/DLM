import unittest

from crystal_dlm.diagnostic_remask import (
    active_coord_positions,
    anti_high_symmetry_failures,
    build_prefill_token_ids_by_position,
    composition_preserved,
    geometry_degeneracy_record,
    remask_positions_for_mode,
)
from crystal_dlm.fixed_slot import MASK_TOKEN_ID, arrays_to_tokens, build_special_tokens


class TinyTokenizer:
    def __init__(self):
        self._vocab = {token: idx for idx, token in enumerate(build_special_tokens())}

    def get_vocab(self):
        return self._vocab


class DiagnosticRemaskTests(unittest.TestCase):
    def sample_arrays(self):
        tokens, _ = arrays_to_tokens(
            lengths=[3.0, 3.0, 3.0],
            angles=[90.0, 90.0, 90.0],
            species=["Na", "Cl"],
            frac_coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        return {
            "num_atoms": 2,
            "lengths": [3.0, 3.0, 3.0],
            "angles": [90.0, 90.0, 90.0],
            "species": ["Na", "Cl"],
            "frac_coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            "tokens": tokens,
        }

    def test_remask_positions_for_geometry(self):
        arrays = self.sample_arrays()
        self.assertEqual(remask_positions_for_mode(arrays, "lattice_only"), [1, 2, 3, 4, 5, 6])
        self.assertEqual(active_coord_positions(2), [9, 10, 11, 14, 15, 16])
        self.assertEqual(
            remask_positions_for_mode(arrays, "geometry"),
            [1, 2, 3, 4, 5, 6, 9, 10, 11, 14, 15, 16],
        )

    def test_prefill_uses_mask_for_remasked_active_coords_and_freezes_inactive_pads(self):
        tokenizer = TinyTokenizer()
        arrays = self.sample_arrays()
        prefill = build_prefill_token_ids_by_position(tokenizer, [arrays], mode="geometry")
        vocab = tokenizer.get_vocab()
        self.assertEqual(prefill[0], [vocab["<N_002>"]])
        self.assertEqual(prefill[7], [vocab["<S00>"]])
        self.assertEqual(prefill[8], [vocab["<E_Na>"]])
        self.assertNotIn(9, prefill)
        self.assertEqual(prefill[17], [vocab["<S02>"]])
        self.assertEqual(prefill[18], [vocab["<EMPTY>"]])
        self.assertEqual(prefill[19], [vocab["<X_PAD>"]])

    def test_prefill_mixed_batch_masks_active_and_freezes_inactive_same_position(self):
        tokenizer = TinyTokenizer()
        first = self.sample_arrays()
        second = self.sample_arrays()
        second["num_atoms"] = 1
        second["species"] = ["Na"]
        second["frac_coords"] = [[0.0, 0.0, 0.0]]
        second["tokens"], _ = arrays_to_tokens(
            lengths=second["lengths"],
            angles=second["angles"],
            species=second["species"],
            frac_coords=second["frac_coords"],
        )
        prefill = build_prefill_token_ids_by_position(tokenizer, [first, second], mode="geometry")
        vocab = tokenizer.get_vocab()
        self.assertEqual(prefill[14], [MASK_TOKEN_ID, vocab["<X_PAD>"]])
        self.assertEqual(prefill[15], [MASK_TOKEN_ID, vocab["<Y_PAD>"]])
        self.assertEqual(prefill[16], [MASK_TOKEN_ID, vocab["<Z_PAD>"]])

    def test_geometry_degeneracy_and_composition_preservation(self):
        arrays = self.sample_arrays()
        record = geometry_degeneracy_record(arrays)
        self.assertTrue(record["all_lengths_equal"])
        self.assertTrue(record["all_angles_90"])
        self.assertEqual(record["high_symmetry_coord_fraction"], 1.0)
        self.assertEqual(
            anti_high_symmetry_failures(
                arrays,
                max_high_symmetry_coord_fraction=0.75,
                reject_all_lengths_equal=True,
                reject_all_angles_90=True,
            ),
            ["all_lengths_equal", "all_angles_90", "high_symmetry_coord_fraction"],
        )
        self.assertTrue(composition_preserved(arrays, dict(arrays)))


if __name__ == "__main__":
    unittest.main()

