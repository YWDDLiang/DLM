import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from crystal_dlm.llada_generation import (
        _apply_lightweight_decoding_masks,
        _prepare_atom_count_grammar,
        lattice_angle_rad,
    )
else:
    _apply_lightweight_decoding_masks = None
    _prepare_atom_count_grammar = None
    lattice_angle_rad = None


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class LLaDAGenerationMaskTest(unittest.TestCase):
    def test_duplicate_coordinate_mask_bans_matching_z_only(self):
        prompt_length = 3
        gen_length = 107
        vocab_size = 400
        mask_id = 399
        count_id = 10
        x25, y25, z50, z75 = 25, 125, 250, 275
        x33, y33 = 33, 133
        logits = torch.zeros((1, prompt_length + gen_length, vocab_size))
        x = torch.full((1, prompt_length + gen_length), mask_id, dtype=torch.long)
        x[0, prompt_length + 0] = count_id
        x[0, prompt_length + 9] = x25
        x[0, prompt_length + 10] = y25
        x[0, prompt_length + 11] = z50
        x[0, prompt_length + 14] = x25
        x[0, prompt_length + 15] = y25
        x[0, prompt_length + 16] = mask_id
        x[0, prompt_length + 19] = x33
        x[0, prompt_length + 20] = y33
        x[0, prompt_length + 21] = mask_id
        constraints = {
            "duplicate_coordinate_mask": True,
            "max_atoms": 3,
            "count_token_to_n": {count_id: 3},
            "coord_token_to_bin": {
                "X": {x25: 25, x33: 33},
                "Y": {y25: 25, y33: 33},
                "Z": {z50: 50, z75: 75},
            },
            "z_bin_to_token_id": {50: z50, 75: z75},
        }

        _apply_lightweight_decoding_masks(logits, x, prompt_length, gen_length, constraints)
        min_value = torch.finfo(logits.dtype).min

        self.assertEqual(logits[0, prompt_length + 16, z50].item(), min_value)
        self.assertEqual(logits[0, prompt_length + 16, z75].item(), 0.0)
        self.assertEqual(logits[0, prompt_length + 21, z50].item(), 0.0)

    def test_duplicate_coordinate_mask_ignores_empty_slots_beyond_n(self):
        prompt_length = 0
        gen_length = 107
        vocab_size = 300
        mask_id = 299
        count_id = 10
        x25, y25, z50 = 25, 125, 250
        logits = torch.zeros((1, gen_length, vocab_size))
        x = torch.full((1, gen_length), mask_id, dtype=torch.long)
        x[0, 0] = count_id
        x[0, 9] = x25
        x[0, 10] = y25
        x[0, 11] = z50
        x[0, 14] = x25
        x[0, 15] = y25
        constraints = {
            "duplicate_coordinate_mask": True,
            "max_atoms": 2,
            "count_token_to_n": {count_id: 1},
            "coord_token_to_bin": {
                "X": {x25: 25},
                "Y": {y25: 25},
                "Z": {z50: 50},
            },
            "z_bin_to_token_id": {50: z50},
        }

        _apply_lightweight_decoding_masks(logits, x, prompt_length, gen_length, constraints)

        self.assertEqual(logits[0, 16, z50].item(), 0.0)

    def test_duplicate_coordinate_mask_treats_boundary_bins_as_pbc_equivalent(self):
        prompt_length = 0
        gen_length = 107
        vocab_size = 500
        mask_id = 499
        count_id = 10
        x0, x100 = 20, 21
        y0, y100 = 120, 121
        z0, z100, z50 = 220, 221, 250
        logits = torch.zeros((1, gen_length, vocab_size))
        x = torch.full((1, gen_length), mask_id, dtype=torch.long)
        x[0, 0] = count_id
        x[0, 9] = x0
        x[0, 10] = y100
        x[0, 11] = z0
        x[0, 14] = x100
        x[0, 15] = y0
        x[0, 16] = mask_id
        constraints = {
            "duplicate_coordinate_mask": True,
            "max_atoms": 2,
            "count_token_to_n": {count_id: 2},
            "coord_period": 100,
            "coord_token_to_bin": {
                "X": {x0: 0, x100: 100},
                "Y": {y0: 0, y100: 100},
                "Z": {z0: 0, z100: 100, z50: 50},
            },
            "z_bin_to_token_id": {0: z0, 100: z100, 50: z50},
        }

        _apply_lightweight_decoding_masks(logits, x, prompt_length, gen_length, constraints)
        min_value = torch.finfo(logits.dtype).min

        self.assertEqual(logits[0, 16, z0].item(), min_value)
        self.assertEqual(logits[0, 16, z100].item(), min_value)
        self.assertEqual(logits[0, 16, z50].item(), 0.0)

    def test_duplicate_coordinate_mask_respects_body_offset(self):
        prompt_length = 0
        body_offset = 10
        gen_length = 117
        vocab_size = 500
        mask_id = 499
        count_id = 10
        x0, x100 = 20, 21
        y0, y100 = 120, 121
        z0, z100, z50 = 220, 221, 250
        logits = torch.zeros((1, gen_length, vocab_size))
        x = torch.full((1, gen_length), mask_id, dtype=torch.long)
        x[0, body_offset + 0] = count_id
        x[0, body_offset + 9] = x0
        x[0, body_offset + 10] = y100
        x[0, body_offset + 11] = z0
        x[0, body_offset + 14] = x100
        x[0, body_offset + 15] = y0
        x[0, body_offset + 16] = mask_id
        constraints = {
            "body_offset": body_offset,
            "duplicate_coordinate_mask": True,
            "max_atoms": 2,
            "count_token_to_n": {count_id: 2},
            "coord_period": 100,
            "coord_token_to_bin": {
                "X": {x0: 0, x100: 100},
                "Y": {y0: 0, y100: 100},
                "Z": {z0: 0, z100: 100, z50: 50},
            },
            "z_bin_to_token_id": {0: z0, 100: z100, 50: z50},
        }

        _apply_lightweight_decoding_masks(logits, x, prompt_length, gen_length, constraints)
        min_value = torch.finfo(logits.dtype).min

        self.assertEqual(logits[0, body_offset + 16, z0].item(), min_value)
        self.assertEqual(logits[0, body_offset + 16, z100].item(), min_value)
        self.assertEqual(logits[0, body_offset + 16, z50].item(), 0.0)

    def test_lattice_volume_mask_bans_invalid_gamma_and_zero_lengths(self):
        prompt_length = 0
        gen_length = 107
        vocab_size = 400
        alpha60, beta60 = 301, 302
        gamma90, gamma179 = 190, 279
        la0, lb0, lc0 = 1, 2, 3
        logits = torch.zeros((1, gen_length, vocab_size))
        x = torch.zeros((1, gen_length), dtype=torch.long)
        x[0, 4] = alpha60
        x[0, 5] = beta60
        constraints = {
            "lattice_volume_mask": True,
            "min_lattice_rad": 1e-4,
            "zero_length_token_ids_by_position": {1: la0, 2: lb0, 3: lc0},
            "angle_token_to_bin": {"AA": {alpha60: 60}, "AB": {beta60: 60}},
            "gamma_bin_to_token_id": {90: gamma90, 179: gamma179},
        }

        _apply_lightweight_decoding_masks(logits, x, prompt_length, gen_length, constraints)
        min_value = torch.finfo(logits.dtype).min

        self.assertGreater(lattice_angle_rad(60, 60, 90), 1e-4)
        self.assertLessEqual(lattice_angle_rad(60, 60, 179), 1e-4)
        self.assertEqual(logits[0, 1, la0].item(), min_value)
        self.assertEqual(logits[0, 2, lb0].item(), min_value)
        self.assertEqual(logits[0, 3, lc0].item(), min_value)
        self.assertEqual(logits[0, 6, gamma90].item(), 0.0)
        self.assertEqual(logits[0, 6, gamma179].item(), min_value)

    def test_lattice_volume_mask_respects_body_offset(self):
        prompt_length = 0
        body_offset = 10
        gen_length = 117
        vocab_size = 400
        alpha60, beta60 = 301, 302
        gamma90, gamma179 = 190, 279
        la0, lb0, lc0 = 1, 2, 3
        logits = torch.zeros((1, gen_length, vocab_size))
        x = torch.zeros((1, gen_length), dtype=torch.long)
        x[0, body_offset + 4] = alpha60
        x[0, body_offset + 5] = beta60
        constraints = {
            "body_offset": body_offset,
            "lattice_volume_mask": True,
            "min_lattice_rad": 1e-4,
            "zero_length_token_ids_by_position": {
                body_offset + 1: la0,
                body_offset + 2: lb0,
                body_offset + 3: lc0,
            },
            "angle_token_to_bin": {"AA": {alpha60: 60}, "AB": {beta60: 60}},
            "gamma_bin_to_token_id": {90: gamma90, 179: gamma179},
        }

        _apply_lightweight_decoding_masks(logits, x, prompt_length, gen_length, constraints)
        min_value = torch.finfo(logits.dtype).min

        self.assertEqual(logits[0, body_offset + 1, la0].item(), min_value)
        self.assertEqual(logits[0, body_offset + 2, lb0].item(), min_value)
        self.assertEqual(logits[0, body_offset + 3, lc0].item(), min_value)
        self.assertEqual(logits[0, body_offset + 6, gamma90].item(), 0.0)
        self.assertEqual(logits[0, body_offset + 6, gamma179].item(), min_value)

    def test_lattice_volume_mask_fails_closed_when_no_gamma_is_legal(self):
        prompt_length = 0
        gen_length = 107
        vocab_size = 400
        alpha1, beta179 = 301, 302
        gamma178, gamma179 = 278, 279
        logits = torch.zeros((1, gen_length, vocab_size))
        x = torch.zeros((1, gen_length), dtype=torch.long)
        x[0, 4] = alpha1
        x[0, 5] = beta179
        constraints = {
            "lattice_volume_mask": True,
            "min_lattice_rad": 1e-4,
            "zero_length_token_ids_by_position": {},
            "angle_token_to_bin": {"AA": {alpha1: 1}, "AB": {beta179: 179}},
            "gamma_bin_to_token_id": {178: gamma178, 179: gamma179},
        }

        _apply_lightweight_decoding_masks(logits, x, prompt_length, gen_length, constraints)
        min_value = torch.finfo(logits.dtype).min

        self.assertEqual(logits[0, 6, gamma178].item(), min_value)
        self.assertEqual(logits[0, 6, gamma179].item(), min_value)

    def test_dynamic_duplicate_coordinate_mask_uses_four_token_sites(self):
        prompt_length = 0
        gen_length = 87
        vocab_size = 500
        mask_id = 499
        count_id = 10
        x0, x100 = 20, 21
        y0, y100 = 120, 121
        z0, z100, z50 = 220, 221, 250
        logits = torch.zeros((1, gen_length, vocab_size))
        x = torch.full((1, gen_length), mask_id, dtype=torch.long)
        x[0, 0] = count_id
        x[0, 8] = x0
        x[0, 9] = y100
        x[0, 10] = z0
        x[0, 12] = x100
        x[0, 13] = y0
        x[0, 14] = mask_id
        constraints = {
            "representation": "dynamic_v1",
            "duplicate_coordinate_mask": True,
            "max_atoms": 2,
            "count_token_to_n": {count_id: 2},
            "coord_period": 100,
            "coord_token_to_bin": {
                "X": {x0: 0, x100: 100},
                "Y": {y0: 0, y100: 100},
                "Z": {z0: 0, z100: 100, z50: 50},
            },
            "z_bin_to_token_id": {0: z0, 100: z100, 50: z50},
        }

        _apply_lightweight_decoding_masks(logits, x, prompt_length, gen_length, constraints)
        min_value = torch.finfo(logits.dtype).min

        self.assertEqual(logits[0, 14, z0].item(), min_value)
        self.assertEqual(logits[0, 14, z100].item(), min_value)
        self.assertEqual(logits[0, 14, z50].item(), 0.0)

    def test_dynamic_duplicate_coordinate_mask_checks_later_generated_slots(self):
        prompt_length = 0
        gen_length = 87
        vocab_size = 500
        mask_id = 499
        count_id = 10
        x0, y0 = 20, 120
        z0, z50 = 220, 250
        logits = torch.zeros((1, gen_length, vocab_size))
        x = torch.full((1, gen_length), mask_id, dtype=torch.long)
        x[0, 0] = count_id
        x[0, 8] = x0
        x[0, 9] = y0
        x[0, 10] = mask_id
        x[0, 12] = x0
        x[0, 13] = y0
        x[0, 14] = z0
        constraints = {
            "representation": "dynamic_v1",
            "duplicate_coordinate_mask": True,
            "max_atoms": 2,
            "count_token_to_n": {count_id: 2},
            "coord_period": 100,
            "coord_token_to_bin": {
                "X": {x0: 0},
                "Y": {y0: 0},
                "Z": {z0: 0, z50: 50},
            },
            "z_bin_to_token_id": {0: z0, 50: z50},
        }

        _apply_lightweight_decoding_masks(logits, x, prompt_length, gen_length, constraints)
        min_value = torch.finfo(logits.dtype).min

        self.assertEqual(logits[0, 10, z0].item(), min_value)
        self.assertEqual(logits[0, 10, z50].item(), 0.0)

    def test_dynamic_atom_count_grammar_does_not_require_empty_token(self):
        grammar = {
            "representation": "dynamic_v1",
            "max_atoms": 2,
            "count_token_to_n": {10: 1},
            "element_token_ids": [20, 21],
            "coord_token_ids": {
                "X": [100],
                "Y": [110],
                "Z": [120],
            },
            "eos_token_id": 2,
        }

        prepared = _prepare_atom_count_grammar(grammar, vocab_size=200, device=torch.device("cpu"))

        self.assertIn("eos_mask", prepared)
        self.assertNotIn("empty_mask", prepared)
        self.assertNotIn("pad_coord_masks", prepared)


if __name__ == "__main__":
    unittest.main()
