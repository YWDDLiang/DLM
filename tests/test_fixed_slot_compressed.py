import unittest

from crystal_dlm.fixed_slot import ANSWER_TOKEN_COUNT, FixedSlotError, arrays_to_tokens
from crystal_dlm.fixed_slot_compressed import (
    CompressedFixedSlotConfig,
    arrays_to_compressed_tokens,
    build_compressed_special_tokens,
    compressed_token_sources,
    compressed_tokens_to_arrays,
    compressed_tokens_to_full_tokens,
    full_tokens_to_compressed_tokens,
    parse_compressed_fixed_slot_answer,
    schema_allowed_token_strings,
)


class FixedSlotCompressedTests(unittest.TestCase):
    def test_shared_xyz_round_trip(self):
        config = CompressedFixedSlotConfig(share_coordinates=True)
        tokens, diagnostics = arrays_to_compressed_tokens(
            lengths=[4.13, 5.2, 18.42],
            angles=[90.0, 91.0, 120.0],
            species=["Ga", "Te"],
            frac_coords=[[1.0, 0.666, -0.1], [0.25, 0.75, 0.5]],
            config=config,
        )
        self.assertEqual(len(tokens), ANSWER_TOKEN_COUNT)
        self.assertIn("<C_000>", tokens)
        self.assertIn("<C_PAD>", tokens)
        self.assertNotIn("<X_PAD>", tokens)
        self.assertGreaterEqual(diagnostics.coord_wraps, 1)
        arrays = compressed_tokens_to_arrays(tokens, config=config)
        self.assertEqual(arrays["num_atoms"], 2)
        self.assertEqual(arrays["species"], ["Ga", "Te"])
        self.assertEqual(arrays["atom_types"], [31, 52])

    def test_shared_xyz_length_angle_round_trip(self):
        config = CompressedFixedSlotConfig(
            share_coordinates=True,
            share_lengths=True,
            share_angles=True,
        )
        tokens, _ = arrays_to_compressed_tokens(
            lengths=[3.1, 4.2, 5.3],
            angles=[80, 90, 100],
            species=["Li", "O"],
            frac_coords=[[0.0, 0.5, 1.0], [0.12, 0.34, 0.56]],
            config=config,
        )
        self.assertIn("<L_031>", tokens)
        self.assertIn("<A_080>", tokens)
        parsed = parse_compressed_fixed_slot_answer("".join(tokens), config=config)
        self.assertEqual(parsed["species"], ["Li", "O"])
        self.assertAlmostEqual(parsed["lengths"][1], 4.2)
        self.assertEqual(parsed["angles"][2], 100.0)

    def test_full_tokens_convert_to_compressed_and_back(self):
        config = CompressedFixedSlotConfig(
            share_coordinates=True,
            share_lengths=True,
            share_angles=False,
        )
        full_tokens, _ = arrays_to_tokens(
            lengths=[3, 4, 5],
            angles=[70, 80, 90],
            species=["Na"],
            frac_coords=[[0.1, 0.2, 0.3]],
        )
        compressed = full_tokens_to_compressed_tokens(full_tokens, config=config)
        self.assertIn("<L_030>", compressed)
        self.assertIn("<C_010>", compressed)
        expanded = compressed_tokens_to_full_tokens(compressed, config=config)
        self.assertEqual(expanded[0], full_tokens[0])
        self.assertEqual(expanded[1:7], full_tokens[1:7])
        self.assertEqual(expanded[7], "<S00>")
        self.assertEqual(expanded[8], "<E_Na>")

    def test_empty_slots_require_shared_pad(self):
        config = CompressedFixedSlotConfig(share_coordinates=True)
        tokens, _ = arrays_to_compressed_tokens(
            lengths=[3, 3, 3],
            angles=[90, 90, 90],
            species=["Na"],
            frac_coords=[[0, 0, 0]],
            config=config,
        )
        tokens[14] = "<C_000>"
        with self.assertRaises(FixedSlotError):
            compressed_tokens_to_arrays(tokens, config=config)

    def test_vocab_counts_match_ablation_expectations(self):
        full_like = build_compressed_special_tokens(
            CompressedFixedSlotConfig(
                share_coordinates=False,
                share_lengths=False,
                share_angles=False,
            )
        )
        abl1 = build_compressed_special_tokens(
            CompressedFixedSlotConfig(
                share_coordinates=True,
                share_lengths=False,
                share_angles=False,
            )
        )
        abl2 = build_compressed_special_tokens(
            CompressedFixedSlotConfig(
                share_coordinates=True,
                share_lengths=True,
                share_angles=False,
            )
        )
        abl3 = build_compressed_special_tokens(
            CompressedFixedSlotConfig(
                share_coordinates=True,
                share_lengths=True,
                share_angles=True,
            )
        )
        self.assertEqual(len(full_like) - len(abl1), 204)
        self.assertEqual(len(abl1) - len(abl2), 1002)
        self.assertEqual(len(abl2) - len(abl3), 358)
        self.assertEqual(len(abl3), len(set(abl3)))

    def test_token_sources_for_embedding_merge(self):
        config = CompressedFixedSlotConfig(
            share_coordinates=True,
            share_lengths=True,
            share_angles=True,
        )
        sources = compressed_token_sources(config)
        self.assertEqual(sources["<C_050>"], ["<X_050>", "<Y_050>", "<Z_050>"])
        self.assertEqual(sources["<L_120>"], ["<LA_120>", "<LB_120>", "<LC_120>"])
        self.assertEqual(sources["<A_090>"], ["<AA_090>", "<AB_090>", "<AG_090>"])

    def test_schema_positions_use_shared_axis_tokens(self):
        config = CompressedFixedSlotConfig(
            share_coordinates=True,
            share_lengths=True,
            share_angles=True,
        )
        allowed = schema_allowed_token_strings(config)
        self.assertIn("<L_000>", allowed[1])
        self.assertIn("<L_000>", allowed[2])
        self.assertIn("<A_090>", allowed[4])
        self.assertIn("<A_090>", allowed[6])
        self.assertIn("<C_050>", allowed[9])
        self.assertIn("<C_050>", allowed[10])
        self.assertIn("<C_050>", allowed[11])


if __name__ == "__main__":
    unittest.main()
