import unittest

from crystal_dlm.fixed_slot import (
    ANSWER_TOKEN_COUNT,
    FixedSlotError,
    FixedSlotConfig,
    answer_token_count,
    arrays_to_answer,
    arrays_to_tokens,
    build_special_tokens,
    parse_fixed_slot_answer,
    tokens_to_arrays,
)


class FixedSlotTests(unittest.TestCase):
    def test_round_trip_two_atoms(self):
        tokens, diagnostics = arrays_to_tokens(
            lengths=[4.13, 4.13, 18.42],
            angles=[90.0, 90.0, 120.0],
            species=["Ga", "Te"],
            frac_coords=[[1.0, 0.666, -0.1], [0.25, 0.75, 0.5]],
        )
        self.assertEqual(len(tokens), ANSWER_TOKEN_COUNT)
        self.assertGreaterEqual(diagnostics.coord_wraps, 1)
        arrays = tokens_to_arrays(tokens)
        self.assertEqual(arrays["num_atoms"], 2)
        self.assertEqual(arrays["species"], ["Ga", "Te"])
        self.assertEqual(arrays["atom_types"], [31, 52])
        self.assertEqual(len(arrays["frac_coords"]), 2)

    def test_empty_slots_are_required_after_num_atoms(self):
        tokens, _ = arrays_to_tokens(
            lengths=[3, 3, 3],
            angles=[90, 90, 90],
            species=["Na"],
            frac_coords=[[0, 0, 0]],
        )
        tokens[9] = "<E_Na>"
        with self.assertRaises(FixedSlotError):
            tokens_to_arrays(tokens)

    def test_extracts_tokens_from_decoded_text(self):
        tokens, _ = arrays_to_tokens(
            lengths=[3, 3, 3],
            angles=[90, 90, 90],
            species=["Na"],
            frac_coords=[[0, 0, 0]],
        )
        parsed = parse_fixed_slot_answer("prefix\n" + " ".join(tokens) + "\ntrailing")
        self.assertEqual(parsed["num_atoms"], 1)
        self.assertEqual(parsed["species"], ["Na"])

    def test_ignores_model_control_tokens(self):
        tokens, _ = arrays_to_tokens(
            lengths=[3, 3, 3],
            angles=[90, 90, 90],
            species=["Na"],
            frac_coords=[[0, 0, 0]],
        )
        decoded = "<|eot_id|><|endoftext|>\n" + " ".join(tokens)
        parsed = parse_fixed_slot_answer(decoded)
        self.assertEqual(parsed["num_atoms"], 1)
        self.assertEqual(parsed["species"], ["Na"])

    def test_compact_answer_round_trip(self):
        answer, _ = arrays_to_answer(
            lengths=[3, 3, 3],
            angles=[90, 90, 90],
            species=["Na"],
            frac_coords=[[0, 0, 0]],
            separator="",
        )
        self.assertNotIn(" ", answer)
        parsed = parse_fixed_slot_answer(answer)
        self.assertEqual(parsed["num_atoms"], 1)
        self.assertEqual(len(parsed["tokens"]), ANSWER_TOKEN_COUNT)

    def test_rejects_unsupported_element(self):
        with self.assertRaises(FixedSlotError):
            arrays_to_tokens(
                lengths=[3, 3, 3],
                angles=[90, 90, 90],
                species=["Am"],
                frac_coords=[[0, 0, 0]],
            )

    def test_special_tokens_cover_schema(self):
        specials = build_special_tokens()
        for token in ["<N_020>", "<LA_500>", "<AG_179>", "<S19>", "<E_Pu>", "<Z_100>", "<EMPTY>"]:
            self.assertIn(token, specials)
        self.assertEqual(len(specials), len(set(specials)))

    def test_eighty_slot_config_round_trip(self):
        config = FixedSlotConfig(max_atoms=80)
        species = ["Cs"] * 16 + ["I"] * 48 + ["Pb"] * 12 + ["Mn", "Ba", "Ca", "Sr"]
        coords = [[(i * 0.013) % 1.0, (i * 0.017) % 1.0, (i * 0.019) % 1.0] for i in range(80)]
        tokens, _ = arrays_to_tokens(
            lengths=[21.3, 9.5, 17.9],
            angles=[90.0, 90.0, 90.0],
            species=species,
            frac_coords=coords,
            config=config,
        )
        self.assertEqual(answer_token_count(config), 407)
        self.assertEqual(len(tokens), 407)
        self.assertIn("<N_080>", build_special_tokens(config))
        self.assertIn("<S79>", build_special_tokens(config))
        parsed = tokens_to_arrays(tokens, config=config)
        self.assertEqual(parsed["num_atoms"], 80)
        self.assertEqual(parsed["species"][0], "Cs")
        self.assertEqual(parsed["species"][-1], "Sr")


if __name__ == "__main__":
    unittest.main()
