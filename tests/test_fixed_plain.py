import unittest

from crystal_dlm.fixed_plain import (
    assemble_plain_answer,
    parse_coords_text,
    parse_count_text,
    parse_elements_text,
    parse_lattice_text,
    split_plain_tokens,
    tokenize_plain_answer,
)


class FixedPlainTest(unittest.TestCase):
    def test_parse_and_assemble_fixed_plain(self):
        arrays = assemble_plain_answer(
            "N_003",
            "LA_030 LB_031 LC_050 AA_090 AB_090 AG_120",
            "E_Li E_V E_O",
            "X_000 Y_000 Z_000 X_050 Y_050 Z_050 X_025 Y_025 Z_025",
        )

        self.assertEqual(arrays["num_atoms"], 3)
        self.assertEqual(arrays["species"], ["Li", "V", "O"])
        self.assertEqual(len(tokenize_plain_answer(arrays["answer"])), 107)

    def test_split_plain_tokens_extracts_modules(self):
        arrays = assemble_plain_answer(
            "N_002",
            "LA_030 LB_030 LC_030 AA_090 AB_090 AG_090",
            "E_Na E_Cl",
            "X_000 Y_000 Z_000 X_050 Y_050 Z_050",
        )
        modules = split_plain_tokens(tokenize_plain_answer(arrays["answer"]))

        self.assertEqual(modules["count_text"], "N_002")
        self.assertEqual(modules["elements_text"], "E_Na E_Cl")
        self.assertEqual(modules["coords_text"], "X_000 Y_000 Z_000 X_050 Y_050 Z_050")

    def test_staged_parsers_skip_invalid_prefixes(self):
        self.assertEqual(parse_count_text("text N_004 trailing")["num_atoms"], 4)
        self.assertEqual(
            parse_lattice_text("LA_030 LB_030 LC_030 AA_179 AB_001 AG_179 LA_040 LB_040 LC_040 AA_090 AB_090 AG_090")[
                "lattice_text"
            ],
            "LA_040 LB_040 LC_040 AA_090 AB_090 AG_090",
        )
        self.assertEqual(parse_elements_text("E_Li E_O E_O extra", 3)["species"], ["Li", "O", "O"])

    def test_coords_parser_deduplicates_pbc(self):
        parsed = parse_coords_text(
            "X_000 Y_000 Z_000 X_100 Y_000 Z_000 X_050 Y_050 Z_050",
            2,
            deduplicate_pbc=True,
        )

        self.assertEqual(parsed["coords_text"], "X_000 Y_000 Z_000 X_050 Y_050 Z_050")

    def test_coords_parser_normalizes_short_plain_bins(self):
        parsed = parse_coords_text(
            "X_00 Y_0 Z_50 X_25 Y_75 Z_05",
            2,
            deduplicate_pbc=True,
        )

        self.assertEqual(parsed["coords_text"], "X_000 Y_000 Z_050 X_025 Y_075 Z_005")


if __name__ == "__main__":
    unittest.main()
