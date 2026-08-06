import unittest

from crystal_dlm.crysllmgen_text import (
    CifLiteError,
    parse_composition_text,
    parse_coords_text,
    parse_lattice_text,
)


class CrysLLMGenTextParsingTest(unittest.TestCase):
    def test_lattice_parser_scans_for_first_valid_pair(self):
        parsed = parse_lattice_text(
            "\n".join(
                [
                    "7.3 7.3 7.3",
                    "137 137 137",
                    "7.3 7.3 7.3",
                    "137 137 67",
                ]
            )
        )

        self.assertEqual(parsed["text"], "7.3 7.3 7.3\n137 137 67")

    def test_composition_parser_keeps_valid_prefix_before_atom_cap(self):
        parsed = parse_composition_text("O 12\nF 12\nNa 1\n")

        self.assertEqual(parsed["composition"], {"O": 12})
        self.assertEqual(len(parsed["species"]), 12)

    def test_coords_parser_can_skip_bad_lines_and_pbc_duplicates(self):
        parsed = parse_coords_text(
            "\n".join(
                [
                    "0.00 0.00 0.00",
                    "not a coordinate",
                    "1.00 0.00 0.00",
                    "0.25 0.50 0.75",
                    "0.33 0.44 0.55",
                ]
            ),
            expected_count=3,
            skip_invalid=True,
            deduplicate_pbc=True,
        )

        self.assertEqual(len(parsed["frac_coords"]), 3)
        self.assertEqual(parsed["text"], "0.00 0.00 0.00\n0.25 0.50 0.75\n0.33 0.44 0.55")

    def test_coords_parser_remains_strict_by_default(self):
        with self.assertRaises(CifLiteError):
            parse_coords_text("0.00 0.00 0.00\nnot a coordinate\n0.25 0.50 0.75", expected_count=2)


if __name__ == "__main__":
    unittest.main()
