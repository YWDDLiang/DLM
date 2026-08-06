import unittest

from crystal_dlm.cif_lite import (
    CifLiteError,
    assert_no_pbc_duplicate,
    format_composition_block,
    format_lattice_block,
    format_sites_block,
    parse_cif_lite_modules,
    parse_composition_block,
    truncate_module_text,
)


class CifLiteTest(unittest.TestCase):
    def test_encode_parse_round_trip(self):
        composition = format_composition_block(["Li", "Li", "V", "O", "O", "O", "O"])
        lattice = format_lattice_block([5.43, 5.44, 7.55], [90.1, 90.0, 120.0])
        sites = format_sites_block(
            ["Li", "Li", "V", "O", "O", "O", "O"],
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25],
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6],
            ],
        )

        arrays = parse_cif_lite_modules(composition, lattice, sites)

        self.assertEqual(arrays["num_atoms"], 7)
        self.assertEqual(arrays["composition"], {"Li": 2, "O": 4, "V": 1})
        self.assertEqual(arrays["atom_types"][0], 3)
        self.assertAlmostEqual(arrays["lengths"][0], 5.4)

    def test_species_multiset_mismatch_is_rejected(self):
        composition = format_composition_block(["Li", "O"])
        lattice = format_lattice_block([4.0, 4.0, 4.0], [90, 90, 90])
        sites = format_sites_block(["Li", "Li"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        with self.assertRaises(CifLiteError):
            parse_cif_lite_modules(composition, lattice, sites)

    def test_pbc_duplicate_treats_zero_and_one_as_equivalent(self):
        with self.assertRaises(CifLiteError):
            assert_no_pbc_duplicate([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])

    def test_composition_parser_sums_duplicate_element_lines(self):
        parsed = parse_composition_block("composition:\nLi 1\nO 2\nLi 1\nend\n")
        self.assertEqual(parsed["counts"], {"Li": 2, "O": 2})

    def test_truncate_module_text_keeps_first_end(self):
        text = "noise\nsites:\nLi\n0 0 0\nend\nlattice:\n1 1 1\n90 90 90\nend\n"
        self.assertEqual(truncate_module_text(text, "sites"), "sites:\nLi\n0 0 0\nend\n")


if __name__ == "__main__":
    unittest.main()
