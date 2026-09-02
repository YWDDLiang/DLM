import unittest

from crystal_dlm.canonical_site_order import canonicalize_dynamic_answer_to_plan
from crystal_dlm.dynamic_crystal import arrays_to_dynamic_answer, parse_dynamic_answer


class CanonicalSiteOrderTests(unittest.TestCase):
    def test_pure_site_permutation(self) -> None:
        answer, _ = arrays_to_dynamic_answer(
            [5.0, 6.0, 7.0],
            [80.0, 90.0, 100.0],
            ["Na", "O", "Na"],
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )
        canonical, diagnostics = canonicalize_dynamic_answer_to_plan(
            answer,
            {"N": 3, "elements": ["O", "Na"], "counts": [1, 2]},
        )
        parsed = parse_dynamic_answer(canonical, strict=True)
        self.assertEqual(parsed["species"], ["O", "Na", "Na"])
        self.assertEqual(
            parsed["frac_coords"],
            [[0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
        )
        self.assertTrue(diagnostics["changed"])
        self.assertEqual(diagnostics["mismatched_element_slots"], 2)

    def test_already_canonical_is_exact_identity(self) -> None:
        answer, _ = arrays_to_dynamic_answer(
            [5.0, 5.0, 5.0],
            [90.0, 90.0, 90.0],
            ["O", "Na", "Na"],
            [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4]],
        )
        canonical, diagnostics = canonicalize_dynamic_answer_to_plan(
            answer,
            {"N": 3, "elements": ["O", "Na"], "counts": [1, 2]},
        )
        self.assertEqual(canonical, answer)
        self.assertFalse(diagnostics["changed"])

    def test_species_multiset_mismatch_fails(self) -> None:
        answer, _ = arrays_to_dynamic_answer(
            [5.0, 5.0, 5.0],
            [90.0, 90.0, 90.0],
            ["O", "Na"],
            [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
        )
        with self.assertRaisesRegex(ValueError, "species multisets differ"):
            canonicalize_dynamic_answer_to_plan(
                answer,
                {"N": 2, "elements": ["O", "K"], "counts": [1, 1]},
            )


if __name__ == "__main__":
    unittest.main()
