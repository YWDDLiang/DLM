import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.doping import (
    CANONICAL_BSITE_INDICES,
    DOPANT_SYMBOLS,
    all_candidate_combos,
    band_gap_bin,
    compact_answer,
    compact_special_tokens,
    combo_name,
    feature_vector,
    objective_tuple_from_properties,
    parse_compact_answer,
    ranking_tuple,
)


class DopingCompactTests(unittest.TestCase):
    def test_compact_answer_is_six_sorted_tokens(self):
        answer = compact_answer(["Zn", "Al", "In"])
        parsed = parse_compact_answer(answer)
        self.assertEqual(answer.split(), ["<DOPANT_Al>", "<DOPANT_In>", "<DOPANT_Zn>", "<B00>", "<B01>", "<B14>"])
        self.assertEqual(len(answer.split()), 6)
        self.assertEqual(parsed["dopants"], ["Al", "In", "Zn"])
        self.assertEqual(tuple(parsed["dopant_site_indices"]), CANONICAL_BSITE_INDICES)

    def test_reject_duplicate_or_unsorted_dopants(self):
        with self.assertRaises(ValueError):
            parse_compact_answer("<DOPANT_Al> <DOPANT_Al> <DOPANT_Zn> <B00> <B01> <B14>")
        with self.assertRaises(ValueError):
            parse_compact_answer("<DOPANT_Zn> <DOPANT_Al> <DOPANT_In> <B00> <B01> <B14>")

    def test_band_gap_boundaries_are_target(self):
        self.assertEqual(band_gap_bin(1.6999), "<BG_LOW>")
        self.assertEqual(band_gap_bin(1.7), "<BG_TARGET>")
        self.assertEqual(band_gap_bin(2.7), "<BG_TARGET>")
        self.assertEqual(band_gap_bin(2.7001), "<BG_HIGH>")

    def test_ranking_directions(self):
        better = {
            "p_success": 0.9,
            "p_band_gap_target": 0.8,
            "pred_formation_energy": -1.0,
            "pred_defect_formation_energy": 5.0,
            "uncertainty": 0.1,
        }
        worse_fe = {**better, "pred_formation_energy": 0.0}
        worse_dfe = {**better, "pred_defect_formation_energy": -5.0}
        self.assertGreater(ranking_tuple(better), ranking_tuple(worse_fe))
        self.assertGreater(ranking_tuple(better), ranking_tuple(worse_dfe))
        self.assertEqual(
            objective_tuple_from_properties(
                {"band_gap": 2.0, "formation_energy": -1.0, "defect_formation_energy": 4.0}
            ),
            (1, 1.0, 4.0),
        )

    def test_candidate_space_and_features(self):
        self.assertEqual(len(DOPANT_SYMBOLS), 13)
        self.assertEqual(len(all_candidate_combos()), 286)
        features = feature_vector(["Al", "In", "Zn"])
        self.assertEqual(len(features), 91)
        self.assertEqual(sum(features[:13]), 3.0)
        self.assertEqual(sum(features[13:]), 3.0)

    def test_vocab_contains_expected_tokens(self):
        vocab = compact_special_tokens()
        self.assertIn("<DOPANT_Al>", vocab)
        self.assertIn("<B15>", vocab)
        self.assertIn("<DOPING_COMPACT>", vocab)
        self.assertIn("<BG_TARGET>", vocab)
        self.assertIn("<FE_Q1>", vocab)
        self.assertIn("<DFE_Q4>", vocab)


if __name__ == "__main__":
    unittest.main()
