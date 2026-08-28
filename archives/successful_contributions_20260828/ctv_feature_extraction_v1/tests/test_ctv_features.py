from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ctv_features import (
    exact_prompt_length,
    geometry_token_family,
    selected_probability_error,
)


class CTVFeaturesTest(unittest.TestCase):
    def test_exact_prompt_length(self):
        self.assertEqual(exact_prompt_length(70, 4), 47)
        with self.assertRaises(ValueError):
            exact_prompt_length(20, 20)

    def test_geometry_token_family(self):
        self.assertEqual(geometry_token_family("<LA_050>"), 0)
        self.assertEqual(geometry_token_family("<Z_999>"), 8)
        with self.assertRaises(ValueError):
            geometry_token_family("<E_Fe>")

    def test_selected_probability_error(self):
        error = selected_probability_error(
            selected_token_ids=[4, 2],
            selected_probabilities=[0.4, 0.2],
            legal_token_ids=[1, 2, 3, 4],
            legal_probabilities=[0.1, 0.2, 0.3, 0.4],
        )
        self.assertEqual(error, 0.0)
        with self.assertRaises(ValueError):
            selected_probability_error(
                selected_token_ids=[9],
                selected_probabilities=[1.0],
                legal_token_ids=[1],
                legal_probabilities=[1.0],
            )
