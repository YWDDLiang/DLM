import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.ccfd import CCFDState, FormulaToken, legal_next_tokens, replay_tokens


class CCFDTest(unittest.TestCase):
    def test_magnetite_mixed_valence_conserves_atoms_and_charge(self) -> None:
        tokens = sorted(
            [
                FormulaToken.from_symbol("O", -2, 4),
                FormulaToken.from_symbol("Fe", 2, 1),
                FormulaToken.from_symbol("Fe", 3, 2),
            ]
        )
        state = replay_tokens(7, tokens)
        self.assertTrue(state.eos_legal)
        self.assertEqual(state.remaining_atoms, 0)
        self.assertEqual(state.remaining_charge, 0)

    def test_alloy_zero_valence_branch(self) -> None:
        tokens = sorted(
            [
                FormulaToken.from_symbol("Fe", 0, 1),
                FormulaToken.from_symbol("Ni", 0, 3),
            ]
        )
        state = replay_tokens(4, tokens)
        self.assertEqual(state.branch, "alloy")
        self.assertTrue(state.eos_legal)

    def test_ionic_and_alloy_tokens_cannot_mix(self) -> None:
        state = CCFDState.start(2).append(FormulaToken.from_symbol("O", -2, 1))
        with self.assertRaisesRegex(ValueError, "cannot share a branch"):
            state.append(FormulaToken.from_symbol("Fe", 0, 1))

    def test_one_element_cannot_mix_valence_signs(self) -> None:
        state = CCFDState.start(2).append(FormulaToken.from_symbol("Fe", -2, 1))
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            state.append(FormulaToken.from_symbol("Fe", 2, 1))

    def test_lookahead_masks_dead_end_action(self) -> None:
        catalog = tuple(
            sorted(
                [
                    FormulaToken.from_symbol("O", -2, 1),
                    FormulaToken.from_symbol("O", -2, 2),
                    FormulaToken.from_symbol("Fe", 2, 1),
                    FormulaToken.from_symbol("Fe", 3, 1),
                ]
            )
        )
        legal = legal_next_tokens(CCFDState.start(2), catalog, max_species=2)
        self.assertIn(FormulaToken.from_symbol("O", -2, 1), legal)
        self.assertNotIn(FormulaToken.from_symbol("O", -2, 2), legal)

    def test_noncanonical_order_fails_closed(self) -> None:
        fe = FormulaToken.from_symbol("Fe", 2, 1)
        oxygen = FormulaToken.from_symbol("O", -2, 1)
        state = CCFDState.start(2).append(fe)
        with self.assertRaisesRegex(ValueError, "canonical"):
            state.append(oxygen)


if __name__ == "__main__":
    unittest.main()
