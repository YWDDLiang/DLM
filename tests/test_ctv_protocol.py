from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.composition_identity import (
    canonical_symbol_counts,
    identity_text,
    reduced_composition_identity,
)
from crystal_dlm.ctv_protocol import (
    apply_energy_guidance,
    branch_record_id,
    counter_seed,
    select_eight_legal_actions,
    validate_branch_ledger,
)


class CTVProtocolTest(unittest.TestCase):
    def test_reduced_identity_is_order_and_scale_invariant(self):
        left = reduced_composition_identity(["Fe", "O"], [4, 6])
        right = reduced_composition_identity(["O", "Fe"], [3, 2])
        self.assertEqual(left, right)
        self.assertEqual(identity_text(left), "8:3|26:2")
        self.assertEqual(canonical_symbol_counts(["Fe", "O"], [2, 3]), (("O", 3), ("Fe", 2)))

    def test_counter_rng_and_branch_id_are_order_independent(self):
        first = counter_seed("8:3|26:2", 7, 10, 42, 9)
        second = counter_seed("8:3|26:2", 7, 10, 42, 9)
        changed = counter_seed("8:3|26:2", 7, 10, 43, 9)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        branch = branch_record_id(
            composition_id="8:3|26:2",
            sample_idx=7,
            milestone=0.6,
            position=10,
            action_token=42,
            continuation_seed=9,
        )
        self.assertEqual(len(branch), 64)

    def test_action_quantiles_project_to_distinct_legal_tokens(self):
        with self.assertRaisesRegex(ValueError, "at least eight"):
            select_eight_legal_actions([0.95, 0.05], [10, 11])
        probabilities = [0.04] * 8 + [0.12] + [0.04] * 14
        selected = select_eight_legal_actions(
            probabilities, list(range(100, 123))
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(set(selected)), 8)
        concentrated = select_eight_legal_actions(
            [0.93] + [0.01] * 7,
            list(range(200, 208)),
        )
        self.assertEqual(concentrated[0], 200)
        self.assertEqual(set(concentrated), set(range(200, 208)))

    def test_gamma_zero_is_bit_exact_and_unsupported_is_base(self):
        base = (1.0, 2.0, 3.0)
        costs = (0.2, -0.3, 5.0)
        supported = (True, True, False)
        self.assertEqual(
            apply_energy_guidance(base, costs, supported, gamma=0.0),
            base,
        )
        self.assertEqual(
            apply_energy_guidance(base, costs, supported, gamma=5.0),
            (0.0, 3.5, 3.0),
        )

    def test_branch_ledger_requires_eight_actions_per_state(self):
        rows = []
        for state in ("s0", "s1"):
            for action in range(8):
                rows.append(
                    {
                        "branch_id": f"{state}-{action}",
                        "state_id": state,
                        "action_token": action,
                    }
                )
        summary = validate_branch_ledger(rows, expected_rows=16)
        self.assertEqual(summary, {"rows": 16, "unique_branch_ids": 16, "states": 2})
        rows[-1]["action_token"] = 6
        with self.assertRaisesRegex(ValueError, "repeats action"):
            validate_branch_ledger(rows, expected_rows=16)


if __name__ == "__main__":
    unittest.main()
