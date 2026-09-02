from __future__ import annotations

import unittest

from crystal_dlm.dynamic_crystal import (
    arrays_to_dynamic_answer,
    dynamic_answer_token_count,
    parse_dynamic_answer,
)
from crystal_dlm.fixed_slot import FixedSlotError


class DynamicGuardTests(unittest.TestCase):
    def _answer(self) -> str:
        answer, _ = arrays_to_dynamic_answer(
            lengths=[4.1, 4.2, 4.3],
            angles=[90.0, 91.0, 92.0],
            species=["Li", "O"],
            frac_coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        return answer

    def test_dynamic_length_rejects_out_of_domain_counts(self) -> None:
        for value in (0, 21, -1):
            with self.assertRaises(FixedSlotError):
                dynamic_answer_token_count(value)

    def test_strict_parse_rejects_surrounding_text(self) -> None:
        answer = self._answer()
        with self.assertRaises(FixedSlotError):
            parse_dynamic_answer("garbage" + answer, strict=True)
        with self.assertRaises(FixedSlotError):
            parse_dynamic_answer(answer + "garbage", strict=True)

    def test_strict_parse_allows_only_whitespace_between_tokens(self) -> None:
        answer = self._answer()
        spaced = answer.replace("><", "> \n <")
        parsed = parse_dynamic_answer(spaced, strict=True)
        self.assertEqual(parsed["num_atoms"], 2)
        self.assertEqual(len(parsed["tokens"]), dynamic_answer_token_count(2))


try:
    from crystal_dlm.llada_generation import _validate_generation_position_groups
except ModuleNotFoundError:
    _validate_generation_position_groups = None


@unittest.skipIf(_validate_generation_position_groups is None, "torch unavailable")
class ScheduleGuardTests(unittest.TestCase):
    def test_noncontiguous_complete_schedule_is_valid(self) -> None:
        groups = _validate_generation_position_groups(
            [[4, 1], [3], [0, 2]],
            5,
        )
        self.assertEqual(groups, [[4, 1], [3], [0, 2]])

    def test_incomplete_schedule_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not cover every position"):
            _validate_generation_position_groups([[4, 1], [3]], 5)


if __name__ == "__main__":
    unittest.main()
