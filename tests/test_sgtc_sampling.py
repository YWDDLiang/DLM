from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.sgtc_sampling import (
    matched_base_noise_group,
    validate_sgtc_attempts,
    validate_sgtc_denominator,
)


class SGTCSamplingTest(unittest.TestCase):
    def test_noise_group_is_arm_independent(self):
        first = matched_base_noise_group(seed=17, composition_id="8:2|26:1", sample_idx=4)
        second = matched_base_noise_group(seed=17, composition_id="8:2|26:1", sample_idx=4)
        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            matched_base_noise_group(seed=18, composition_id="8:2|26:1", sample_idx=4),
        )

    def test_attempt_accounting(self):
        rows = [
            {"ordinal": index, "sample_idx": index, "parsed": index != 2}
            for index in range(4)
        ]
        self.assertEqual(
            validate_sgtc_attempts(rows, expected=4),
            {"requested": 4, "parsed": 3, "failed": 1},
        )

    def test_frozen_screen_denominators(self):
        self.assertEqual(validate_sgtc_denominator(256), 256)
        self.assertEqual(validate_sgtc_denominator(1000), 1000)
        with self.assertRaisesRegex(ValueError, "L6=256 or L7=1000"):
            validate_sgtc_denominator(512)


if __name__ == "__main__":
    unittest.main()
