from __future__ import annotations

import time
import unittest

from crystal_dlm.wqcodiff.deadlines import (
    WalltimeLimitExceeded,
    run_with_walltime_limit,
)


class WalltimeDeadlineTests(unittest.TestCase):
    def test_fast_result_is_returned(self) -> None:
        self.assertEqual(run_with_walltime_limit(lambda: 17, 0.5), 17)

    def test_slow_call_is_interrupted_and_timer_is_reusable(self) -> None:
        started = time.monotonic()
        with self.assertRaises(WalltimeLimitExceeded):
            run_with_walltime_limit(lambda: time.sleep(1.0), 0.02)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(run_with_walltime_limit(lambda: "ok", 0.5), "ok")

    def test_nonpositive_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            run_with_walltime_limit(lambda: None, 0.0)


if __name__ == "__main__":
    unittest.main()
