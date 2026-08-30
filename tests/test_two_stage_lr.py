import math
import unittest

from src.crystal_dlm.two_stage_lr import two_stage_lr_multiplier


class TwoStageLRTest(unittest.TestCase):
    def value(self, step):
        return two_stage_lr_multiplier(
            step,
            total_steps=3392,
            stage_boundary=1696,
            stage1_warmup=100,
            stage1_min_ratio=0.2,
            stage2_base_ratio=0.2,
            stage2_warmup=100,
            stage2_min_ratio=0.1,
        )

    def test_stage_warmups_and_absolute_scale(self):
        self.assertAlmostEqual(self.value(0), 0.01)
        self.assertAlmostEqual(self.value(99), 1.0)
        self.assertGreaterEqual(self.value(1695), 0.2)
        self.assertAlmostEqual(self.value(1696), 0.002)
        self.assertAlmostEqual(self.value(1795), 0.2)
        self.assertGreaterEqual(self.value(3392), 0.02)

    def test_schedule_is_finite_and_nonnegative(self):
        values = [self.value(step) for step in range(3393)]
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in values))
        self.assertLessEqual(max(values), 1.0)

    def test_invalid_boundary_fails(self):
        with self.assertRaises(ValueError):
            two_stage_lr_multiplier(
                0,
                total_steps=10,
                stage_boundary=10,
                stage1_warmup=0,
                stage1_min_ratio=0.2,
                stage2_base_ratio=0.2,
                stage2_warmup=0,
                stage2_min_ratio=0.1,
            )


if __name__ == "__main__":
    unittest.main()
