import unittest

from scripts.audit_c3fd_native_teacher_sft import quantile


class TeacherSFTTokenAuditTest(unittest.TestCase):
    def test_quantile_is_deterministic(self):
        values = [5, 1, 3, 2, 4]
        self.assertEqual(quantile(values, 0.5), 3)
        self.assertEqual(quantile(values, 0.95), 5)

    def test_empty_quantile_fails(self):
        with self.assertRaises(ValueError):
            quantile([], 0.5)


if __name__ == "__main__":
    unittest.main()
