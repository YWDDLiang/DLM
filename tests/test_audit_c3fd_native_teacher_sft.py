import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_c3fd_native_teacher_sft",
    ROOT / "scripts" / "audit_c3fd_native_teacher_sft.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import audit_c3fd_native_teacher_sft.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TeacherSFTTokenAuditTest(unittest.TestCase):
    def test_quantile_is_deterministic(self):
        values = [5, 1, 3, 2, 4]
        self.assertEqual(MODULE.quantile(values, 0.5), 3)
        self.assertEqual(MODULE.quantile(values, 0.95), 5)

    def test_empty_quantile_fails(self):
        with self.assertRaises(ValueError):
            MODULE.quantile([], 0.5)


if __name__ == "__main__":
    unittest.main()
