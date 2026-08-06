import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("assemble_terminal_v2.py")
SPEC = importlib.util.spec_from_file_location("assemble_terminal_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ArrayAwareSchedulerTests(unittest.TestCase):
    def test_display_job_id_maps_distinct_raw_allocations(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "scheduler.txt"
            record.write_text(
                "29337_0|29339|COMPLETED|0:0|00:06:51\n"
                "29337_1|29337|COMPLETED|0:0|00:04:00\n",
                encoding="utf-8",
            )
            parsed = MODULE.parse_scheduler(record, "29337")
        self.assertEqual(parsed["B1"]["job_id"], "29337_0")
        self.assertEqual(parsed["B1"]["job_id_raw"], "29339")
        self.assertEqual(parsed["B2"]["job_id"], "29337_1")
        self.assertEqual(parsed["B2"]["job_id_raw"], "29337")
        self.assertEqual(parsed["B1"]["exit_code"], "0:0")
        self.assertEqual(parsed["B2"]["state"], "COMPLETED")

    def test_duplicate_display_job_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "scheduler.txt"
            record.write_text(
                "29337_0|29339|COMPLETED|0:0|00:06:51\n"
                "29337_0|29340|COMPLETED|0:0|00:06:52\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                MODULE.parse_scheduler(record, "29337")


if __name__ == "__main__":
    unittest.main()
