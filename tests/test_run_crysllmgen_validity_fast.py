import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_crysllmgen_validity_fast",
    ROOT / "scripts/run_crysllmgen_validity_fast.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import fast validity screen")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FastValidityScreenTest(unittest.TestCase):
    def test_attempt_ids_must_be_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                json.dumps({"attempt_id": "same"}) + "\n"
                + json.dumps({"attempt_id": "same"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                MODULE.read_rows(path)


if __name__ == "__main__":
    unittest.main()
