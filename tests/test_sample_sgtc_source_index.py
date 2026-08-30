from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "scripts" / "sample_sgtc_l6.py"


class SGTCSamplerSourceIndexTest(unittest.TestCase):
    def test_original_source_index_precedes_execution_ordinal_fallback(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            'row.get("source_sample_idx", row.get("sample_idx", ordinal))',
            text,
        )


if __name__ == "__main__":
    unittest.main()
