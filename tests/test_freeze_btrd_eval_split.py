from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_btrd_eval_split.py"


class FreezeBtrdEvalSplitTest(unittest.TestCase):
    def test_freezes_prefix_and_remainder_without_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "main1000").mkdir(parents=True)
            (source / "remainder").mkdir()
            rows = [
                {
                    "reduced_composition_identity": f"id-{index}",
                    "source_sample_idx": index,
                }
                for index in range(1159)
            ]
            (source / "main1000/plans_for_dlm.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows[:1000])
            )
            (source / "remainder/plans_for_dlm.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows[1000:])
            )
            (source / "manifest.json").write_text("{}\n")
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--known-splits",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["development_rows"], 256)
            self.assertEqual(manifest["confirmation_rows"], 903)
            self.assertFalse(manifest["selection_outcomes_read"])
            development = (output / "development256/plans_for_dlm.jsonl").read_text().splitlines()
            confirmation = (output / "confirmation903/plans_for_dlm.jsonl").read_text().splitlines()
            self.assertEqual(len(development), 256)
            self.assertEqual(len(confirmation), 903)


if __name__ == "__main__":
    unittest.main()
