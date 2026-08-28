import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "assemble_c3fd_sample_segments",
    ROOT / "scripts" / "assemble_c3fd_sample_segments.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import assemble_c3fd_sample_segments.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AssembleC3FDSampleSegmentsTest(unittest.TestCase):
    def make_segment(self, root: Path, name: str, start: int, count: int) -> Path:
        segment = root / name
        segment.mkdir()
        raw = []
        plans = []
        for sample_idx in range(start, start + count):
            parsed = sample_idx != 1
            row = {
                "sample_idx": sample_idx,
                "parsed": parsed,
                "failure": None if parsed else "ValueError:synthetic",
                "target_proposal": {"family": "oxide", "N": 2, "arity": 2},
                "certificate": (
                    {"certificate_class": "benchmark_compatible"} if parsed else None
                ),
            }
            raw.append(row)
            if parsed:
                plans.append(
                    {
                        "sample_idx": sample_idx,
                        "plan_text": f"plan-{sample_idx}",
                        "plan_state": {"N": 2},
                    }
                )
        (segment / "raw_generations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8"
        )
        (segment / "plans_for_dlm.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in plans), encoding="utf-8"
        )
        (segment / "sample_metrics.json").write_text(
            json.dumps(
                {
                    "seed": 17,
                    "requested_samples": count,
                    "start_index": start,
                    "end_index_exclusive": start + count,
                    "reachability_mode": "pauling_bitset",
                    "elapsed_sec": 1.5,
                }
            ),
            encoding="utf-8",
        )
        (segment / "_SUCCESS").touch()
        return segment

    def test_assembles_exact_global_ledger_without_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.make_segment(root, "first", 0, 2)
            second = self.make_segment(root, "second", 2, 2)
            output = root / "output"
            argv = [
                "assemble_c3fd_sample_segments.py",
                "--segment",
                str(first),
                "--segment",
                str(second),
                "--output-dir",
                str(output),
                "--requested",
                "4",
                "--seed",
                "17",
            ]
            previous = sys.argv
            try:
                sys.argv = argv
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.main()
            finally:
                sys.argv = previous
            raw = list(MODULE.iter_jsonl(output / "raw_generations.jsonl"))
            metrics = json.loads((output / "sample_metrics.json").read_text())
            self.assertEqual([row["sample_idx"] for row in raw], [0, 1, 2, 3])
            self.assertEqual(metrics["requested_samples"], 4)
            self.assertEqual(metrics["parsed_samples"], 3)
            self.assertEqual(metrics["failures"], {"ValueError:synthetic": 1})
            self.assertFalse(metrics["replacement"])
            self.assertEqual(metrics["elapsed_sec"], 3.0)


if __name__ == "__main__":
    unittest.main()
