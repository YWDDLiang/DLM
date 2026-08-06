import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.a800.run_day7_lane import (
    execute_lane,
    parse_checkpoint_map,
    resolve_lane,
)


class Day7LaneTests(unittest.TestCase):
    def test_lane_is_modulo_partitioned_and_placeholders_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "val.jsonl"
            checkpoint = root / "geo.pt"
            dataset.write_text("{}\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            jobs = []
            for ordinal in range(9):
                stem = root / f"cell-{ordinal}"
                jobs.append(
                    {
                        "phase": "day7-primary",
                        "cell_id": f"cell-{ordinal}",
                        "experiment_id": f"experiment-{ordinal}",
                        "method": "M-WQ-STRAT-GEO",
                        "attempts": 16,
                        "backbone_calls_per_attempt": 16,
                        "argv": [
                            "python",
                            "-m",
                            "crystal_dlm.wqcodiff",
                            "recovery",
                            "--checkpoint",
                            "${CHECKPOINT_M_WQ_STRAT_GEO}",
                            "--dataset",
                            "${DAY7_VAL_WQ}",
                            "--output",
                            str(stem.with_suffix(".jsonl")),
                            "--ledger",
                            str(stem.with_suffix(".attempts.jsonl")),
                            "--revision-threshold",
                            "${REVISION_THRESHOLD}",
                        ],
                    }
                )
            plan = {
                "schema": "wqcodiff_materialized_job_plan_v1",
                "maximum_concurrent_lanes": 4,
                "registry_sha256": "a" * 64,
                "protocol_sha256": "b" * 64,
                "source_bundle_sha256": "c" * 64,
                "run_id": "unit-day7",
                "phase_summary": {"day7-primary": {"jobs": 9}},
                "jobs": jobs,
            }
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            lane = resolve_lane(
                plan_path=plan_path,
                phase="day7-primary",
                lane_index=1,
                lane_count=4,
                dataset_path=dataset,
                checkpoints={"M-WQ-STRAT-GEO": checkpoint},
                revision_threshold=0.8,
            )
            self.assertEqual([job["phase_ordinal"] for job in lane["jobs"]], [1, 5])
            self.assertEqual(lane["cells"], 2)
            self.assertEqual(lane["attempts"], 32)
            for job in lane["jobs"]:
                self.assertNotIn("${", " ".join(job["argv"]))
                index = job["argv"].index("--revision-threshold")
                self.assertEqual(job["argv"][index + 1], "0.8")

    def test_checkpoint_map_rejects_duplicate_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "unique"):
                parse_checkpoint_map(
                    [
                        f"M-WQ-STRAT-GEO={checkpoint}",
                        f"M-WQ-STRAT-GEO={checkpoint}",
                    ]
                )

    def test_execute_lane_writes_terminal_hashes_and_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "cell.jsonl"
            ledger = root / "cell.attempts.jsonl"
            writer = (
                "from pathlib import Path; import sys; "
                "o=Path(sys.argv[1]); l=Path(sys.argv[2]); "
                "o.write_text('{}\\n'); "
                "o.with_suffix('.summary.json').write_text('{}\\n'); "
                "l.write_text('{}\\n')"
            )
            lane = {
                "schema": "wqcodiff_day7_lane_plan_v1",
                "phase": "threshold-calibration",
                "lane_index": 0,
                "cells": 1,
                "attempts": 1,
                "backbone_calls": 16,
                "jobs": [
                    {
                        "cell_id": "unit-cell",
                        "phase_ordinal": 0,
                        "argv": [sys.executable, "-c", writer, str(output), str(ledger)],
                        "output": str(output),
                        "ledger": str(ledger),
                    }
                ],
            }
            manifest = root / "lane.manifest.json"
            events = root / "lane.events.jsonl"
            complete = root / "lane.complete.json"
            result = execute_lane(
                lane,
                manifest_path=manifest,
                events_path=events,
                complete_path=complete,
                project_root=root,
            )
            self.assertTrue(result["ok"])
            terminal = json.loads(events.read_text().splitlines()[-1])
            self.assertEqual(terminal["returncode"], 0)
            self.assertIsNotNone(terminal["output_sha256"])
            self.assertIsNotNone(terminal["summary_sha256"])
            self.assertIsNotNone(terminal["ledger_sha256"])
            with self.assertRaises(FileExistsError):
                execute_lane(
                    lane,
                    manifest_path=root / "second.manifest.json",
                    events_path=root / "second.events.jsonl",
                    complete_path=root / "second.complete.json",
                    project_root=root,
                )


if __name__ == "__main__":
    unittest.main()
