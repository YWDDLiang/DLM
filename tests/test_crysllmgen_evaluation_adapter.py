from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen import evaluation_adapter
from crystal_dlm.wqcodiff.evaluation import (
    _load_evaluation_protocol,
    _read_generation,
    _trace_summary,
)


class CrysLLMGenEvaluationAdapterTests(unittest.TestCase):
    def test_rich_evaluator_accepts_active_protocol_v4(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol = _load_evaluation_protocol(
            root / "configs/experiments/wyckoff_codiffusion/protocol_v4.yaml"
        )
        self.assertEqual(protocol.name, "crysllmgen_wyckoff_georev_v4")

    def test_common_generation_schema_reuses_rich_evaluator(self) -> None:
        row = {
            "schema": "wqcodiff_generation_attempt_v1",
            "attempt_id": "a-1",
            "pair_id": "p-1",
            "method": "C-WQ-GEOREV",
            "training_seed": 11,
            "sampling_seed": 101,
            "status": "succeeded",
            "trace": [{"action": "llama_direct_edit"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(_read_generation(path), [row])
        self.assertEqual(_trace_summary(row)["revision_events"], 1)

    def test_r5c_aggregation_restores_every_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            structures = root / "input.extxyz"
            structures.write_text("frozen-input\n", encoding="utf-8")
            mapping = [
                {
                    "extxyz_index": index,
                    "attempt_id": f"a-{index}",
                    "generation_ordinal": index,
                    "structure_hash": f"h-{index}",
                }
                for index in range(3)
            ]
            manifest = {
                "schema": "crysllmgen_r5c_input_manifest_v1",
                "method": "C-WQ-GEOREV",
                "structure_stage": "mlip_relaxed",
                "total_attempts": 4,
                "materialized_structures": 3,
                "generation_failures": [
                    {
                        "attempt_id": "a-3",
                        "generation_ordinal": 3,
                        "status": "failed",
                        "reason": "parse",
                    }
                ],
                "index_mapping": mapping,
                "structures_path": str(structures),
                "structures_sha256": hashlib.sha256(structures.read_bytes()).hexdigest(),
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "r5c.json").write_text(
                json.dumps({"num_structures": 3}), encoding="utf-8"
            )
            (root / "detailed.json").write_text(
                json.dumps(
                    {
                        "energy_above_hull": [-0.01],
                        "novel_unique": [True],
                    }
                ),
                encoding="utf-8",
            )
            (root / "unsupported.json").write_text(
                json.dumps(
                    {
                        "unsupported_records": [
                            {"index": 1, "formula": "X"}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "relax.json").write_text(
                json.dumps({"relax_failed_indices": [2]}), encoding="utf-8"
            )
            script = root / "r5c.py"
            script.write_text("# frozen\n", encoding="utf-8")
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"model")
            previous = evaluation_adapter.R5C_SCRIPT_SHA256
            evaluation_adapter.R5C_SCRIPT_SHA256 = hashlib.sha256(
                script.read_bytes()
            ).hexdigest()
            try:
                report = evaluation_adapter.aggregate_r5c_output(
                    input_manifest_path=root / "manifest.json",
                    r5c_summary_path=root / "r5c.json",
                    detailed_metrics_path=root / "detailed.json",
                    unsupported_path=root / "unsupported.json",
                    relax_failures_path=root / "relax.json",
                    output_jsonl=root / "attempts.jsonl",
                    output_summary=root / "summary.json",
                    evaluator="MatterSim-v1.0.0-5M",
                    evaluator_checkpoint=checkpoint,
                    r5c_script=script,
                )
            finally:
                evaluation_adapter.R5C_SCRIPT_SHA256 = previous
            self.assertEqual(report["counts"]["total"], 4)
            self.assertEqual(report["counts"]["generation_failed"], 1)
            self.assertEqual(report["counts"]["unsupported"], 1)
            self.assertEqual(report["counts"]["nonconverged"], 1)
            self.assertEqual(report["counts"]["sun_0p0"], 1)
            self.assertEqual(report["rates"]["sun_0p0"], 0.25)
            self.assertEqual(
                len((root / "attempts.jsonl").read_text().splitlines()), 4
            )


if __name__ == "__main__":
    unittest.main()
