from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.gate import sha256_file
from crystal_dlm.wqcodiff.crysllmgen.screening import (
    _configuration,
    _paired_effect,
)


class CrysLLMGenScreeningTests(unittest.TestCase):
    def _write_evidence(self, root: Path, method: str, outcomes: list[bool]) -> dict:
        generation = root / f"{method}-generation.jsonl"
        r5c = root / f"{method}-r5c.jsonl"
        with generation.open("w", encoding="utf-8") as gen, r5c.open(
            "w", encoding="utf-8"
        ) as evaluated:
            for ordinal, outcome in enumerate(outcomes):
                attempt = f"{method}-attempt-{ordinal}"
                pair = f"pair-{ordinal}"
                gen.write(
                    json.dumps(
                        {
                            "schema": "wqcodiff_generation_attempt_v1",
                            "attempt_id": attempt,
                            "pair_id": pair,
                            "method": method,
                            "training_seed": 11,
                            "sampling_seed": 101,
                            "status": "succeeded",
                            "retry_or_replacement_used": False,
                            "calls": {"joint": 64},
                            "generation_flops_lower_bound": 100.0,
                            "revision_total": int(outcome),
                            "handoff_tau": 0.5,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                evaluated.write(
                    json.dumps(
                        {
                            "schema": "crysllmgen_r5c_attempt_result_v1",
                            "attempt_id": attempt,
                            "method": method,
                            "status": "succeeded",
                            "metrics": {
                                "sun_0p1": outcome,
                                "novel_unique": True,
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        metrics = root / f"{method}-metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "schema": "crysllmgen_generation_metrics_report_v1",
                    "ok": True,
                    "method": method,
                    "denominator": "all_generation_attempts",
                    "attempts": len(outcomes),
                    "generation_jsonl_sha256": sha256_file(generation),
                    "retry_or_replacement_used": False,
                    "metrics_unchanged_upstream": {
                        "comp_valid": 90.0,
                        "struct_valid": 99.0,
                        "valid": 89.0,
                        "cov_recall": 92.0,
                        "cov_precision": 98.0,
                        "wdist_density": 0.2,
                        "wdist_num_elems": 0.1,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "configuration_id": method,
            "method": method,
            "handoff_tau": 0.5,
            "generation_jsonl": str(generation),
            "r5c_attempt_jsonl": str(r5c),
            "crysllmgen_metrics_report": str(metrics),
        }

    def test_configuration_closes_full_attempt_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entry = self._write_evidence(
                Path(temporary), "C-WQ-HANDOFF", [True, False]
            )
            result = _configuration(entry, expected_attempts=2)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertEqual(result["sun_0p1"], 0.5)
        self.assertEqual(result["sampling_seed_counts"], {101: 2})

    def test_paired_effect_counts_both_directions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _configuration(
                self._write_evidence(root, "C-WQ-HANDOFF", [False, True, False]),
                expected_attempts=3,
            )
            candidate_entry = self._write_evidence(
                root, "C-WQ-GEOREV", [True, False, True]
            )
            candidate = _configuration(candidate_entry, expected_attempts=3)
            result = _paired_effect(candidate, baseline)
        self.assertEqual(result["wrong_to_right_pairs"], 2)
        self.assertEqual(result["right_to_wrong_pairs"], 1)
        self.assertAlmostEqual(result["outcome_revision_precision"], 2 / 3)
        self.assertGreater(result["gain_pp"], 0.0)


if __name__ == "__main__":
    unittest.main()
