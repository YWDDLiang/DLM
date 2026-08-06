from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from diagnostics.complete_wq_existing22_mp_hull import (
    recompute_completed_attempts,
    scientific_decision,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_existing22_mp_completion_v1.json"
)
RUN_ONCE = (
    ROOT
    / "scripts"
    / "a800"
    / "wq_existing22_mp_completion_v1"
    / "run_once.sh"
)
QUERY_SCRIPT = ROOT / "diagnostics" / "complete_wq_existing22_mp_hull.py"


def original_attempts() -> tuple[list[dict], list[str]]:
    unknown_ids = [f"unknown-{index}" for index in range(8)]
    rows: list[dict] = []
    for index in range(22):
        generated = index < 17
        unknown = index < 8
        meta = 8 <= index < 11
        rows.append(
            {
                "schema": "crysllmgen_r5c_a100_sun_attempt_v1",
                "attempt_id": (
                    unknown_ids[index] if unknown else f"attempt-{index}"
                ),
                "generation_ordinal": index,
                "generation_status": (
                    "succeeded" if generated else "failed"
                ),
                "evaluation_status": (
                    "relaxation_or_hull_unknown"
                    if unknown
                    else ("evaluated" if generated else "generation_failed")
                ),
                "execution_patch_sha256": "1" * 64,
                "metrics": {
                    "novel": generated,
                    "unique_representative": generated,
                    "novel_unique": generated,
                    "energy_per_atom": -1.0 if generated else None,
                    "e_above_hull": 0.05 if meta else None,
                    "strict_full_sun": False,
                    "meta_full_sun": meta,
                },
                "retry_or_replacement_used": False,
            }
        )
    return rows, unknown_ids


class FrozenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.runner = RUN_ONCE.read_text(encoding="utf-8")
        cls.query = QUERY_SCRIPT.read_text(encoding="utf-8")

    def test_exact_eight_record_scope(self) -> None:
        records = self.contract["unknown_records"]
        self.assertEqual(len(records), 8)
        self.assertEqual(
            [row["source_ordinal"] for row in records],
            [276, 285, 292, 298, 316, 397, 479, 508],
        )
        self.assertEqual(len({row["attempt_id"] for row in records}), 8)
        self.assertEqual(len({row["chemsys"] for row in records}), 8)
        ordinal397 = next(
            row for row in records if row["source_ordinal"] == 397
        )
        self.assertEqual(
            ordinal397["composition"],
            {"Ba": 1.0, "O": 11.0, "Yb": 3.0},
        )

    def test_login_node_only_and_no_scientific_mutation(self) -> None:
        execution = self.contract["execution"]
        scope = self.contract["scope"]
        self.assertEqual(execution["location"], "A800_login_node")
        self.assertFalse(execution["slurm_allowed"])
        self.assertFalse(execution["gpu_allowed"])
        self.assertEqual(scope["slurm_jobs"], 0)
        self.assertEqual(scope["gpu_jobs"], 0)
        self.assertEqual(scope["chgnet_calls"], 0)
        self.assertEqual(scope["geometry_changes"], 0)
        self.assertFalse(scope["new_generation"])
        self.assertFalse(scope["training"])
        self.assertFalse(scope["sample_retry_or_replacement"])
        self.assertFalse(scope["mp_query_retry_or_replacement"])

    def test_pinned_official_client_and_a100_semantics(self) -> None:
        mp = self.contract["materials_project"]
        self.assertEqual(mp["client"], "mp_api.client.MPRester")
        self.assertEqual(mp["method"], "get_entries_in_chemsys")
        self.assertEqual(mp["mp_api_version"], "0.45.13")
        self.assertEqual(mp["emmet_core_version"], "0.85.1")
        self.assertEqual(mp["pymatgen_version"], "2025.6.14")
        self.assertEqual(mp["thermo_type"], "GGA_GGA+U")
        self.assertTrue(mp["compatible_only"])
        self.assertEqual(mp["maximum_queries"], 8)
        self.assertFalse(mp["api_key_serialized"])

    def test_runner_fails_closed_before_fixed_identity_reuse(self) -> None:
        self.assertIn('[[ -n "${SLURM_JOB_ID:-}"', self.runner)
        self.assertIn('[[ -z "${MP_API_KEY:-}"', self.runner)
        self.assertIn('[[ -e "${CLAIM}" || -e "${OUTPUT}"', self.runner)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', self.runner)
        self.assertNotIn("sbatch", self.runner)
        self.assertNotIn("srun", self.runner)

    def test_query_has_no_database_version_call_or_key_serialization(self) -> None:
        self.assertNotIn("get_database_version(", self.query)
        self.assertIn('os.environ.get("MP_API_KEY", "")', self.query)
        self.assertIn('"api_key_serialized": False', self.query)
        self.assertNotIn("print(api_key", self.query)
        self.assertNotIn("json.dump(api_key", self.query)


class CompletedDecisionTests(unittest.TestCase):
    def test_completed_all_eight_can_pass_fixed_threshold(self) -> None:
        attempts, unknown_ids = original_attempts()
        original = copy.deepcopy(attempts)
        results = []
        for index, attempt_id in enumerate(unknown_ids):
            e_hull = 0.0 if index < 2 else 0.05
            results.append(
                {
                    "attempt_id": attempt_id,
                    "query_status": "resolved",
                    "posthoc_category": (
                        "strict_stable"
                        if index < 2
                        else "meta_only_stable"
                    ),
                    "e_above_hull": e_hull,
                }
            )
        completed, counts = recompute_completed_attempts(
            original_attempts=attempts,
            query_results=results,
            completion_execution_patch_sha256="2" * 64,
        )
        self.assertEqual(attempts, original)
        self.assertEqual(len(completed), 22)
        self.assertEqual(counts["strict_full_sun"], 2)
        self.assertEqual(counts["meta_full_sun"], 11)
        self.assertEqual(counts["relaxation_or_hull_unknown"], 0)
        self.assertEqual(
            scientific_decision(
                strict_count=counts["strict_full_sun"],
                meta_count=counts["meta_full_sun"],
                unknown_count=counts["relaxation_or_hull_unknown"],
                minimum_strict=2,
                minimum_meta=11,
            ),
            "PASS",
        )

    def test_structured_query_error_remains_unknown_without_retry(self) -> None:
        attempts, unknown_ids = original_attempts()
        results = []
        for index, attempt_id in enumerate(unknown_ids):
            if index == 7:
                results.append(
                    {
                        "attempt_id": attempt_id,
                        "query_status": "query_error",
                        "posthoc_category": "hull_unknown",
                        "e_above_hull": None,
                    }
                )
            else:
                e_hull = 0.0 if index < 2 else 0.05
                results.append(
                    {
                        "attempt_id": attempt_id,
                        "query_status": "resolved",
                        "posthoc_category": (
                            "strict_stable"
                            if index < 2
                            else "meta_only_stable"
                        ),
                        "e_above_hull": e_hull,
                    }
                )
        _, counts = recompute_completed_attempts(
            original_attempts=attempts,
            query_results=results,
            completion_execution_patch_sha256="2" * 64,
        )
        self.assertEqual(counts["strict_full_sun"], 2)
        self.assertEqual(counts["meta_full_sun"], 10)
        self.assertEqual(counts["relaxation_or_hull_unknown"], 1)
        self.assertEqual(
            scientific_decision(
                strict_count=counts["strict_full_sun"],
                meta_count=counts["meta_full_sun"],
                unknown_count=counts["relaxation_or_hull_unknown"],
                minimum_strict=2,
                minimum_meta=11,
            ),
            "INCONCLUSIVE_MP_COVERAGE",
        )

    def test_fail_when_residual_unknown_cannot_reach_minimum(self) -> None:
        self.assertEqual(
            scientific_decision(
                strict_count=0,
                meta_count=3,
                unknown_count=0,
                minimum_strict=2,
                minimum_meta=11,
            ),
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
