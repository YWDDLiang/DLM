from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from complete_sun import (  # noqa: E402
    audit_source_cache_coverage,
    chemsys_query_values,
    exact_mcnemar,
    paired_effect,
    strip_unavailable_optional_mson_tags,
)


class CompletionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cls.runner = (ROOT / "run_once.sh").read_text(encoding="utf-8")
        cls.source = (ROOT / "complete_sun.py").read_text(encoding="utf-8")

    def test_frozen_all_attempt_scope(self) -> None:
        self.assertEqual(self.config["sun"]["attempts_per_arm"], 256)
        self.assertEqual(
            self.config["sun"]["denominator"], "all_registered_attempts"
        )
        self.assertEqual(
            self.config["source_run"]["expected_distinct_missing_chemsys_union"],
            204,
        )
        self.assertEqual(
            self.config["source_run"]["expected_distinct_missing_chemsys_overlap"],
            7,
        )
        self.assertEqual(
            self.config["source_run"]["arms"]["P0"]["expected_counts"][
                "source_hull_unknown"
            ],
            101,
        )
        self.assertEqual(
            self.config["source_run"]["arms"]["P1"]["expected_counts"][
                "source_hull_unknown"
            ],
            110,
        )
        self.assertTrue(self.config["sun"]["generation_reused"])
        self.assertTrue(self.config["sun"]["chgnet_energy_reused"])
        self.assertFalse(self.config["sun"]["retry_or_replacement_used"])

    def test_no_gpu_slurm_or_downstream(self) -> None:
        self.assertFalse(self.config["execution"]["slurm_allowed"])
        self.assertFalse(self.config["execution"]["gpu_allowed"])
        self.assertFalse(
            self.config["authorization"]["new_generation_authorized"]
        )
        self.assertFalse(
            self.config["authorization"]["chgnet_rerun_authorized"]
        )
        self.assertFalse(
            self.config["authorization"]["automatic_downstream_authorized"]
        )
        self.assertIn("login-node-only", self.runner)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', self.runner)
        self.assertNotIn("sbatch", self.runner)
        self.assertNotIn("srun", self.runner)

    def test_secret_is_file_only_unlinked_and_not_serialized(self) -> None:
        self.assertIn("read_and_destroy_api_key", self.source)
        self.assertIn("location.unlink(missing_ok=True)", self.source)
        self.assertIn('"api_key_serialized": False', self.source)
        self.assertNotIn("print(api_key", self.source)
        self.assertNotIn("json.dump(api_key", self.source)
        self.assertNotIn("os.environ.get(\"MP_API_KEY\"", self.source)
        self.assertIn("unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY", self.runner)
        self.assertIn('final_error["http_status"] in {401, 403}', self.source)

    def test_exact_mcnemar_direction(self) -> None:
        result = exact_mcnemar(
            [True, True, True, False],
            [False, False, True, False],
        )
        self.assertEqual(result["candidate_only"], 2)
        self.assertEqual(result["baseline_only"], 0)
        self.assertEqual(result["discordant"], 2)

    def test_paired_effect_retains_256_denominator(self) -> None:
        baseline = [False] * 256
        candidate = [True] + [False] * 255
        result = paired_effect(candidate, baseline, seed_offset=8)
        self.assertEqual(result["attempts"], 256)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["baseline_count"], 0)
        self.assertAlmostEqual(result["difference_percentage_points"], 100 / 256)

    def test_populated_stale_cache_row_does_not_block_unknown_refresh(self) -> None:
        result = audit_source_cache_coverage(
            cached={
                "A-B": [{"composition": {"A": 1}, "energy": -1.0}],
                "C-D": [{"composition": {"C": 1}, "energy": -2.0}],
            },
            all_cached_chemsys={"A-B", "C-D"},
            wanted_chemsys={"A-B", "C-D"},
            source_unknown_chemsys={"C-D"},
        )
        self.assertEqual(result["source_evaluated_chemsys"], 1)
        self.assertEqual(result["source_unknown_chemsys"], 1)
        self.assertEqual(result["base_cache_records_for_source_unknown"], 1)
        self.assertEqual(
            result["base_cache_populated_records_for_source_unknown"],
            1,
        )

    def test_chemsys_query_includes_all_subsystems(self) -> None:
        self.assertEqual(
            chemsys_query_values("Ag-As-Co"),
            ["Ag", "As", "Co", "Ag-As", "Ag-Co", "As-Co", "Ag-As-Co"],
        )

    def test_optional_mson_metadata_can_remain_plain_json(self) -> None:
        stripped: set[str] = set()
        result = strip_unavailable_optional_mson_tags(
            {
                "payload": {
                    "@module": "emmet.module_that_is_not_installed",
                    "@class": "Metadata",
                    "value": 7,
                }
            },
            stripped,
        )
        self.assertEqual(result, {"payload": {"value": 7}})
        self.assertEqual(stripped, {"emmet.module_that_is_not_installed"})

    def test_legacy_pymatgen_mson_module_is_redirected(self) -> None:
        stripped: set[str] = set()
        redirected: set[str] = set()
        with mock.patch("complete_sun.importlib.import_module"):
            result = strip_unavailable_optional_mson_tags(
                {
                    "@module": "pymatgen.core.entries",
                    "@class": "ComputedEntry",
                    "energy": -1.0,
                },
                stripped,
                redirected,
            )
        self.assertEqual(
            result["@module"],
            "pymatgen.entries.computed_entries",
        )
        self.assertEqual(
            redirected,
            {
                "pymatgen.core.entries"
                "->pymatgen.entries.computed_entries"
            },
        )
        self.assertEqual(stripped, set())


if __name__ == "__main__":
    unittest.main()
