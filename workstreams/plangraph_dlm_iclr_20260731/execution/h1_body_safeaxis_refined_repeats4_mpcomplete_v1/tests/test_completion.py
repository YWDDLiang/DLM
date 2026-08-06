from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from complete_sun import (  # noqa: E402
    ARM_KEYS,
    CompletionError,
    audit_source_cache_coverage,
    chemsys_query_values,
    complete_arm_attempts,
    exact_mcnemar,
    hierarchical_paired_bootstrap,
    read_and_destroy_api_key,
    strip_unavailable_optional_mson_tags,
)


class CompletionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cls.runner = (ROOT / "run_once.sh").read_text(encoding="utf-8")
        cls.preflight = (ROOT / "preflight.sh").read_text(encoding="utf-8")
        cls.source = (ROOT / "complete_sun.py").read_text(encoding="utf-8")

    def test_frozen_eight_arm_scope(self) -> None:
        self.assertEqual(set(self.config["source_run"]["arms"]), set(ARM_KEYS))
        self.assertEqual(self.config["sun"]["attempts_per_arm_per_repeat"], 256)
        self.assertEqual(self.config["sun"]["repeat_count"], 4)
        self.assertEqual(self.config["sun"]["denominator"], "raw_all_attempts")
        self.assertEqual(
            self.config["source_run"]["expected_distinct_missing_chemsys_union"],
            107,
        )
        self.assertEqual(
            self.config["source_run"][
                "expected_distinct_missing_chemsys_union_sha256"
            ],
            "e977016940c04bd5635aeb15c0970806c721e0be41803152b9f72b1af6816a38",
        )
        self.assertEqual(
            sum(
                arm["expected_counts"]["source_hull_unknown"]
                for arm in self.config["source_run"]["arms"].values()
            ),
            825,
        )

    def test_strict_meta_relaxation_is_pinned_identically(self) -> None:
        for arm in self.config["source_run"]["arms"].values():
            self.assertEqual(
                arm["strict_relax_results"]["sha256"],
                arm["meta_relax_results"]["sha256"],
            )

    def test_no_gpu_slurm_rerun_or_downstream(self) -> None:
        authorization = self.config["authorization"]
        self.assertFalse(self.config["execution"]["slurm_allowed"])
        self.assertFalse(self.config["execution"]["gpu_allowed"])
        for name in (
            "new_generation_authorized",
            "refinement_rerun_authorized",
            "chgnet_rerun_authorized",
            "direct_metrics_rerun_authorized",
            "novelty_rerun_authorized",
            "checkpoint_reselection_authorized",
            "automatic_training_authorized",
            "automatic_promotion_authorized",
            "automatic_downstream_authorized",
        ):
            self.assertFalse(authorization[name])
        self.assertIn("login-node-only", self.runner)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', self.runner)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', self.preflight)
        self.assertNotIn("sbatch", self.runner + self.preflight)
        self.assertNotIn("srun", self.runner + self.preflight)

    def test_secret_is_file_only_unlinked_and_not_serialized(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            carrier = Path(directory) / "key"
            carrier.write_text("a" * 32 + "\n", encoding="ascii")
            os.chmod(carrier, 0o600)
            self.assertEqual(read_and_destroy_api_key(carrier), "a" * 32)
            self.assertFalse(carrier.exists())
        self.assertNotIn("print(api_key", self.source)
        self.assertNotIn("json.dump(api_key", self.source)
        self.assertNotIn('os.environ.get("MP_API_KEY"', self.source)
        self.assertIn("unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY", self.runner)
        self.assertIn('final_error["http_status"] in {401, 403}', self.source)

    def test_chemsys_query_includes_all_subsystems(self) -> None:
        self.assertEqual(
            chemsys_query_values("Ag-As-Co"),
            ["Ag", "As", "Co", "Ag-As", "Ag-Co", "As-Co", "Ag-As-Co"],
        )

    def test_exact_mcnemar_direction(self) -> None:
        result = exact_mcnemar(
            [True, True, True, False],
            [False, False, True, False],
        )
        self.assertEqual(result["candidate_only"], 2)
        self.assertEqual(result["baseline_only"], 0)
        self.assertEqual(result["discordant"], 2)

    def test_hierarchical_bootstrap_is_deterministic(self) -> None:
        differences = [
            [
                [float((repeat + ordinal) % 2), -float(ordinal % 3 == 0)]
                for ordinal in range(256)
            ]
            for repeat in range(4)
        ]
        first = hierarchical_paired_bootstrap(
            differences, seed=17, replicates=200
        )
        second = hierarchical_paired_bootstrap(
            differences, seed=17, replicates=200
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"strict_full_sun", "meta_full_sun"})

    def test_completed_attempts_preserve_parity_and_resolve_unknown(self) -> None:
        attempts = []
        for ordinal in range(256):
            novel_unique = ordinal < 2
            attempts.append(
                {
                    "attempt_id": f"h1-r03e-r0-control-{ordinal:04d}",
                    "generation_ordinal": ordinal,
                    "generation_status": "succeeded",
                    "evaluation_status": (
                        "evaluated"
                        if ordinal == 0
                        else (
                            "relaxation_or_hull_unknown"
                            if ordinal == 1
                            else "not_applicable"
                        )
                    ),
                    "metrics": {
                        "novel": novel_unique,
                        "unique_representative": True,
                        "novel_unique": novel_unique,
                        "strict_full_sun": False,
                        "meta_full_sun": ordinal == 0,
                        "e_above_hull": 0.05 if ordinal == 0 else None,
                    },
                }
            )
        context = {
            "arm_key": "r0_control",
            "arm": "control",
            "attempts": attempts,
            "targets": [
                {
                    "attempt_id": attempts[0]["attempt_id"],
                    "chemsys": "A-B",
                    "source_evaluation_status": "evaluated",
                    "source_e_above_hull": 0.05,
                },
                {
                    "attempt_id": attempts[1]["attempt_id"],
                    "chemsys": "A-C",
                    "source_evaluation_status": "relaxation_or_hull_unknown",
                    "source_e_above_hull": None,
                },
            ],
        }
        hulls = {
            ("r0_control", attempts[0]["attempt_id"]): 0.05,
            ("r0_control", attempts[1]["attempt_id"]): 0.02,
        }
        completed, counts = complete_arm_attempts(
            context=context,
            hulls=hulls,
            strict_threshold=0.0,
            meta_threshold=0.1,
            execution_manifest_sha256="a" * 64,
            queried_chemsys={"A-C"},
        )
        self.assertEqual(len(completed), 256)
        self.assertEqual(counts["source_evaluated_parity"], 1)
        self.assertEqual(counts["source_unknown_resolved"], 1)
        self.assertEqual(counts["relaxation_or_hull_unknown"], 0)
        self.assertTrue(completed[1]["metrics"]["meta_full_sun"])

    def test_existing_hull_parity_is_hard_failure(self) -> None:
        context = {
            "arm_key": "r0_control",
            "arm": "control",
            "attempts": [
                {
                    "attempt_id": f"x-{ordinal}",
                    "generation_status": "succeeded",
                    "metrics": {
                        "novel": ordinal == 0,
                        "unique_representative": True,
                        "novel_unique": ordinal == 0,
                        "e_above_hull": 0.05 if ordinal == 0 else None,
                    },
                }
                for ordinal in range(256)
            ],
            "targets": [
                {
                    "attempt_id": "x-0",
                    "chemsys": "A-B",
                    "source_evaluation_status": "evaluated",
                    "source_e_above_hull": 0.05,
                }
            ],
        }
        with self.assertRaises(CompletionError):
            complete_arm_attempts(
                context=context,
                hulls={("r0_control", "x-0"): 0.051},
                strict_threshold=0.0,
                meta_threshold=0.1,
                execution_manifest_sha256="a" * 64,
                queried_chemsys=set(),
            )

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
            result["@module"], "pymatgen.entries.computed_entries"
        )
        self.assertEqual(
            redirected,
            {"pymatgen.core.entries->pymatgen.entries.computed_entries"},
        )


if __name__ == "__main__":
    unittest.main()
