from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.wtb_confirmatory import (
    ARM_METHODS,
    ATTEMPTS,
    BOOTSTRAP_DRAWS,
    END_ORDINAL_INCLUSIVE,
    IDENTITY,
    START_ORDINAL,
    build_confirmatory_cells,
    exact_mcnemar,
    paired_binary_effect,
    paired_numeric_effect,
)
from scripts.a800.summarize_wq_wyckoff_chart_retraction_confirmatory256_v1 import (
    _run as summarize_confirmatory,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "wq_wyckoff_chart_retraction_confirmatory256_v1.json"
)
SOURCE_RUNNER = (
    ROOT
    / "scripts"
    / "a800"
    / "run_wq_wyckoff_chart_retraction_sources256_v1.py"
)
ARMS_RUNNER = (
    ROOT
    / "scripts"
    / "a800"
    / "run_wq_wyckoff_chart_retraction_arms256_v1.py"
)
SUMMARIZER = (
    ROOT
    / "scripts"
    / "a800"
    / "summarize_wq_wyckoff_chart_retraction_confirmatory256_v1.py"
)
PIPELINE = (
    ROOT
    / "scripts"
    / "a800"
    / "wq_wyckoff_chart_retraction_confirmatory256_v1"
    / "pipeline.sbatch"
)
SUBMIT = PIPELINE.with_name("submit_once.sh")
AUTHORIZATION = (
    ROOT
    / "diagnostics"
    / "authorization_records"
    / "wq_wyckoff_chart_retraction_confirmatory256_v1_local_preparation.json"
)
INSTALLER = ROOT / "scripts" / "a800" / "install_authorized_patch.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class WTBConfirmatory256IdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_panel_is_exact_disjoint_and_deterministic(self) -> None:
        first = build_confirmatory_cells()
        second = build_confirmatory_cells()
        self.assertEqual(first, second)
        self.assertEqual(len(first), ATTEMPTS)
        self.assertEqual(first[0].ordinal, START_ORDINAL)
        self.assertEqual(first[-1].ordinal, END_ORDINAL_INCLUSIVE)
        self.assertEqual(
            [cell.ordinal for cell in first],
            list(range(512, 768)),
        )
        self.assertTrue(all(cell.ordinal > 511 for cell in first))
        self.assertEqual(len({cell.cell_id for cell in first}), ATTEMPTS)
        self.assertEqual(len({cell.pair_id for cell in first}), ATTEMPTS)
        self.assertEqual(
            len({cell.source_attempt_id for cell in first}),
            ATTEMPTS,
        )
        self.assertEqual(
            len(
                {
                    attempt_id
                    for cell in first
                    for attempt_id in cell.arm_attempt_ids.values()
                }
            ),
            3 * ATTEMPTS,
        )
        self.assertTrue(
            all(set(cell.arm_attempt_ids) == set(ARM_METHODS) for cell in first)
        )

    def test_panel_rejects_compatible_looking_replacements(self) -> None:
        for kwargs in (
            {"training_seed": 12},
            {"sampling_seed": 102},
            {"start_ordinal": 513},
            {"attempts": 255},
        ):
            with self.assertRaisesRegex(ValueError, "frozen"):
                build_confirmatory_cells(**kwargs)

    def test_contract_is_local_only_and_freezes_scientific_scope(self) -> None:
        self.assertEqual(self.contract["identity"], IDENTITY)
        self.assertEqual(
            self.contract["status"],
            "local_built_remote_execution_not_authorized",
        )
        self.assertFalse(
            self.contract["authorization"]["remote_transfer_authorized"]
        )
        self.assertFalse(
            self.contract["authorization"]["remote_install_authorized"]
        )
        self.assertFalse(
            self.contract["authorization"]["slurm_submission_authorized"]
        )
        self.assertEqual(self.contract["panel"]["attempts"], ATTEMPTS)
        self.assertEqual(self.contract["panel"]["start_ordinal"], 512)
        self.assertEqual(
            self.contract["panel"]["end_ordinal_inclusive"],
            767,
        )
        self.assertFalse(
            self.contract["development_evidence"]["development_panel_reused"]
        )
        self.assertEqual(self.contract["matrix"]["arms"], ARM_METHODS)
        self.assertTrue(all(self.contract["forbidden_actions"].values()))

    def test_local_authorization_does_not_expand_to_remote_execution(self) -> None:
        authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
        self.assertEqual(
            authorization["user_quote"],
            "开始吧，写成已给plan，按计划推进",
        )
        not_authorized = authorization["interpretation"]["not_authorized_yet"]
        self.assertTrue(any("transfer" in item for item in not_authorized))
        self.assertTrue(any("install" in item for item in not_authorized))
        self.assertTrue(any("submit" in item for item in not_authorized))
        self.assertTrue(any("train" in item for item in not_authorized))
        self.assertIn(
            (
                "user_wq_wyckoff_chart_retraction_confirmatory256_v1_"
                "local_preparation_2026-07-26"
            ),
            INSTALLER.read_text(encoding="utf-8"),
        )

    def test_contract_binds_every_inference_and_evaluation_source(self) -> None:
        for entry in self.contract["implementation"].values():
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), entry["path"])
            self.assertEqual(_sha256(path), entry["sha256"], entry["path"])

    def test_exact_statistics_use_one_paired_denominator(self) -> None:
        left = [True, True, False, False]
        right = [True, False, True, False]
        exact = exact_mcnemar(left, right)
        self.assertEqual(exact["attempts"], 4)
        self.assertEqual(exact["left_only"], 1)
        self.assertEqual(exact["right_only"], 1)
        self.assertEqual(exact["discordant"], 2)
        self.assertEqual(exact["two_sided_exact_p_value"], 1.0)
        effect = paired_binary_effect(
            [True] * 192 + [False] * 64,
            [True] * 160 + [False] * 96,
        )
        self.assertEqual(effect["attempts"], ATTEMPTS)
        self.assertEqual(effect["bootstrap"]["draws"], BOOTSTRAP_DRAWS)
        self.assertAlmostEqual(
            effect["difference_percentage_points"],
            12.5,
        )
        with self.assertRaisesRegex(ValueError, "10,000"):
            paired_binary_effect(left, right, draws=9999)
        numeric = paired_numeric_effect(
            [2.0, 3.0, 4.0],
            [1.0, 1.5, 2.0],
        )
        self.assertEqual(numeric["common_observed_pairs"], 3)
        self.assertAlmostEqual(numeric["mean_difference"], 1.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_numeric_effect([1.0, float("nan")], [1.0, 2.0])


class WTBConfirmatory256ExecutionSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.source_text = SOURCE_RUNNER.read_text(encoding="utf-8")
        cls.arms_text = ARMS_RUNNER.read_text(encoding="utf-8")
        cls.summary_text = SUMMARIZER.read_text(encoding="utf-8")
        cls.pipeline_text = PIPELINE.read_text(encoding="utf-8")
        cls.submit_text = SUBMIT.read_text(encoding="utf-8")

    def test_python_sources_parse_and_have_no_hidden_repeat_loops(self) -> None:
        for path, text in (
            (SOURCE_RUNNER, self.source_text),
            (ARMS_RUNNER, self.arms_text),
            (SUMMARIZER, self.summary_text),
        ):
            tree = ast.parse(text, filename=str(path))
            self.assertFalse(
                any(isinstance(node, ast.While) for node in ast.walk(tree)),
                str(path),
            )

    def test_generation_runners_cannot_submit_train_or_query_mp(self) -> None:
        for text in (self.source_text, self.arms_text):
            lowered = text.lower()
            for forbidden in (
                "sbatch",
                "scancel",
                "scontrol update",
                "mprester",
                "requests.",
                ".backward(",
                "optimizer.step",
            ):
                self.assertNotIn(forbidden, lowered)
        self.assertIn("one constrained WQ proposal", self.source_text)
        self.assertIn("Failures remain terminal rows", self.source_text)
        self.assertIn("failures remain in all downstream denominators", self.arms_text)

    def test_pipeline_is_one_gpu_eight_cpu_and_evaluation_only(self) -> None:
        resources = self.contract["resources"]
        self.assertEqual(resources["a800"], 1)
        self.assertEqual(resources["cpus"], 8)
        self.assertLessEqual(resources["cpus"], 8 * resources["a800"])
        self.assertIn("#SBATCH --partition=gpu", self.pipeline_text)
        self.assertIn("#SBATCH --cpus-per-task=8", self.pipeline_text)
        self.assertIn(
            "#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1",
            self.pipeline_text,
        )
        self.assertNotIn("#SBATCH --array", self.pipeline_text)
        self.assertIn("conda activate diff_meets_diff", self.pipeline_text)
        self.assertIn("run_crysllmgen_metrics.py", self.pipeline_text)
        self.assertIn("run_crysllmgen_a100_sun.py", self.pipeline_text)
        self.assertNotIn("train_crysllmgen", self.pipeline_text)
        self.assertNotIn("optimizer", self.pipeline_text.lower())
        self.assertIn('"training_performed": False', self.pipeline_text)
        self.assertIn('"automatic_followup_submission": False', self.pipeline_text)

    def test_submit_is_fail_closed_before_exclusive_claim(self) -> None:
        gate = self.submit_text.index("wtb256_preclaim_resource_gate=PASS")
        claim = self.submit_text.index('with Path(sys.argv[1]).open("x"')
        sbatch = self.submit_text.index("sbatch --parsable \\\n")
        self.assertLess(gate, claim)
        self.assertLess(claim, sbatch)
        self.assertIn(r"(\d+)\s*$", self.submit_text)
        self.assertNotIn(r"(\d+)\\s*$", self.submit_text)
        self.assertIn("cpus > 8 * gpus", self.submit_text)
        self.assertIn("submission_failed_no_retry", self.submit_text)
        self.assertIn("same WTB-256 job identity already exists", self.submit_text)
        self.assertNotIn("scancel", self.submit_text)
        self.assertNotIn("scontrol update", self.submit_text)
        self.assertIn('or "#SBATCH --array" in text', self.submit_text)
        self.assertNotIn("train_crysllmgen", self.submit_text)

    def test_pipeline_and_submit_bind_exact_contract_hash(self) -> None:
        contract_sha = _sha256(CONTRACT)
        self.assertIn(f"CONTRACT_SHA256={contract_sha}", self.pipeline_text)
        self.assertIn(f"CONTRACT_SHA256={contract_sha}", self.submit_text)

    def test_shells_are_bash_syntax_clean(self) -> None:
        for path in (PIPELINE, SUBMIT):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_summary_cannot_automatically_authorize_training(self) -> None:
        self.assertIn('"automatic_training_authorized": False', self.summary_text)
        self.assertIn('"training_performed": False', self.summary_text)
        self.assertIn('"training_free_promotion_to_l3"', self.summary_text)
        self.assertIn(
            '"design_separate_mlip_free_adapter_gate_and_request_authorization"',
            self.summary_text,
        )
        self.assertIn('"point_estimate_only"', self.summary_text)
        self.assertIn(
            "replicate-wise uniqueness is not approximated",
            self.summary_text,
        )


class WTBConfirmatory256SummaryFixtureTests(unittest.TestCase):
    def test_full_256_by_three_fixture_promotes_only_training_free_l3(
        self,
    ) -> None:
        contract_sha = _sha256(CONTRACT)
        patch_sha = "a" * 64
        cells = build_confirmatory_cells()
        counts = {
            "R": {"joint": 100, "strict": 20, "meta": 100, "novel": 126},
            "U": {"joint": 108, "strict": 24, "meta": 112, "novel": 128},
            "T": {"joint": 120, "strict": 32, "meta": 120, "novel": 128},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arms = root / "arms"
            evaluation = root / "evaluation"
            output = root / "summary"
            mechanics: list[dict] = []
            for arm, method in ARM_METHODS.items():
                generation: list[dict] = []
                direct: list[dict] = []
                sun: list[dict] = []
                for index, cell in enumerate(cells):
                    attempt_id = cell.arm_attempt_ids[arm]
                    generation.append(
                        {
                            "schema": "wqcodiff_generation_attempt_v1",
                            "identity": IDENTITY,
                            "arm": arm,
                            "method": method,
                            "attempt_id": attempt_id,
                            "source_attempt_id": cell.source_attempt_id,
                            "pair_id": cell.pair_id,
                            "ordinal": cell.ordinal,
                            "forward_noise_seed": cell.forward_noise_seed,
                            "reverse_noise_seed": cell.reverse_noise_seed,
                            "training_seed": 11,
                            "sampling_seed": 101,
                            "contract_sha256": contract_sha,
                            "execution_patch_sha256": patch_sha,
                            "retry_or_replacement_used": False,
                            "best_of_or_rerank_used": False,
                            "status": "succeeded",
                            "composition_signature": "c" * 64,
                            "volume": 16.0,
                            "minimum_pair_distance_angstrom": (
                                1.5 + 0.01 * index
                            ),
                            "volume_per_atom_angstrom3": 16.0,
                            "density_g_cm3": 3.0,
                            "collision_free_at_0p5_angstrom": True,
                            "structure": {"synthetic": True},
                        }
                    )
                    joint = index < counts[arm]["joint"]
                    direct.append(
                        {
                            "schema": "crysllmgen_metric_attempt_v1",
                            "method": method,
                            "attempt_id": attempt_id,
                            "comp_valid": joint,
                            "struct_valid": joint,
                            "valid": joint,
                        }
                    )
                    novel = index < counts[arm]["novel"]
                    sun.append(
                        {
                            "schema": "crysllmgen_r5c_a100_sun_attempt_v1",
                            "method": method,
                            "attempt_id": attempt_id,
                            "execution_patch_sha256": patch_sha,
                            "retry_or_replacement_used": False,
                            "metrics": {
                                "novel": novel,
                                "unique_representative": novel,
                                "novel_unique": novel,
                                "strict_full_sun": index
                                < counts[arm]["strict"],
                                "meta_full_sun": index < counts[arm]["meta"],
                            },
                        }
                    )
                    is_t = arm == "T"
                    mechanics.append(
                        {
                            "schema": (
                                "wq_wyckoff_chart_retraction_arm_mechanics_v1"
                            ),
                            "identity": IDENTITY,
                            "arm": arm,
                            "method": method,
                            "attempt_id": attempt_id,
                            "status": "succeeded",
                            "parent_decoder_calls": 64 if arm != "R" else 0,
                            "projection_calls": 64 if is_t else 0,
                            "details": {
                                "topology_hash_unchanged": True,
                                "mechanics": {
                                    "topology_hash_unchanged": True,
                                    "lattice_projection_methods": (
                                        ["global_chart_retraction_v1"]
                                        if is_t
                                        else []
                                    ),
                                    (
                                        "all_chart_retraction_audit_values_finite"
                                    ): True,
                                },
                            },
                        }
                    )

                generation_path = arms / f"{arm.lower()}_generation.jsonl"
                direct_path = (
                    evaluation / arm / "crysllmgen_metrics" / "attempt_metrics.jsonl"
                )
                sun_path = (
                    evaluation / arm / "r5c_a100_sun" / "attempt_results.jsonl"
                )
                _write_jsonl(generation_path, generation)
                _write_jsonl(direct_path, direct)
                _write_jsonl(sun_path, sun)
                _write_json(
                    direct_path.with_name("report.json"),
                    {
                        "schema": "crysllmgen_generation_metrics_report_v1",
                        "ok": True,
                        "method": method,
                        "attempts": ATTEMPTS,
                        "denominator": "all_generation_attempts",
                        "generation_jsonl_sha256": _sha256(generation_path),
                        "attempt_metrics_sha256": _sha256(direct_path),
                        "retry_or_replacement_used": False,
                    },
                )
                _write_json(
                    sun_path.with_name("attempt_summary.json"),
                    {
                        "schema": "crysllmgen_r5c_a100_sun_summary_v1",
                        "ok": True,
                        "method": method,
                        "denominator": "all_generation_attempts",
                        "counts": {"total_attempts": ATTEMPTS},
                        "execution_patch_sha256": patch_sha,
                        "coverage_adjusted_selection_role": (
                            "report_only_never_checkpoint_selection"
                        ),
                        "retry_or_replacement_used": False,
                    },
                )

            _write_jsonl(arms / "arm_mechanics.jsonl", mechanics)
            _write_jsonl(arms / "t_trajectory_evidence.jsonl", [])
            _write_json(
                arms / "arms_report.json",
                {
                    "schema": "wq_wyckoff_chart_retraction_arms_report_v1",
                    "ok": True,
                    "acceptance": "PASS",
                    "contract_sha256": contract_sha,
                    "execution_patch_sha256": patch_sha,
                    "attempts_per_arm": ATTEMPTS,
                    "retry_or_replacement_used": False,
                },
            )
            args = argparse.Namespace(
                contract=CONTRACT.resolve(),
                arms_dir=arms.resolve(),
                evaluation_dir=evaluation.resolve(),
                output_dir=output.resolve(),
            )
            result = summarize_confirmatory(args, output.resolve())
            self.assertTrue(result["integrity_pass"])
            self.assertTrue(result["scientific_promotion_pass"])
            self.assertEqual(
                result["decision"],
                "training_free_promotion_to_l3",
            )
            self.assertEqual(
                result["arm_results"]["T"]["joint_valid"]["count"],
                120,
            )
            self.assertEqual(
                result["failure_taxonomy"]["T"]["terminal_categories"][
                    "joint_valid"
                ],
                120,
            )
            self.assertEqual(
                result["failure_taxonomy"]["T"]["terminal_categories"][
                    "composition_and_structure_invalid"
                ],
                136,
            )
            self.assertEqual(
                result["paired_geometry_descriptors"]["T_vs_U"][
                    "minimum_pair_distance_angstrom"
                ]["common_observed_pairs"],
                ATTEMPTS,
            )
            self.assertGreaterEqual(
                result["paired_comparisons"]["T_vs_R"]["joint_valid"][
                    "difference_percentage_points"
                ],
                3.0,
            )
            self.assertEqual(
                result["paired_comparisons"]["T_vs_U"]["novel_unique"][
                    "inference"
                ],
                "point_estimate_only",
            )
            lock = json.loads(
                (output / "promotion_lock.json").read_text(encoding="utf-8")
            )
            self.assertFalse(lock["automatic_training_authorized"])
            self.assertFalse(lock["retry_or_replacement_allowed"])
            self.assertEqual(
                lock["allowed_next_action"],
                "prepare_separate_multiseed_1000_contract",
            )
            with self.assertRaises(FileExistsError):
                summarize_confirmatory(args, output.resolve())


if __name__ == "__main__":
    unittest.main()
