from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[3]
for location in (ROOT, PROJECT_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from assemble_terminal import _exact_mcnemar  # noqa: E402
from protocol import (  # noqa: E402
    canonical_sha256,
    plan_body_eligible,
    require_source_manifest,
    write_json_exclusive,
    write_jsonl_exclusive,
)


class ProtocolTests(unittest.TestCase):
    def test_config_freezes_two_arm_all_attempt_screen(self) -> None:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["panel"]["attempts_per_arm"], 256)
        self.assertEqual(config["panel"]["sample_idx_start"], 0)
        self.assertEqual(config["panel"]["sample_idx_end_inclusive"], 255)
        self.assertEqual(
            config["panel"]["expected_body_eligible_by_arm"],
            {"P0": 254, "P1": 253},
        )
        self.assertTrue(
            config["source_plan_run"]["arms"]["P1"]["planner_checkpoint"].endswith(
                "checkpoint-000050"
            )
        )
        self.assertEqual(
            config["source_plan_run"]["arms"]["P0"]["planner_model"],
            config["source_plan_run"]["arms"]["P1"]["planner_model"],
        )
        self.assertEqual(
            config["panel"]["failure_denominator"], "all_registered_attempts"
        )
        for key in ("retry", "replacement", "repair", "filter", "rerank"):
            self.assertIs(config["panel"][key], False)
        self.assertIs(config["panel"]["sample_id_in_prompt"], False)
        self.assertIn(
            "treatment-dependent",
            config["panel"]["treatment_boundary"],
        )
        self.assertNotIn(
            "common Planner prompt",
            " ".join(config["panel"]["pair_on"]),
        )
        self.assertIs(
            config["decision_firewall"]["automatic_downstream_authorized"], False
        )
        self.assertIs(
            config["decision_firewall"]["automatic_promotion_authorized"], False
        )
        self.assertIs(
            config["authorization"]["formal_jointchem_promotion_override"], False
        )
        self.assertIs(
            config["authorization"]["manual_crystal_evaluation_authorized"], True
        )
        self.assertIs(
            config["authorization"][
                "manual_authorization_includes_afterok_sun_evaluation"
            ],
            True,
        )
        self.assertIs(
            config["authorization"]["automatic_crystal_evaluation_authorized"], False
        )
        self.assertEqual(config["body"]["adapter_bytes"], 6_391_016_776)
        self.assertEqual(len(config["body"]["adapter_sha256"]), 64)
        self.assertTrue(
            all(
                character in "0123456789abcdef"
                for character in config["body"]["adapter_sha256"]
            )
        )
        self.assertEqual(config["parent_refiner"]["diffusion_steps"], 800)
        self.assertEqual(config["parent_refiner"]["maximum_atoms_for_common_noise"], 20)

    def test_plan_ineligibility_is_terminal_not_replaceable(self) -> None:
        self.assertEqual(
            plan_body_eligible({"parsed": False, "plan_state": None}),
            (False, "planner_parse_failed"),
        )
        valid = {
            "parsed": True,
            "plan_state": {"N": 3, "elements": ["Li", "O"], "counts": [2, 1]},
        }
        self.assertEqual(plan_body_eligible(valid), (True, ""))
        invalid = {
            "parsed": True,
            "plan_state": {"N": 4, "elements": ["Li", "O"], "counts": [2, 1]},
        }
        self.assertEqual(
            plan_body_eligible(invalid),
            (False, "planner_composition_shape_invalid"),
        )

    def test_exclusive_writers_reject_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json_exclusive(root / "one.json", {"x": 1})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(root / "one.json", {"x": 2})
            write_jsonl_exclusive(root / "rows.jsonl", [{"x": 1}, {"x": 2}])
            with self.assertRaises(FileExistsError):
                write_jsonl_exclusive(root / "rows.jsonl", [{"x": 3}])

    def test_source_manifest_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "SOURCE_SHA256.txt"
            manifest.write_text("0" * 64 + "  ../escape\n", encoding="utf-8")
            manifest_sha = (
                __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
            )
            with self.assertRaises(ValueError):
                require_source_manifest(root, manifest_sha)

    def test_source_manifest_rejects_unlisted_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            listed = root / "listed.txt"
            listed.write_text("listed\n", encoding="utf-8")
            (root / "extra.txt").write_text("extra\n", encoding="utf-8")
            listed_sha = __import__("hashlib").sha256(listed.read_bytes()).hexdigest()
            manifest = root / "SOURCE_SHA256.txt"
            manifest.write_text(f"{listed_sha}  listed.txt\n", encoding="utf-8")
            manifest_sha = (
                __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
            )
            with self.assertRaises(ValueError):
                require_source_manifest(root, manifest_sha)

    def test_exact_mcnemar_direction_and_tie(self) -> None:
        tied = _exact_mcnemar([True, False], [True, False])
        self.assertEqual(tied["discordant"], 0)
        self.assertEqual(tied["two_sided_exact_p_value"], 1.0)
        directional = _exact_mcnemar(
            [True, True, True, False],
            [False, False, True, False],
        )
        self.assertEqual(directional["candidate_only"], 2)
        self.assertEqual(directional["baseline_only"], 0)
        self.assertEqual(directional["discordant"], 2)

    def test_generation_uses_semantic_body_and_three_parent_noise_roles(self) -> None:
        llada = (ROOT / "paired_llada.py").read_text(encoding="utf-8")
        runner = (ROOT / "run_paired_body_refine.py").read_text(encoding="utf-8")
        self.assertIn("body_gumbel_suffix_group_", llada)
        self.assertIn("step_in_group=step_in_group", llada)
        for role in (
            'role="coord_corrector"',
            'role="coord_predictor"',
            'role="lattice_predictor"',
        ):
            self.assertIn(role, runner)
        self.assertIn('"structure": structure', runner)
        self.assertIn('"retry_or_replacement_used": False', runner)

    def test_slurm_dag_respects_cpu_per_a800_and_no_followup(self) -> None:
        generation = (ROOT / "slurm" / "generate.sbatch").read_text(encoding="utf-8")
        evaluation = (ROOT / "slurm" / "evaluate.sbatch").read_text(encoding="utf-8")
        submission = (ROOT / "submit_once.sh").read_text(encoding="utf-8")
        for text in (generation, evaluation):
            self.assertIn("#SBATCH --cpus-per-task=8", text)
            self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", text)
            self.assertIn("#SBATCH --array=0-1%2", text)
            self.assertIn("A800", text)
        self.assertIn('--dependency="afterok:$data_job"', submission)
        self.assertIn('--dependency="afterok:$generation_job"', submission)
        self.assertIn('--dependency="afterok:$evaluation_job"', submission)
        self.assertNotIn("scancel", submission)
        self.assertNotIn("1000", submission)

    def test_canonical_hash_is_order_independent(self) -> None:
        self.assertEqual(
            canonical_sha256({"b": 2, "a": 1}),
            canonical_sha256({"a": 1, "b": 2}),
        )


if __name__ == "__main__":
    unittest.main()
