#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
for location in (RUNTIME, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from assemble_schedule32 import (  # noqa: E402
    output_agreement,
    paired_identity_mismatches,
    summarize,
)
from crystal_dlm.dynamic_crystal import (  # noqa: E402
    arrays_to_dynamic_answer,
    parse_dynamic_answer,
)
from crystal_dlm.planned_corruption import h1a2_generation_schedule  # noqa: E402
from crystal_dlm.r5_dynamic_length import (  # noqa: E402
    exact_body_token_count,
    exact_dynamic_generation_schedule,
)
from crystal_dlm.r5_plan_state import plan_state_from_arrays  # noqa: E402
from safe_axis_schedule import (  # noqa: E402
    analyze_axis_schedule,
    h1a2_safe_axis_generation_schedule,
    require_safe_axis_schedule,
)


class H1BodySafeAxis32ProtocolTests(unittest.TestCase):
    def test_frozen_h1_body_runtime_is_byte_exact(self) -> None:
        identity = json.loads(
            (HERE / "H1_BODY_RUNTIME_SHA256.json").read_text(encoding="utf-8")
        )
        for relative, expected in identity["files"].items():
            observed = hashlib.sha256((HERE / relative).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, relative)

    def test_authorization_and_config_change_only_schedule(self) -> None:
        authorization = json.loads(
            (HERE / "AUTHORIZATION.json").read_text(encoding="utf-8")
        )
        scope = authorization["scope"]
        self.assertEqual(scope["principal_variable"], "body_generation_schedule_only")
        self.assertEqual(scope["attempts"], 32)
        self.assertTrue(scope["frozen_h1_p0_plans"])
        self.assertTrue(scope["frozen_b0_checkpoint"])
        for key in (
            "training",
            "planner_sampling",
            "body_checkpoint_change",
            "diffusion_refinement",
            "direct_metrics",
            "sun",
            "checkpoint_reselection",
            "promotion",
            "automatic_downstream",
        ):
            self.assertFalse(scope[key])

        config = json.loads((HERE / "CONFIG.json").read_text(encoding="utf-8"))
        self.assertEqual(config["denominator"], 32)
        self.assertEqual(config["treatment"]["control_policy"], "d1")
        self.assertEqual(config["treatment"]["candidate_policy"], "d2_safe_axis")
        self.assertTrue(config["treatment"]["all_xy_must_precede_all_z"])
        self.assertFalse(
            config["treatment"]["mixed_axis_coordinate_groups_allowed"]
        )
        self.assertEqual(
            config["treatment"]["required_z_before_xy_count"],
            0,
        )
        self.assertEqual(config["body"]["max_batch_size"], 8)
        self.assertTrue(config["body"]["exact_length_generation"])
        self.assertEqual(config["body"]["answer_token_count_formula"], "7+4N")
        self.assertFalse(config["retry"])
        self.assertFalse(config["replacement"])
        self.assertFalse(config["repair"])
        self.assertFalse(config["filter"])
        self.assertFalse(config["rerank"])
        self.assertEqual(
            config["gate"]["control_completion_drop_vs_historical_max_count"],
            0,
        )

    def test_safe_axis_is_inference_available_and_d1_is_h1_exact(self) -> None:
        answer, _ = arrays_to_dynamic_answer(
            lengths=[3.1, 3.1, 5.2],
            angles=[90.0, 90.0, 120.0],
            species=["Li", "O", "Li"],
            frac_coords=[
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
            ],
        )
        arrays = parse_dynamic_answer(answer, strict=True)
        plan = plan_state_from_arrays(
            arrays,
            metadata={"spacegroup.number": 194},
        )
        d1 = h1a2_generation_schedule(plan, policy="d1")
        safe_axis = h1a2_safe_axis_generation_schedule(plan)
        self.assertEqual(d1, exact_dynamic_generation_schedule(plan["N"]))
        self.assertNotEqual(d1, safe_axis)
        self.assertEqual(
            sorted(position for group in safe_axis for position in group),
            list(range(7 + 4 * int(plan["N"]))),
        )
        report = require_safe_axis_schedule(
            safe_axis,
            num_atoms=int(plan["N"]),
        )
        self.assertTrue(report["gate_passed"])
        self.assertTrue(report["all_xy_precede_all_z"])
        self.assertEqual(report["z_before_xy_count"], 0)
        self.assertEqual(report["mixed_axis_coordinate_groups"], 0)

    def test_safe_axis_preserves_r5c_exact_length_for_n1_to_n20(self) -> None:
        for num_atoms in range(1, 21):
            formula = "Li" if num_atoms == 1 else f"Li{num_atoms}"
            plan = {
                "N": num_atoms,
                "elements": ["Li"],
                "counts": [num_atoms],
                "formula": formula,
            }
            expected_length = 7 + 4 * num_atoms
            self.assertEqual(exact_body_token_count(plan), expected_length)
            d1 = exact_dynamic_generation_schedule(num_atoms)
            safe_axis = h1a2_safe_axis_generation_schedule(plan)
            for schedule in (d1, safe_axis):
                flattened = [
                    position for group in schedule for position in group
                ]
                self.assertEqual(len(flattened), expected_length)
                self.assertEqual(
                    sorted(flattened),
                    list(range(expected_length)),
                )

    def test_mixed_axis_schedule_fails_before_model_work(self) -> None:
        num_atoms = 3
        mixed = [
            [0, 7, 11, 15],
            [1, 2, 3, 4, 5, 6],
            [8, 9, 10, 12, 13, 14, 16, 17, 18],
        ]
        report = analyze_axis_schedule(mixed, num_atoms=num_atoms)
        self.assertFalse(report["gate_passed"])
        self.assertGreater(report["z_before_xy_count"], 0)
        self.assertGreater(report["mixed_axis_coordinate_groups"], 0)
        with self.assertRaisesRegex(ValueError, "safe-axis schedule invariant"):
            require_safe_axis_schedule(mixed, num_atoms=num_atoms)

    def test_paired_identity_and_failure_summary_are_fail_closed(self) -> None:
        control = [
            {
                "ordinal": 0,
                "attempt_id": "a",
                "pair_id": "p",
                "body_noise_seed": 7,
                "plan_state_sha256": "x",
                "status": "succeeded",
                "reason": "",
            },
            {
                "ordinal": 1,
                "attempt_id": "b",
                "pair_id": "q",
                "body_noise_seed": 8,
                "plan_state_sha256": "y",
                "status": "failed",
                "reason": "body:FixedSlotError:duplicate coordinate",
            },
        ]
        candidate = [dict(row) for row in control]
        self.assertEqual(paired_identity_mismatches(control, candidate), [])
        self.assertEqual(summarize(control)["duplicate_coordinate_failures"], 1)
        self.assertEqual(output_agreement(control, candidate)["status_matches"], 2)
        candidate[0]["body_noise_seed"] = 99
        self.assertEqual(
            paired_identity_mismatches(control, candidate)[0]["field"],
            "body_noise_seed",
        )

    def test_single_slurm_job_has_no_forbidden_downstream(self) -> None:
        script = (HERE / "schedule32.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1", script)
        self.assertNotIn("#SBATCH --array", script)
        self.assertIn("run_schedule32.py", script)
        self.assertIn("assemble_schedule32.py", script)
        runner = (HERE / "run_schedule32.py").read_text(encoding="utf-8")
        self.assertIn("shared_batch_partition_applied_identically", runner)
        self.assertIn('("control", "d1", "control_schedule")', runner)
        self.assertIn(
            '("candidate", "d2_safe_axis", "candidate_schedule")',
            runner,
        )
        self.assertIn("schedule_invariants.json", runner)
        self.assertIn("all_candidate_invariants_passed", runner)
        self.assertNotIn("refine_", script)
        self.assertNotIn("run_crysllmgen_metrics", script)
        self.assertNotIn("run_crysllmgen_a100_sun", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
