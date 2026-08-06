#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path

from assemble_terminal import _interaction, _paired_effect
from protocol import (
    ARM_COMPONENTS,
    ARM_ORDER,
    attempt_id,
    require_runtime_manifest,
    sha256_file,
    validate_arm,
)


HERE = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def test_registered_arms(self) -> None:
        self.assertEqual(ARM_ORDER, ("M00", "M10", "M01", "M11"))
        self.assertEqual(ARM_COMPONENTS["M11"], ("Pstar", "B2"))

    def test_attempt_id_is_ordinal_stable(self) -> None:
        self.assertEqual(
            attempt_id("M01", 7),
            "h1a2-v3-poststop-sun256:0007:M01",
        )
        with self.assertRaises(ValueError):
            validate_arm("shuffle")

    def test_config_is_diagnostic_and_refines_every_arm(self) -> None:
        config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        authorization = json.loads(
            (HERE / "AUTHORIZATION.json").read_text(encoding="utf-8")
        )
        self.assertFalse(config["decision_firewall"]["formal_g3"])
        self.assertFalse(config["decision_firewall"]["automatic_downstream"])
        self.assertEqual(config["refiner"]["diffusion_steps"], 800)
        self.assertEqual(set(config["arms"]), set(ARM_ORDER))
        self.assertTrue(
            authorization["scope"]["diffusion_refinement_required_for_every_arm"]
        )
        self.assertEqual(authorization["scope"]["diffusion_reverse_steps"], 800)

    def test_isolated_runtime_snapshot_matches_manifest(self) -> None:
        manifest = require_runtime_manifest(HERE.parents[3], HERE)
        self.assertTrue((HERE / "runtime/crystal_dlm/__init__.py").is_file())
        self.assertEqual(len(sha256_file(manifest)), 64)

    def test_packed_arm_orders_refinement_before_all_evaluation(self) -> None:
        script = (HERE / "arm_pipeline.sbatch").read_text(encoding="utf-8")
        stages = [
            "prepare_plans.py",
            "sample_llada_h1a2_factorial_body.py",
            "refine_h1a2_factorial_with_crysllmgen.py",
            "finalize_generation.py",
            "run_crysllmgen_metrics.py",
            "run_crysllmgen_a100_sun.py",
            "validate_evaluation.py",
        ]
        offsets = [script.index(stage) for stage in stages]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("#SBATCH --array=0-3%2", script)
        self.assertIn("--time=1-12:00:00", script)

    def test_paired_effect_uses_all_attempts(self) -> None:
        baseline = [False] * 256
        candidate = [True] * 16 + [False] * 240
        effect = _paired_effect(
            candidate,
            baseline,
            candidate_arm="M11",
            baseline_arm="M00",
            seed_offset=0,
        )
        self.assertEqual(effect["attempts"], 256)
        self.assertEqual(effect["difference_percentage_points"], 6.25)

    def test_factorial_interaction(self) -> None:
        vectors = {
            "M00": [False] * 256,
            "M10": [False] * 256,
            "M01": [False] * 256,
            "M11": [True] * 256,
        }
        result = _interaction(vectors, seed_offset=1)
        self.assertEqual(result["interaction_percentage_points"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
