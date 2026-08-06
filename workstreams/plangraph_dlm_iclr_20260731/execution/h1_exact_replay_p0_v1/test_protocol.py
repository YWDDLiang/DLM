#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


class H1ExactReplayProtocolTests(unittest.TestCase):
    def test_authorization_is_zero_change_and_no_downstream(self) -> None:
        authorization = json.loads(
            (HERE / "AUTHORIZATION.json").read_text(encoding="utf-8")
        )
        scope = authorization["scope"]
        self.assertTrue(scope["frozen_h1_p0_replay"])
        self.assertEqual(scope["arms"], ["P0"])
        self.assertEqual(scope["attempts"], 256)
        self.assertEqual(scope["diffusion_reverse_steps"], 800)
        for key in (
            "direct_metrics",
            "sun",
            "training",
            "checkpoint_reselection",
            "planner_change",
            "body_checkpoint_change",
            "body_schedule_change",
            "automatic_downstream",
        ):
            self.assertFalse(scope[key])

    def test_execution_reuses_frozen_h1_source_and_output_comparison(self) -> None:
        manifest = json.loads(
            (HERE / "EXECUTION_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["scientific_changes"], [])
        self.assertEqual(manifest["frozen_input"]["planner_arm"], "P0")
        self.assertEqual(manifest["frozen_input"]["attempts"], 256)
        self.assertEqual(manifest["frozen_input"]["reverse_steps"], 800)
        self.assertFalse(manifest["sun"])
        script = (HERE / "replay_p0.sbatch").read_text(encoding="utf-8")
        self.assertIn("run_paired_body_refine.py", script)
        self.assertIn("--arm P0", script)
        self.assertIn("compare_replay.py", script)
        self.assertNotIn("run_crysllmgen_metrics.py", script)
        self.assertNotIn("run_crysllmgen_a100_sun.py", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
