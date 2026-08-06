from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import assemble_planner512
from evaluate_planner_arm import validate_sampler_config


TVD_KEYS = assemble_planner512.TVD_KEYS


def arm_report(arm: str, *, comp: float, unique: float = 0.9) -> dict:
    checkpoint = assemble_planner512.EXPECTED_CHECKPOINTS[arm]
    audit = {
        "parse_rate": 1.0,
        "completion_rate": 1.0,
        "composition_valid_rate": comp,
        "unique_formula_rate": unique,
        "mean_N": 10.0,
        "all_metal_rate": 0.2,
        "single_element_rate": 0.01,
    }
    return {
        "schema": "h1a2_v3_planner512_arm_report_v1",
        "status": "complete",
        "arm": arm,
        "denominator": 512,
        "checkpoint_identity_sha256": checkpoint,
        "attempt_audit": audit,
        "distribution_comparison": {key: 0.1 for key in TVD_KEYS},
    }


class Planner512Tests(unittest.TestCase):
    def test_sampler_contract(self) -> None:
        validate_sampler_config(
            {
                "num_samples": 512,
                "max_new_tokens": 96,
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 50,
                "max_atoms": 20,
                "prompt_style": "h1_rich_plan_v1",
                "include_sample_id": False,
                "seed": 17029,
                "seed_mode": "stateless_ordinal_v1",
                "rank_independent_sampling": True,
                "effective_generation_batch_size": 1,
            }
        )

    def _run_assembly(self, pstar_comp: float) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm, comp in (
                ("P0", 0.90),
                ("P-control", 0.91),
                ("P-star", pstar_comp),
            ):
                path = root / "arms" / arm / "plan_report.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(arm_report(arm, comp=comp)) + "\n")
            training = {
                "status": "complete",
                "training_complete": True,
                "selection_gate_passed": True,
                "p0_initial_target_nll": 0.294,
                "selections": {
                    "pstar": {
                        "selected": {"target_nll": 0.291},
                        "candidates": [
                            {
                                "step": 400,
                                "checkpoint_manifest_sha256": assemble_planner512.EXPECTED_CHECKPOINTS[
                                    "P-star"
                                ],
                                "target_nll": 0.292,
                            }
                        ],
                    }
                },
            }
            training_path = root / "training.json"
            training_path.write_text(json.dumps(training) + "\n")
            output = root / "terminal.json"
            argv = [
                "assemble_planner512.py",
                "--run-root",
                str(root),
                "--planner-training-terminal",
                str(training_path),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(assemble_planner512.main(), 0)
            return json.loads(output.read_text())

    def test_pass(self) -> None:
        terminal = self._run_assembly(0.93)
        self.assertTrue(terminal["planner_gate_passed"])
        self.assertEqual(
            terminal["decision"],
            "select_Pstar_for_future_authorized_body_evaluation",
        )

    def test_scientific_stop(self) -> None:
        terminal = self._run_assembly(0.915)
        self.assertFalse(terminal["planner_gate_passed"])
        self.assertEqual(terminal["decision"], "scientific_stop_retain_P0")


if __name__ == "__main__":
    unittest.main()
