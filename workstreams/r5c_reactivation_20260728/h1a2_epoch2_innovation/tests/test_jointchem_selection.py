from __future__ import annotations

from pathlib import Path
import sys
import unittest


THIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = THIS_DIR.parent / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from select_jointchem import select_checkpoint, select_plan_arm  # noqa: E402


def checkpoint_event(step, *, positive, chemistry, joint, loss):
    return {
        "step": step,
        "checkpoint_dir": f"/checkpoint-{step}",
        "checkpoint_manifest_sha256": f"{step:064x}"[-64:],
        "eval": {
            "positive_nll": positive,
            "chemistry_negative_nll": chemistry,
            "joint_negative_nll": joint,
            "chemistry_paired_margin": chemistry - positive,
            "joint_paired_margin": joint - positive,
            "loss": loss,
        },
    }


def training_report(arm, checkpoints, *, initial_positive=1.0):
    return {
        "ok": True,
        "global_step": 400,
        "arm": arm,
        "initial_epoch2_eval": {"positive_nll": initial_positive},
        "checkpoints": checkpoints,
    }


def plan_report(identity, arm, *, comp, all_metal=0.25, mean_n=11.0, tvd=0.1):
    comparison = {
        key: tvd
        for key in (
            "n_tvd",
            "num_elements_tvd",
            "element_presence_tvd",
            "anion_framework_tvd",
            "charge_bucket_tvd",
            "lattice_system_tvd",
            "spacegroup_bucket_tvd",
            "volume_per_atom_bin_tvd",
        )
    }
    return {
        "identity": identity,
        "arm": arm,
        "step": 200,
        "denominator": 512,
        "composition": {
            "parse_rate": 0.99,
            "composition_valid_rate": comp,
            "anion_match_rate": 0.99,
            "charge_match_rate": 0.99,
            "all_metal_rate": all_metal,
        },
        "generated_distribution": {"mean_N": mean_n},
        "distribution_comparison": comparison,
    }


class JointChemSelectionTests(unittest.TestCase):
    def test_checkpoint_selection_requires_positive_margins_for_jointchem(self):
        checkpoints = []
        for step in range(50, 401, 50):
            if step == 100:
                checkpoints.append(
                    checkpoint_event(
                        step,
                        positive=1.0,
                        chemistry=1.2,
                        joint=1.1,
                        loss=0.8,
                    )
                )
            else:
                checkpoints.append(
                    checkpoint_event(
                        step,
                        positive=1.0,
                        chemistry=0.9,
                        joint=1.1,
                        loss=0.7,
                    )
                )
        result = select_checkpoint(
            training_report("jointchem", checkpoints)
        )
        self.assertEqual(result["selected"]["step"], 100)

    def test_checkpoint_selection_uses_direct_paired_margin_not_difference_of_means(self):
        checkpoints = []
        for step in range(50, 401, 50):
            event = checkpoint_event(
                step,
                positive=1.0,
                chemistry=0.8,
                joint=0.8,
                loss=0.9,
            )
            if step == 150:
                event["eval"]["chemistry_paired_margin"] = 0.15
                event["eval"]["joint_paired_margin"] = 0.10
                event["eval"]["loss"] = 0.7
            checkpoints.append(event)
        result = select_checkpoint(training_report("jointchem", checkpoints))
        self.assertEqual(result["selected"]["step"], 150)

    def test_valid_replay_chooses_lowest_positive_nll(self):
        checkpoints = [
            checkpoint_event(
                step,
                positive=1.0 - step / 10000,
                chemistry=1.2,
                joint=1.1,
                loss=1.0,
            )
            for step in range(50, 401, 50)
        ]
        result = select_checkpoint(
            training_report("valid_replay", checkpoints)
        )
        self.assertEqual(result["selected"]["step"], 400)

    def test_plan_selection_accepts_comp_gain_without_tvd_or_metal_gaming(self):
        baseline = plan_report("P0", "baseline", comp=0.88)
        candidate = plan_report("P2", "jointchem", comp=0.91)
        checkpoint = {
            "selected": {
                "margins": {"chemistry": 0.2, "joint": 0.1},
                "nll_noninferior": True,
            }
        }
        result = select_plan_arm(baseline, [(candidate, checkpoint)])
        self.assertEqual(result["decision"], "selected_for_paired256_crystal_screen")
        self.assertEqual(result["selected"]["identity"], "P2")

    def test_plan_selection_rejects_all_metal_inflation(self):
        baseline = plan_report("P0", "baseline", comp=0.88, all_metal=0.25)
        candidate = plan_report("P2", "jointchem", comp=0.92, all_metal=0.28)
        checkpoint = {
            "selected": {
                "margins": {"chemistry": 0.2, "joint": 0.1},
                "nll_noninferior": True,
            }
        }
        result = select_plan_arm(baseline, [(candidate, checkpoint)])
        self.assertEqual(result["decision"], "stop_no_plan_candidate")
        self.assertIn("all_metal_inflation_above_2pp", result["candidates"][0]["reasons"])

    def test_checkpoint_selection_rejects_nll_regression_vs_epoch2(self):
        checkpoints = [
            checkpoint_event(
                step,
                positive=1.02,
                chemistry=1.3,
                joint=1.2,
                loss=0.8,
            )
            for step in range(50, 401, 50)
        ]
        result = select_checkpoint(
            training_report("jointchem", checkpoints, initial_positive=1.0)
        )
        self.assertEqual(result["decision"], "no_eligible_checkpoint")
        self.assertTrue(
            all(
                "positive_nll_degradation_above_1pct" in value["reasons"]
                for value in result["candidates"]
            )
        )

    def test_plan_selection_rejects_missing_margin(self):
        baseline = plan_report("P0", "baseline", comp=0.88)
        candidate = plan_report("P2", "jointchem", comp=0.91)
        checkpoint = {
            "selected": {
                "margins": {"chemistry": 0.2, "joint": None},
                "nll_noninferior": True,
            }
        }
        result = select_plan_arm(baseline, [(candidate, checkpoint)])
        self.assertEqual(result["decision"], "stop_no_plan_candidate")
        self.assertIn(
            "joint_likelihood_margin_missing",
            result["candidates"][0]["reasons"],
        )

    def test_plan_selection_rejects_nll_noninferiority_failure(self):
        baseline = plan_report("P0", "baseline", comp=0.88)
        candidate = plan_report("P2", "jointchem", comp=0.91)
        checkpoint = {
            "selected": {
                "margins": {"chemistry": 0.2, "joint": 0.1},
                "nll_noninferior": False,
            }
        }
        result = select_plan_arm(baseline, [(candidate, checkpoint)])
        self.assertEqual(result["decision"], "stop_no_plan_candidate")
        self.assertIn(
            "validation_nll_not_noninferior_to_epoch2",
            result["candidates"][0]["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
