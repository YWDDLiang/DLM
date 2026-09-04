from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.build_spad_basin_preflight_actions import (  # noqa: E402
    ATTEMPT_SOURCES,
    retain_fixed_order_actions,
    summarize_groups,
)


def attempt(source, ids, *, legal=True, status="supplied"):
    return {
        "source": source,
        "action_token_ids": list(ids),
        "action_tokens": [str(value) for value in ids],
        "legal_supplied_action": legal,
        "legality_reason": None if legal else "synthetic_invalid",
        "status": status,
        "reason": "synthetic",
        "step": None,
        "proposal_status": status,
        "proposal_reason": "synthetic",
        "proposal_step": None,
    }


class FixedOrderRetentionTest(unittest.TestCase):
    def test_deduplicates_in_fixed_order_without_replacement(self):
        attempts = [
            attempt("no_op", (10, 11, 12)),
            attempt("reference_dlm", (20, 21, 22)),
            attempt("physics_downhill", (20, 21, 22), status="accepted"),
            attempt("physics_reverse", (30, 31, 32), legal=False, status="invalid"),
        ]
        retained, audited = retain_fixed_order_actions(attempts)
        self.assertEqual([row["source"] for row in retained], ["no_op", "reference_dlm"])
        self.assertEqual(
            [row["retention_status"] for row in audited],
            ["retained", "retained", "duplicate", "invalid"],
        )
        self.assertEqual(audited[2]["retention_reason"], "same_action_as:reference_dlm")

    def test_retains_zero_headroom_k1_group(self):
        attempts = [
            attempt("no_op", (10, 11, 12)),
            attempt("reference_dlm", (10, 11, 12)),
            attempt("physics_downhill", (), legal=False, status="invalid"),
            attempt("physics_reverse", (), legal=False, status="invalid"),
        ]
        retained, audited = retain_fixed_order_actions(attempts)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["source"], "no_op")
        self.assertEqual(len(audited), 4)

    def test_source_and_action_accounting(self):
        group = {
            "candidate_attempts": [
                attempt(source, (index,)) | {"retention_status": "retained"}
                for index, source in enumerate(ATTEMPT_SOURCES)
            ],
            "candidates": [
                {
                    "source": "no_op",
                    "terminal_legal": True,
                },
                {
                    "source": "reference_dlm",
                    "terminal_legal": False,
                    "reference_replay_matches_final": True,
                },
            ],
            "state_diagnostics": {"efsm_known": True},
            "group_failure": None,
        }
        report = summarize_groups([group])
        self.assertEqual(report["candidate_count_histogram"], {"2": 1})
        self.assertEqual(
            report["candidate_attempt_sources"],
            {source: 1 for source in ATTEMPT_SOURCES},
        )
        self.assertEqual(report["retained_candidates"], 2)
        self.assertEqual(report["terminal_valid_candidates"], 1)
        self.assertEqual(report["reference_replay"], {"retained": 1, "matched": 1, "mismatched": 0})


class WrapperContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = (ROOT / "slurm/210_build_spad_basin_preflight_actions.sbatch").read_text(
            encoding="utf-8"
        )

    def test_resources_and_torchrun_contract(self):
        self.assertIn("#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2", self.wrapper)
        self.assertIn("#SBATCH --cpus-per-task=8", self.wrapper)
        self.assertIn("--nproc_per_node=2", self.wrapper)
        self.assertIn("envs/diff_meets_diff/bin/torchrun", self.wrapper)
        self.assertIn("--chgnet-batch-size 16", self.wrapper)
        self.assertIn("--original-batch-size 8", self.wrapper)
        self.assertIn("spad_basin_preflight_states_v1_20260904/states.jsonl", self.wrapper)

    def test_merge_and_reference_replay_contract(self):
        self.assertIn("groups_rank{rank}.jsonl", self.wrapper)
        self.assertIn("len(groups) != 128", self.wrapper)
        self.assertIn("list(range(128))", self.wrapper)
        self.assertIn("reference_mismatch", self.wrapper)
        self.assertIn("ACTIONS_FINAL.json", self.wrapper)
        self.assertIn("touch \"${RUN}/_SUCCESS\"", self.wrapper)

    def test_wrapper_does_not_launch_downstream_evaluators(self):
        lowered = self.wrapper.lower()
        for forbidden in ("model494", "relax", "hull", "direct"):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
