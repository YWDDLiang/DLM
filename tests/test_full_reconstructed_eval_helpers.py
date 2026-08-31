import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "eval_runtime"
EVALUATOR_PATH = RUNTIME / "run_full_reconstructed_eval.py"
os.environ.setdefault("H1_ACTIVE_DENOMINATOR", "256")
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location(
    "full_reconstructed_eval_helper_test_module", EVALUATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)


class FakeChild:
    def __init__(self, pid, *, exits_on_join=False, exits_on_terminate=True):
        self.pid = pid
        self.name = f"fake-{pid}"
        self._alive = True
        self.exits_on_join = exits_on_join
        self.exits_on_terminate = exits_on_terminate

    def is_alive(self):
        return self._alive

    def join(self, _timeout):
        if self.exits_on_join:
            self._alive = False

    def terminate(self):
        if self.exits_on_terminate:
            self._alive = False

    def kill(self):
        self._alive = False


class FakeComposition:
    def __init__(self, payload):
        self.payload = dict(payload)

    def as_dict(self):
        return dict(self.payload)


class FakeResumable:
    @staticmethod
    def decode_composition(payload):
        return FakeComposition(payload)


class FullReconstructedEvalHelperTest(unittest.TestCase):
    def test_cleanup_joins_then_terminates_then_kills_only_lingering_children(self):
        joined = FakeChild(101, exits_on_join=True)
        terminated = FakeChild(102)
        killed = FakeChild(103, exits_on_terminate=False)

        report = EVALUATOR.cleanup_multiprocessing_children(
            [joined, terminated, killed],
            join_timeout_seconds=0,
            terminate_timeout_seconds=0,
        )

        self.assertEqual(report["initial_count"], 3)
        self.assertEqual(report["terminated_pids"], [102, 103])
        self.assertEqual(report["killed_pids"], [103])
        self.assertEqual(report["surviving_pids"], [])
        self.assertTrue(report["clean"])

    def test_exact_reuse_guard_rejects_model494_refined_rows(self):
        raw = {
            "method": "H1-A2-DLM-RAW-BODY-NO-MODEL494",
            "diffusion_refinement_applied": False,
            "diffusion_refinement_steps": 0,
            "refiner_noise_seed": None,
        }
        EVALUATOR._require_raw_generation([raw], role="F", arm="control")

        refined = {
            **raw,
            "method": "H1-A2-DLM-CE-CONTROL",
            "diffusion_refinement_applied": True,
            "diffusion_refinement_steps": 800,
            "refiner_noise_seed": 101117,
        }
        with self.assertRaisesRegex(ValueError, "restricted to unrefined raw"):
            EVALUATOR._require_raw_generation(
                [refined], role="F", arm="control"
            )

    def test_exact_result_shard_round_trips_content_and_output_identities(self):
        first = "1" * 64
        views = {
            "F": EVALUATOR.exact_raw_reuse.ManifestIdentities(2, (first, first)),
            "M": EVALUATOR.exact_raw_reuse.ManifestIdentities(2, (first,)),
        }
        plan = EVALUATOR.exact_raw_reuse.build_pair_plan(views)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "F.results.jsonl"
            manifest = root / "F.results.json"
            EVALUATOR._write_exact_result_shard(
                path=shard,
                manifest_path=manifest,
                role="F",
                plan=plan,
                identities=plan.owned_identities("F"),
                results=[(-1.25, FakeComposition({"Li": 1.0}))],
            )

            loaded = EVALUATOR._load_exact_result_shard(
                path=shard,
                manifest_path=manifest,
                role="F",
                plan=plan,
                resumable=FakeResumable,
            )

            self.assertEqual(set(loaded), {first})
            self.assertEqual(loaded[first][0], -1.25)
            self.assertEqual(loaded[first][1].as_dict(), {"Li": 1.0})


if __name__ == "__main__":
    unittest.main()
