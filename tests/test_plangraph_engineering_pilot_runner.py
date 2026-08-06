from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "workstreams/plangraph_dlm_iclr_20260731"
    / "execution/engineering_pilot_32_v1/run_engineering_pilot.py"
)
MANIFEST_PATH = (
    ROOT
    / "workstreams/plangraph_dlm_iclr_20260731"
    / "ENGINEERING_PILOT_32_MANIFEST_V1.json"
)
AUTHORIZATION_PATH = (
    ROOT
    / "workstreams/plangraph_dlm_iclr_20260731"
    / "ENGINEERING_PILOT_32_AUTHORIZATION_V1.json"
)
SBATCH_PATH = RUNNER_PATH.with_name("engineering_pilot.sbatch")


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "test_plangraph_engineering_pilot_runner_module",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlanGraphEngineeringPilotRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.authorization = json.loads(
            AUTHORIZATION_PATH.read_text(encoding="utf-8")
        )

    def test_frozen_manifest_and_external_authorization_are_exact(self) -> None:
        self.assertEqual(
            self.runner.sha256_file(MANIFEST_PATH),
            self.runner.FROZEN_MANIFEST_SHA256,
        )
        self.runner.validate_manifest(self.manifest)
        self.runner.validate_authorization(
            self.authorization,
            manifest_sha256=self.runner.FROZEN_MANIFEST_SHA256,
        )
        self.assertFalse(self.manifest["authorization"]["job_submission"])
        self.assertTrue(
            self.authorization["authorized_scope"]["job_submission"]
        )
        self.assertFalse(
            self.authorization["continuing_locks"]["automatic_downstream"]
        )

    def test_first_nonempty_raw_line_hash_is_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_bytes(b"\n{\"x\":1}\r\n  \n{\"x\":2}\n{\"x\":3}\n")
            observed = self.runner.sha256_first_nonempty_raw_lines(path, 2)
            import hashlib

            expected = hashlib.sha256(b"{\"x\":1}\r\n{\"x\":2}\n").hexdigest()
            self.assertEqual(observed, expected)

    def test_every_arm_uses_the_frozen_bounded_arguments(self) -> None:
        for arm, expected in self.runner.ARM_POLICIES.items():
            with self.subTest(arm=arm):
                argv = self.runner.build_training_argv(
                    manifest=self.manifest,
                    arm=arm,
                    output_dir="/tmp/new-run/arms/example/training",
                )
                mapping = {
                    argv[index]: argv[index + 1]
                    for index in range(1, len(argv) - 1)
                    if argv[index].startswith("--")
                    and not argv[index + 1].startswith("--")
                }
                policy, iid_fraction, planned_fraction = expected
                self.assertEqual(
                    mapping["--planned-corruption-policy"], policy
                )
                self.assertEqual(float(mapping["--iid-fraction"]), iid_fraction)
                self.assertEqual(
                    float(mapping["--planned-fraction"]), planned_fraction
                )
                self.assertEqual(int(mapping["--limit-train"]), 32)
                self.assertEqual(int(mapping["--limit-val"]), 32)
                self.assertEqual(int(mapping["--max-train-steps"]), 4)
                self.assertEqual(int(mapping["--grad-accum"]), 8)
                self.assertEqual(int(mapping["--max-length"]), 768)
                self.assertIn("--use-lora", argv)
                for forbidden in (
                    "weighted-sampling",
                    "retry",
                    "replacement",
                    "repair",
                    "sun",
                    "chgnet",
                    "mp-api",
                ):
                    self.assertNotIn(forbidden, " ".join(argv).lower())

    def test_acceptance_requires_exact_microbatches_updates_and_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training_log = root / "training.jsonl"
            gradient_log = root / "gradients.jsonl"
            events = [{"event": "start"}]
            events.extend(
                {
                    "event": "train",
                    "step": step,
                    "loss": 1.0 + step,
                    "task_loss": 2.0 + step,
                }
                for step in range(1, 5)
            )
            events.append({"event": "eval", "step": 4, "val_loss": 3.0})
            training_log.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            gradient_log.write_text(
                "".join(
                    json.dumps(
                        {
                            "optimizer_step": step,
                            "pre_clip_grad_norm": 0.5 + step,
                        }
                    )
                    + "\n"
                    for step in range(1, 5)
                ),
                encoding="utf-8",
            )
            report = self.runner.validate_completed_training(
                training_log=training_log,
                gradient_log=gradient_log,
                instrumentation={
                    "train_compute_calls": 32,
                    "optimizer_steps": 4,
                },
            )
            self.assertEqual(report["train_microbatches"], 32)
            self.assertEqual(report["optimizer_updates"], 4)

    def test_slurm_array_is_single_a800_sequential_and_has_no_downstream(self) -> None:
        script = SBATCH_PATH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-3%1", script)
        self.assertIn("#SBATCH --cpus-per-task=8", script)
        self.assertIn("#SBATCH --mem=64G", script)
        self.assertIn("#SBATCH --time=01:00:00", script)
        self.assertIn(
            "#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1",
            script,
        )
        self.assertIn(
            "RUNNER_SHA256="
            "175f33c3c3a9b4b1f3ec1d36ecddf127846bef3d1a74449ffe2494a22835057a",
            script,
        )
        self.assertIn("plangraph_engineering32_automatic_downstream=false", script)
        for forbidden in (
            "afterok",
            "sun_evaluation",
            "crystal_generation",
            "automatic_promotion=true",
            "sbatch ",
            "scancel",
        ):
            self.assertNotIn(forbidden, script)


if __name__ == "__main__":
    unittest.main()
