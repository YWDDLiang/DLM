import json
from pathlib import Path
import tempfile
import unittest

from scripts.validate_h1a2_pstar_smoke import validate


def metrics(arm: str) -> dict[str, float]:
    result = {
        "target_nll": 1.1,
        "field_loss": 1.2,
        "total_loss": 1.2,
    }
    if arm == "pstar":
        result["lookahead_loss"] = 1.3
    return result


class H1A2PstarSmokeValidatorTests(unittest.TestCase):
    def fixture(self, root: Path, arm: str) -> tuple[Path, Path]:
        output = root / arm
        output.mkdir()
        report = {
            "status": "complete",
            "arm": arm,
            "engineering_smoke": True,
            "microbatches": 32,
            "optimizer_updates": 4,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "max_length": 768,
            "seed": 17,
            "shuffle": False,
            "generation_or_sun_selection": False,
            "initial_validation": metrics(arm),
            "checkpoints": [{"step": 4, "metrics": metrics(arm)}],
            "auxiliary_heads_discarded_for_inference": arm == "pstar",
            "cuda": {
                "device_name": "NVIDIA A800-SXM4-80GB",
                "peak_memory_allocated_bytes": 1024,
                "peak_memory_reserved_bytes": 2048,
            },
            "elapsed_sec": 3.0,
        }
        (output / "training_report.json").write_text(json.dumps(report))
        (output / "events.jsonl").write_text(
            json.dumps(
                {
                    "event": "training",
                    "train_loss_recent": 1.0,
                    "grad_norm": 0.5,
                }
            )
            + "\n"
        )
        gpu_csv = root / f"{arm}.csv"
        gpu_csv.write_text("time,0,NVIDIA A800,100,80000,50,10\n")
        return output, gpu_csv

    def test_both_registered_arms_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ("pcontrol", "pstar"):
                output, gpu_csv = self.fixture(root, arm)
                result = validate(output, arm=arm, gpu_csv=gpu_csv)
                self.assertTrue(result["engineering_gate_passed"])

    def test_nonfinite_or_zero_gradient_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, gpu_csv = self.fixture(root, "pcontrol")
            (output / "events.jsonl").write_text(
                json.dumps(
                    {
                        "event": "training",
                        "train_loss_recent": 1.0,
                        "grad_norm": 0.0,
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                validate(output, arm="pcontrol", gpu_csv=gpu_csv)


if __name__ == "__main__":
    unittest.main()
