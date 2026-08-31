from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm" / "116_train_compact_v2_geometry_g2.sbatch"


class GeometryG2SlurmTest(unittest.TestCase):
    def test_single_seed_same_schedule_and_relation_contract(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("--gres=gpu:NVIDIAA800-SXM4-80GB:1", text)
        self.assertIn("--checkpoint-path \"${BASE}\"", text)
        self.assertIn("--dynamic-geometry-only", text)
        self.assertIn("--max-train-steps 348", text)
        self.assertIn("--grad-accum 16", text)
        self.assertIn("--seed 81017", text)
        self.assertIn("--periodic-relation-rank 64", text)
        self.assertIn("relation_forward\tacyclic_q0_soft_geometry_residual_q1", text)
        self.assertIn("development_full_endpoint_required\ttrue", text)
        self.assertIn("official_query\tfalse", text)

    def test_pins_all_implementation_and_requires_step0_equality(self) -> None:
        text = SCRIPT.read_text()
        for needle in (
            "TRAINER_SHA=cc9ba741",
            "OBJECTIVE_SHA=b4d1df1b",
            "ADAPTER_SHA=a72c47e5",
            "RUNTIME_SHA=ec6f4444",
            'relation["step0_max_logit_delta"] == 0.0',
            'range(1, 11)',
            'output_projection.weight',
        ):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
