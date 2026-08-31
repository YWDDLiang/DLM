from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm" / "115_train_compact_v2_geometry_g1.sbatch"


class GeometryG1SlurmTest(unittest.TestCase):
    def test_contract_is_single_seed_endpoint_only_and_geometry_only(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("--gres=gpu:NVIDIAA800-SXM4-80GB:2", text)
        self.assertIn("--checkpoint-path \"${BASE}\"", text)
        self.assertIn("--dynamic-geometry-only", text)
        self.assertIn("--max-train-steps 348", text)
        self.assertIn("--save-steps 348", text)
        self.assertIn("--skip-final-alias", text)
        self.assertIn("--seed 81017", text)
        self.assertIn("--periodic-metric-weight 0.1", text)
        self.assertIn("--periodic-pair-rdf-weight 0.1", text)
        self.assertIn("--periodic-overlap-weight 0.2", text)
        self.assertIn("--periodic-coordination-weight 0.05", text)
        self.assertIn("weight_selection\tfixed_scale_normalized_no_grid", text)
        self.assertIn("official_query\tfalse", text)

    def test_pins_base_data_and_code(self) -> None:
        text = SCRIPT.read_text()
        for needle in (
            "BASE_ADAPTER_SHA=06cd5465",
            "DATA_MANIFEST_SHA=b77a1d0",
            "TRAINER_SHA=e2744ab9",
            "OBJECTIVE_SHA=ccde406c",
            "H1A2_CODE_COMMIT",
        ):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
