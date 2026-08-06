import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = (
    ROOT
    / "configs"
    / "experiments"
    / "wyckoff_codiffusion"
    / "training_evaluation_inventory_v1.json"
)


class TrainingEvaluationInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(INVENTORY.read_text(encoding="utf-8"))

    def test_training_counts_are_derived_from_registered_runs(self) -> None:
        training = self.payload["training"]
        main_runs = sum(
            int(family["training_runs"])
            for family in training["main_model_families"]
        )
        ablation_runs = sum(
            int(family["training_runs"])
            for family in training["trained_ablation_families"]
        )
        summary = training["summary"]
        self.assertEqual(len(training["main_model_families"]), 3)
        self.assertEqual(main_runs, 9)
        self.assertEqual(ablation_runs, 1)
        self.assertEqual(summary["main_training_runs"], main_runs)
        self.assertEqual(summary["trained_ablation_runs"], ablation_runs)
        self.assertEqual(
            summary["maximum_scheduled_training_runs"],
            main_runs + ablation_runs,
        )

    def test_main_families_use_exactly_the_three_registered_seeds(self) -> None:
        training = self.payload["training"]
        registered_seeds = training["seeds"]
        self.assertEqual(registered_seeds, [11, 23, 47])
        for family in training["main_model_families"]:
            self.assertEqual(family["seeds"], registered_seeds)
            self.assertEqual(family["training_runs"], len(registered_seeds))

    def test_controls_do_not_create_additional_training_runs(self) -> None:
        training = self.payload["training"]
        evaluation = self.payload["evaluation"]
        self.assertEqual(
            training["summary"]["separately_trained_models_for_inference_controls"],
            0,
        )
        self.assertTrue(self.payload["execution_rules"]["controls_reuse_main_checkpoints"])
        self.assertEqual(len(evaluation["additional_inference_controls"]), 4)

    def test_evaluation_counts_are_derived_from_registered_entries(self) -> None:
        evaluation = self.payload["evaluation"]
        summary = evaluation["summary"]
        self.assertEqual(len(evaluation["families"]), 10)
        self.assertEqual(len(evaluation["primary_methods"]), 5)
        self.assertEqual(len(evaluation["additional_inference_controls"]), 4)
        self.assertEqual(len(evaluation["validation_only_diagnostics"]), 4)
        self.assertEqual(summary["evaluation_families"], 10)
        self.assertEqual(
            summary["total_registered_configuration_ids"],
            len(evaluation["primary_methods"])
            + len(evaluation["additional_inference_controls"])
            + len(evaluation["validation_only_diagnostics"]),
        )

    def test_sun_contract_stays_mlip_only_and_blas_is_one(self) -> None:
        axes = self.payload["evaluation"]["shared_axes"]
        rules = self.payload["execution_rules"]
        self.assertEqual(len(axes["mlip_evaluators"]), 3)
        self.assertEqual(axes["sun_thresholds_eV_per_atom"], [0.0, 0.1])
        self.assertFalse(rules["new_dft_allowed"])
        self.assertEqual(rules["blas_thread_count"], 1)


if __name__ == "__main__":
    unittest.main()
