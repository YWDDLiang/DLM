import unittest

from h1a2_repro.science import INFERENCE, PUBLIC_RESULT, SEEDS


class PublicContractTests(unittest.TestCase):
    def test_public_result(self) -> None:
        self.assertEqual(PUBLIC_RESULT["entries"], 1000)
        self.assertEqual(PUBLIC_RESULT["strict_sun"]["numerator"], 105)
        self.assertEqual(PUBLIC_RESULT["meta_sun"]["numerator"], 488)

    def test_exact_audit_view(self) -> None:
        exact = PUBLIC_RESULT["exact_all_attempt_view"]
        self.assertEqual(exact["strict_sun"]["numerator"], 103)
        self.assertEqual(exact["meta_sun"]["numerator"], 553)
        self.assertEqual(exact["hull_known"], 1132)

    def test_known_seeds(self) -> None:
        self.assertEqual(SEEDS.planner_train, 17)
        self.assertEqual(SEEDS.planner_sample_base, 17)
        self.assertEqual(SEEDS.quick_planner_sample, 17029)
        self.assertEqual(SEEDS.dlm_data, 20260515)

    def test_inference_shape(self) -> None:
        self.assertEqual(INFERENCE.planner_attempts, 1200)
        self.assertEqual(INFERENCE.refined_target, 1000)
        self.assertEqual(INFERENCE.quick_attempts, 256)
        self.assertEqual(INFERENCE.quick_repeats, 4)


if __name__ == "__main__":
    unittest.main()
