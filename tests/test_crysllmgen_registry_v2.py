from __future__ import annotations

import copy
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.protocol import (
    load_registry_v2,
    validate_registry_v2,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/experiments/wyckoff_codiffusion/experiment_registry_v2.yaml"


class CrysLLMGenRegistryV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_registry_v2(REGISTRY)

    def test_training_and_evaluation_counts_are_frozen(self) -> None:
        counts = self.registry.data["training_inventory"]["counts"]
        self.assertEqual(counts["main_runs"], 9)
        self.assertEqual(counts["maximum_training_runs"], 10)
        self.assertEqual(
            self.registry.data["evaluation_inventory"]["total_configuration_ids"],
            13,
        )

    def test_gate_a_blocks_training_and_has_four_gpu_lanes_max(self) -> None:
        self.assertTrue(self.registry.data["gate_a"]["training_blocked_until_all_pass"])
        self.assertEqual(
            self.registry.data["execution_contract"]["maximum_concurrent_gpu_lanes"],
            4,
        )

    def test_registry_rejects_added_control_model_or_retry(self) -> None:
        changed = copy.deepcopy(self.registry.data)
        changed["training_inventory"]["counts"]["inference_control_training_runs"] = 1
        with self.assertRaisesRegex(ValueError, "control training"):
            validate_registry_v2(changed, self.registry.protocol)
        changed = copy.deepcopy(self.registry.data)
        changed["execution_contract"]["retry_or_replacement"] = True
        with self.assertRaisesRegex(ValueError, "retry"):
            validate_registry_v2(changed, self.registry.protocol)


if __name__ == "__main__":
    unittest.main()

