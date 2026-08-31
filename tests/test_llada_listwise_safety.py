from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "scripts" / "llada_listwise_safety.py"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
MODULE = None
if TORCH_AVAILABLE:
    SPEC = importlib.util.spec_from_file_location("llada_listwise_safety", SCRIPT)
    MODULE = importlib.util.module_from_spec(SPEC)
    assert SPEC and SPEC.loader
    sys.modules[SPEC.name] = MODULE
    SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is unavailable")
class LLaDAListwiseSafetyTest(unittest.TestCase):
    def test_frozen_optimization_constants(self):
        self.assertEqual(MODULE.TOTAL_UPDATES, 348)
        self.assertEqual(MODULE.GRADIENT_ACCUMULATION, 8)
        self.assertEqual(MODULE.LEARNING_RATE, 5e-6)
        self.assertEqual(MODULE.BEST_ANCHOR_WEIGHT, 0.2)
        self.assertEqual(MODULE.QUADRATIC_WEIGHT, 0.05)

    def test_collator_requires_one_kway_group(self):
        item = {
            "input_ids": MODULE.torch.zeros((4, 9), dtype=MODULE.torch.long),
            "attention_mask": MODULE.torch.ones((4, 9), dtype=MODULE.torch.long),
            "prompt_length": 1,
            "num_atoms": 2,
            "target_energies": MODULE.torch.tensor([-3.0, -2.0, -1.0, 0.0]),
            "best_index": 0,
            "group_id": "g",
        }
        self.assertEqual(MODULE.collate_one([item])["input_ids"].shape, (4, 9))
        with self.assertRaises(ValueError):
            MODULE.collate_one([item, item])

    def test_seed_ledger_is_one_to_one(self):
        self.assertEqual(MODULE.ALLOWED_POLICY_SEEDS, (82017, 82018))
        self.assertEqual(MODULE.ALLOWED_TRAINING_SEEDS, (83017, 83018))


if __name__ == "__main__":
    unittest.main()
