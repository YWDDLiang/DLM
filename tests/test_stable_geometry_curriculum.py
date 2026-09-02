from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crystal_dlm.stable_geometry_curriculum import (
    dynamic_geometry_relative_positions,
    forbidden_training_paths,
    source_ehull,
    strip_training_outcomes,
)


class StableGeometryCurriculumTest(unittest.TestCase):
    def test_dynamic_geometry_positions_exclude_n_and_elements(self):
        self.assertEqual(
            dynamic_geometry_relative_positions(2),
            (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14),
        )

    def test_strip_outcomes_is_recursive(self):
        row = {
            "prompt": "composition only",
            "metadata": {"e_above_hull": 0.0, "material_id": "mp-1"},
            "nested": [{"formation_energy_per_atom": -1.0, "keep": 2}],
        }
        clean = strip_training_outcomes(row)
        self.assertEqual(forbidden_training_paths(clean), [])
        self.assertEqual(clean["metadata"], {"material_id": "mp-1"})
        self.assertEqual(clean["nested"], [{"keep": 2}])

    def test_source_ehull_requires_finite_metadata(self):
        self.assertEqual(source_ehull({"metadata": {"e_above_hull": 0.01}}), 0.01)
        with self.assertRaises(ValueError):
            source_ehull({"metadata": {}})


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class SGTCLLaDAMaskTest(unittest.TestCase):
    def load_trainer(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "llada_sft_sgtc", ROOT / "src/scripts/llada_sft.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import llada_sft")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_forward_process_candidate_mask_is_geometry_only(self):
        module = self.load_trainer()
        input_ids = torch.arange(15, dtype=torch.long).reshape(1, -1)
        result = module.forward_process(
            input_ids,
            torch.ones_like(input_ids),
            torch.tensor([0]),
            mask_policy_ids=torch.tensor([module.MASK_POLICY_TO_ID["normal"]]),
            empty_token_id=999,
            dynamic_geometry_only=True,
        )
        observed = tuple(
            int(value)
            for value in torch.nonzero(
                result["candidate_mask"][0], as_tuple=False
            ).reshape(-1)
        )
        self.assertEqual(observed, dynamic_geometry_relative_positions(2))

    def test_paired_source_supplies_unmasked_tokens_only(self):
        module = self.load_trainer()
        target = torch.tensor([[10, 11, 20, 21]], dtype=torch.long)
        source = torch.tensor([[10, 11, 30, 31]], dtype=torch.long)
        processed = {
            "noisy": target.clone(),
            "masked_indices": torch.tensor([[False, False, True, False]]),
        }
        observed = module.apply_paired_source_tokens(
            processed,
            source,
            target,
            torch.tensor([2]),
            mask_id=999,
        )
        self.assertEqual(observed["noisy"].tolist(), [[10, 11, 999, 31]])

    def test_paired_source_rejects_prompt_mismatch(self):
        module = self.load_trainer()
        target = torch.tensor([[10, 11, 20]], dtype=torch.long)
        source = torch.tensor([[10, 12, 30]], dtype=torch.long)
        processed = {
            "noisy": target.clone(),
            "masked_indices": torch.tensor([[False, False, True]]),
        }
        with self.assertRaisesRegex(ValueError, "prompt tokens differ"):
            module.apply_paired_source_tokens(
                processed,
                source,
                target,
                torch.tensor([2]),
            )

    def test_paired_dynamic_contract_locks_n_and_element_order(self):
        module = self.load_trainer()
        target = list(range(17))
        source = list(target)
        source[2 + 1] = 999  # lattice token may differ
        source[2 + 8] = 998  # coordinate token may differ
        module.validate_paired_dynamic_ids(
            target,
            source,
            prompt_length=2,
            num_atoms=2,
        )
        source[2 + 7] = 997  # first element token may not differ
        with self.assertRaisesRegex(ValueError, "element order"):
            module.validate_paired_dynamic_ids(
                target,
                source,
                prompt_length=2,
                num_atoms=2,
            )

    def test_collator_preserves_optional_source_tensor(self):
        module = self.load_trainer()

        class Tokenizer:
            pad_token_id = 0

        target = torch.tensor([1, 2, 3, 4])
        source = torch.tensor([1, 2, 8, 9])
        batch = module.DataCollator(Tokenizer())(
            [
                {
                    "input_ids": target,
                    "source_input_ids": source,
                    "prompt_length": 2,
                }
            ]
        )
        self.assertEqual(batch["source_input_ids"].tolist(), [[1, 2, 8, 9]])

    def test_forced_rollout_masks_separate_model_and_loss_masks(self):
        module = self.load_trainer()
        source = torch.tensor([[10, 11, 30, 31, 32]], dtype=torch.long)
        forced = torch.tensor([[False, False, True, True, True]])
        supervised = torch.tensor([[False, False, True, False, False]])
        processed = {
            "noisy": source.clone(),
            "masked_indices": torch.zeros_like(source, dtype=torch.bool),
            "candidate_mask": torch.zeros_like(source, dtype=torch.bool),
            "p_mask": torch.full(source.shape, 0.5),
        }
        observed = module.apply_forced_rollout_masks(
            processed,
            source,
            forced,
            supervised,
            mask_id=999,
        )
        self.assertEqual(observed["noisy"].tolist(), [[10, 11, 999, 999, 999]])
        self.assertTrue(torch.equal(observed["masked_indices"], forced))
        self.assertTrue(torch.equal(observed["loss_indices"], supervised))
        self.assertTrue(torch.equal(observed["candidate_mask"], supervised))
        self.assertTrue(torch.equal(observed["p_mask"], torch.ones_like(observed["p_mask"])))

    def test_forced_rollout_process_bypasses_random_corruption(self):
        module = self.load_trainer()
        source = torch.tensor([[10, 11, 30, 31, 32]], dtype=torch.long)
        forced = torch.tensor([[False, False, True, True, True]])
        supervised = torch.tensor([[False, False, True, False, False]])
        observed = module.forced_rollout_process(
            source,
            torch.ones_like(source),
            torch.tensor([2]),
            forced,
            supervised,
            mask_id=999,
        )
        self.assertEqual(observed["noisy"].tolist(), [[10, 11, 999, 999, 999]])
        self.assertEqual(observed["answer_mask"].tolist(), [[False, False, True, True, True]])


if __name__ == "__main__":
    unittest.main()
