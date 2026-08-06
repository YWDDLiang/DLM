import importlib.util
import sys
import unittest
from pathlib import Path

from crystal_dlm.composition_validity import pbc_duplicate_record, replace_active_elements, smact_validity


class CompositionValidityTest(unittest.TestCase):
    def test_negative_element_replacement_preserves_schema_slots(self):
        prompt_length = 3
        seq = [900, 901, 902] + [0] * 107
        seq[prompt_length + 0] = 301
        seq[prompt_length + 7] = 401
        seq[prompt_length + 8] = 101
        seq[prompt_length + 9] = 501
        seq[prompt_length + 10] = 502
        seq[prompt_length + 11] = 503
        seq[prompt_length + 12] = 402
        seq[prompt_length + 13] = 102
        seq[prompt_length + 14] = 504
        seq[prompt_length + 15] = 505
        seq[prompt_length + 16] = 506
        for rel in range(17, 107):
            seq[prompt_length + rel] = 700 + rel

        negative, meta = replace_active_elements(
            seq,
            prompt_length=prompt_length,
            element_token_ids=[101, 102, 103],
            token_id_to_atomic_number={101: 11, 102: 17, 103: 2},
            attempts=20,
            rng=__import__("random").Random(7),
            validity_fn=lambda atom_types: list(atom_types) == [11, 17],
        )

        self.assertIsNotNone(negative, meta)
        self.assertEqual(len(negative), len(seq))
        self.assertEqual(negative[:prompt_length], seq[:prompt_length])
        self.assertEqual(negative[prompt_length + 0], seq[prompt_length + 0])
        self.assertEqual(negative[prompt_length + 7], seq[prompt_length + 7])
        self.assertEqual(negative[prompt_length + 12], seq[prompt_length + 12])
        self.assertEqual(negative[prompt_length + 9], seq[prompt_length + 9])
        self.assertNotEqual(
            [negative[prompt_length + 8], negative[prompt_length + 13]],
            [seq[prompt_length + 8], seq[prompt_length + 13]],
        )

    def test_smact_wrapper_matches_crysllmgen_reference_when_available(self):
        if importlib.util.find_spec("smact") is None:
            self.skipTest("smact is not installed")
        eval_utils_path = Path(__file__).resolve().parents[1] / "reference/crysllmgen/eval_utils.py"
        spec = importlib.util.spec_from_file_location("crysllmgen_eval_utils_for_test", eval_utils_path)
        module = importlib.util.module_from_spec(spec)
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(eval_utils_path.parent))
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path

        examples = [
            ([11, 17], [1, 1]),
            ([6, 8], [1, 2]),
            ([26, 28], [1, 1]),
            ([1], [1]),
        ]
        for elems, counts in examples:
            self.assertEqual(smact_validity(elems, counts), module.smact_validity(elems, counts))

    def test_pbc_duplicate_record_treats_000_and_100_as_equivalent(self):
        record = pbc_duplicate_record([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0], [0.25, 0.5, 0.75]])
        self.assertFalse(record["has_exact_duplicate"])
        self.assertTrue(record["has_pbc_equivalent_duplicate"])
        self.assertEqual(record["pbc_only_duplicate_site_count"], 1)


if __name__ == "__main__":
    unittest.main()
