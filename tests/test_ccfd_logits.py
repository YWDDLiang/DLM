import unittest

from crystal_dlm.ccfd_logits import CCFDFormulaLogitsProcessor
from crystal_dlm.h1_llm_planner import (
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL,
    clean_generated_plan_text,
)


class TinyTokenizer:
    eos_token_id = 0
    pieces = {
        0: "",
        1: " Fe",
        2: "2",
        3: "O",
        4: "3",
        5: "\nanion:",
        6: " Xx",
        7: "21",
        8: "Fe",
    }

    def __len__(self):
        return len(self.pieces)

    def decode(self, token_ids, **_kwargs):
        return "".join(self.pieces[int(value)] for value in token_ids)


def only_fe2o3(elements, counts):
    return tuple(elements) == ("Fe", "O") and tuple(counts) == (2, 3)


class CCFDLogitsTest(unittest.TestCase):
    def processor(self):
        return CCFDFormulaLogitsProcessor(
            TinyTokenizer(),
            start_length=2,
            eos_token_id=0,
            validator=only_fe2o3,
        )

    def test_masks_formula_prefix_without_resizing_tokenizer(self):
        processor = self.processor()
        self.assertIn(1, processor.allowed_token_ids([]))
        self.assertNotIn(6, processor.allowed_token_ids([]))
        self.assertIn(2, processor.allowed_token_ids([1]))
        self.assertIn(3, processor.allowed_token_ids([1, 2]))

    def test_newline_requires_terminal_valid_formula(self):
        processor = self.processor()
        self.assertNotIn(5, processor.allowed_token_ids([1, 2, 3]))
        self.assertIn(5, processor.allowed_token_ids([1, 2, 3, 4]))
        self.assertIsNone(processor.allowed_token_ids([1, 2, 3, 4, 5]))

    def test_rich_prefill_reconstructs_formula_label_without_repair(self):
        text = " Fe2O3\nanion: oxide\ncharge: neutral_plausible\nlattice: cubic\nspacegroup: sg_195_230\nvolume: volpa_010_014\nend: plan"
        cleaned = clean_generated_plan_text(
            text,
            prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL,
        )
        self.assertTrue(cleaned.startswith("formula: Fe2O3\n"))


if __name__ == "__main__":
    unittest.main()
