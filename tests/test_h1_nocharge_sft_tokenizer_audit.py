from __future__ import annotations

import unittest

from scripts.audit_h1_nocharge_sft_tokenizer import audit_record


class FakeTokenizer:
    eos_token = "~"
    chat_template = None

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        value = str(text)
        result = {"input_ids": [ord(char) for char in value]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(idx, idx + 1) for idx in range(len(value))]
        return result


def record(*, task="direct_nocharge_plan", answer="formula: Li2O\nend: plan"):
    return {
        "schema": "h1_nocharge_ion_aux_v1",
        "record_id": "train:0000:test",
        "task": task,
        "source_row_idx": 1,
        "infill_cursor": None,
        "formula": "Li2O",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "answer": answer,
        "loss_mode": "sft",
        "weighted_answer_spans": [
            {"start": 9, "end": 13, "weight": 2.0, "label": "formula"}
        ],
    }


class TokenizerAuditTests(unittest.TestCase):
    def test_weighted_answer_and_prompt_are_audited(self):
        result = audit_record(FakeTokenizer(), record(), max_length=128)
        self.assertGreater(result["prompt_token_count"], 0)
        self.assertGreater(result["answer_token_count"], 0)
        self.assertEqual(result["weight_policy"][0]["weight"], 2.0)

    def test_direct_nocharge_rejects_charge_target(self):
        value = record(answer="formula: Li2O\ncharge: neutral_plausible\nend: plan")
        value["weighted_answer_spans"] = []
        with self.assertRaises(ValueError):
            audit_record(FakeTokenizer(), value, max_length=128)

    def test_conditional_anchor_rejects_formula_in_answer(self):
        value = record(task="conditional_mp20_anchor", answer="formula: Li2O\nend: anchor")
        value["weighted_answer_spans"] = []
        with self.assertRaises(ValueError):
            audit_record(FakeTokenizer(), value, max_length=128)


if __name__ == "__main__":
    unittest.main()
