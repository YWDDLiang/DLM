from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from crystal_dlm.fixed_slot import MASK_TOKEN_ID, build_special_tokens


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_sscd_token_checkpoint",
    ROOT / "scripts" / "audit_sscd_token_checkpoint.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load audit_sscd_token_checkpoint.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class FakeTokenizer:
    def __init__(self) -> None:
        tokens = build_special_tokens()
        self._vocab = {token: 200000 + index for index, token in enumerate(tokens)}
        self._reverse = {value: key for key, value in self._vocab.items()}
        self.pad_token_id = 1
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = MASK_TOKEN_ID

    def __len__(self) -> int:
        return 300000

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<|mdm_mask|>":
            return {"input_ids": [MASK_TOKEN_ID]}
        if text in self._vocab:
            return {"input_ids": [self._vocab[text]]}
        return {"input_ids": [999, 998]}

    def decode(self, token_ids, **kwargs):
        del kwargs
        return self._reverse[int(token_ids[0])]

    def convert_ids_to_tokens(self, token_id):
        if int(token_id) == MASK_TOKEN_ID:
            return "<|mdm_mask|>"
        return self._reverse.get(int(token_id), "<unknown>")


class TokenAuditTests(unittest.TestCase):
    def test_complete_atomic_tokenizer_passes(self) -> None:
        rows, report = AUDIT.audit_tokenizer(FakeTokenizer())
        self.assertEqual(len(rows), 2481)
        self.assertEqual(report["dynamic_special_tokens"], 2457)
        self.assertEqual(report["failures"], {})
        self.assertTrue(report["mask_contract"]["distinct_from_crystal_ids"])

    def test_runtime_tokenizer_contract_passes(self) -> None:
        from crystal_dlm.r5_dynamic_length import (
            validate_dynamic_tokenizer_contract,
        )

        report = validate_dynamic_tokenizer_contract(FakeTokenizer())
        self.assertEqual(report["atomic_crystal_tokens"], 2481)
        self.assertEqual(report["mask_token_id"], MASK_TOKEN_ID)

    def test_non_atomic_token_is_reported(self) -> None:
        class BrokenTokenizer(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False):
                if text == "<X_037>":
                    return {"input_ids": [7, 8]}
                return super().__call__(text, add_special_tokens=add_special_tokens)

        _rows, report = AUDIT.audit_tokenizer(BrokenTokenizer())
        self.assertEqual(report["failures"]["non_atomic"], 1)


if __name__ == "__main__":
    unittest.main()
