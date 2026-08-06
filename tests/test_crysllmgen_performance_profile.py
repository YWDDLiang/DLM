from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from crystal_dlm.wqcodiff.crysllmgen.lora import (
    FixedLengthSFTCollator,
    JsonlSFTDataset,
    PretokenizedMemmapSFTDataset,
    materialize_pretokenized_sft_cache,
)
from crystal_dlm.wqcodiff.crysllmgen.performance_profile import (
    load_lora_ddp_profile_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion/ddp_profile_matrix_v1.json"
)
FLASH_MATRIX = (
    ROOT
    / "configs/experiments/wyckoff_codiffusion/ddp_flash_followup_v1.json"
)
BASE_SOURCE = "6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908"
PROTOCOL_SHA = "22efb63d3b7e37353206022d3ce9e43f0c19ed02dbd5fa75a71a053339707af0"


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1

    def __len__(self) -> int:
        return 256

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        value = ""
        for message in messages:
            value += f"{message['role'][0]}:{message['content']}|"
        if add_generation_prompt:
            value += "a:"
        assert tokenize
        return list(value.encode("utf-8"))


def _example(example_id: str, answer: str) -> dict[str, str]:
    return {
        "example_id": example_id,
        "system_prompt": "system",
        "user_prompt": "user",
        "answer": answer,
    }


class CrysLLMGenDDPPerformanceProfileTests(unittest.TestCase):
    def test_profile_matrix_is_exact_and_preserves_global_batch(self) -> None:
        matrix = load_lora_ddp_profile_matrix(
            MATRIX,
            base_source_bundle_sha256=BASE_SOURCE,
            protocol_v4_sha256=PROTOCOL_SHA,
        )
        self.assertEqual(len(matrix.variants), 6)
        self.assertEqual(
            matrix.select("cache4_mb8_acc4_gc_off_sdpa").global_effective_batch,
            64,
        )
        self.assertTrue(
            matrix.select("cache4_mb4_acc8_gc_reentrant_sdpa")
            .gradient_checkpointing_use_reentrant
        )

    def test_profile_matrix_rejects_scientific_or_batch_drift(self) -> None:
        payload = json.loads(MATRIX.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(payload)
        changed["scientific_attempt"] = True
        mutations.append(changed)
        changed = copy.deepcopy(payload)
        changed["variants"][4]["gradient_accumulation"] = 8
        mutations.append(changed)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "matrix.json"
                path.write_text(json.dumps(mutation), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_lora_ddp_profile_matrix(
                        path,
                        base_source_bundle_sha256=BASE_SOURCE,
                        protocol_v4_sha256=PROTOCOL_SHA,
                    )

    def test_flash_followup_changes_only_attention_backend(self) -> None:
        matrix = load_lora_ddp_profile_matrix(
            FLASH_MATRIX,
            base_source_bundle_sha256=BASE_SOURCE,
            protocol_v4_sha256=PROTOCOL_SHA,
        )
        self.assertEqual(len(matrix.variants), 1)
        variant = matrix.select("cache4_mb8_acc4_gc_off_flash2")
        self.assertEqual(variant.attention_implementation, "flash_attention_2")
        self.assertEqual(variant.per_device_microbatch, 8)
        self.assertEqual(variant.gradient_accumulation, 4)
        self.assertFalse(variant.gradient_checkpointing)
        self.assertEqual(variant.data_mode, "pretokenized_memmap")

        payload = json.loads(FLASH_MATRIX.read_text(encoding="utf-8"))
        payload["variants"][0]["gradient_checkpointing"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flash.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "FlashAttention"):
                load_lora_ddp_profile_matrix(
                    path,
                    base_source_bundle_sha256=BASE_SOURCE,
                    protocol_v4_sha256=PROTOCOL_SHA,
                )

    def test_pretokenized_memmap_cache_matches_lazy_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "examples.jsonl"
            examples = [_example("one", "abc"), _example("two", "xyz")]
            data.write_text(
                "".join(json.dumps(value) + "\n" for value in examples),
                encoding="utf-8",
            )
            cache = root / "cache"
            report = materialize_pretokenized_sft_cache(
                data_path=data,
                tokenizer=tokenizer,
                max_length=64,
                output_dir=cache,
            )
            self.assertEqual(report["examples"], 2)
            lazy = JsonlSFTDataset(data, tokenizer, max_length=64)
            input_ids = np.load(cache / "input_ids.npy", mmap_mode="r")
            labels = np.load(cache / "labels.npy", mmap_mode="r")
            attention = np.load(cache / "attention_mask.npy", mmap_mode="r")
            for index in range(2):
                expected = lazy[index]
                length = len(expected["input_ids"])
                self.assertEqual(
                    input_ids[index, :length].tolist(), expected["input_ids"]
                )
                self.assertEqual(labels[index, :length].tolist(), expected["labels"])
                self.assertEqual(
                    attention[index, :length].tolist(), expected["attention_mask"]
                )
                self.assertTrue(np.all(input_ids[index, length:] == 0))
                self.assertTrue(np.all(labels[index, length:] == -100))
                self.assertTrue(np.all(attention[index, length:] == 0))
            cached = PretokenizedMemmapSFTDataset(
                cache,
                data_path=data,
                tokenizer=tokenizer,
                max_length=64,
            )
            self.assertEqual(len(cached), len(lazy))
            self.assertEqual(cached.identity()["mode"], "pretokenized_memmap")
            with self.assertRaises(FileExistsError):
                materialize_pretokenized_sft_cache(
                    data_path=data,
                    tokenizer=tokenizer,
                    max_length=64,
                    output_dir=cache,
                )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is server-only")
    def test_cached_collator_is_tensor_identical_to_lazy_collator(self) -> None:
        import torch

        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "examples.jsonl"
            data.write_text(json.dumps(_example("one", "abc")) + "\n", encoding="utf-8")
            cache = root / "cache"
            materialize_pretokenized_sft_cache(
                data_path=data,
                tokenizer=tokenizer,
                max_length=64,
                output_dir=cache,
            )
            lazy = JsonlSFTDataset(data, tokenizer, max_length=64)
            cached = PretokenizedMemmapSFTDataset(
                cache,
                data_path=data,
                tokenizer=tokenizer,
                max_length=64,
            )
            collator = FixedLengthSFTCollator(pad_token_id=0, max_length=64)
            lazy_batch = collator([lazy[0]])
            cached_batch = collator([cached[0]])
            for key in ("input_ids", "labels", "attention_mask"):
                self.assertTrue(torch.equal(lazy_batch[key], cached_batch[key]))


if __name__ == "__main__":
    unittest.main()
