from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.lora import validate_sft_artifacts


class CrysLLMGenLoraArtifactTests(unittest.TestCase):
    def test_data_manifest_and_fixed_optimizer_tokens_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "coarse_sft.jsonl"
            data.write_text('{"example_id":"one"}\n', encoding="utf-8")
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "crysllmgen_sft_manifest_v1",
                        "representation": "wyckoff",
                        "training_seed": 11,
                        "examples": 1,
                        "canonical_orbit_order": False,
                        "jsonl_sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "schema": "crysllmgen_sft_token_audit_v1",
                        "ok": True,
                        "representation": "wyckoff",
                        "training_seed": 11,
                        "max_length": 256,
                        "examples_tokenized": 1,
                        "fixed_padded_optimizer_tokens": 256,
                        "jsonl_sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            identity = validate_sft_artifacts(
                data_path=data,
                manifest_path=manifest,
                token_audit_path=audit,
                representation="wyckoff",
                training_seed=11,
                max_length=256,
            )
            self.assertEqual(identity["fixed_padded_optimizer_tokens"], 256)
            data.write_text('{"example_id":"mutated"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                validate_sft_artifacts(
                    data_path=data,
                    manifest_path=manifest,
                    token_audit_path=audit,
                    representation="wyckoff",
                    training_seed=11,
                    max_length=256,
                )


if __name__ == "__main__":
    unittest.main()

