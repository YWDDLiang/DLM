#!/usr/bin/env python3
"""Materialize the immutable fixed-length token cache used by DDP profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crystal_dlm.wqcodiff.crysllmgen.lora import (
    materialize_pretokenized_sft_cache,
    validate_sft_artifacts,
)
from crystal_dlm.wqcodiff.crysllmgen.protocol import load_protocol_v4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import transformers
    from transformers import AutoTokenizer

    protocol = load_protocol_v4(args.protocol.resolve())
    max_length = int(protocol.data["llama_training"]["max_sequence_length"])
    artifact_identity = validate_sft_artifacts(
        data_path=args.data,
        manifest_path=args.data_manifest,
        token_audit_path=args.token_audit,
        representation="wyckoff",
        training_seed=args.training_seed,
        max_length=max_length,
        dataset_stage="mixed_edit",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_root.resolve(),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
        model_max_length=max_length,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("registered tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    report = materialize_pretokenized_sft_cache(
        data_path=args.data,
        tokenizer=tokenizer,
        max_length=max_length,
        output_dir=args.output_dir,
    )
    if (
        int(report["examples"]) != int(artifact_identity["examples"])
        or report["source_data"]["sha256"] != artifact_identity["data_sha256"]
    ):
        raise RuntimeError("materialized cache denominator/source identity mismatch")
    print(
        json.dumps(
            {
                "ok": True,
                "scientific_attempt": False,
                "protocol_sha256": protocol.sha256,
                "data": artifact_identity,
                "cache": report,
                "transformers": transformers.__version__,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
