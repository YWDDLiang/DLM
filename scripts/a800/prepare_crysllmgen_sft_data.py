#!/usr/bin/env python3
"""Materialize deterministic atom or Wyckoff coarse-proposal SFT JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog
from crystal_dlm.wqcodiff.crysllmgen.sft_data import (
    tokenize_sft_example,
    write_coarse_sft_jsonl,
    write_mixed_wq_sft_jsonl,
)


def _require_environment() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SFT data preparation must run through Slurm")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=("atom", "wyckoff"), required=True)
    parser.add_argument("--dataset-stage", choices=("coarse", "mixed_edit"), default="coarse")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--canonical-orbit-order", action="store_true")
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    _require_environment()
    if args.canonical_orbit_order and args.representation != "wyckoff":
        raise ValueError("canonical orbit order applies only to Wyckoff data")
    if args.dataset_stage == "mixed_edit":
        if args.representation != "wyckoff" or args.canonical_orbit_order:
            raise ValueError("mixed edit data requires non-canonical Wyckoff representation")
        if args.epochs != 1:
            raise ValueError("mixed edit data is a balanced one-pass materialization")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = output_dir / "coarse_sft.jsonl"
    manifest_path = output_dir / "coarse_sft.manifest.json"
    catalog = PyXtalChartCatalog() if args.representation == "wyckoff" else None
    if args.dataset_stage == "mixed_edit":
        assert catalog is not None
        manifest = write_mixed_wq_sft_jsonl(
            input_paths=[value.resolve() for value in args.input],
            output=data_path,
            manifest=manifest_path,
            training_seed=args.training_seed,
            catalog=catalog,
        )
    else:
        manifest = write_coarse_sft_jsonl(
            input_paths=[value.resolve() for value in args.input],
            output=data_path,
            manifest=manifest_path,
            representation=args.representation,
            epochs=args.epochs,
            training_seed=args.training_seed,
            catalog=catalog,
            canonical_orbit_order=args.canonical_orbit_order,
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_root.resolve(),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
        model_max_length=args.max_length,
        padding_side="right",
    )
    lengths: list[int] = []
    supervised: list[int] = []
    overlength: list[str] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            example = json.loads(line)
            try:
                tokenized = tokenize_sft_example(
                    tokenizer, example, max_length=args.max_length
                )
            except ValueError as exc:
                if "exceeds frozen sequence length" not in str(exc):
                    raise
                overlength.append(str(example["example_id"]))
                continue
            lengths.append(len(tokenized["input_ids"]))
            supervised.append(sum(value != -100 for value in tokenized["labels"]))
    audit = {
        "schema": "crysllmgen_sft_token_audit_v1",
        "ok": not overlength and len(lengths) == int(manifest["examples"]),
        "representation": args.representation,
        "dataset_stage": args.dataset_stage,
        "training_seed": args.training_seed,
        "canonical_orbit_order": bool(args.canonical_orbit_order),
        "max_length": args.max_length,
        "examples_expected": manifest["examples"],
        "examples_tokenized": len(lengths),
        "overlength_count": len(overlength),
        "overlength_example_ids": overlength,
        "input_tokens_min": min(lengths, default=None),
        "input_tokens_max": max(lengths, default=None),
        "input_tokens_mean": sum(lengths) / len(lengths) if lengths else None,
        "supervised_tokens_min": min(supervised, default=None),
        "supervised_tokens_max": max(supervised, default=None),
        "supervised_tokens_total": sum(supervised),
        "fixed_padded_optimizer_tokens": int(manifest["examples"]) * args.max_length,
        "jsonl_sha256": _sha256(data_path),
        "manifest_sha256": _sha256(manifest_path),
        "llama_root": str(args.llama_root.resolve()),
        "thread_count": 1,
        "offline": True,
    }
    audit_path = output_dir / "token_audit.json"
    with audit_path.open("x", encoding="utf-8") as output:
        json.dump(audit, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({"manifest": manifest, "token_audit": audit}, indent=2, sort_keys=True))
    if not audit["ok"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
