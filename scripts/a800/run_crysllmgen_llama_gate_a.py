#!/usr/bin/env python3
"""Verify the frozen Llama/LoRA assets with one offline BF16 Slurm forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_environment() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Llama Gate A must run through Slurm")
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


def _verify_registered_files(root: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed: list[dict[str, Any]] = []
    for registered in files:
        path = root / str(registered["name"])
        if not path.is_file():
            raise FileNotFoundError(f"registered model asset is missing: {path}")
        size = path.stat().st_size
        digest = _sha256(path)
        if size != int(registered["bytes"]) or digest != str(registered["sha256"]):
            raise RuntimeError(f"registered model asset identity changed: {path}")
        observed.append({"name": path.name, "bytes": size, "sha256": digest})
    return observed


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-preflight", type=Path, required=True)
    parser.add_argument("--adapter-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require_environment()

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the registered Slurm job")
    llama_preflight_path = args.llama_preflight.resolve()
    adapter_preflight_path = args.adapter_preflight.resolve()
    llama_record = json.loads(llama_preflight_path.read_text(encoding="utf-8"))
    adapter_record = json.loads(adapter_preflight_path.read_text(encoding="utf-8"))
    base_root = Path(llama_record["source_path"]).resolve()
    adapter = adapter_record["selected"]
    adapter_root = Path(adapter["path"]).resolve()
    started = time.time()
    base_files = _verify_registered_files(base_root, list(llama_record["files"]))
    adapter_files = _verify_registered_files(adapter_root, list(adapter["files"]))
    identity_verified_s = time.time() - started

    tokenizer = AutoTokenizer.from_pretrained(
        base_root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
        model_max_length=512,
        padding_side="right",
    )
    if not tokenizer.chat_template:
        raise RuntimeError("registered instruction tokenizer has no chat template")
    messages = [
        {
            "role": "system",
            "content": "Return exactly one crystal record in the registered grammar.",
        },
        {
            "role": "user",
            "content": "Generate one unconditional MP20 crystal. Return only the record.",
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )
    device = torch.device("cuda", 0)
    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_root,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    model.config.use_cache = False
    load_seconds = time.time() - load_started
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    with torch.inference_mode():
        base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1]
    if not torch.isfinite(base_logits).all():
        raise RuntimeError("base Llama forward produced non-finite logits")
    base_logits_hash = hashlib.sha256(
        base_logits.float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()

    adapter_started = time.time()
    model = PeftModel.from_pretrained(
        model,
        adapter_root,
        local_files_only=True,
        is_trainable=False,
    )
    model.eval()
    with torch.inference_mode():
        adapter_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits[:, -1]
    if not torch.isfinite(adapter_logits).all():
        raise RuntimeError("adapter Llama forward produced non-finite logits")
    if torch.equal(base_logits, adapter_logits):
        raise RuntimeError("selected LoRA has no observable effect on the registered prompt")
    adapter_logits_hash = hashlib.sha256(
        adapter_logits.float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    first_parameter = next(model.parameters())
    report = {
        "schema": "crysllmgen_llama_gate_a_report_v1",
        "ok": True,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "host": os.uname().nodename,
        "offline": True,
        "blas_threads": 1,
        "base_root": str(base_root),
        "adapter_root": str(adapter_root),
        "llama_preflight_sha256": _sha256(llama_preflight_path),
        "adapter_preflight_sha256": _sha256(adapter_preflight_path),
        "verified_base_files": base_files,
        "verified_adapter_files": adapter_files,
        "identity_verification_s": identity_verified_s,
        "model_load_s": load_seconds,
        "base_plus_adapter_forward_s": time.time() - adapter_started,
        "prompt": {
            "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "input_tokens": int(input_ids.shape[1]),
            "max_sequence_length": 512,
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode("utf-8")
            ).hexdigest(),
        },
        "model": {
            "architecture": type(model.base_model.model).__name__,
            "parameter_count_with_adapter": sum(value.numel() for value in model.parameters()),
            "first_parameter_dtype": str(first_parameter.dtype),
            "first_parameter_device": str(first_parameter.device),
            "base_logits_sha256_f32": base_logits_hash,
            "adapter_logits_sha256_f32": adapter_logits_hash,
            "adapter_changes_logits": True,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
            "walltime_s": time.time() - started,
        },
    }
    _write_exclusive(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
