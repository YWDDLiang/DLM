#!/usr/bin/env python3
"""Gate A: 256 one-shot constrained WQ generations must all parse legally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog
from crystal_dlm.wqcodiff.crysllmgen.constrained import ProposalTokenConstraint
from crystal_dlm.wqcodiff.crysllmgen.sft_data import (
    PROTOCOL_NAME,
    UNCONDITIONAL_USER_PROMPT,
    WQ_SYSTEM_PROMPT,
)
from crystal_dlm.wqcodiff.crysllmgen.wq_text import parse_wq_proposal


ATTEMPTS = 256
MAX_SEQUENCE_LENGTH = 512
SAMPLING_SEED = 2026072002


def _require_environment() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("constrained generation Gate A must run through Slurm")
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
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _derive_seed(ordinal: int) -> int:
    payload = json.dumps(
        [PROTOCOL_NAME, "gate_a_constrained_256", SAMPLING_SEED, ordinal],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def _attempt_id(ordinal: int) -> str:
    payload = json.dumps(
        {
            "protocol": PROTOCOL_NAME,
            "experiment": "gate_a_constrained_256",
            "sampling_seed": SAMPLING_SEED,
            "ordinal": ordinal,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "gate-a-" + hashlib.sha256(payload).hexdigest()[:24]


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=ATTEMPTS)
    args = parser.parse_args()
    _require_environment()
    if args.attempts != ATTEMPTS:
        raise ValueError(f"Gate A denominator is frozen at {ATTEMPTS}")

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the registered Slurm job")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    llama_root = args.llama_root.resolve()
    adapter_root = args.adapter.resolve()
    adapter_model = adapter_root / "adapter_model.safetensors"
    adapter_config = adapter_root / "adapter_config.json"
    if not adapter_model.is_file() or not adapter_config.is_file():
        raise FileNotFoundError("registered WQ smoke adapter is incomplete")

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        llama_root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
        model_max_length=MAX_SEQUENCE_LENGTH,
        padding_side="right",
    )
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise RuntimeError("registered tokenizer lacks EOS or official chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    messages = [
        {"role": "system", "content": WQ_SYSTEM_PROMPT},
        {"role": "user", "content": UNCONDITIONAL_USER_PROMPT},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prompt_width = int(prompt_ids.shape[1])
    if prompt_width <= 0 or prompt_width >= MAX_SEQUENCE_LENGTH:
        raise RuntimeError("registered prompt width is outside the sequence contract")
    max_new_tokens = MAX_SEQUENCE_LENGTH - prompt_width
    catalog = PyXtalChartCatalog(hall_style="spglib")
    constraint = ProposalTokenConstraint(
        tokenizer,
        catalog,
        prompt_width=prompt_width,
    )
    model = AutoModelForCausalLM.from_pretrained(
        llama_root,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(
        model,
        adapter_root,
        local_files_only=True,
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    prompt_ids = prompt_ids.to(torch.device("cuda", 0))

    attempts_path = output_dir / "attempts.jsonl"
    terminal = 0
    parsed = 0
    topology_legal = 0
    failure_reasons: dict[str, int] = {}
    generated_token_counts: list[int] = []
    with attempts_path.open("xb") as binary:
        for ordinal in range(ATTEMPTS):
            attempt_started = time.time()
            attempt_id = _attempt_id(ordinal)
            seed = _derive_seed(ordinal)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            text = ""
            generated_tokens = 0
            record: dict[str, Any] = {
                "schema": "crysllmgen_constrained_attempt_v1",
                "attempt_id": attempt_id,
                "ordinal": ordinal,
                "sampling_seed": SAMPLING_SEED,
                "derived_seed": seed,
                "status": "failed",
                "reason": "uninitialized",
                "retry_or_replacement_used": False,
                "llama_invocations": 1,
            }
            try:
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=prompt_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        renormalize_logits=True,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        prefix_allowed_tokens_fn=constraint,
                    )
                generated = output[0, prompt_width:]
                generated_tokens = int(generated.numel())
                if generated_tokens and int(generated[-1]) == tokenizer.eos_token_id:
                    generated = generated[:-1]
                text = tokenizer.decode(
                    generated,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                state = parse_wq_proposal(text, catalog, attempt_id=attempt_id)
                if not 1 <= state.atom_count <= 20 or not state.orbits:
                    raise ValueError("parsed state is outside MP20 topology support")
                record.update(
                    {
                        "status": "success",
                        "reason": "",
                        "space_group": state.space_group,
                        "atom_count": state.atom_count,
                        "orbit_count": len(state.orbits),
                        "topology_hash": state.topology_hash(include_geometry=True),
                    }
                )
                parsed += 1
                topology_legal += 1
            except Exception as exc:
                reason = f"{type(exc).__name__}:{exc}"
                record["reason"] = reason
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            record.update(
                {
                    "generated_tokens": generated_tokens,
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "walltime_s": time.time() - attempt_started,
                }
            )
            terminal += 1
            generated_token_counts.append(generated_tokens)
            binary.write(
                (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
            binary.flush()
        os.fsync(binary.fileno())

    attempts_sha = _sha256(attempts_path)
    report = {
        "schema": "crysllmgen_constrained_gate_report_v1",
        "ok": terminal == parsed == topology_legal == ATTEMPTS,
        "submitted_attempts": ATTEMPTS,
        "terminal_attempts": terminal,
        "parsed_attempts": parsed,
        "topology_legal_attempts": topology_legal,
        "retry_or_replacement_used": False,
        "failure_reasons": failure_reasons,
        "attempts_jsonl": str(attempts_path),
        "attempts_sha256": attempts_sha,
        "model": {
            "llama_root": str(llama_root),
            "adapter_root": str(adapter_root),
            "adapter_model_sha256": _sha256(adapter_model),
            "adapter_config_sha256": _sha256(adapter_config),
            "prompt_width": prompt_width,
            "tokenizer_size": len(tokenizer),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "max_new_tokens": max_new_tokens,
        },
        "generation": {
            "sampling_seed": SAMPLING_SEED,
            "temperature": 0.8,
            "top_p": 0.95,
            "generated_tokens_total": sum(generated_token_counts),
            "generated_tokens_max": max(generated_token_counts),
        },
        "runtime": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "host": os.uname().nodename,
            "walltime_s": time.time() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "python": sys.version,
            "threads": 1,
            "offline": True,
        },
    }
    _write_exclusive(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
