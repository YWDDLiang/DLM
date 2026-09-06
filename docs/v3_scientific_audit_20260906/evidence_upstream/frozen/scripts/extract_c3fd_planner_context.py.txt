#!/usr/bin/env python3
"""Extract the frozen P0 Planner context used by the C³FD semantic head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crystal_dlm.h1_llm_planner import (  # noqa: E402
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL,
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    format_planner_prompt,
    load_llama3_compatible_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_source = (
        args.checkpoint_path
        if (Path(args.checkpoint_path) / "tokenizer_config.json").exists()
        else args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    config = load_llama3_compatible_config(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.checkpoint_path, is_trainable=False)
    model.to(device)
    model.eval()
    prompt = format_planner_prompt(
        tokenizer,
        sample_idx=None,
        prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL,
    )
    tokenized = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    with torch.inference_mode():
        output = model(**tokenized, output_hidden_states=True, use_cache=False)
    context = output.hidden_states[-1][:, -1, :].detach().float().cpu()
    payload = {
        "schema": "h1a2_c3fd_frozen_planner_context_v1",
        "context": context,
        "context_size": int(context.shape[-1]),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_style": H1_PLANNER_PROMPT_STYLE_RICH_PLAN_PREFILL,
        "model_path": str(Path(args.model_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "weights_trainable": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    manifest = {key: value for key, value in payload.items() if key != "context"}
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
