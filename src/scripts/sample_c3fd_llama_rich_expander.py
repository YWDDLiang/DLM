#!/usr/bin/env python3
"""Sample strict H1-A2 rich Plans from C3FD-conditioned Llama Expanders."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crystal_dlm.c3fd_rich_expander import (  # noqa: E402
    FEATURE_DIM,
    ROUTE_FORMULA,
    ROUTE_SOFT_PREFIX,
    ROUTES,
    SoftPrefixProjector,
    SoftPrefixProjectorConfig,
    assemble_expanded_plan,
    build_expander_prompt,
)
from crystal_dlm.h1_llm_planner import (  # noqa: E402
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    load_llama3_compatible_config,
)
from crystal_dlm.r5_plan_body import (  # noqa: E402
    has_plan_end_marker,
    has_plan_tail_after_end_marker,
)
from crystal_dlm.r5_plan_state import build_body_prompt  # noqa: E402


END_MARKER = re.compile(r"(?im)^\s*end\s*:\s*plan\s*$")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def load_model(model_path: Path, checkpoint_path: Path, device: torch.device):
    tokenizer_source = (
        checkpoint_path
        if (checkpoint_path / "tokenizer_config.json").is_file()
        else model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = load_llama3_compatible_config(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to(device).eval()
    return model, tokenizer


def load_projector(checkpoint_path: Path, device: torch.device):
    config_path = checkpoint_path / "soft_prefix_projector_config.json"
    state_path = checkpoint_path / "soft_prefix_projector.pt"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("M checkpoint lacks soft-prefix projector files")
    config = SoftPrefixProjectorConfig(
        **json.loads(config_path.read_text(encoding="utf-8"))
    )
    if config.input_dim != FEATURE_DIM:
        raise ValueError("soft-prefix feature width changed")
    projector = SoftPrefixProjector(config).to(device)
    projector.load_state_dict(torch.load(state_path, map_location=device))
    projector.eval()
    return projector


def _sample_next(
    logits: torch.Tensor,
    *,
    generator: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    values = logits.float() / max(float(temperature), 1e-6)
    if int(top_k) > 0:
        cutoff = min(int(top_k), values.shape[-1])
        threshold = torch.topk(values, cutoff, dim=-1).values[..., -1:]
        values = values.masked_fill(values < threshold, float("-inf"))
    probabilities = torch.softmax(values, dim=-1)
    if float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch.sort(
            probabilities, descending=True, dim=-1
        )
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs > float(top_p)
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        sampled_sorted = torch.multinomial(
            sorted_probs, 1, generator=generator
        )
        return torch.gather(sorted_indices, -1, sampled_sorted)
    return torch.multinomial(probabilities, 1, generator=generator)


@torch.no_grad()
def generate_suffix(
    *,
    model,
    tokenizer,
    prompt: str,
    projector,
    features: list[float] | None,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    device = model_device(model)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_embeddings = model.get_input_embeddings()(input_ids)
    if projector is not None:
        if features is None or len(features) != FEATURE_DIM:
            raise ValueError("M sampling requires one valid feature vector")
        feature_tensor = torch.tensor(
            [features], dtype=torch.float32, device=device
        )
        prefix = projector(feature_tensor).to(dtype=token_embeddings.dtype)
        inputs_embeds = torch.cat([prefix, token_embeddings], dim=1)
        prefix_mask = torch.ones(
            (1, prefix.shape[1]), dtype=attention_mask.dtype, device=device
        )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
    else:
        if features is not None:
            raise ValueError("F sampling must not consume M feature vectors")
        inputs_embeds = token_embeddings
    output = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past = output.past_key_values
    logits = output.logits[:, -1, :]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    generated: list[int] = []
    for _step in range(int(max_new_tokens)):
        next_token = _sample_next(
            logits,
            generator=generator,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        token_id = int(next_token.item())
        generated.append(token_id)
        text = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if END_MARKER.search(text):
            return text[: END_MARKER.search(text).end()].strip()
        if tokenizer.eos_token_id is not None and token_id == int(
            tokenizer.eos_token_id
        ):
            break
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
            ],
            dim=1,
        )
        output = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        logits = output.logits[:, -1, :]
    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def plan_from_condition(row: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = row.get("expander_plan_state") or row.get("plan_state")
    if not isinstance(plan, Mapping):
        raise ValueError("condition row lacks plan_state")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--condition-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = list(iter_jsonl(args.condition_jsonl))[: int(args.num_samples)]
    if len(rows) != int(args.num_samples):
        raise ValueError("condition ledger is shorter than requested denominator")
    sample_indices = [int(row.get("sample_idx", index)) for index, row in enumerate(rows)]
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("condition sample_idx values are not unique")
    args.output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model_path, args.checkpoint_path, device)
    projector = (
        load_projector(args.checkpoint_path, device)
        if args.route == ROUTE_SOFT_PREFIX
        else None
    )
    raw_path = args.output_dir / "raw_generations.jsonl"
    plans_path = args.output_dir / "plans_for_dlm.jsonl"
    failure_path = args.output_dir / "failure_cases.jsonl"
    metrics = {
        "schema": "c3fd_llama_rich_expander_sampling_v1",
        "route": args.route,
        "requested": len(rows),
        "decoded": 0,
        "parsed": 0,
        "failed": 0,
        "formula_changed": 0,
        "retry_replacement_rerank": False,
        "failures": {},
    }
    started = time.time()
    with raw_path.open("x", encoding="utf-8") as raw_handle, plans_path.open(
        "x", encoding="utf-8"
    ) as plan_handle, failure_path.open("x", encoding="utf-8") as failure_handle:
        for index, row in enumerate(rows):
            sample_idx = sample_indices[index]
            plan = plan_from_condition(row)
            features = row.get("soft_prefix_features")
            if args.route == ROUTE_FORMULA:
                features = None
            record = {
                "sample_idx": sample_idx,
                "route": args.route,
                "parsed": False,
                "text": None,
                "plan_state": None,
                "failure": None,
            }
            try:
                suffix = generate_suffix(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=build_expander_prompt(tokenizer, plan),
                    projector=projector,
                    features=features,
                    seed=int(args.seed) * 1_000_003 + sample_idx,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                )
                metrics["decoded"] += 1
                assembled = assemble_expanded_plan(plan, suffix)
                if not has_plan_end_marker(assembled["plan_text"]):
                    raise ValueError("generated Plan lacks end marker")
                if has_plan_tail_after_end_marker(assembled["plan_text"]):
                    raise ValueError("generated Plan has a tail after end marker")
                record.update(
                    {
                        "parsed": True,
                        "text": suffix,
                        "plan_state": assembled["plan_state"],
                        "plan_text": assembled["plan_text"],
                    }
                )
                metrics["parsed"] += 1
                plan_handle.write(
                    json.dumps(
                        {
                            "sample_idx": sample_idx,
                            "plan_text": assembled["plan_text"],
                            "plan_state": assembled["plan_state"],
                            "prompt": build_body_prompt(
                                assembled["plan_state"]
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            except Exception as exc:  # noqa: BLE001 - all attempts are retained.
                reason = f"{type(exc).__name__}:{exc}"
                record["failure"] = reason
                metrics["failed"] += 1
                metrics["failures"][reason] = int(
                    metrics["failures"].get(reason, 0)
                ) + 1
                failure_handle.write(
                    json.dumps(
                        {
                            "sample_idx": sample_idx,
                            "reason": type(exc).__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            raw_handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    metrics["elapsed_sec"] = time.time() - started
    (args.output_dir / "sample_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_SUCCESS").touch(exist_ok=False)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
