#!/usr/bin/env python3
"""Sample compact count/count-valence Plans and render canonical rich Plans."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from crystal_dlm.fixed_slot import write_json  # noqa: E402
from crystal_dlm.h1_llm_planner import (  # noqa: E402
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    load_llama3_compatible_config,
)
from crystal_dlm.r5_plan_body import H1_RICH_PLAN_FORMAT, format_composition_plan  # noqa: E402
from crystal_dlm.r5_plan_state import (  # noqa: E402
    build_body_prompt,
    build_countfields_plan_prompt,
    build_countvalence_plan_prompt,
    parse_countfields_plan_state,
    parse_countvalence_plan_state,
    validate_plan_state,
)


LATTICE_TO_SPACEGROUP = {
    "triclinic": "sg_001_002",
    "monoclinic": "sg_003_015",
    "orthorhombic": "sg_016_074",
    "tetragonal": "sg_075_142",
    "trigonal": "sg_143_167",
    "hexagonal": "sg_168_194",
    "cubic": "sg_195_230",
}


def load_planner(model_path: str, checkpoint_path: str, device: torch.device):
    tokenizer_source = (
        checkpoint_path
        if (Path(checkpoint_path) / "tokenizer_config.json").exists()
        else model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=load_llama3_compatible_config(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to(device).eval()
    return model, tokenizer


def extract_compact_line(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    start = cleaned.upper().find("P01=")
    if start < 0:
        raise ValueError("generated continuation has no P01 field")
    line = cleaned[start:].splitlines()[0].strip()
    line = re.split(r"<\|endoftext\|>|</s>|<s>", line, maxsplit=1)[0].strip()
    return line


def n_bin(value: int) -> str:
    if value <= 4:
        return "01_04"
    if value <= 8:
        return "05_08"
    if value <= 12:
        return "09_12"
    return "13_20"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=("countfields", "countvalence"),
        required=True,
    )
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--max-atoms", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    if args.representation == "countvalence":
        prompt = build_countvalence_plan_prompt()
        parse_plan: Callable[..., dict[str, Any]] = parse_countvalence_plan_state
    else:
        prompt = build_countfields_plan_prompt()
        parse_plan = parse_countfields_plan_state

    model, tokenizer = load_planner(args.model_path, args.checkpoint_path, device)
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config.update(
        {
            "schema": "h1a2_llama_compact_planner_sampling_v1",
            "prompt": prompt,
            "public_output": "h1_rich_plan_v1",
            "rerank": False,
            "repair": False,
        }
    )
    write_json(str(args.output_dir / "run_config.json"), run_config)

    metrics: dict[str, Any] = {
        "requested": int(args.num_samples),
        "decoded": 0,
        "parsed": 0,
        "composition_valid": 0,
        "smact_valid": 0,
        "generated_charge_known": 0,
        "generated_charge_neutral": 0,
        "charge_bucket_match": 0,
        "lattice_spacegroup_match": 0,
        "failures": Counter(),
        "family": Counter(),
        "charge": Counter(),
        "lattice": Counter(),
        "spacegroup": Counter(),
        "volume": Counter(),
        "n_bin": Counter(),
        "formula": Counter(),
        "plan_text": Counter(),
    }
    start_time = time.time()
    raw_path = args.output_dir / "raw_generations.jsonl"
    plan_path = args.output_dir / "plans_for_dlm.jsonl"
    failure_path = args.output_dir / "failure_cases.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_handle, plan_path.open(
        "w", encoding="utf-8"
    ) as plan_handle, failure_path.open("w", encoding="utf-8") as failure_handle:
        progress = tqdm(total=int(args.num_samples), desc=f"Planner {args.representation}")
        offset = 0
        while offset < int(args.num_samples):
            batch_size = min(int(args.batch_size), int(args.num_samples) - offset)
            encoded = tokenizer(
                [prompt] * batch_size,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(
                outputs[:, input_ids.shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for local_idx, generated_text in enumerate(decoded):
                sample_idx = offset + local_idx
                metrics["decoded"] += 1
                raw_record: dict[str, Any] = {
                    "sample_idx": sample_idx,
                    "raw_model_text": generated_text,
                    "representation": args.representation,
                    "parsed": False,
                }
                try:
                    compact_text = extract_compact_line(generated_text)
                    plan = parse_plan(compact_text, max_atoms=int(args.max_atoms))
                    validation = validate_plan_state(plan, max_atoms=int(args.max_atoms))
                    if not validation.valid:
                        raise ValueError(f"invalid composition {validation.to_dict()}")
                    rich_text = format_composition_plan(
                        plan,
                        plan_style=H1_RICH_PLAN_FORMAT,
                    )
                    record = {
                        "sample_idx": sample_idx,
                        "plan_text": rich_text,
                        "compact_plan_text": compact_text,
                        "parsed_plan": plan,
                        "plan_state": plan,
                        "prompt": build_body_prompt(plan).rstrip() + "\n",
                        "representation": args.representation,
                    }
                    plan_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raw_record.update(record)
                    raw_record["parsed"] = True
                    metrics["parsed"] += 1
                    metrics["composition_valid"] += 1
                    metrics["smact_valid"] += int(
                        plan.get("validator", {}).get("valid") is True
                    )
                    if plan.get("generated_charge_sum_known") is True:
                        metrics["generated_charge_known"] += 1
                        metrics["generated_charge_neutral"] += int(
                            int(plan.get("generated_charge_sum") or 0) == 0
                        )
                    metrics["charge_bucket_match"] += int(
                        plan.get("charge_bucket_match") is True
                    )
                    metrics["lattice_spacegroup_match"] += int(
                        LATTICE_TO_SPACEGROUP.get(str(plan.get("lattice_system")))
                        == str(plan.get("spacegroup_bucket"))
                    )
                    metrics["family"][str(plan.get("anion_framework"))] += 1
                    metrics["charge"][str(plan.get("charge_bucket"))] += 1
                    metrics["lattice"][str(plan.get("lattice_system"))] += 1
                    metrics["spacegroup"][str(plan.get("spacegroup_bucket"))] += 1
                    metrics["volume"][str(plan.get("volume_per_atom_bin"))] += 1
                    metrics["n_bin"][n_bin(int(plan["N"]))] += 1
                    metrics["formula"][str(plan["formula"])] += 1
                    metrics["plan_text"][rich_text] += 1
                except Exception as exc:  # noqa: BLE001
                    reason = type(exc).__name__
                    metrics["failures"][reason] += 1
                    raw_record.update({"reason": reason, "message": str(exc)})
                    failure_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                progress.update(1)
            offset += batch_size

    decoded_denominator = max(1, int(metrics["decoded"]))
    parsed_denominator = max(1, int(metrics["parsed"]))
    metrics["elapsed_sec"] = time.time() - start_time
    metrics["parse_rate"] = int(metrics["parsed"]) / decoded_denominator
    metrics["composition_valid_rate"] = (
        int(metrics["composition_valid"]) / decoded_denominator
    )
    metrics["smact_valid_rate"] = int(metrics["smact_valid"]) / parsed_denominator
    metrics["generated_charge_neutral_rate"] = (
        int(metrics["generated_charge_neutral"]) / parsed_denominator
        if args.representation == "countvalence"
        else None
    )
    metrics["charge_bucket_match_rate"] = (
        int(metrics["charge_bucket_match"]) / parsed_denominator
        if args.representation == "countvalence"
        else None
    )
    metrics["lattice_spacegroup_match_rate"] = (
        int(metrics["lattice_spacegroup_match"]) / parsed_denominator
    )
    metrics["unique_formula_count"] = len(metrics.pop("formula"))
    metrics["unique_plan_count"] = len(metrics.pop("plan_text"))
    for key, value in list(metrics.items()):
        if isinstance(value, Counter):
            metrics[key] = dict(sorted(value.items()))
    write_json(str(args.output_dir / "sample_metrics.json"), metrics)
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
