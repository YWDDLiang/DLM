#!/usr/bin/env python3
"""Sample one frozen all-attempt G1 Planner arm with ordinal-keyed noise."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

SOURCE_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crystal_dlm.composition_validity import (
    classify_smact_validity,
    formula_from_composition,
)
from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.h1_llm_planner import (
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    canonical_plan_record_for_style,
    clean_generated_plan_text,
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    format_planner_prompt,
    load_llama3_compatible_config,
)
from crystal_dlm.plangraph_v1 import (
    PLANGRAPH_PLANNER_PROMPT,
    validate_plangraph,
)
from g1_protocol import (
    G1_ATTEMPTS,
    composition_fields,
    extract_first_json,
    plangraph_completion,
    rate,
    sha256_file,
    write_json,
)


PLAN_MARKER = re.compile(r"(?i)\bend\s*:\s*plan\b")


def load_ledger(path: Path, expected_sha256: str) -> list[dict[str, int]]:
    if sha256_file(path) != str(expected_sha256):
        raise ValueError("G1 seed ledger SHA mismatch")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(
                    {
                        "ordinal": int(row["ordinal"]),
                        "seed": int(row["seed"]),
                    }
                )
    if len(rows) != G1_ATTEMPTS:
        raise ValueError(f"G1 seed ledger has {len(rows)} rows")
    if [row["ordinal"] for row in rows] != list(range(G1_ATTEMPTS)):
        raise ValueError("G1 seed ledger ordinals are not exactly 0..511")
    return rows


def load_model(model_path: str, checkpoint_path: str, device: torch.device):
    tokenizer_source = (
        checkpoint_path
        if (Path(checkpoint_path) / "tokenizer_config.json").exists()
        else model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = load_llama3_compatible_config(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to(device).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model, tokenizer


def stateless_uniform(seed: int, step: int) -> float:
    digest = hashlib.sha256(
        f"plangraph_g1_sampling_uniform_v1\0{int(seed)}\0{int(step)}".encode(
            "utf-8"
        )
    ).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 0.5) / float(1 << 64)


def sample_from_logits(
    logits: torch.Tensor,
    *,
    uniform: float,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    values = logits.float() / float(temperature)
    k = min(int(top_k), int(values.numel()))
    if k > 0 and k < int(values.numel()):
        threshold = torch.topk(values, k=k).values[-1]
        values = values.masked_fill(values < threshold, float("-inf"))
    sorted_logits, sorted_indices = torch.sort(values, descending=True)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    if float(top_p) < 1.0:
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = (cumulative - probabilities) >= float(top_p)
        probabilities = probabilities.masked_fill(remove, 0.0)
        probabilities = probabilities / probabilities.sum().clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
    cumulative = torch.cumsum(probabilities, dim=-1)
    u = torch.tensor(float(uniform), device=cumulative.device)
    position = int(torch.searchsorted(cumulative, u, right=False).item())
    position = min(position, int(sorted_indices.numel()) - 1)
    return int(sorted_indices[position].item())


def complete_continuation(arm: str, text: str) -> bool:
    if arm == "P0":
        return PLAN_MARKER.search(text) is not None
    try:
        extract_first_json(text)
        return True
    except Exception:
        return False


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    *,
    arm: str,
    prompt: str,
    seeds: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: torch.device,
) -> list[str]:
    prompts = [prompt] * len(seeds)
    encoded = tokenizer(
        prompts,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past = output.past_key_values
    logits = output.logits[:, -1, :]
    generated: list[list[int]] = [[] for _ in seeds]
    finished = [False] * len(seeds)
    eos = tokenizer.eos_token_id
    pad = int(tokenizer.pad_token_id)
    for step in range(int(max_new_tokens)):
        next_tokens = []
        for row, seed in enumerate(seeds):
            if finished[row]:
                next_tokens.append(pad)
                continue
            token = sample_from_logits(
                logits[row],
                uniform=stateless_uniform(seed, step),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
            if eos is not None and token == int(eos):
                finished[row] = True
                next_tokens.append(pad)
                continue
            generated[row].append(token)
            next_tokens.append(token)
            text = tokenizer.decode(
                generated[row],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if complete_continuation(arm, text):
                finished[row] = True
        if all(finished):
            break
        next_tensor = torch.tensor(
            next_tokens,
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        active = torch.tensor(
            [0 if done else 1 for done in finished],
            dtype=attention_mask.dtype,
            device=device,
        ).unsqueeze(1)
        attention_mask = torch.cat([attention_mask, active], dim=1)
        output = model(
            input_ids=next_tensor,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        logits = output.logits[:, -1, :]
    return [
        tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for tokens in generated
    ]


def canonical_formula(symbols: list[str], counts: list[int]) -> str:
    pairs = sorted(
        ((SYMBOL_TO_Z[symbol], int(count)) for symbol, count in zip(symbols, counts)),
        key=lambda item: item[0],
    )
    divisor = 0
    for _, count in pairs:
        divisor = math.gcd(divisor, int(count))
    divisor = max(1, divisor)
    return formula_from_composition(
        [atomic_number for atomic_number, _ in pairs],
        [count // divisor for _, count in pairs],
    )


def chemistry(
    symbols: list[str],
    counts: list[int],
) -> tuple[dict[str, Any], str]:
    if len(symbols) != len(counts) or not symbols:
        raise ValueError("invalid composition arity")
    atomic_numbers = [int(SYMBOL_TO_Z[symbol]) for symbol in symbols]
    classification = dict(
        classify_smact_validity(
            atomic_numbers,
            [int(value) for value in counts],
        )
    )
    return classification, canonical_formula(symbols, counts)


def parse_attempt(arm: str, raw_text: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "parsed": False,
        "plan_complete": False,
        "strict_schema_valid": None,
        "composition_valid": False,
        "failure_stage": None,
        "failure_reason": None,
    }
    if arm == "P0":
        cleaned = clean_generated_plan_text(
            raw_text,
            prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
            truncate_after_marker=True,
        )
        plan_record = canonical_plan_record_for_style(
            cleaned,
            sample_idx=None,
            max_atoms=20,
            prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
        )
        plan = plan_record["plan_state"]
        symbols = [str(value) for value in plan["elements"]]
        counts = [int(value) for value in plan["counts"]]
        classification, formula = chemistry(symbols, counts)
        record.update(
            {
                "parsed": True,
                "plan_complete": True,
                "composition_valid": classification.get("valid") is True,
                "chemistry_reason": classification.get("reason"),
                "formula": formula,
                "num_atoms": int(sum(counts)),
                "element_arity": len(symbols),
                "single_element": len(symbols) == 1,
                "all_metal": classification.get("reason")
                == "all_metal_shortcut",
                "lattice_system": str(plan.get("lattice_system") or "unknown"),
                "spacegroup_bucket": str(
                    plan.get("spacegroup_bucket") or "sg_unknown"
                ),
                "anion_framework": str(plan.get("anion_framework") or "other"),
                "charge_bucket": str(
                    plan.get("charge_bucket") or "validator_unavailable"
                ),
                "canonical_plan": plan,
                "raw_plan_text": cleaned,
            }
        )
        return record

    graph, consumed = extract_first_json(raw_text)
    complete, completion_errors = plangraph_completion(graph)
    validation = validate_plangraph(graph)
    symbols, counts = composition_fields(graph)
    classification, formula = chemistry(symbols, counts)
    composition = graph["composition"]
    oxidation_candidates = composition.get("oxidation_candidates")
    oxidation_neutral = False
    if isinstance(oxidation_candidates, list):
        for candidate in oxidation_candidates:
            if (
                isinstance(candidate, list)
                and len(candidate) == len(counts)
                and all(isinstance(value, int) for value in candidate)
                and sum(
                    int(count) * int(oxidation)
                    for count, oxidation in zip(counts, candidate)
                )
                == 0
            ):
                oxidation_neutral = True
                break
    record.update(
        {
            "parsed": True,
            "plan_complete": complete,
            "strict_schema_valid": validation.valid,
            "schema_errors": list(validation.errors),
            "completion_errors": completion_errors,
            "composition_valid": classification.get("valid") is True,
            "chemistry_reason": classification.get("reason"),
            "oxidation_candidate_neutral": oxidation_neutral,
            "formula": formula,
            "num_atoms": int(sum(counts)),
            "element_arity": len(symbols),
            "single_element": len(symbols) == 1,
            "all_metal": classification.get("reason") == "all_metal_shortcut",
            "lattice_system": str(
                (graph.get("symmetry") or {}).get("lattice_system") or "unknown"
            ),
            "spacegroup_bucket": str(
                (graph.get("symmetry") or {}).get("spacegroup_bucket")
                or "sg_unknown"
            ),
            "anion_framework": str(
                composition.get("anion_framework") or "other"
            ),
            "charge_bucket": str(
                composition.get("charge_bucket") or "validator_unavailable"
            ),
            "canonical_plan": graph,
            "json_consumed_characters": consumed,
        }
    )
    return record


def histogram(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(
        str(record[field])
        for record in records
        if record.get("parsed") and record.get(field) is not None
    )
    return {key: int(counts[key]) for key in sorted(counts)}


def summarize(
    arm: str,
    attempts: list[dict[str, Any]],
    *,
    ledger_sha256: str,
    checkpoint_path: str,
    checkpoint_identity_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    parsed = [record for record in attempts if record.get("parsed")]
    formulas = {
        str(record["formula"])
        for record in parsed
        if record.get("formula")
    }
    failures = Counter(
        f"{record.get('failure_stage')}:{record.get('failure_reason')}"
        for record in attempts
        if not record.get("parsed")
    )
    chemistry_failures = Counter(
        str(record.get("chemistry_reason"))
        for record in parsed
        if record.get("composition_valid") is not True
    )
    report = {
        "schema": "plangraph-dlm-g1-planner-report@1",
        "status": "complete",
        "arm": arm,
        "attempts": len(attempts),
        "all_attempt_denominator": len(attempts),
        "ordinal_start": 0,
        "ordinal_end_inclusive": len(attempts) - 1,
        "seed_ledger_sha256": ledger_sha256,
        "checkpoint_path": checkpoint_path,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "parse_count": len(parsed),
        "parse_rate": rate(len(parsed), len(attempts)),
        "plan_completion_count": sum(
            int(record.get("plan_complete") is True) for record in attempts
        ),
        "plan_completion_rate": rate(
            sum(int(record.get("plan_complete") is True) for record in attempts),
            len(attempts),
        ),
        "strict_schema_valid_count": (
            None
            if arm == "P0"
            else sum(
                int(record.get("strict_schema_valid") is True)
                for record in attempts
            )
        ),
        "composition_valid_count": sum(
            int(record.get("composition_valid") is True) for record in attempts
        ),
        "composition_valid_rate": rate(
            sum(
                int(record.get("composition_valid") is True)
                for record in attempts
            ),
            len(attempts),
        ),
        "unique_formula_count": len(formulas),
        "unique_formula_rate_all_attempt": rate(len(formulas), len(attempts)),
        "single_element_rate": rate(
            sum(int(record.get("single_element") is True) for record in attempts),
            len(attempts),
        ),
        "all_metal_rate": rate(
            sum(int(record.get("all_metal") is True) for record in attempts),
            len(attempts),
        ),
        "oxidation_candidate_neutral_rate": (
            None
            if arm == "P0"
            else rate(
                sum(
                    int(record.get("oxidation_candidate_neutral") is True)
                    for record in attempts
                ),
                len(attempts),
            )
        ),
        "num_atoms_histogram": histogram(attempts, "num_atoms"),
        "element_arity_histogram": histogram(attempts, "element_arity"),
        "formula_histogram": histogram(attempts, "formula"),
        "lattice_system_histogram": histogram(attempts, "lattice_system"),
        "spacegroup_bucket_histogram": histogram(
            attempts, "spacegroup_bucket"
        ),
        "anion_framework_histogram": histogram(attempts, "anion_framework"),
        "charge_bucket_histogram": histogram(attempts, "charge_bucket"),
        "parse_failures": dict(sorted(failures.items())),
        "chemistry_failures": dict(sorted(chemistry_failures.items())),
        "elapsed_seconds": float(elapsed_seconds),
        "sample_id_in_prompt": False,
        "retry": False,
        "replacement": False,
        "repair": False,
        "filter": False,
        "rerank": False,
        "crystal_generation": False,
        "energy_or_sun_signal": False,
    }
    if report["attempts"] != G1_ATTEMPTS:
        raise ValueError("G1 report lost attempts")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("P0", "PG", "PG-shuffle"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-identity-sha256", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    frozen = {
        "batch_size": 8,
        "max_new_tokens": 640,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 50,
    }
    for key, expected in frozen.items():
        if getattr(args, key) != expected:
            raise ValueError(f"G1 freezes {key}={expected}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("G1 sampling requires exactly one visible GPU")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    ledger = load_ledger(args.ledger, args.ledger_sha256)
    loaded = time.time()
    model, tokenizer = load_model(
        args.model_path,
        args.checkpoint_path,
        device,
    )
    model_load_seconds = time.time() - loaded
    if args.arm == "P0":
        prompt = format_planner_prompt(
            tokenizer,
            sample_idx=None,
            prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
        )
    else:
        prompt = PLANGRAPH_PLANNER_PROMPT
    if "sample_id" in prompt.lower():
        raise ValueError("G1 prompt contains sample_id")

    attempts: list[dict[str, Any]] = []
    started = time.time()
    for offset in range(0, len(ledger), int(args.batch_size)):
        batch = ledger[offset : offset + int(args.batch_size)]
        decoded = generate_batch(
            model,
            tokenizer,
            arm=args.arm,
            prompt=prompt,
            seeds=[row["seed"] for row in batch],
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            device=device,
        )
        for ledger_row, text in zip(batch, decoded):
            attempt: dict[str, Any] = {
                "ordinal": int(ledger_row["ordinal"]),
                "seed": int(ledger_row["seed"]),
                "arm": args.arm,
                "raw_model_text": text,
            }
            try:
                attempt.update(parse_attempt(args.arm, text))
            except Exception as exc:
                attempt.update(
                    {
                        "parsed": False,
                        "plan_complete": False,
                        "strict_schema_valid": None,
                        "composition_valid": False,
                        "failure_stage": "parse_or_validate",
                        "failure_reason": type(exc).__name__,
                        "failure_message": str(exc)[:1000],
                    }
                )
            attempts.append(attempt)
        print(
            json.dumps(
                {
                    "event": "sampling_progress",
                    "arm": args.arm,
                    "attempts_complete": len(attempts),
                    "attempts_total": G1_ATTEMPTS,
                    "elapsed_seconds": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if len(attempts) != G1_ATTEMPTS:
        raise RuntimeError("G1 sampling lost attempts")
    attempts_path = args.output_dir / "attempts.jsonl"
    with attempts_path.open("x", encoding="utf-8") as handle:
        for attempt in attempts:
            handle.write(
                json.dumps(
                    attempt,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    report = summarize(
        args.arm,
        attempts,
        ledger_sha256=str(args.ledger_sha256),
        checkpoint_path=args.checkpoint_path,
        checkpoint_identity_sha256=args.checkpoint_identity_sha256,
        elapsed_seconds=time.time() - started,
    )
    report.update(
        {
            "model_path": args.model_path,
            "model_load_seconds": model_load_seconds,
            "attempts_sha256": sha256_file(attempts_path),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "sampling": frozen,
            "ordinal_noise": "sha256(seed, token_step) inverse-CDF",
        }
    )
    write_json(args.output_dir / "planner_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

