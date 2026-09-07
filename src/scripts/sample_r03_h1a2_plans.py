#!/usr/bin/env python3
"""Sample original H1A2 rich Plans, optionally constraining formula via C3FD.

The native prompt is read verbatim from a prepared asset.  P0, its tokenizer,
sampling defaults, EOS handling, and batch-wide end-marker rule are unchanged.
Both output ledgers retain every requested row, including failed rich Plans or
failed terminal certificates.  No formula/rich values are supplied as repairs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from crystal_dlm.h1_llm_planner import (
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    H1_PLANNER_PROMPT_VERSION,
    canonical_plan_record_for_style,
    clean_generated_plan_text,
    format_planner_prompt,
)
from crystal_dlm.r03_control_pointer import sha256_file, stable_digest
from crystal_dlm.r03_formula_bridge import C3FDFormulaOracle, R03C3FDFormulaLogitsProcessor
from crystal_dlm.r5_plan_body import has_plan_end_marker, has_plan_tail_after_end_marker, symbol_counts_from_formula
from scripts.train_r03_control_pointer import load_frozen_p0, write_json


SCHEMA = "r03_native_h1a2_plan_sampling_v1"
SAMPLING_DEFAULTS = {"max_new_tokens": 96, "temperature": 0.9, "top_p": 0.95,
                     "top_k": 50, "do_sample": True}
FORMULA_LINE = re.compile(r"(?im)^[ \t]*formula[ \t]*:[ \t]*([^\r\n]+)")


class NativePlanEndStoppingCriteria:
    """The original batch-wide end:plan predicate, without per-row forcing."""

    def __init__(self, tokenizer: Any, start_length: int):
        self.tokenizer = tokenizer
        self.start_length = int(start_length)
        self.marker = re.compile(r"(?i)\bend\s*:\s*plan\b")

    def __call__(self, input_ids: Any, scores: Any, **kwargs) -> bool:
        generated_ids = input_ids[:, self.start_length:]
        if generated_ids.numel() == 0:
            return False
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        return all(self.marker.search(text) is not None for text in decoded)


def read_domain(path: Path) -> tuple[C3FDFormulaOracle, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), dict):
        raise ValueError("C3FD domain must contain a declared symbol-to-oxidation-states nodes mapping")
    eneg = payload["electronegativities"] if "electronegativities" in payload else payload["eneg"]
    metals = payload["metal_symbols"] if "metal_symbols" in payload else payload["metals"]
    max_atoms = payload["max_atoms"] if "max_atoms" in payload else payload["max"]
    max_species = payload["max_species"]
    strata = payload["allowed_strata"]
    if (not isinstance(eneg, dict) or not isinstance(metals, list) or not isinstance(strata, list)
            or not strata or max_atoms != 20 or not isinstance(max_species, int) or not 1 <= max_species <= 20):
        raise ValueError("C3FD domain must explicitly preserve MP20 bounds, eneg/metals, and nonempty allowed strata")
    oracle = C3FDFormulaOracle(
        nodes=payload["nodes"], electronegativities=eneg, metal_symbols=metals,
        max_atoms=max_atoms, max_species=max_species, allowed_strata=strata,
    )
    return oracle, {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "schema": payload.get("schema"), "max_atoms": max_atoms, "max_species": max_species,
        "symbols": len(payload["nodes"]), "declared_strata": len(oracle.strata),
    }


def generation_arguments(
    tokenizer: Any, input_ids: Any, attention_mask: Any, *, processor: Any = None, containers: Any = None,
) -> dict[str, Any]:
    if containers is None:
        from transformers import LogitsProcessorList, StoppingCriteriaList
    else:
        LogitsProcessorList = containers.LogitsProcessorList
        StoppingCriteriaList = containers.StoppingCriteriaList
    arguments = {
        "input_ids": input_ids, "attention_mask": attention_mask, **SAMPLING_DEFAULTS,
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
        "stopping_criteria": StoppingCriteriaList([
            NativePlanEndStoppingCriteria(tokenizer, int(input_ids.shape[1])),
        ]),
    }
    if processor is not None:
        if int(processor.start_length) != int(input_ids.shape[1]):
            raise ValueError("C3FD formula processor prompt boundary changed")
        arguments["logits_processor"] = LogitsProcessorList([processor])
    return arguments


def decode_record(
    *, tokenizer: Any, prompt_ids: Sequence[int], generated_ids: Sequence[int],
    sample_idx: int, local_idx: int, seed: int, oracle: C3FDFormulaOracle | None,
    processor: Any = None,
) -> dict[str, Any]:
    exact_decoded = str(tokenizer.decode(list(generated_ids), skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False))
    raw_model_text = clean_generated_plan_text(exact_decoded, prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
                                               truncate_after_marker=False)
    raw_plan_text = clean_generated_plan_text(raw_model_text, prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
                                              truncate_after_marker=True)
    record = {
        "schema": SCHEMA, "sample_idx": int(sample_idx), "local_sample_idx": int(local_idx),
        "seed": int(seed), "attempt_id": f"r03-planner:{seed}:{sample_idx}",
        "prompt_version": H1_PLANNER_PROMPT_VERSION, "prompt_style": H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
        "include_sample_id": False, "decoded_continuation": exact_decoded,
        "raw_model_text": raw_model_text, "raw_plan_text": raw_plan_text,
        "prompt_input_ids": [int(value) for value in prompt_ids],
        "generated_token_ids": [int(value) for value in generated_ids],
        "parsed": False, "plan_parse_success": False, "body_eligible": False,
        "attempt_status": "planner_parse_failure", "reason": None,
        "plan_end_marker_present": has_plan_end_marker(raw_plan_text),
        "plan_tail_after_end_marker": has_plan_tail_after_end_marker(raw_model_text),
        "c3fd_enabled": oracle is not None, "c3fd_terminal_valid": None, "c3fd_failure_reason": None,
        "retry_used": False, "replacement_used": False,
    }
    # A formula may close correctly even when another rich field is missing.
    # Record the chemical certificate separately from complete Plan parsing.
    if oracle is not None:
        match = FORMULA_LINE.search(raw_model_text)
        formula = match.group(1).strip() if match else None
        terminal_valid = formula is not None and oracle.is_terminal_valid(formula)
        record.update(c3fd_checked_formula=formula, c3fd_terminal_valid=bool(terminal_valid))
        if not terminal_valid:
            record["c3fd_failure_reason"] = "missing_formula_field" if formula is None else "terminal_formula_outside_declared_support"
        if processor is not None:
            matching = [(prefix, reason) for prefix, reason in processor.failures.items() if exact_decoded.startswith(prefix)]
            if matching:
                prefix, reason = max(matching, key=lambda pair: len(pair[0]))
                record["c3fd_processor_failure"] = {"reason": reason, "generated_prefix": prefix}
                record["c3fd_failure_reason"] = reason
    try:
        canonical = canonical_plan_record_for_style(
            raw_plan_text, sample_idx=sample_idx, max_atoms=20, prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
        )
        record.update(canonical, plan_parse_success=True)
        record["body_prompt"] = canonical["prompt"]
        record["body_prompt_sha256"] = hashlib.sha256(canonical["prompt"].encode("utf-8")).hexdigest()
        if oracle is not None:
            # The historical parser may accept a later duplicate formula
            # field.  Rich decoding is unconstrained after the first formula
            # newline, so verify what the downstream parser actually selected
            # and refuse any composition change after the certified closure.
            parsed_formula = canonical["plan_state"]["formula"]
            parsed_terminal = oracle.is_terminal_valid(parsed_formula)
            record["c3fd_parsed_formula_terminal_valid"] = parsed_terminal
            record["c3fd_composition_preserved_after_closure"] = False
            if not parsed_terminal:
                if record["c3fd_failure_reason"] is None:
                    record["c3fd_failure_reason"] = "parsed_formula_outside_declared_support"
                record["c3fd_terminal_valid"] = False
            elif record["c3fd_terminal_valid"] is True:
                raw_composition = symbol_counts_from_formula(record["c3fd_checked_formula"])
                parsed_composition = symbol_counts_from_formula(parsed_formula)
                unchanged = raw_composition == parsed_composition
                record["c3fd_composition_preserved_after_closure"] = unchanged
                if not unchanged:
                    record["c3fd_failure_reason"] = "parsed_composition_changed_after_formula_closure"
        if oracle is not None and (record["c3fd_terminal_valid"] is not True or record["c3fd_failure_reason"] is not None):
            record.update(attempt_status="c3fd_terminal_failure", reason=record["c3fd_failure_reason"])
        else:
            record.update(parsed=True, body_eligible=True, attempt_status="complete", reason=None)
    except (ValueError, TypeError, KeyError) as exc:
        record.update(reason=type(exc).__name__, message=str(exc), attempt_status="planner_parse_failure")
    if not record["body_eligible"]:
        record["ineligible_reason"] = record["reason"] or record["c3fd_failure_reason"] or "planner_parse_failure"
    return record


def sample_batch(
    model: Any, tokenizer: Any, *, native_prompt: str, global_ids: Sequence[int],
    local_ids: Sequence[int], seed: int, oracle: C3FDFormulaOracle | None,
    processor: Any = None, containers: Any = None,
) -> list[dict[str, Any]]:
    import torch
    if len(global_ids) != len(local_ids) or not 1 <= len(global_ids) <= 4:
        raise ValueError("native Planner batch needs aligned IDs and at most four rows")
    encoded = tokenizer([native_prompt] * len(global_ids), padding=True,
                        add_special_tokens=False, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device)
    kwargs = generation_arguments(tokenizer, input_ids, attention, processor=processor, containers=containers)
    with torch.no_grad():
        outputs = model.generate(**kwargs)
    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
    if sequences.ndim != 2 or sequences.shape[0] != len(global_ids) or sequences.shape[1] < input_ids.shape[1]:
        raise RuntimeError("native Planner changed the one-result-per-request output shape")
    if not torch.equal(sequences[:, :input_ids.shape[1]], input_ids):
        raise RuntimeError("native Planner returned a changed prompt prefix")
    generated = sequences[:, input_ids.shape[1]:].detach().cpu().tolist()
    prompts = input_ids.detach().cpu().tolist()
    return [decode_record(
        tokenizer=tokenizer, prompt_ids=prompt, generated_ids=tokens, sample_idx=sample_idx,
        local_idx=local_idx, seed=seed, oracle=oracle, processor=processor,
    ) for prompt, tokens, sample_idx, local_idx in zip(prompts, generated, global_ids, local_ids)]


def summarize(rows: Sequence[Mapping[str, Any]], *, requested: int, elapsed: float) -> dict[str, Any]:
    method = any(row["c3fd_enabled"] for row in rows)
    terminal_rows = sum(row["c3fd_terminal_valid"] is True for row in rows)
    successful = sum(row["body_eligible"] is True for row in rows)
    return {
        "schema": SCHEMA, "requested_samples": requested, "decoded_samples": len(rows),
        "completed_request_rows": len(rows), "not_completed_requests": requested - len(rows),
        "plan_parse_success": sum(row["plan_parse_success"] for row in rows),
        "body_eligible": successful, "plan_parse_rate": successful / max(1, requested),
        "c3fd_enabled": method, "c3fd_terminal_valid": terminal_rows if method else None,
        "c3fd_terminal_failure_rows": sum(row["c3fd_failure_reason"] is not None for row in rows) if method else 0,
        "plan_end_marker_success": sum(row["plan_end_marker_present"] for row in rows),
        "plan_tail_after_end_marker": sum(row["plan_tail_after_end_marker"] for row in rows),
        "failures": dict(Counter(str(row["reason"] or "unknown") for row in rows if not row["body_eligible"])),
        "denominator_policy": "all_requested_rows_including_failed_rich_plans_or_terminal_certificates",
        "time_sec": elapsed, "replacement_sampling": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c3fd-domain", type=Path)
    parser.add_argument("--native-prompt-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-index-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.num_samples < 1 or args.sample_index_offset < 0 or not 1 <= args.batch_size <= 4:
        raise ValueError("positive request count, nonnegative global offset and batch size in 1..4 required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("native entry uses one process/GPU; caller registers separate seed/shard commands")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    native_prompt = args.native_prompt_file.read_bytes().decode("utf-8")
    if not native_prompt:
        raise ValueError("the native historical prompt file is empty")
    oracle, domain_identity = read_domain(args.c3fd_domain) if args.c3fd_domain else (None, None)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    requested_ids = list(range(args.sample_index_offset, args.sample_index_offset + args.num_samples))
    with (args.output_dir / "requested_attempts.jsonl").open("x", encoding="utf-8") as handle:
        for local_idx, sample_idx in enumerate(requested_ids):
            handle.write(json.dumps({"sample_idx": sample_idx, "local_sample_idx": local_idx,
                                     "seed": args.seed, "attempt_status": "registered"}) + "\n")
    import torch

    # Preserve native seeding order: seed once before loading P0, then let the
    # same process RNG advance across successive batches without per-row reset.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = []
    started = time.monotonic()
    try:
        model, tokenizer, identity = load_frozen_p0(args.model_path, args.checkpoint_path, device=args.device)
        if native_prompt != format_planner_prompt(tokenizer, sample_idx=None, prompt_style=H1_PLANNER_PROMPT_STYLE_RICH_PLAN):
            raise ValueError("prepared prompt differs from the original no-sample-ID H1A2 rich prompt")
        prompt_ids = list(tokenizer(native_prompt, add_special_tokens=False)["input_ids"])
        processor = (R03C3FDFormulaLogitsProcessor(tokenizer, oracle=oracle, start_length=len(prompt_ids),
                                                  eos_token_id=int(tokenizer.eos_token_id)) if oracle else None)
        run_config = {
            "schema": SCHEMA, "planner_identity": identity, "native_prompt_file": str(args.native_prompt_file.resolve()),
            "native_prompt_sha256": hashlib.sha256(native_prompt.encode("utf-8")).hexdigest(),
            "native_prompt_input_ids_sha256": stable_digest(prompt_ids), "native_prompt_tokens": len(prompt_ids),
            "c3fd_domain": domain_identity, "sampling": dict(SAMPLING_DEFAULTS),
            "num_samples": args.num_samples, "sample_index_offset": args.sample_index_offset,
            "seed": args.seed, "batch_size": args.batch_size,
            "seed_mode": "single_process_seed_once_before_P0_load_no_ordinal_reseed",
            "sample_index_offset_changes_prompt_or_rng": False,
            "prompt_style": H1_PLANNER_PROMPT_STYLE_RICH_PLAN, "include_sample_id": False,
            "stop_after_plan_marker": True, "truncate_after_plan_marker": True,
            "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
            "planner_weights_changed": False, "tokenizer_resized": False,
            "formula_prefill_added": False, "external_formula_composer": False,
            "formula_bridge_source_sha256": sha256_file(PROJECT_ROOT / "src/crystal_dlm/r03_formula_bridge.py"),
        }
        write_json(args.output_dir / "run_config.json", run_config)
        with (args.output_dir / "raw_generations.jsonl").open("x", encoding="utf-8") as raw_handle, \
                (args.output_dir / "plans_for_dlm.jsonl").open("x", encoding="utf-8") as plans_handle, \
                (args.output_dir / "failure_cases.jsonl").open("x", encoding="utf-8") as failure_handle:
            for begin in range(0, args.num_samples, args.batch_size):
                ids = requested_ids[begin:begin + args.batch_size]
                batch_rows = sample_batch(
                    model, tokenizer, native_prompt=native_prompt, global_ids=ids,
                    local_ids=list(range(begin, begin + len(ids))), seed=args.seed,
                    oracle=oracle, processor=processor,
                )
                for row in batch_rows:
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                    raw_handle.write(line)
                    plans_handle.write(line)
                    if not row["body_eligible"]:
                        failure_handle.write(line)
                for handle in (raw_handle, plans_handle, failure_handle):
                    handle.flush()
                rows.extend(batch_rows)
                progress = summarize(rows, requested=args.num_samples, elapsed=time.monotonic() - started)
                write_json(args.output_dir / "sample_metrics.json", progress)
                print(json.dumps({"event": "r03_native_planner_progress", **progress}, ensure_ascii=False), flush=True)
        if [row["sample_idx"] for row in rows] != requested_ids:
            raise RuntimeError("native sampling dropped, duplicated or reordered a registered request")
        final = summarize(rows, requested=args.num_samples, elapsed=time.monotonic() - started)
        if processor is not None:
            final["c3fd_processor_stats"] = processor.stats()
            final["c3fd_oracle_cache_stats"] = oracle.stats()
        write_json(args.output_dir / "sample_metrics.json", final)
        (args.output_dir / "_SUCCESS").touch()
        print(json.dumps(final, ensure_ascii=False, sort_keys=True), flush=True)
    except BaseException as exc:
        write_json(args.output_dir / "RUN_ERROR.json", {
            "error_type": type(exc).__name__, "message": str(exc), "requested_samples": args.num_samples,
            "completed_rows": len(rows), "not_completed_requests": args.num_samples - len(rows),
            "unstarted_or_interrupted_are_scientific_failures": False,
        })
        (args.output_dir / "_FAILED").touch()
        raise


if __name__ == "__main__":
    main()
