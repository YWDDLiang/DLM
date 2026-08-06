#!/usr/bin/env python3
"""Sample H1 Llama formula plans for LLM-plan + DLM-body hybrid generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from crystal_dlm.h1_crplan import (  # noqa: E402
    CRPLAN_MODES,
    CRPLAN_SCHEMA,
    MISSING_STATE_POLICIES,
    CRPlanDeadEndError,
    CRPlanIdentityError,
    CRPlanLogitsProcessor,
    CRPlanTokenVocabulary,
    certificate_for_symbol_counts,
    load_frozen_smact_table,
    validate_crplan_parsed_identity,
)
from crystal_dlm.h1_llm_planner import (  # noqa: E402
    H1_PLANNER_PROMPT_VERSION,
    H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL,
    H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    canonical_plan_record_for_style,
    clean_generated_plan_text,
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    format_planner_prompt,
    load_llama3_compatible_config,
    normalize_prompt_style,
)
from crystal_dlm.ordinal_rng import (  # noqa: E402
    derive_ordinal_seed,
    ordered_ordinal_records,
)
from crystal_dlm.r5_plan_body import has_plan_end_marker, has_plan_tail_after_end_marker  # noqa: E402
from crystal_dlm.fixed_slot import write_json  # noqa: E402
from scripts.sample_llada_dynamic_crystals import init_distributed, rank_path  # noqa: E402


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def load_planner(model_path: str, checkpoint_path: str | None, device: torch.device):
    tokenizer_source = checkpoint_path if checkpoint_path and (Path(checkpoint_path) / "tokenizer_config.json").exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = load_llama3_compatible_config(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if checkpoint_path:
        ensure_peft_cache_compat()
        disable_peft_bnb_autodetect()
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to(device).eval()
    return model, tokenizer


class GeneratedPlanEndStoppingCriteria(StoppingCriteria):
    """Stop the batch once every generated continuation contains end: plan."""

    def __init__(self, tokenizer, start_length: int) -> None:
        self.tokenizer = tokenizer
        self.start_length = int(start_length)
        self.marker = re.compile(r"(?i)\bend\s*:\s*plan\b")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # noqa: D401
        generated_ids = input_ids[:, self.start_length :]
        if generated_ids.numel() == 0:
            return False
        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return all(self.marker.search(text) is not None for text in decoded)


def add_failure(
    metrics: Dict[str, Any],
    failure_handle,
    sample_idx: int,
    stage: str,
    exc: Exception,
    raw_text: str,
    *,
    sampling_seed: int | None = None,
) -> None:
    reason = f"{stage}:{type(exc).__name__}"
    metrics["failures"][reason] = int(metrics["failures"].get(reason, 0)) + 1
    failure_handle.write(
        json.dumps(
            {
                "sample_idx": int(sample_idx),
                "stage": stage,
                "reason": type(exc).__name__,
                "message": str(exc),
                "raw_plan_text": raw_text,
                "planner_sampling_seed": sampling_seed,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def finalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    decoded = max(1, int(metrics.get("decoded_samples", 0)))
    metrics["plan_parse_rate"] = float(metrics.get("plan_parse_success", 0)) / decoded
    metrics["formula_parse_rate"] = float(metrics.get("formula_parse_success", 0)) / decoded
    metrics["valid_formula_rate"] = float(metrics.get("valid_formula_success", 0)) / decoded
    metrics["valid_n_rate"] = float(metrics.get("valid_n_success", 0)) / decoded
    metrics["plan_end_marker_rate"] = float(metrics.get("plan_end_marker_success", 0)) / decoded
    metrics["plan_tail_after_end_marker_rate"] = float(metrics.get("plan_tail_after_end_marker", 0)) / decoded
    metrics["single_element_rate"] = float(metrics.get("single_element_plans", 0)) / decoded
    metrics["rich_field_valid_rate"] = float(metrics.get("rich_field_valid_success", 0)) / decoded
    for key in ("anion", "charge", "lattice", "spacegroup", "volume"):
        metrics[f"rich_{key}_valid_rate"] = float(metrics.get(f"rich_{key}_valid_success", 0)) / decoded
    metrics["rich_charge_derived_rate"] = float(metrics.get("rich_charge_derived_success", 0)) / decoded
    metrics["generated_charge_field_rate"] = float(metrics.get("generated_charge_field_count", 0)) / decoded
    metrics["valid_plan_count"] = int(metrics.get("plan_parse_success", 0))
    return metrics


def _read_rank_records(output_dir: Path, filename: str, world_size: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = rank_path(output_dir, filename, rank, True)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise TypeError(f"{path} contains a non-object JSONL row")
                    records.append(payload)
    return records


def _merge_rank_jsonl(
    output_dir: Path,
    filename: str,
    world_size: int,
    *,
    sort_by_ordinal: bool,
    expected_count: int,
    require_complete: bool,
) -> None:
    output_path = output_dir / filename
    if not sort_by_ordinal:
        with output_path.open("w", encoding="utf-8") as output_handle:
            for rank in range(world_size):
                path = rank_path(output_dir, filename, rank, True)
                if path.exists():
                    output_handle.write(path.read_text(encoding="utf-8"))
        return
    records = ordered_ordinal_records(
        _read_rank_records(output_dir, filename, world_size),
        expected_count=expected_count,
        require_complete=require_complete,
    )
    with output_path.open("w", encoding="utf-8") as output_handle:
        for record in records:
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_distributed_outputs(
    output_dir: Path,
    world_size: int,
    *,
    expected_samples: int,
    sort_by_ordinal: bool,
) -> None:
    merged = {
        "requested_samples": 0,
        "decoded_samples": 0,
        "plan_parse_success": 0,
        "formula_parse_success": 0,
        "valid_formula_success": 0,
        "valid_n_success": 0,
        "plan_end_marker_success": 0,
        "plan_tail_after_end_marker": 0,
        "single_element_plans": 0,
        "rich_field_required": False,
        "rich_field_valid_success": 0,
        "rich_anion_valid_success": 0,
        "rich_charge_valid_success": 0,
        "rich_charge_derived_success": 0,
        "generated_charge_field_count": 0,
        "rich_lattice_valid_success": 0,
        "rich_spacegroup_valid_success": 0,
        "rich_volume_valid_success": 0,
        "constraint_dead_end_count": 0,
        "constraint_identity_failure_count": 0,
        "constraint_masked_step_count": 0,
        "constraint_preterminal_support_difference_steps": 0,
        "constraint_primary_witness_count": 0,
        "constraint_mixed_valence_only_count": 0,
        "constraint_non_applicable_count": 0,
        "failures": {},
        "time_sec": 0.0,
        "distributed": True,
        "world_size": world_size,
    }
    for rank in range(world_size):
        metrics_path = rank_path(output_dir, "sample_metrics.json", rank, True)
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key in (
                "requested_samples",
                "decoded_samples",
                "plan_parse_success",
                "formula_parse_success",
                "valid_formula_success",
                "valid_n_success",
                "plan_end_marker_success",
                "plan_tail_after_end_marker",
                "single_element_plans",
                "rich_field_valid_success",
                "rich_anion_valid_success",
                "rich_charge_valid_success",
                "rich_charge_derived_success",
                "generated_charge_field_count",
                "rich_lattice_valid_success",
                "rich_spacegroup_valid_success",
                "rich_volume_valid_success",
                "constraint_dead_end_count",
                "constraint_identity_failure_count",
                "constraint_masked_step_count",
                "constraint_preterminal_support_difference_steps",
                "constraint_primary_witness_count",
                "constraint_mixed_valence_only_count",
                "constraint_non_applicable_count",
            ):
                merged[key] += int(metrics.get(key, 0))
            merged["rich_field_required"] = bool(merged.get("rich_field_required")) or bool(
                metrics.get("rich_field_required")
            )
            merged["time_sec"] = max(float(merged["time_sec"]), float(metrics.get("time_sec") or 0.0))
            for reason, count in metrics.get("failures", {}).items():
                merged["failures"][reason] = int(merged["failures"].get(reason, 0)) + int(count)
    _merge_rank_jsonl(
        output_dir,
        "raw_generations.jsonl",
        world_size,
        sort_by_ordinal=sort_by_ordinal,
        expected_count=expected_samples,
        require_complete=sort_by_ordinal,
    )
    _merge_rank_jsonl(
        output_dir,
        "failure_cases.jsonl",
        world_size,
        sort_by_ordinal=sort_by_ordinal,
        expected_count=expected_samples,
        require_complete=False,
    )
    _merge_rank_jsonl(
        output_dir,
        "plans_for_dlm.jsonl",
        world_size,
        sort_by_ordinal=sort_by_ordinal,
        expected_count=expected_samples,
        require_complete=False,
    )
    write_json(str(output_dir / "sample_metrics.json"), finalize_metrics(merged))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/Llama-3.1-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-atoms", type=int, default=20)
    parser.add_argument("--prompt-style", default="chat_formula_end_v1")
    parser.add_argument("--include-sample-id", dest="include_sample_id", action="store_true", default=True)
    parser.add_argument("--no-include-sample-id", dest="include_sample_id", action="store_false")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--seed-mode",
        choices=("legacy_rank", "stateless_ordinal_v1"),
        default="legacy_rank",
        help=(
            "legacy_rank preserves historical rank-seeded batching; "
            "stateless_ordinal_v1 forces one sample per generation call and "
            "derives RNG only from seed, sample_idx, and stage"
        ),
    )
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=True)
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--stop-after-plan-marker", dest="stop_after_plan_marker", action="store_true", default=True)
    parser.add_argument("--no-stop-after-plan-marker", dest="stop_after_plan_marker", action="store_false")
    parser.add_argument("--truncate-after-plan-marker", dest="truncate_after_plan_marker", action="store_true", default=True)
    parser.add_argument("--no-truncate-after-plan-marker", dest="truncate_after_plan_marker", action="store_false")
    parser.add_argument(
        "--formula-constraint-mode",
        choices=("off", *CRPLAN_MODES),
        default="off",
        help=(
            "off preserves the frozen P0 sampler; other modes change only the "
            "legal token support of the first generated formula value"
        ),
    )
    parser.add_argument(
        "--formula-missing-state-policy",
        choices=MISSING_STATE_POLICIES,
        default="allow_non_applicable",
        help=(
            "terminal/full-prefix handling for elements absent from the frozen "
            "SMACT oxidation-state table; four-arm mechanism studies use "
            "fail_closed in both endpoint-constrained arms"
        ),
    )
    args = parser.parse_args()
    args.prompt_style = normalize_prompt_style(args.prompt_style)
    rich_field_required = args.prompt_style in (
        H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
        H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE,
    )
    nocharge_rich_plan = args.prompt_style == H1_PLANNER_PROMPT_STYLE_RICH_NOCHARGE
    generated_rich_field_names = (
        ("anion", "lattice", "spacegroup", "volume")
        if nocharge_rich_plan
        else ("anion", "charge", "lattice", "spacegroup", "volume")
    )
    if (
        args.formula_constraint_mode != "off"
        and args.seed_mode != "stateless_ordinal_v1"
    ):
        raise ValueError(
            "CR-Plan requires stateless_ordinal_v1 batch-size-one sampling"
        )
    if (
        args.formula_constraint_mode != "off"
        and args.prompt_style == H1_PLANNER_PROMPT_STYLE_FORMULA_PREFILL
    ):
        raise ValueError(
            "CR-Plan forbids formula-prefill prompts because the masked FSM "
            "must observe the generated formula label and value"
        )

    dist_info = init_distributed()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    distributed = bool(dist_info["distributed"])
    is_main = bool(dist_info["is_main"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_seed = (
        int(args.seed) + rank
        if args.seed_mode == "legacy_rank"
        else derive_ordinal_seed(
            int(args.seed),
            sample_idx=0,
            stage="planner_process_initialization",
            role="shared",
        )
    )
    torch.manual_seed(initial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initial_seed)

    model, tokenizer = load_planner(args.model_path, args.checkpoint_path, dist_info["device"])
    cr_reachability = None
    cr_token_vocabulary = None
    if args.formula_constraint_mode != "off":
        cr_reachability = load_frozen_smact_table(
            max_atoms=int(args.max_atoms),
            missing_state_policy=args.formula_missing_state_policy,
        )
        cr_token_vocabulary = CRPlanTokenVocabulary.from_tokenizer(tokenizer)
    tasks = [idx for idx in range(int(args.num_samples)) if idx % world_size == rank]

    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update(
        {
            "method": (
                "h1_llm_formula_planner"
                if args.formula_constraint_mode == "off"
                else "h1_llm_formula_planner_crplan"
            ),
            "prompt_version": H1_PLANNER_PROMPT_VERSION,
            "prompt_style": args.prompt_style,
            "include_sample_id": bool(args.include_sample_id),
            "distributed": distributed,
            "world_size": world_size,
            "effective_do_sample": bool(args.do_sample and float(args.temperature) > 0.0),
            "rich_field_required": bool(rich_field_required),
            "generated_charge_field_expected": bool(
                args.prompt_style == H1_PLANNER_PROMPT_STYLE_RICH_PLAN
            ),
            "charge_taxonomy_source": (
                "derived_from_formula_frozen_evaluator"
                if nocharge_rich_plan
                else "generated_field"
                if args.prompt_style == H1_PLANNER_PROMPT_STYLE_RICH_PLAN
                else "not_applicable"
            ),
            "rank_independent_sampling": args.seed_mode == "stateless_ordinal_v1",
            "effective_generation_batch_size": (
                1 if args.seed_mode == "stateless_ordinal_v1" else int(args.batch_size)
            ),
            "formula_constraint_mode": args.formula_constraint_mode,
            "formula_constraint_changes_prompt": False,
            "formula_constraint_changes_weights": False,
            "formula_constraint_changes_body_prompt": False,
            "formula_constraint_retry_replacement_repair_filter_or_rerank": False,
        }
    )
    if cr_reachability is not None and cr_token_vocabulary is not None:
        run_config.update(
            {
                "crplan_schema": CRPLAN_SCHEMA,
                "oxidation_table_sha256": cr_reachability.table_sha256,
                "constraint_contract_sha256": (
                    cr_reachability.constraint_contract_sha256
                ),
                "formula_missing_state_policy": (
                    cr_reachability.missing_state_policy
                ),
                "token_fragment_sha256": cr_token_vocabulary.fragment_sha256,
                "mixed_valence_only_is_primary_gain": False,
                "pauling_is_hard_constraint": False,
            }
        )
    if is_main:
        write_json(str(args.output_dir / "run_config.json"), run_config)
        write_json(
            str(args.output_dir / "tokenizer_report.json"),
            {
                "vocab_size": len(tokenizer),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "padding_side": tokenizer.padding_side,
                "decoded_token_fragment_sha256": (
                    None
                    if cr_token_vocabulary is None
                    else cr_token_vocabulary.fragment_sha256
                ),
            },
        )
        if cr_reachability is not None:
            write_json(
                str(args.output_dir / "oxidation_table_report.json"),
                cr_reachability.table_report(),
            )

    raw_path = rank_path(args.output_dir, "raw_generations.jsonl", rank, distributed)
    plans_path = rank_path(args.output_dir, "plans_for_dlm.jsonl", rank, distributed)
    failure_path = rank_path(args.output_dir, "failure_cases.jsonl", rank, distributed)
    metrics: Dict[str, Any] = {
        "requested_samples": len(tasks),
        "decoded_samples": 0,
        "plan_parse_success": 0,
        "formula_parse_success": 0,
        "valid_formula_success": 0,
        "valid_n_success": 0,
        "plan_end_marker_success": 0,
        "plan_tail_after_end_marker": 0,
        "single_element_plans": 0,
        "rich_field_required": bool(rich_field_required),
        "rich_field_valid_success": 0,
        "rich_anion_valid_success": 0,
        "rich_charge_valid_success": 0,
        "rich_charge_derived_success": 0,
        "generated_charge_field_count": 0,
        "rich_lattice_valid_success": 0,
        "rich_spacegroup_valid_success": 0,
        "rich_volume_valid_success": 0,
        "constraint_dead_end_count": 0,
        "constraint_identity_failure_count": 0,
        "constraint_masked_step_count": 0,
        "constraint_preterminal_support_difference_steps": 0,
        "constraint_primary_witness_count": 0,
        "constraint_mixed_valence_only_count": 0,
        "constraint_non_applicable_count": 0,
        "failures": {},
        "time_sec": None,
        "rank": rank,
        "world_size": world_size,
    }

    start = time.time()
    with raw_path.open("w", encoding="utf-8") as raw_handle, plans_path.open("w", encoding="utf-8") as plans_handle, failure_path.open(
        "w", encoding="utf-8"
    ) as failure_handle:
        progress = tqdm(total=len(tasks), desc=f"H1 planner rank{rank}", disable=distributed and not is_main)
        offset = 0
        generation_batch_size = (
            1 if args.seed_mode == "stateless_ordinal_v1" else int(args.batch_size)
        )
        while offset < len(tasks):
            batch_ids = tasks[offset : offset + generation_batch_size]
            offset += generation_batch_size
            sampling_seed = None
            if args.seed_mode == "stateless_ordinal_v1":
                sampling_seed = derive_ordinal_seed(
                    int(args.seed),
                    sample_idx=int(batch_ids[0]),
                    stage="planner_sampling",
                    role="shared",
                )
                torch.manual_seed(sampling_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sampling_seed)
            prompts = [
                format_planner_prompt(
                    tokenizer,
                    sample_idx=sample_idx if bool(args.include_sample_id) else None,
                    prompt_style=args.prompt_style,
                )
                for sample_idx in batch_ids
            ]
            encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model_device(model))
            attention_mask = encoded["attention_mask"].to(model_device(model))
            planner_prompt_sha256 = [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in prompts
            ]
            planner_input_ids_sha256 = [
                hashlib.sha256(
                    json.dumps(
                        [int(value) for value in row.tolist()],
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest()
                for row in encoded["input_ids"]
            ]
            processor = None
            if args.formula_constraint_mode != "off":
                assert len(batch_ids) == 1
                assert cr_reachability is not None
                assert cr_token_vocabulary is not None
                processor = CRPlanLogitsProcessor(
                    tokenizer,
                    cr_token_vocabulary,
                    cr_reachability,
                    prompt_width=int(input_ids.shape[1]),
                    mode=args.formula_constraint_mode,
                    attempt_ordinal=int(batch_ids[0]),
                )
            generation_error: CRPlanDeadEndError | None = None
            generation_started = time.perf_counter()
            try:
                with torch.no_grad():
                    effective_do_sample = bool(args.do_sample and float(args.temperature) > 0.0)
                    generate_kwargs: Dict[str, Any] = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": int(args.max_new_tokens),
                        "do_sample": effective_do_sample,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                    if effective_do_sample:
                        generate_kwargs["temperature"] = float(args.temperature)
                        generate_kwargs["top_p"] = float(args.top_p)
                        generate_kwargs["top_k"] = int(args.top_k)
                    if bool(args.stop_after_plan_marker):
                        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                            [GeneratedPlanEndStoppingCriteria(tokenizer, input_ids.shape[1])]
                        )
                    if processor is not None:
                        generate_kwargs["logits_processor"] = LogitsProcessorList(
                            [processor]
                        )
                        generate_kwargs["renormalize_logits"] = True
                    outputs = model.generate(**generate_kwargs)
                generated_ids = outputs[:, input_ids.shape[1] :]
                decoded = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                generated_token_rows: list[list[int] | None] = [
                    [int(value) for value in row.tolist()]
                    for row in generated_ids
                ]
            except CRPlanDeadEndError as exc:
                generation_error = exc
                decoded = [""]
                generated_token_rows = [None]
            generation_latency_sec = time.perf_counter() - generation_started
            for row_position, (sample_idx, text) in enumerate(zip(batch_ids, decoded)):
                constraint_diagnostics = (
                    None
                    if processor is None
                    else processor.diagnostics(
                        generated_token_rows[row_position]
                    )
                )
                if constraint_diagnostics is not None:
                    metrics["constraint_masked_step_count"] += int(
                        constraint_diagnostics["masked_step_count"]
                    )
                    metrics[
                        "constraint_preterminal_support_difference_steps"
                    ] += int(
                        constraint_diagnostics[
                            "preterminal_support_difference_steps"
                        ]
                    )
                raw_model_text = clean_generated_plan_text(
                    text,
                    prompt_style=args.prompt_style,
                    truncate_after_marker=False,
                )
                raw_text = clean_generated_plan_text(
                    raw_model_text,
                    prompt_style=args.prompt_style,
                    truncate_after_marker=bool(args.truncate_after_plan_marker),
                )
                marker = has_plan_end_marker(raw_text)
                tail = has_plan_tail_after_end_marker(raw_model_text)
                metrics["decoded_samples"] += 1
                if marker:
                    metrics["plan_end_marker_success"] += 1
                if tail:
                    metrics["plan_tail_after_end_marker"] += 1
                raw_record: Dict[str, Any] = {
                    "sample_idx": int(sample_idx),
                    "raw_plan_text": raw_text,
                    "raw_model_text": raw_model_text,
                    "planner_model_path": args.model_path,
                    "planner_checkpoint_path": args.checkpoint_path,
                    "plan_end_marker_present": marker,
                    "plan_tail_after_end_marker": tail,
                    "prompt_version": H1_PLANNER_PROMPT_VERSION,
                    "prompt_style": args.prompt_style,
                    "planner_sampling_seed": sampling_seed,
                    "planner_input_prompt_sha256": (
                        planner_prompt_sha256[row_position]
                    ),
                    "planner_input_ids_sha256": (
                        planner_input_ids_sha256[row_position]
                    ),
                    "seed_mode": args.seed_mode,
                    "formula_constraint_mode": args.formula_constraint_mode,
                    "crplan_diagnostics": constraint_diagnostics,
                    "planner_generation_latency_sec": generation_latency_sec,
                    "parsed": False,
                    "formula_parse": False,
                }
                if generation_error is not None:
                    metrics["constraint_dead_end_count"] += 1
                    add_failure(
                        metrics,
                        failure_handle,
                        int(sample_idx),
                        "crplan_dead_end",
                        generation_error,
                        raw_text,
                        sampling_seed=sampling_seed,
                    )
                    raw_record.update(
                        {
                            "reason": type(generation_error).__name__,
                            "message": str(generation_error),
                            "fail_closed": True,
                        }
                    )
                    raw_handle.write(
                        json.dumps(raw_record, ensure_ascii=False) + "\n"
                    )
                    progress.update(1)
                    continue
                try:
                    plan_record = canonical_plan_record_for_style(
                        raw_text,
                        sample_idx=sample_idx,
                        max_atoms=int(args.max_atoms),
                        prompt_style=args.prompt_style,
                    )
                    plan_record["planner_sampling_seed"] = sampling_seed
                    plan_record["seed_mode"] = args.seed_mode
                    plan_record["formula_constraint_mode"] = (
                        args.formula_constraint_mode
                    )
                    plan_record["crplan_diagnostics"] = constraint_diagnostics
                    plan = plan_record["plan_state"]
                    if cr_reachability is not None:
                        identity = validate_crplan_parsed_identity(
                            raw_model_text=raw_model_text,
                            prompt_style=args.prompt_style,
                            parsed_symbols=plan.get("elements") or (),
                            parsed_counts=plan.get("counts") or (),
                            diagnostics=constraint_diagnostics,
                            mode=args.formula_constraint_mode,
                        )
                        plan_record["crplan_identity"] = identity
                        certificate = certificate_for_symbol_counts(
                            cr_reachability,
                            plan.get("elements") or (),
                            plan.get("counts") or (),
                        )
                        if (
                            not isinstance(constraint_diagnostics, dict)
                            or constraint_diagnostics.get(
                                "terminal_certificate"
                            )
                            != certificate
                        ):
                            raise CRPlanIdentityError(
                                "independently recomputed terminal certificate "
                                "does not equal the decoder certificate"
                            )
                        plan_record["crplan_terminal_certificate"] = certificate
                        stratum = str(certificate["stratum"])
                        if certificate["primary_charge_witness"] is True:
                            metrics["constraint_primary_witness_count"] += 1
                        elif stratum == "charge_applicable_mixed_valence_only":
                            metrics["constraint_mixed_valence_only_count"] += 1
                        elif stratum.startswith("charge_not_applicable_"):
                            metrics["constraint_non_applicable_count"] += 1
                    metrics["plan_parse_success"] += 1
                    metrics["formula_parse_success"] += 1
                    metrics["valid_formula_success"] += 1
                    metrics["valid_n_success"] += 1
                    if len(plan.get("elements") or []) == 1:
                        metrics["single_element_plans"] += 1
                    rich_fields = plan.get("generated_rich_fields")
                    if isinstance(rich_fields, dict) and "charge" in rich_fields:
                        metrics["generated_charge_field_count"] += 1
                    if isinstance(rich_fields, dict) and all(
                        key in rich_fields for key in generated_rich_field_names
                    ):
                        metrics["rich_field_valid_success"] += 1
                        for key in generated_rich_field_names:
                            metrics[f"rich_{key}_valid_success"] += 1
                        if plan.get("derived_charge_bucket_from_formula") is True:
                            metrics["rich_charge_derived_success"] += 1
                    raw_record.update(
                        {
                            **plan_record,
                            "parsed": True,
                            "formula_parse": True,
                            "valid_formula": True,
                            "valid_N": True,
                        }
                    )
                    plans_handle.write(json.dumps(plan_record, ensure_ascii=False) + "\n")
                except CRPlanIdentityError as exc:
                    metrics["constraint_identity_failure_count"] += 1
                    add_failure(
                        metrics,
                        failure_handle,
                        int(sample_idx),
                        "crplan_identity",
                        exc,
                        raw_text,
                        sampling_seed=sampling_seed,
                    )
                    raw_record.update(
                        {
                            "reason": type(exc).__name__,
                            "message": str(exc),
                            "fail_closed": True,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    add_failure(
                        metrics,
                        failure_handle,
                        int(sample_idx),
                        "parse_plan",
                        exc,
                        raw_text,
                        sampling_seed=sampling_seed,
                    )
                    raw_record.update({"reason": type(exc).__name__, "message": str(exc)})
                raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                progress.update(1)

    metrics["time_sec"] = time.time() - start
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), finalize_metrics(metrics))
    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_outputs(
                args.output_dir,
                world_size,
                expected_samples=int(args.num_samples),
                sort_by_ordinal=args.seed_mode == "stateless_ordinal_v1",
            )
        dist.barrier()


if __name__ == "__main__":
    main()
