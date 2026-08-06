#!/usr/bin/env python3
"""Sample H1 Llama formula plans for LLM-plan + DLM-body hybrid generation."""

from __future__ import annotations

import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from crystal_dlm.h1_llm_planner import (  # noqa: E402
    H1_PLANNER_PROMPT_VERSION,
    canonical_plan_record_for_style,
    clean_generated_plan_text,
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    format_planner_prompt,
    load_llama3_compatible_config,
    normalize_prompt_style,
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


def add_failure(metrics: Dict[str, Any], failure_handle, sample_idx: int, stage: str, exc: Exception, raw_text: str) -> None:
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
    metrics["valid_plan_count"] = int(metrics.get("plan_parse_success", 0))
    return metrics


def merge_distributed_outputs(output_dir: Path, world_size: int) -> None:
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
        "rich_lattice_valid_success": 0,
        "rich_spacegroup_valid_success": 0,
        "rich_volume_valid_success": 0,
        "failures": {},
        "time_sec": 0.0,
        "distributed": True,
        "world_size": world_size,
    }
    with (output_dir / "raw_generations.jsonl").open("w", encoding="utf-8") as raw_out, (
        output_dir / "failure_cases.jsonl"
    ).open("w", encoding="utf-8") as failure_out, (output_dir / "plans_for_dlm.jsonl").open(
        "w", encoding="utf-8"
    ) as plans_out:
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
                    "rich_lattice_valid_success",
                    "rich_spacegroup_valid_success",
                    "rich_volume_valid_success",
                ):
                    merged[key] += int(metrics.get(key, 0))
                merged["rich_field_required"] = bool(merged.get("rich_field_required")) or bool(
                    metrics.get("rich_field_required")
                )
                merged["time_sec"] = max(float(merged["time_sec"]), float(metrics.get("time_sec") or 0.0))
                for reason, count in metrics.get("failures", {}).items():
                    merged["failures"][reason] = int(merged["failures"].get(reason, 0)) + int(count)
            for filename, handle in (
                ("raw_generations.jsonl", raw_out),
                ("failure_cases.jsonl", failure_out),
                ("plans_for_dlm.jsonl", plans_out),
            ):
                path = rank_path(output_dir, filename, rank, True)
                if path.exists():
                    handle.write(path.read_text(encoding="utf-8"))
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
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=True)
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--stop-after-plan-marker", dest="stop_after_plan_marker", action="store_true", default=True)
    parser.add_argument("--no-stop-after-plan-marker", dest="stop_after_plan_marker", action="store_false")
    parser.add_argument("--truncate-after-plan-marker", dest="truncate_after_plan_marker", action="store_true", default=True)
    parser.add_argument("--no-truncate-after-plan-marker", dest="truncate_after_plan_marker", action="store_false")
    args = parser.parse_args()
    args.prompt_style = normalize_prompt_style(args.prompt_style)
    rich_field_required = args.prompt_style == "h1_rich_plan_v1"

    dist_info = init_distributed()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    distributed = bool(dist_info["distributed"])
    is_main = bool(dist_info["is_main"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed) + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + rank)

    model, tokenizer = load_planner(args.model_path, args.checkpoint_path, dist_info["device"])
    tasks = [idx for idx in range(int(args.num_samples)) if idx % world_size == rank]

    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update(
        {
            "method": "h1_llm_formula_planner",
            "prompt_version": H1_PLANNER_PROMPT_VERSION,
            "prompt_style": args.prompt_style,
            "include_sample_id": bool(args.include_sample_id),
            "distributed": distributed,
            "world_size": world_size,
            "effective_do_sample": bool(args.do_sample and float(args.temperature) > 0.0),
            "rich_field_required": bool(rich_field_required),
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
            },
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
        "rich_lattice_valid_success": 0,
        "rich_spacegroup_valid_success": 0,
        "rich_volume_valid_success": 0,
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
        while offset < len(tasks):
            batch_ids = tasks[offset : offset + int(args.batch_size)]
            offset += int(args.batch_size)
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
                outputs = model.generate(**generate_kwargs)
            generated_ids = outputs[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for sample_idx, text in zip(batch_ids, decoded):
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
                    "parsed": False,
                    "formula_parse": False,
                }
                try:
                    plan_record = canonical_plan_record_for_style(
                        raw_text,
                        sample_idx=sample_idx,
                        max_atoms=int(args.max_atoms),
                        prompt_style=args.prompt_style,
                    )
                    plan = plan_record["plan_state"]
                    metrics["plan_parse_success"] += 1
                    metrics["formula_parse_success"] += 1
                    metrics["valid_formula_success"] += 1
                    metrics["valid_n_success"] += 1
                    if len(plan.get("elements") or []) == 1:
                        metrics["single_element_plans"] += 1
                    rich_fields = plan.get("generated_rich_fields")
                    if isinstance(rich_fields, dict) and all(
                        key in rich_fields for key in ("anion", "charge", "lattice", "spacegroup", "volume")
                    ):
                        metrics["rich_field_valid_success"] += 1
                        for key in ("anion", "charge", "lattice", "spacegroup", "volume"):
                            metrics[f"rich_{key}_valid_success"] += 1
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
                except Exception as exc:  # noqa: BLE001
                    add_failure(metrics, failure_handle, int(sample_idx), "parse_plan", exc, raw_text)
                    raw_record.update({"reason": type(exc).__name__, "message": str(exc)})
                raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                progress.update(1)

    metrics["time_sec"] = time.time() - start
    write_json(str(rank_path(args.output_dir, "sample_metrics.json", rank, distributed)), finalize_metrics(metrics))
    if distributed:
        dist.barrier()
        if is_main:
            merge_distributed_outputs(args.output_dir, world_size)
        dist.barrier()


if __name__ == "__main__":
    main()
