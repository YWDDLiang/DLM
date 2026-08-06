#!/usr/bin/env python3
"""Train a tiny DOPING_STRUCT20 adapter in memory and sample without reloading it."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from crystal_dlm.doping import arrays_to_cif_text, combo_name, write_json, write_jsonl
from crystal_dlm.doping_structure import (
    STRUCT20_DIRECTED_PROMPT,
    bsite_dopants_from_arrays,
    expand_structure20_arrays,
    load_full80_template,
    parse_structure20_answer,
)
from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.llada_generation import generate
from scripts.llada_sft import (
    DataCollator,
    JsonlSftDataset,
    build_loss_config,
    compute_loss,
    load_tokenizer_and_model,
)
from scripts.sample_doping_structure20 import (
    build_geometry_constraints,
    graph_from_arrays,
    required_token_id,
    sample_dopant_tokens,
)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def make_sft_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        skip_data_vocab_resize=True,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules="q_proj,k_proj,v_proj,ff_proj,up_proj",
        modules_to_save="model.transformer.wte,model.transformer.ff_out",
        answer_token_count=107,
        atom_count_loss_weight=1.0,
        slot_marker_loss_weight=1.0,
        empty_slot_loss_weight=1.0,
        nonempty_slot_loss_weight=1.0,
        late_slot_start=4,
        late_nonempty_slot_loss_weight=None,
        coordinate_loss_weight=1.0,
        pad_coordinate_loss_weight=1.0,
    )


def make_metrics(
    args: argparse.Namespace,
    start: float,
    raw_records: List[Dict[str, Any]],
    failure_records: List[Dict[str, Any]],
    parse_success: int,
    reconstruction_success: int,
    composition_exact: int,
    graph_success: int,
    unique_signatures: Counter[str],
) -> Dict[str, Any]:
    decoded_count = len(raw_records)
    return {
        "num_samples": args.num_samples,
        "decoded_samples": decoded_count,
        "parse_success": parse_success,
        "parse_rate": parse_success / max(1, decoded_count),
        "reconstruction_success": reconstruction_success,
        "reconstruction_rate": reconstruction_success / max(1, decoded_count),
        "composition_exact_count": composition_exact,
        "composition_exact_rate": composition_exact / max(1, decoded_count),
        "graph_success": graph_success,
        "graph_build_rate": None if args.skip_graph else graph_success / max(1, reconstruction_success),
        "unique_expanded_structure_count": len(unique_signatures),
        "failure_count": len(failure_records),
        "time_sec": time.time() - start,
        "prompt": args.prompt,
        "temperature": args.temperature,
        "steps": args.steps,
        "mode": "doping_structure20_inline_tiny",
        "graph_skipped": args.skip_graph,
        "train_limit": args.limit_train,
        "epochs": args.epochs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/doping_structure20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full80-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/full80_success.jsonl")
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--prompt", default=STRUCT20_DIRECTED_PROMPT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--dopant-mode", choices=["model", "fixed"], default="model")
    parser.add_argument("--fixed-dopants", default="Ca,Fe,Ni")
    parser.add_argument(
        "--sample-train-mode",
        action="store_true",
        help="Keep model.train() during sampling. This diagnoses eval-mode forward hangs in the remote LLaDA implementation.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = args.output_dir / "expanded_cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.jsonl"
    raw_path = args.output_dir / "raw_generations.jsonl"
    failure_path = args.output_dir / "failure_cases.jsonl"
    expanded_path = args.output_dir / "expanded_arrays.jsonl"
    partial_metrics_path = args.output_dir / "partial_metrics.json"

    def log_progress(event: str, **payload: Any) -> None:
        append_jsonl(progress_path, {"event": event, "elapsed_sec": time.time() - start, **payload})

    log_progress("start", num_samples=args.num_samples, steps=args.steps, skip_graph=args.skip_graph)
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint_path: {args.checkpoint_path}")

    hidden_names = set()
    stats_path = args.data_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        hidden_names = {row["name"] for row in stats.get("hidden_good_holdout", [])}
    template = load_full80_template(args.full80_jsonl, exclude_names=hidden_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress("model_load_start", device=str(device), checkpoint_path=str(args.checkpoint_path))
    sft_args = make_sft_args(args)
    tokenizer, model, num_new_tokens, tokenizer_source, model_source = load_tokenizer_and_model(sft_args, is_main=True)
    model.to(device)
    model.train()
    log_progress(
        "model_load_done",
        vocab_size=len(tokenizer),
        num_new_tokens=num_new_tokens,
        tokenizer_source=tokenizer_source,
        model_source=model_source,
    )
    if tokenizer.pad_token_id == MASK_TOKEN_ID:
        raise RuntimeError("tokenizer.pad_token_id must not equal LLaDA mask token id 126336")

    train_ds = JsonlSftDataset(args.data_dir / "train.jsonl", tokenizer, args.max_length)
    train_ds.rows = train_ds.rows[: args.limit_train]
    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=DataCollator(tokenizer),
        num_workers=0,
    )
    loss_config = build_loss_config(tokenizer, sft_args)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    log_progress("train_start", train_rows=len(train_ds), train_batch_size=args.train_batch_size)
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = compute_loss(model, batch, loss_config) / max(1, args.grad_accum)
            loss.backward()
            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                log_progress(
                    "train_step",
                    epoch=epoch,
                    step=global_step,
                    loss=float((loss.detach() * max(1, args.grad_accum)).cpu()),
                )
    log_progress("train_done", global_step=global_step)

    if args.sample_train_mode:
        model.train()
    else:
        model.eval()
    log_progress("sample_mode", training=bool(model.training), dopant_mode=args.dopant_mode)
    raw_records: List[Dict[str, Any]] = []
    failure_records: List[Dict[str, Any]] = []
    expanded_records: List[Dict[str, Any]] = []
    parse_success = 0
    reconstruction_success = 0
    composition_exact = 0
    graph_success = 0
    unique_signatures: Counter[str] = Counter()
    prompt_text = args.prompt.rstrip() + "\n"

    for batch_start in range(0, args.num_samples, args.sample_batch_size):
        current_batch = min(args.sample_batch_size, args.num_samples - batch_start)
        batch_t0 = time.time()
        log_progress("batch_start", batch_start=batch_start, batch_size=current_batch)
        with torch.no_grad():
            if args.dopant_mode == "fixed":
                fixed_dopants = [item.strip() for item in args.fixed_dopants.split(",") if item.strip()]
                if len(fixed_dopants) != 3:
                    raise ValueError("--fixed-dopants must contain exactly three comma-separated elements")
                fixed_slots = (0, 1, 14)
                sampled_dopants = {
                    slot_idx: [required_token_id(tokenizer, f"<E_{symbol}>")] * current_batch
                    for slot_idx, symbol in zip(fixed_slots, fixed_dopants)
                }
            else:
                sampled_dopants = sample_dopant_tokens(
                    model,
                    tokenizer,
                    prompt_text,
                    current_batch,
                    args.temperature,
                    args.seed + batch_start,
                )
            log_progress("dopant_sample_done", batch_start=batch_start, elapsed_batch_sec=time.time() - batch_t0)
            log_progress("constraints_start", batch_start=batch_start)
            constraints, prefill, _ = build_geometry_constraints(tokenizer, sampled_dopants)
            log_progress("constraints_done", batch_start=batch_start, elapsed_batch_sec=time.time() - batch_t0)
            encoded = tokenizer([prompt_text] * current_batch, add_special_tokens=False, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            log_progress("generate_start", batch_start=batch_start, prompt_tokens=int(input_ids.shape[1]))
            outputs = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
                steps=args.steps,
                gen_length=107,
                block_length=107,
                temperature=args.temperature,
                remasking="low_confidence",
                mask_id=MASK_TOKEN_ID,
                allowed_token_ids_by_generation_pos=constraints,
                prefill_token_ids_by_generation_pos=prefill,
            )
            log_progress("generate_done", batch_start=batch_start, elapsed_batch_sec=time.time() - batch_t0)
        generated_ids = outputs[:, input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        for offset, text in enumerate(decoded):
            sample_idx = batch_start + offset
            record: Dict[str, Any] = {"sample_idx": sample_idx, "text": text}
            try:
                compressed = parse_structure20_answer(text)
                parse_success += 1
                expanded = expand_structure20_arrays(compressed, template)
                reconstruction_success += 1
                composition_exact += 1
                dopants = bsite_dopants_from_arrays(expanded)
                name = combo_name(dopants)
                signature = "|".join(
                    [
                        name,
                        ",".join(f"{float(value):.3f}" for value in expanded["lengths"]),
                        ";".join(
                            ",".join(f"{float(coord):.2f}" for coord in expanded["frac_coords"][idx])
                            for idx in expanded["bsite_structure_indices"]
                        ),
                    ]
                )
                unique_signatures[signature] += 1
                cif_path = cif_dir / f"{sample_idx:05d}_{name}.cif"
                cif_path.write_text(arrays_to_cif_text(expanded, data_name=f"struct20_inline_{sample_idx}_{name}"), encoding="utf-8")
                if not args.skip_graph:
                    try:
                        graph_from_arrays(expanded, args.crysllmgen_dir)
                        graph_success += 1
                    except Exception as graph_exc:
                        record["graph_reason"] = type(graph_exc).__name__
                        record["graph_message"] = str(graph_exc)
                expanded_record = {
                    "sample_idx": sample_idx,
                    "name": name,
                    "dopants": dopants,
                    "answer": compressed["answer"],
                    "expanded_arrays": expanded,
                    "expanded_cif": str(cif_path),
                }
                expanded_records.append(expanded_record)
                append_jsonl(expanded_path, expanded_record)
                record.update({"parsed": True, "name": name, "dopants": dopants, "expanded_cif": str(cif_path)})
            except Exception as exc:
                failure = {"sample_idx": sample_idx, "reason": type(exc).__name__, "message": str(exc), "text": text}
                failure_records.append(failure)
                append_jsonl(failure_path, failure)
                record.update({"parsed": False, **failure})
            raw_records.append(record)
            append_jsonl(raw_path, record)
        metrics = make_metrics(
            args,
            start,
            raw_records,
            failure_records,
            parse_success,
            reconstruction_success,
            composition_exact,
            graph_success,
            unique_signatures,
        )
        write_json(partial_metrics_path, metrics)
        log_progress("batch_done", batch_start=batch_start, decoded_samples=len(raw_records), parse_success=parse_success)

    metrics = make_metrics(
        args,
        start,
        raw_records,
        failure_records,
        parse_success,
        reconstruction_success,
        composition_exact,
        graph_success,
        unique_signatures,
    )
    write_jsonl(raw_path, raw_records)
    write_jsonl(failure_path, failure_records)
    write_jsonl(expanded_path, expanded_records)
    write_json(args.output_dir / "sample_metrics.json", metrics)
    write_json(
        args.output_dir / "run_config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    write_json(
        args.output_dir / "tokenizer_report.json",
        {
            "model_path": args.model_path,
            "checkpoint_path": str(args.checkpoint_path),
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "mask_token_id": MASK_TOKEN_ID,
            "pad_token_id_ne_mask_token_id": tokenizer.pad_token_id != MASK_TOKEN_ID,
        },
    )
    with (args.output_dir / "training_log.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "inline_complete", **metrics}, ensure_ascii=False) + "\n")
    log_progress("complete", decoded_samples=len(raw_records), parse_success=parse_success)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
