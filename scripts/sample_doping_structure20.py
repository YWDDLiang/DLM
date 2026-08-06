#!/usr/bin/env python3
"""Sample DOPING_STRUCT20 compressed structural codes and expand to full80 CIFs."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.doping import DOPANT_SYMBOLS, arrays_to_cif_text, read_jsonl, write_json, write_jsonl
from crystal_dlm.doping_structure import (
    FULL80_CONFIG,
    STRUCT20_CONFIG,
    STRUCT20_DIRECTED_PROMPT,
    bsite_dopants_from_arrays,
    composition_counts,
    expand_structure20_arrays,
    load_full80_template,
    parse_structure20_answer,
)
from crystal_dlm.fixed_slot import MASK_TOKEN_ID, Z_TO_SYMBOL
from crystal_dlm.llada_generation import generate
from crystal_dlm.llada_resize import ensure_llada_vocab_size


def load_model_and_tokenizer(args, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    tokenizer_source = checkpoint_path if checkpoint_path and checkpoint_path.exists() else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if checkpoint_path and (checkpoint_path / "adapter_config.json").exists():
        from peft import PeftModel

        model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model.resize_token_embeddings(len(tokenizer))
        ensure_llada_vocab_size(model, len(tokenizer))
        model = PeftModel.from_pretrained(model, str(checkpoint_path))
    else:
        model = AutoModel.from_pretrained(
            str(checkpoint_path or args.model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        with Path(args.data_dir, "vocab_tokens.txt").open(encoding="utf-8") as handle:
            tokens = [line.strip() for line in handle if line.strip()]
        num_new = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        if num_new:
            model.resize_token_embeddings(len(tokenizer))
        ensure_llada_vocab_size(model, len(tokenizer))
    model.to(device).eval()
    return model, tokenizer


def required_token_id(tokenizer, token: str) -> int:
    vocab = getattr(tokenizer, "_crystal_dlm_cached_vocab", None)
    if vocab is None:
        vocab = tokenizer.get_vocab()
        setattr(tokenizer, "_crystal_dlm_cached_vocab", vocab)
    if token not in vocab:
        raise RuntimeError(f"Tokenizer is missing required token {token}")
    return int(vocab[token])


def required_token_ids(tokenizer, tokens: Sequence[str]) -> List[int]:
    return [required_token_id(tokenizer, token) for token in tokens]


def coord_tokens(axis: str) -> List[str]:
    return [f"<{axis}_{idx:03d}>" for idx in range(STRUCT20_CONFIG.coord_min_bin, STRUCT20_CONFIG.coord_max_bin + 1)]


def build_geometry_constraints(tokenizer, dopant_token_ids_by_pos: Mapping[int, Sequence[int]]) -> tuple[List[List[int]], Dict[int, Any], List[int]]:
    allowed: List[List[int]] = []
    prefill: Dict[int, Any] = {}
    geometry_positions: List[int] = []

    n_id = required_token_id(tokenizer, "<N_020>")
    allowed.append([n_id])
    prefill[0] = n_id
    for prefix in ("LA", "LB", "LC"):
        allowed.append(required_token_ids(tokenizer, [f"<{prefix}_{idx:03d}>" for idx in range(0, 501)]))
        geometry_positions.append(len(allowed) - 1)
    for prefix in ("AA", "AB", "AG"):
        allowed.append(required_token_ids(tokenizer, [f"<{prefix}_{idx:03d}>" for idx in range(1, 180)]))
        geometry_positions.append(len(allowed) - 1)

    fixed_species = {}
    for slot_idx in range(16):
        if slot_idx in {0, 1, 14}:
            continue
        fixed_species[slot_idx] = "Mn" if slot_idx == 15 else "Pb"
    fixed_species.update({16: "Cs", 17: "I", 18: "Pb", 19: "Mn"})

    for slot_idx in range(20):
        slot_token_id = required_token_id(tokenizer, f"<S{slot_idx:02d}>")
        allowed.append([slot_token_id])
        prefill[len(allowed) - 1] = slot_token_id
        element_pos = len(allowed)
        if slot_idx in {0, 1, 14}:
            ids = list(dopant_token_ids_by_pos.get(slot_idx) or required_token_ids(tokenizer, [f"<E_{symbol}>" for symbol in DOPANT_SYMBOLS]))
            allowed.append(ids)
            if slot_idx in dopant_token_ids_by_pos:
                prefill[element_pos] = ids
        else:
            token_id = required_token_id(tokenizer, f"<E_{fixed_species[slot_idx]}>")
            allowed.append([token_id])
            prefill[element_pos] = token_id
        for axis in ("X", "Y", "Z"):
            allowed.append(required_token_ids(tokenizer, coord_tokens(axis)))
            geometry_positions.append(len(allowed) - 1)
    if len(allowed) != 107:
        raise RuntimeError(f"Internal structure20 schema error: {len(allowed)} positions")
    return allowed, prefill, geometry_positions


def sample_dopant_tokens(
    model,
    tokenizer,
    prompt_text: str,
    batch_size: int,
    temperature: float,
    seed: int,
) -> Dict[int, List[int]]:
    import torch

    model_device = getattr(model, "device", None) or next(model.parameters()).device
    rng = torch.Generator(device=model_device)
    rng.manual_seed(seed)
    encoded = tokenizer([prompt_text] * batch_size, add_special_tokens=False, padding=True, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(model_device)
    prompt_len = prompt_ids.shape[1]
    suffix = torch.full((batch_size, 107), MASK_TOKEN_ID, dtype=torch.long, device=model_device)
    dopant_positions = {0: 8, 1: 13, 14: 78}
    base_allowed, base_prefill, _ = build_geometry_constraints(tokenizer, {})
    for pos, token_id in base_prefill.items():
        if isinstance(token_id, int):
            suffix[:, pos] = int(token_id)
    x = torch.cat([prompt_ids, suffix], dim=1)
    attention_mask = torch.ones_like(x)
    dopant_token_ids = [required_token_id(tokenizer, f"<E_{symbol}>") for symbol in DOPANT_SYMBOLS]
    selected: List[set[int]] = [set() for _ in range(batch_size)]
    sampled_by_slot: Dict[int, List[int]] = {slot: [] for slot in dopant_positions}
    for slot_idx, gen_pos in dopant_positions.items():
        outputs = model(input_ids=x, attention_mask=attention_mask)
        logits = outputs.logits[:, prompt_len + gen_pos, :]
        masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
        for row_idx in range(batch_size):
            allowed_ids = [token_id for token_id in dopant_token_ids if token_id not in selected[row_idx]]
            masked[row_idx, allowed_ids] = logits[row_idx, allowed_ids]
        if temperature and temperature > 0:
            probs = torch.softmax(masked / float(temperature), dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(1)
        else:
            next_ids = torch.argmax(masked, dim=-1)
        x[:, prompt_len + gen_pos] = next_ids
        values = [int(token_id) for token_id in next_ids.detach().cpu().tolist()]
        sampled_by_slot[slot_idx] = values
        for row_idx, token_id in enumerate(values):
            selected[row_idx].add(token_id)
    return sampled_by_slot


def graph_from_arrays(arrays: Mapping[str, Any], crysllmgen_dir: Path):
    sys.path.insert(0, str(crysllmgen_dir.resolve()))
    from data_utils import process_one
    from crystal_dlm.fixed_slot import arrays_to_structure

    structure = arrays_to_structure(arrays)
    cif = structure.to(fmt="cif")
    return process_one(cif, True, False, "crystalnn", False, 0.01), cif


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def make_result_md(metrics: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# DOPING_STRUCT20 Sampling Result",
            "",
            "## 结论边界",
            "",
            "- 本结果只验证 compressed20 结构代码能否展开为完整 80 原子结构；没有 DFT relaxation，不能声称性质更好。",
            "",
            "## Metrics",
            "",
            f"- parse_rate: {metrics['parse_rate']:.4f}",
            f"- reconstruction_rate: {metrics['reconstruction_rate']:.4f}",
            f"- composition_exact_rate: {metrics['composition_exact_rate']:.4f}",
            f"- graph_build_rate: {metrics['graph_build_rate']}",
            f"- unique_expanded_structure_count: {metrics['unique_expanded_structure_count']}",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/doping_structure20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full80-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/full80_success.jsonl")
    parser.add_argument("--crysllmgen-dir", type=Path, default=PROJECT_ROOT / "reference/crysllmgen")
    parser.add_argument("--prompt", default=STRUCT20_DIRECTED_PROMPT)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()

    import torch

    random.seed(args.seed)
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = args.output_dir / "expanded_cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.jsonl"
    partial_metrics_path = args.output_dir / "partial_metrics.json"
    raw_path = args.output_dir / "raw_generations.jsonl"
    failure_path = args.output_dir / "failure_cases.jsonl"
    expanded_path = args.output_dir / "expanded_arrays.jsonl"

    def log_progress(event: str, **payload: Any) -> None:
        append_jsonl(
            progress_path,
            {
                "event": event,
                "elapsed_sec": time.time() - start,
                **payload,
            },
        )

    log_progress("start", num_samples=args.num_samples, batch_size=args.batch_size, steps=args.steps)
    hidden_names = set()
    stats_path = args.data_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        hidden_names = {row["name"] for row in stats.get("hidden_good_holdout", [])}
    template = load_full80_template(args.full80_jsonl, exclude_names=hidden_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress("model_load_start", device=str(device), checkpoint_path=args.checkpoint_path)
    model, tokenizer = load_model_and_tokenizer(args, device)
    model_device = getattr(model, "device", None) or next(model.parameters()).device
    log_progress("model_load_done", vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)
    if tokenizer.pad_token_id == MASK_TOKEN_ID:
        raise RuntimeError("tokenizer.pad_token_id must not equal LLaDA mask token id 126336")

    raw_records: List[Dict[str, Any]] = []
    failure_records: List[Dict[str, Any]] = []
    expanded_records: List[Dict[str, Any]] = []
    parse_success = 0
    reconstruction_success = 0
    composition_exact = 0
    graph_success = 0
    unique_signatures: Counter[str] = Counter()
    prompt_text = args.prompt.rstrip() + "\n"
    geometry_groups = None

    def current_metrics() -> Dict[str, Any]:
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
            "mode": "doping_structure20",
            "graph_skipped": args.skip_graph,
        }

    for batch_start in range(0, args.num_samples, args.batch_size):
        current_batch = min(args.batch_size, args.num_samples - batch_start)
        batch_t0 = time.time()
        log_progress("batch_start", batch_start=batch_start, batch_size=current_batch)
        sampled_dopants = sample_dopant_tokens(
            model,
            tokenizer,
            prompt_text,
            current_batch,
            args.temperature,
            args.seed + batch_start,
        )
        log_progress("dopant_sample_done", batch_start=batch_start, elapsed_batch_sec=time.time() - batch_t0)
        constraints, prefill, geometry_positions = build_geometry_constraints(tokenizer, sampled_dopants)
        encoded = tokenizer([prompt_text] * current_batch, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded["attention_mask"].to(model_device)
        log_progress("generate_start", batch_start=batch_start, prompt_tokens=int(input_ids.shape[1]))
        outputs = generate(
            model,
            input_ids,
            attention_mask=attention_mask.to(model.device),
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
            record = {"sample_idx": sample_idx, "text": text}
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
                cif_path.write_text(arrays_to_cif_text(expanded, data_name=f"struct20_{sample_idx}_{name}"), encoding="utf-8")
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
                failure = {
                    "sample_idx": sample_idx,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                    "text": text,
                }
                failure_records.append(failure)
                append_jsonl(failure_path, failure)
                record.update({"parsed": False, **failure})
            raw_records.append(record)
            append_jsonl(raw_path, record)
        write_json(partial_metrics_path, current_metrics())
        log_progress(
            "batch_done",
            batch_start=batch_start,
            decoded_samples=len(raw_records),
            parse_success=parse_success,
            reconstruction_success=reconstruction_success,
            elapsed_batch_sec=time.time() - batch_t0,
        )

    metrics = current_metrics()
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
            "checkpoint_path": args.checkpoint_path,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "mask_token_id": MASK_TOKEN_ID,
            "pad_token_id_ne_mask_token_id": tokenizer.pad_token_id != MASK_TOKEN_ID,
        },
    )
    write_json(args.output_dir / "prompt_pool.json", {"active_prompt": args.prompt})
    (args.output_dir / "result.md").write_text(make_result_md(metrics), encoding="utf-8")
    with (args.output_dir / "training_log.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "sampling_complete", **metrics}, ensure_ascii=False) + "\n")
    log_progress("complete", decoded_samples=len(raw_records), parse_success=parse_success)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
