#!/usr/bin/env python3
"""Sample compact dopant combinations and reconstruct candidate CIFs."""

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

from crystal_dlm.doping import (
    CANONICAL_BSITE_INDICES,
    DIRECTED_PROMPT,
    DOPANT_SYMBOLS,
    compact_answer,
    compact_special_tokens,
    combo_from_name,
    combo_name,
    arrays_to_cif_text,
    load_first_full80_template,
    parse_compact_answer,
    ranking_tuple,
    read_jsonl,
    reconstruct_arrays_from_template,
    write_json,
    write_jsonl,
)
from crystal_dlm.fixed_slot import MASK_TOKEN_ID
from crystal_dlm.llada_resize import ensure_llada_vocab_size


def load_model_and_tokenizer(args, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    tokenizer_source = checkpoint_path if checkpoint_path and checkpoint_path.exists() else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        raise RuntimeError(f"Tokenizer is missing required token {token}")
    return int(vocab[token])


def sample_dopants_batch(
    model,
    tokenizer,
    prompt: str,
    batch_size: int,
    temperature: float,
    seed: int,
):
    import torch

    rng = torch.Generator(device=model.device)
    rng.manual_seed(seed)
    prompt_text = prompt.rstrip() + "\n"
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = prompt_ids.shape[1]
    input_ids = prompt_ids.repeat(batch_size, 1)
    b_ids = [required_token_id(tokenizer, f"<B{idx:02d}>") for idx in CANONICAL_BSITE_INDICES]
    suffix = torch.tensor([[MASK_TOKEN_ID, MASK_TOKEN_ID, MASK_TOKEN_ID, *b_ids]], dtype=torch.long, device=model.device)
    input_ids = torch.cat([input_ids, suffix.repeat(batch_size, 1)], dim=1)
    attention_mask = torch.ones_like(input_ids)

    dopant_token_ids = [required_token_id(tokenizer, f"<DOPANT_{symbol}>") for symbol in DOPANT_SYMBOLS]
    selected_indices: List[List[int]] = [[] for _ in range(batch_size)]
    for pos in range(3):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, prompt_len + pos, :]
        masked_logits = torch.full_like(logits, float("-inf"))
        remaining_after = 2 - pos
        for row_idx in range(batch_size):
            min_idx = selected_indices[row_idx][-1] + 1 if selected_indices[row_idx] else 0
            max_exclusive = len(DOPANT_SYMBOLS) - remaining_after
            allowed_indices = list(range(min_idx, max_exclusive))
            allowed_ids = [dopant_token_ids[idx] for idx in allowed_indices]
            masked_logits[row_idx, allowed_ids] = logits[row_idx, allowed_ids]
        if temperature and temperature > 0:
            probs = torch.softmax(masked_logits / float(temperature), dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(1)
        else:
            next_ids = torch.argmax(masked_logits, dim=-1)
        input_ids[:, prompt_len + pos] = next_ids
        for row_idx, token_id in enumerate(next_ids.detach().cpu().tolist()):
            selected_indices[row_idx].append(dopant_token_ids.index(int(token_id)))
    answer_ids = input_ids[:, prompt_len : prompt_len + 6]
    return tokenizer.batch_decode(answer_ids, skip_special_tokens=False)


def load_surrogate_map(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["candidates"] if isinstance(payload, dict) and "candidates" in payload else payload
    return {row["name"]: row for row in rows}


def compact_record_map(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    return {combo_name(row["dopants"]): row for row in rows}


def reconstruct_candidates(
    candidates: List[Dict[str, Any]],
    output_dir: Path,
    full80_jsonl: Path,
    compact_jsonl: Path,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = output_dir / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)
    template = load_first_full80_template(full80_jsonl)
    known_map = compact_record_map(compact_jsonl)
    reconstructed: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        arrays = reconstruct_arrays_from_template(template, candidate["dopants"])
        name = candidate["name"]
        cif_path = cif_dir / f"{rank:03d}_{name}.cif"
        cif_path.write_text(arrays_to_cif_text(arrays, data_name=name), encoding="utf-8")
        known = known_map.get(name)
        record = {
            **candidate,
            "rank": rank,
            "template_cif": str(cif_path),
            "needs_dft": candidate.get("status") != "SUCCESS",
            "known_success": known is not None,
        }
        if known:
            record["true_properties"] = known["properties"]
            record["source_cif"] = known.get("source_cif")
        reconstructed.append(record)
    return reconstructed


def make_result_md(metrics: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Compact Doping Directed Generation Result",
        "",
        "## 运行结论",
        "",
        f"- Gate B：{'通过' if metrics['gate_b_pass'] else '未通过'}",
        f"- parse_rate：{metrics['parse_rate']:.4f}",
        f"- duplicate_or_invalid_rate：{metrics['duplicate_or_invalid_rate']:.4f}",
        f"- unique_combo_count：{metrics['unique_combo_count']}",
        f"- reconstruction_success_rate：{metrics['reconstruction_success_rate']:.4f}",
        f"- overlap@20：{metrics.get('overlap_at_20')}",
        "",
        "## Top Candidates",
        "",
    ]
    for row in candidates[:20]:
        true = row.get("true_properties") or {}
        lines.append(
            "- `{}` status={} needs_dft={} p_target={:.4f} pred_bg={:.4f} pred_fe={:.4f} pred_dfe={:.4f} true_bg={}".format(
                row["name"],
                row.get("status"),
                row.get("needs_dft"),
                float(row.get("p_band_gap_target", 0.0)),
                float(row.get("pred_band_gap", 0.0)),
                float(row.get("pred_formation_energy", 0.0)),
                float(row.get("pred_defect_formation_energy", 0.0)),
                true.get("band_gap"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct/")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/doping_compact")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surrogate-candidates-json", type=Path, default=None)
    parser.add_argument("--compact-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/compact_records.jsonl")
    parser.add_argument("--full80-jsonl", type=Path, default=PROJECT_ROOT / "data/doping_crystal/full80_success.jsonl")
    parser.add_argument("--prompt", default=DIRECTED_PROMPT)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    import torch

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args, device)
    for token in compact_special_tokens():
        required_token_id(tokenizer, token)
    if tokenizer.pad_token_id == MASK_TOKEN_ID:
        raise RuntimeError("tokenizer.pad_token_id must not equal LLaDA mask token id 126336")

    start = time.time()
    raw_records: List[Dict[str, Any]] = []
    failure_records: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    parse_success = 0
    for batch_start in range(0, args.num_samples, args.batch_size):
        current_batch = min(args.batch_size, args.num_samples - batch_start)
        decoded = sample_dopants_batch(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            batch_size=current_batch,
            temperature=args.temperature,
            seed=args.seed + batch_start,
        )
        for offset, text in enumerate(decoded):
            sample_idx = batch_start + offset
            record = {"sample_idx": sample_idx, "text": text}
            try:
                parsed = parse_compact_answer(text)
                parse_success += 1
                counts[parsed["name"]] += 1
                record.update({"parsed": True, **parsed})
            except Exception as exc:
                record.update({"parsed": False, "reason": type(exc).__name__, "message": str(exc)})
                failure_records.append(record)
            raw_records.append(record)

    surrogate = load_surrogate_map(args.surrogate_candidates_json)
    unique_candidates: List[Dict[str, Any]] = []
    for name, count in counts.items():
        parsed = parse_compact_answer(compact_answer(combo_from_name(name)))
        row = {
            "name": name,
            "dopants": parsed["dopants"],
            "dopant_site_indices": parsed["dopant_site_indices"],
            "sample_count": count,
        }
        if name in surrogate:
            row.update(surrogate[name])
            row["sample_count"] = count
        unique_candidates.append(row)
    unique_candidates.sort(
        key=lambda row: (ranking_tuple(row), int(row.get("sample_count", 0))),
        reverse=True,
    )
    top_candidates = unique_candidates[: args.top_k]
    design_dir = args.output_dir / "doping_design"
    reconstructed = reconstruct_candidates(
        top_candidates,
        output_dir=design_dir,
        full80_jsonl=args.full80_jsonl,
        compact_jsonl=args.compact_jsonl,
    )

    surrogate_top20 = [row["name"] for row in sorted(surrogate.values(), key=ranking_tuple, reverse=True)[:20]]
    sampled_top20 = [row["name"] for row in reconstructed[:20]]
    overlap_at_20 = (
        len(set(surrogate_top20) & set(sampled_top20)) / 20.0
        if surrogate_top20 and sampled_top20
        else None
    )
    known_top20 = [row for row in reconstructed[:20] if row.get("true_properties")]
    known_bandgap_hits = [
        1.7 <= float(row["true_properties"]["band_gap"]) <= 2.7
        for row in known_top20
    ]
    mean_known_fe = (
        sum(float(row["true_properties"]["formation_energy"]) for row in known_top20) / len(known_top20)
        if known_top20
        else None
    )
    metrics = {
        "num_samples": args.num_samples,
        "decoded_samples": len(raw_records),
        "parse_success": parse_success,
        "parse_rate": parse_success / max(1, len(raw_records)),
        "duplicate_or_invalid_count": len(failure_records),
        "duplicate_or_invalid_rate": len(failure_records) / max(1, len(raw_records)),
        "unique_combo_count": len(counts),
        "reconstruction_success_count": len(reconstructed),
        "reconstruction_success_rate": len(reconstructed) / max(1, min(args.top_k, len(unique_candidates))),
        "overlap_at_20": overlap_at_20,
        "known_top20_count": len(known_top20),
        "known_top20_bandgap_hit_rate": sum(known_bandgap_hits) / max(1, len(known_bandgap_hits)),
        "known_top20_mean_formation_energy": mean_known_fe,
        "gate_b_pass": (
            parse_success / max(1, len(raw_records)) >= 0.99
            and len(failure_records) / max(1, len(raw_records)) <= 0.01
            and len(reconstructed) / max(1, min(args.top_k, len(unique_candidates))) >= 0.99
            and len(counts) >= 30
        ),
        "time_sec": time.time() - start,
        "prompt": args.prompt,
        "temperature": args.temperature,
    }
    write_jsonl(args.output_dir / "raw_generations.jsonl", raw_records)
    write_jsonl(args.output_dir / "failure_cases.jsonl", failure_records)
    write_json(args.output_dir / "sample_metrics.json", metrics)
    write_json(design_dir / "candidates.json", {"candidates": reconstructed})
    (design_dir / "result.md").write_text(make_result_md(metrics, reconstructed), encoding="utf-8")
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
    write_json(args.output_dir / "prompt_pool.json", {"active_prompt": args.prompt, "directed_prompt": DIRECTED_PROMPT})
    with (args.output_dir / "training_log.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "sampling_complete", **metrics}, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
