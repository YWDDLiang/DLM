#!/usr/bin/env python3
"""Adapt only an external control head to frozen H1A2 P0 formula features.

All representable MP20 source rows are retained.  Existing SPAD contact labels
are reused by source_row_idx; missing labels can be computed from the original
MP20 CIF (geometry only).  No Typed-Planner hidden states, soft labels, energy,
hull, or generated outcomes are read as training inputs or targets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crystal_dlm.fixed_slot import SYMBOL_TO_Z, Z_TO_SYMBOL
from crystal_dlm.h1_llm_planner import (
    H1_PLANNER_PROMPT_STYLE_RICH_PLAN,
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    format_planner_prompt,
    load_llama3_compatible_config,
)
from crystal_dlm.r03_control_pointer import (
    FEATURE_SCHEMA,
    FEATURE_SOURCE,
    P0_ADAPTER_SHA256,
    POINTER_STATE_SCHEMA,
    FormulaFeatureSpec,
    R03ControlPointer,
    R03ControlPointerConfig,
    formula_hidden_batch,
    plan_composition,
    pointer_batch,
    sha256_file,
    species_pointer_loss,
    stable_digest,
)
from crystal_dlm.species_program_pointer import maximum_contact_tree_order


DATASET_SCHEMA = "r03_control_pointer_row_v1"
TRAIN_SCHEMA = "r03_control_pointer_train_v1"
SEED = 86017


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            yield value


def source_index(row: Mapping[str, Any], *, split: str) -> int:
    for field in ("source_split", "split"):
        if row.get(field) not in (None, "", split):
            raise ValueError(f"row split {row[field]!r} does not match {split!r}")
    value = row.get("source_row_idx")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("MP20 source_row_idx must be a nonnegative integer")
    return value


def index_rows(rows: Iterable[Mapping[str, Any]], *, split: str) -> dict[int, Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        index = source_index(row, split=split)
        if index in indexed:
            raise ValueError(f"duplicate MP20 source_row_idx in {split}: {index}")
        indexed[index] = row
    return indexed


def contact_from_cif(payload: tuple[int, str]) -> tuple[int, tuple[tuple[str, int], ...], tuple[str, ...]]:
    source_idx, cif = payload
    from pymatgen.core import Structure

    structure = Structure.from_str(cif, fmt="cif")
    if not structure.is_ordered:
        raise ValueError(f"MP20 source {source_idx} is not an ordered structure")
    species = [str(site.specie.symbol) for site in structure]
    from collections import Counter

    counts = Counter(species)
    composition = plan_composition({
        "N": len(species), "elements": list(counts), "counts": list(counts.values()),
    })
    order = maximum_contact_tree_order(species, structure.distance_matrix.tolist())
    return source_idx, composition, order


def prepare_rows(
    teacher_rows: Sequence[Mapping[str, Any]],
    cached_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    missing_contacts: Mapping[int, tuple[tuple[tuple[str, int], ...], Sequence[str]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join all teacher rows, fail loudly on a missing or mismatched label."""
    if split not in ("train", "val"):
        raise ValueError("only original MP20 train and validation splits are accepted")
    teacher = index_rows(teacher_rows, split=split)
    cached = index_rows(cached_rows, split=split)
    if not teacher:
        raise ValueError(f"empty MP20 {split} source")
    rows = []
    cache_hits = 0
    generated = 0
    for index, raw in teacher.items():
        plan = raw.get("plan_state")
        if not isinstance(plan, Mapping):
            raise ValueError(f"{split}:{index} lacks original plan_state")
        composition = plan_composition(plan)
        elements = [symbol for symbol, _ in composition]
        if index in cached:
            cached_row = cached[index]
            if cached_row.get("schema") != "spad_species_pointer_row_v1":
                raise ValueError("existing contact cache schema changed")
            if cached_row.get("outcomes_read") is not False:
                raise ValueError("contact cache lacks geometry-only provenance")
            atomic = [int(value) for value in cached_row["canonical_atomic_numbers"]]
            counts = [int(value) for value in cached_row["canonical_element_counts"]]
            if len(atomic) != len(counts) or len(set(atomic)) != len(atomic):
                raise ValueError("cached contact composition arrays are malformed")
            cache_composition = plan_composition({
                "N": sum(counts), "elements": [Z_TO_SYMBOL[z] for z in atomic], "counts": counts,
            })
            if cache_composition != composition:
                raise ValueError(f"cached/teacher composition mismatch at {split}:{index}")
            indices = [int(value) for value in cached_row["contact_tree_order_indices"]]
            if sorted(indices) != list(range(len(atomic))):
                raise ValueError("cached contact target is not a complete permutation")
            order = [Z_TO_SYMBOL[atomic[position]] for position in indices]
            if cached_row.get("contact_tree_order_symbols") not in (None, order):
                raise ValueError("cached contact symbols disagree with their indices")
            cache_hits += 1
            label_source = "existing_mp20_contact_cache"
        else:
            if missing_contacts is None or index not in missing_contacts:
                raise ValueError(f"missing contact target for retained {split}:{index}; supply original MP20 CIFs")
            contact_composition, order = missing_contacts[index]
            if tuple(contact_composition) != composition:
                raise ValueError(f"MP20 CIF/teacher composition mismatch at {split}:{index}")
            order = list(order)
            generated += 1
            label_source = "original_mp20_cif_contact_order"
        if set(order) != set(elements) or len(order) != len(elements):
            raise ValueError(f"contact teacher is not an exact species permutation at {split}:{index}")
        # Select an explicit allowlist.  The rich/Typed fields in source files
        # remain outside controller features and targets.
        rows.append({
            "schema": DATASET_SCHEMA,
            "source_split": split,
            "source_row_idx": index,
            "plan_state": {
                "N": sum(count for _, count in composition),
                "elements": elements,
                "counts": [count for _, count in composition],
            },
            "contact_tree_order_indices": [elements.index(symbol) for symbol in order],
            "contact_tree_order_symbols": list(order),
            "label_source": label_source,
        })
    return rows, {
        "source_rows": len(teacher), "retained_rows": len(rows), "filtered_rows": 0,
        "existing_contact_labels": cache_hits, "missing_contact_labels_computed": generated,
        "unused_cached_rows": len(set(cached) - set(teacher)),
        "source_row_indices_sha256": stable_digest([row["source_row_idx"] for row in rows]),
        "geometry_only_targets": True, "energy_hull_generated_outcome_labels_used": False,
    }


def build_split(
    teacher_path: Path, cache_path: Path, *, split: str, mp20_path: Path | None, workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    teacher_rows = list(iter_jsonl(teacher_path))
    cached_rows = list(iter_jsonl(cache_path))
    teacher = index_rows(teacher_rows, split=split)
    cached = index_rows(cached_rows, split=split)
    missing = sorted(set(teacher) - set(cached))
    contacts = {}
    if missing:
        if mp20_path is None:
            raise ValueError(f"{len(missing)} retained {split} rows lack cached labels; --mp20-dir is required")
        missing_set = set(missing)
        with mp20_path.open(encoding="utf-8", newline="") as handle:
            payloads = [
                (index, str(row["cif"])) for index, row in enumerate(csv.DictReader(handle))
                if index in missing_set
            ]
        if len(payloads) != len(missing):
            raise ValueError("original MP20 CSV does not cover every retained source index")
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(contact_from_cif, payloads, chunksize=16))
        else:
            results = [contact_from_cif(payload) for payload in payloads]
        contacts = {index: (composition, order) for index, composition, order in results}
    rows, report = prepare_rows(
        teacher_rows, cached_rows, split=split, missing_contacts=contacts,
    )
    report["source_files"] = {
        "teacher": {"path": str(teacher_path.resolve()), "sha256": sha256_file(teacher_path)},
        "contact_cache": {"path": str(cache_path.resolve()), "sha256": sha256_file(cache_path)},
    }
    if missing:
        report["source_files"]["mp20_csv"] = {"path": str(mp20_path.resolve()), "sha256": sha256_file(mp20_path)}
    return rows, report


def load_frozen_p0(
    llama_model: Path, planner_checkpoint: Path, *, device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Use the native H1 loader contract and refuse a changed P0 adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = planner_checkpoint / "adapter_model.safetensors"
    adapter_sha = sha256_file(adapter_path)
    if adapter_sha != P0_ADAPTER_SHA256:
        raise ValueError(f"P0 adapter identity changed: {adapter_sha}")
    tokenizer_source = planner_checkpoint if (planner_checkpoint / "tokenizer_config.json").exists() else llama_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = load_llama3_compatible_config(str(llama_model))
    if getattr(config, "model_type", None) != "llama":
        raise ValueError("R03 pointer requires the original Llama P0 base")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        llama_model, config=config, trust_remote_code=True, torch_dtype=dtype,
    )
    ensure_peft_cache_compat()
    disable_peft_bnb_autodetect()
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, planner_checkpoint, is_trainable=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device).eval()
    identity = {
        "adapter_sha256": adapter_sha,
        "adapter_path": str(planner_checkpoint.resolve()),
        "base_model_path": str(llama_model.resolve()),
        "base_config_sha256": sha256_file(llama_model / "config.json"),
        "llama_hidden_size": int(config.hidden_size),
        "tokenizer_source": str(tokenizer_source.resolve()),
        "tokenizer_size": len(tokenizer),
        "tokenizer_files": {
            name: sha256_file(tokenizer_source / name)
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
            if (tokenizer_source / name).exists()
        },
        "p0_trainable_parameters": 0,
    }
    return model, tokenizer, identity


def extract_features(
    model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], *,
    spec: FormulaFeatureSpec, batch_size: int, exact_prompt_text: str | None,
) -> Any:
    import torch

    # Same canonical composition + same prompt gives the same frozen replay.
    # Reuse its feature only; retain every original MP20 row and contact label
    # when expanding back for the one-pass head training.
    unique_rows = []
    lookup = {}
    inverse = []
    for row in rows:
        key = (plan_composition(row["plan_state"]),
               row.get("sample_idx", row.get("source_row_idx")) if spec.include_sample_id else None)
        if key not in lookup:
            lookup[key] = len(unique_rows)
            unique_rows.append(row)
        inverse.append(lookup[key])
    print(json.dumps({"event": "formula_replay_deduplication", "source_rows": len(rows),
                      "unique_replays": len(unique_rows), "source_rows_filtered": 0}), flush=True)
    values = []
    start = time.monotonic()
    for begin in range(0, len(unique_rows), batch_size):
        values.append(formula_hidden_batch(
            model, tokenizer, unique_rows[begin:begin + batch_size],
            spec=spec, exact_prompt_text=exact_prompt_text,
        ))
        if begin == 0 or (begin // batch_size + 1) % 25 == 0 or begin + batch_size >= len(unique_rows):
            print(json.dumps({
                "event": "formula_feature_progress", "completed": min(begin + batch_size, len(unique_rows)),
                "total": len(unique_rows), "source_rows": len(rows),
                "elapsed_seconds": round(time.monotonic() - start, 2),
            }), flush=True)
    result = torch.cat(values)[torch.tensor(inverse, dtype=torch.long)]
    if not torch.isfinite(result).all():
        raise ValueError("P0 replay returned non-finite features")
    return result


def order_counts(predicted: Any, teacher: Any, valid: Any) -> dict[str, int]:
    totals = {"rows": 0, "exact": 0, "root": 0, "noncanonical": 0, "canonical_exact": 0,
              "pairs": 0, "pair_correct": 0}
    for index in range(predicted.shape[0]):
        size = int(valid[index].sum())
        pred = [int(value) for value in predicted[index, :size].tolist()]
        target = [int(value) for value in teacher[index, :size].tolist()]
        if sorted(pred) != list(range(size)) or sorted(target) != list(range(size)):
            raise ValueError("pointer or teacher changed the candidate permutation")
        totals["rows"] += 1
        totals["exact"] += pred == target
        totals["root"] += pred[0] == target[0]
        totals["noncanonical"] += pred != list(range(size))
        totals["canonical_exact"] += target == list(range(size))
        p_rank = {value: rank for rank, value in enumerate(pred)}
        t_rank = {value: rank for rank, value in enumerate(target)}
        for left in range(size):
            for right in range(left + 1, size):
                totals["pairs"] += 1
                totals["pair_correct"] += (p_rank[left] < p_rank[right]) == (t_rank[left] < t_rank[right])
    return totals


def evaluate_pointer(pointer: Any, hidden: Any, rows: Sequence[Mapping[str, Any]], *, batch_size: int, device: str) -> dict[str, Any]:
    import torch

    pointer.eval()
    totals: dict[str, int] = {}
    loss_sum = 0.0
    steps = 0
    with torch.no_grad():
        for begin in range(0, len(rows), batch_size):
            batch = {key: value.to(device) for key, value in pointer_batch(rows[begin:begin + batch_size]).items()}
            features = hidden[begin:begin + batch_size].to(device=device, dtype=torch.float32)
            logits = pointer.permutation_logits(
                features, batch["atomic_numbers"], batch["counts"], batch["valid_mask"],
                teacher_order=batch["teacher_order"],
            )
            active_steps = int(batch["valid_mask"].sum())
            loss_sum += float(species_pointer_loss(logits, batch["teacher_order"], batch["valid_mask"])) * active_steps
            steps += active_steps
            predicted = pointer.decode(features, batch["atomic_numbers"], batch["counts"], batch["valid_mask"])
            for key, value in order_counts(predicted, batch["teacher_order"], batch["valid_mask"]).items():
                totals[key] = totals.get(key, 0) + value
    return {
        "rows": totals["rows"], "pointer_nll_per_active_step": loss_sum / steps,
        "exact_permutation_accuracy": totals["exact"] / totals["rows"],
        "root_accuracy": totals["root"] / totals["rows"],
        "pairwise_order_accuracy": totals["pair_correct"] / max(1, totals["pairs"]),
        "canonical_exact_permutation_accuracy": totals["canonical_exact"] / totals["rows"],
        "noncanonical_program_fraction": totals["noncanonical"] / totals["rows"],
        "counts": totals,
    }


def train_one_pass(pointer: Any, hidden: Any, rows: Sequence[Mapping[str, Any]], *, batch_size: int, lr: float, seed: int, device: str, log_path: Path | None = None) -> dict[str, Any]:
    """Optimize only the head, exactly once through every retained source row."""
    import torch

    if hidden.shape[0] != len(rows):
        raise ValueError("feature/source row alignment changed")
    pointer.train()
    total_steps = math.ceil(len(rows) / batch_size)
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    optimizer = torch.optim.AdamW(pointer.parameters(), lr=lr, weight_decay=1e-4)
    warmup = min(20, max(0, total_steps - 1))
    seen = []
    start = time.monotonic()
    weighted_loss = 0.0
    active_total = 0
    for step, begin in enumerate(range(0, len(rows), batch_size)):
        indices = order[begin:begin + batch_size]
        selected = [rows[index] for index in indices]
        batch = {key: value.to(device) for key, value in pointer_batch(selected).items()}
        features = hidden[indices].detach().to(device=device, dtype=torch.float32)
        if step < warmup:
            multiplier = (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, total_steps - warmup)
            multiplier = 0.5 * (1 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = lr * multiplier
        optimizer.zero_grad(set_to_none=True)
        logits = pointer.permutation_logits(
            features, batch["atomic_numbers"], batch["counts"], batch["valid_mask"],
            teacher_order=batch["teacher_order"],
        )
        loss = species_pointer_loss(logits, batch["teacher_order"], batch["valid_mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite pointer training loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(pointer.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        active = int(batch["valid_mask"].sum())
        weighted_loss += float(loss.detach()) * active
        active_total += active
        seen.extend(int(row["source_row_idx"]) for row in selected)
        record = {
            "event": "pointer_train_step", "step": step + 1, "total_steps": total_steps,
            "seen_rows": len(seen), "loss": float(loss.detach()),
            "lr": lr * multiplier, "gradient_norm": float(grad_norm),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == total_steps:
            print(json.dumps(record), flush=True)
    if len(seen) != len(rows) or len(set(seen)) != len(rows):
        raise RuntimeError("one-pass source coverage failed")
    return {
        "epochs": 1, "updates": total_steps, "rows_seen": len(seen),
        "source_visit_order_sha256": stable_digest(seen),
        "pointer_nll_per_active_step": weighted_loss / active_total,
        "checkpoint_selection": "final_one_pass_only",
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt-style", required=True)
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument("--include-sample-id", action="store_true", dest="include_sample_id")
    flags.add_argument("--no-include-sample-id", action="store_false", dest="include_sample_id")
    parser.add_argument("--prompt-text-file", type=Path, help="Exact historical constant prompt; no trailing-newline normalization")


def validate_approved_feature_spec(spec: FormulaFeatureSpec) -> None:
    if spec.prompt_style != H1_PLANNER_PROMPT_STYLE_RICH_PLAN or spec.include_sample_id:
        raise ValueError("frozen R03 contract requires h1_rich_plan_v1 and --no-include-sample-id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-data-dir", type=Path, required=True)
    parser.add_argument("--pointer-data-dir", type=Path, required=True)
    parser.add_argument("--mp20-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, help="Previously extracted compatible train/val formula_features files")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pointer-size", type=int, default=256)
    parser.add_argument("--contact-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    add_feature_arguments(parser)
    args = parser.parse_args()
    if args.lr != 1e-3 or args.seed != SEED:
        raise ValueError("approved single-pass pointer configuration fixes lr=1e-3 and seed=86017")
    if min(args.batch_size, args.feature_batch_size, args.pointer_size) < 1 or not 1 <= args.contact_workers <= 4:
        raise ValueError("batch sizes must be positive and contact workers must be in 1..4")
    spec = FormulaFeatureSpec(args.prompt_style, bool(args.include_sample_id))
    validate_approved_feature_spec(spec)
    exact_prompt = args.prompt_text_file.read_text(encoding="utf-8") if args.prompt_text_file else None
    if exact_prompt is not None and spec.include_sample_id:
        raise ValueError("constant exact historical prompt cannot contain varying sample IDs")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    datasets = {}
    dataset_reports = {}
    for split in ("train", "val"):
        print(json.dumps({"event": "prepare_contact_split", "split": split}), flush=True)
        datasets[split], dataset_reports[split] = build_split(
            args.teacher_data_dir / f"{split}.jsonl", args.pointer_data_dir / f"{split}.jsonl",
            split=split, mp20_path=args.mp20_dir / f"{split}.csv" if args.mp20_dir else None,
            workers=args.contact_workers,
        )
        with (args.output_dir / f"{split}_rows.jsonl").open("x", encoding="utf-8") as handle:
            for row in datasets[split]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps({"event": "contact_split_ready", "split": split, **dataset_reports[split]}), flush=True)

    # Loading P0 also validates its checkpoint and tokenizer identity.  Cached
    # features avoid repeated forwards, never bypass identity checks.
    model, tokenizer, identity = load_frozen_p0(args.llama_model, args.planner_checkpoint, device=args.device)
    if exact_prompt is None and not spec.include_sample_id:
        exact_prompt = format_planner_prompt(tokenizer, sample_idx=None, prompt_style=spec.prompt_style)
    features = {}
    for split in ("train", "val"):
        metadata = {
            "schema": FEATURE_SCHEMA, "split": split, "feature_spec": asdict(spec),
            "feature_source": FEATURE_SOURCE, "planner_identity": identity,
            "feature_formula_rendering": "canonical_atomic_number_counts_control_only",
            "exact_prompt_text": exact_prompt,
            "rows_sha256": stable_digest(datasets[split]),
            "source_row_indices": [row["source_row_idx"] for row in datasets[split]],
        }
        if args.feature_cache_dir:
            cached = torch.load(args.feature_cache_dir / f"{split}_formula_features.pt", map_location="cpu")
            if cached["metadata"] != metadata:
                raise ValueError(f"{split} feature cache identity/prompt/source alignment changed")
            features[split] = cached["hidden"].float()
        else:
            features[split] = extract_features(
                model, tokenizer, datasets[split], spec=spec, batch_size=args.feature_batch_size,
                exact_prompt_text=exact_prompt,
            )
        if features[split].shape != (len(datasets[split]), identity["llama_hidden_size"]) or not torch.isfinite(features[split]).all():
            raise ValueError(f"{split} feature cache has invalid shape or nonfinite values")
        torch.save({"metadata": metadata, "hidden": features[split]}, args.output_dir / f"{split}_formula_features.pt")
    del model
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    # Seed after loading/extraction so fresh head initialization is independent
    # of cache reuse, feature batching, and model-loader RNG consumption.
    torch.manual_seed(args.seed)
    config = R03ControlPointerConfig(identity["llama_hidden_size"], pointer_size=args.pointer_size)
    pointer = R03ControlPointer(config).to(args.device)
    manifest = {
        "schema": TRAIN_SCHEMA, "feature_spec": asdict(spec), "exact_prompt_text": exact_prompt,
        "planner_identity": identity, "pointer_config": asdict(config),
        "feature_formula_rendering": "canonical_atomic_number_counts_control_only",
        "feature_replay_deduplication": "identical_prompt_and_composition_only_all_source_rows_retained",
        "seed": args.seed, "lr": args.lr, "epochs": 1, "batch_size": args.batch_size,
        "datasets": dataset_reports, "p0_weights_updated": False,
        "rich_soft_fields_used": False, "typed_planner_features_used": False,
        "old_pointer_weights_loaded": False, "initial_r03_schedule_mutated": False,
        "revision_scope": "first_two_species_anchors_one_reverse_sweep",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    train_report = train_one_pass(
        pointer, features["train"], datasets["train"], batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, device=args.device, log_path=args.output_dir / "train_log.jsonl",
    )
    validation = evaluate_pointer(
        pointer, features["val"], datasets["val"], batch_size=args.batch_size, device=args.device,
    )
    torch.save({
        "schema": POINTER_STATE_SCHEMA, "config": asdict(config),
        "state_dict": {key: value.detach().cpu() for key, value in pointer.state_dict().items()},
        "feature_spec": asdict(spec), "exact_prompt_text": exact_prompt,
        "planner_identity": identity, "training_manifest": manifest,
    }, args.output_dir / "r03_control_pointer.pt")
    report = {"train": train_report, "validation": validation,
              "checkpoint_sha256": sha256_file(args.output_dir / "r03_control_pointer.pt")}
    write_json(args.output_dir / "results.json", report)
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
