#!/usr/bin/env python3
"""At most 256 offline repair-only updates from immutable R5-C B0.

Use --prepare-only with --sources-manifest to validate existing K8/K4 native
physics, then --prepared-dir for the one bounded training run. Preparation
loads tokenizers only; neither mode samples or labels any new structure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal_dlm.r03_physics_transfer import (
    B0_ADAPTER_SHA256, REPAIR_SUPPORT_PROTOCOL, REPAIR_VIEW_SCHEMA, TRANSFER_SCHEMA,
    TransferContractError, build_repair_constraints, file_sha256, prepare_transfer,
    read_jsonl, repair_scalar_example, sample_transfer_example,
    supported_reference_kl, supported_scalar_logits, weighted_scalar_ce,
)


B0_TOKENIZER_SHA256 = "3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509"
MAX_UPDATES = 256
LEARNING_RATE = 1e-6
REFERENCE_KL_WEIGHT = 0.1
MP20_CE_WEIGHT = 0.2


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, help="Original LLaDA base; required for training only")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sources-manifest", type=Path)
    source.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--max-length", type=int, default=382)
    parser.add_argument("--engineering-only", action="store_true",
                        help="Emit an ineligible engineering checkpoint, never a selectable policy")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output-dir must be new; existing experiments are never overwritten")
    if not 1 <= args.updates <= MAX_UPDATES or args.learning_rate != LEARNING_RATE:
        parser.error("this route allows 1..256 updates at exactly LR1e-6")
    if args.effective_batch < 1 or args.max_length != 382:
        parser.error("positive effective batch and unchanged max length382 required")
    if args.prepare_only and args.prepared_dir:
        parser.error("prepare-only requires the original pinned sources-manifest")
    if not args.prepare_only and args.model_path is None:
        parser.error("training requires the original --model-path")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("this bounded trainer uses one process/GPU; effective batch comes from accumulation")
    return args


def verify_b0(checkpoint: Path):
    if not (checkpoint / "adapter_config.json").is_file():
        raise TransferContractError("a complete original B0 PEFT checkpoint is required")
    if file_sha256(checkpoint / "adapter_model.safetensors") != B0_ADAPTER_SHA256:
        raise TransferContractError("construction/repair initialization must be the pinned original R5-C B0")
    if file_sha256(checkpoint / "tokenizer.json") != B0_TOKENIZER_SHA256:
        raise TransferContractError("the original trained B0 tokenizer must remain unchanged")
    config = read_json(checkpoint / "adapter_config.json")
    if int(config.get("r", 0)) != 8 or int(config.get("lora_alpha", 0)) != 32:
        raise TransferContractError("B0 LoRA ABI differs from the registered asset")
    if set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"}:
        raise TransferContractError("only B0's original attention/FFN LoRA targets are allowed")
    if not any("wte" in str(name) for name in config.get("modules_to_save", [])) or not any(
            "ff_out" in str(name) for name in config.get("modules_to_save", [])):
        raise TransferContractError("B0 must supply its trained embedding/head tables")
    return {"checkpoint": str(checkpoint.resolve()), "adapter_sha256": B0_ADAPTER_SHA256,
            "tokenizer_json_sha256": B0_TOKENIZER_SHA256, "adapter_config": config,
            "adapter_config_sha256": file_sha256(checkpoint / "adapter_config.json")}


def prepare(args, destination, b0_identity):
    from transformers import AutoTokenizer
    sources = read_json(args.sources_manifest)
    b0_tokenizer = AutoTokenizer.from_pretrained(str(args.b0_checkpoint), trust_remote_code=True,
                                                local_files_only=True)
    source_tokenizers = {
        role: AutoTokenizer.from_pretrained(str(sources[role]["source_tokenizer_path"]),
                                            trust_remote_code=True, local_files_only=True)
        for role in ("k8", "k4") if sources.get(role) is not None
    }
    records, rejected, retention, report = prepare_transfer(
        sources, b0_tokenizer=b0_tokenizer, source_tokenizers=source_tokenizers,
    )
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows in (("records.jsonl", records), ("rejected.jsonl", rejected), ("mp20_retention.jsonl", retention)):
        write_jsonl(destination / name, rows)
    report.update(b0=b0_identity, sources_manifest_sha256=file_sha256(args.sources_manifest),
                  prepared_files_sha256={name: file_sha256(destination / name)
                                         for name in ("records.jsonl", "rejected.jsonl", "mp20_retention.jsonl")})
    write_json(destination / "PREPARATION_FINAL.json", report)
    (destination / "_SUCCESS").touch()
    return destination


def load_prepared(directory, b0_identity):
    if not (directory / "_SUCCESS").is_file() or (directory / "_INVALIDATED").exists():
        raise TransferContractError("prepared compatibility dataset is incomplete or invalidated")
    report = read_json(directory / "PREPARATION_FINAL.json")
    if (report.get("schema") != TRANSFER_SCHEMA or report.get("status") != "complete"
            or report.get("repair_view_schema") != REPAIR_VIEW_SCHEMA
            or report.get("support_protocol") != REPAIR_SUPPORT_PROTOCOL
            or report.get("b0", {}).get("adapter_sha256") != b0_identity["adapter_sha256"]
            or report.get("b0", {}).get("tokenizer_json_sha256") != B0_TOKENIZER_SHA256):
        raise TransferContractError("prepared data belongs to a different B0/repair contract")
    for name, expected in report["prepared_files_sha256"].items():
        if Path(name).name != name or file_sha256(directory / name) != expected:
            raise TransferContractError("prepared dataset changed after compatibility acceptance")
    records = read_jsonl(directory / "records.jsonl")
    retention = read_jsonl(directory / "mp20_retention.jsonl")
    if not records or len(records) != report["normalization"]["compatible_occurrences"]:
        raise TransferContractError("no complete accepted physical transfer dataset")
    if not math.isclose(sum(float(row["dataset_weight"]) for row in records), 1.0, abs_tol=1e-10):
        raise TransferContractError("prepared physical weights do not sum to one")
    return records, retention, report


def set_repair_lora_trainable(model):
    """Existing B0 LoRA only; no added embedding rows, heads or state modules."""
    model.requires_grad_(False)
    selected = []
    for name, parameter in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    if not selected or any("state" in name or "conditioner" in name for name, _ in selected):
        raise TransferContractError("only the retained B0 attention/FFN LoRA can be updated")
    if model.get_input_embeddings().weight.requires_grad or model.get_output_embeddings().weight.requires_grad:
        raise TransferContractError("trained B0 embedding/head tables must remain frozen")
    return selected


def materialize(example, tokenizer, device, max_length=382):
    import torch
    prefix = list(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"])
    body = example["input_body"]
    if not prefix or len(prefix) + len(body) > max_length:
        raise TransferContractError("repair input would be empty or truncated")
    tokens = torch.tensor([prefix + body], dtype=torch.long, device=device)
    return tokens, torch.ones_like(tokens), len(prefix)


def scalar_forward(model, example, tokenizer, constraints, device):
    tokens, attention, prefix_length = materialize(example, tokenizer, device)
    raw = model(tokens, attention_mask=attention).logits
    vector, report = supported_scalar_logits(raw, example["input_body"], prefix_length,
                                              example["num_atoms"], example["position"], constraints=constraints)
    if not report["available"]:
        raise TransferContractError(f"sampled training view is unsupported: {report['reason']}")
    return vector


def _tables_equal(model, reference):
    import torch
    return {
        "embedding": bool(torch.equal(model.get_input_embeddings().weight, reference.get_input_embeddings().weight)),
        "head": bool(torch.equal(model.get_output_embeddings().weight, reference.get_output_embeddings().weight)),
    }


def train(args, prepared, b0_identity):
    import torch
    from scripts.sample_llada_dynamic_crystals import load_model_and_tokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("actual B0 training requires its assigned GPU; CPU mode is fixtures/preparation only")
    records, retention, preparation = load_prepared(prepared, b0_identity)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    model, tokenizer = load_model_and_tokenizer(str(args.model_path), str(args.b0_checkpoint), device)
    reference, reference_tokenizer = load_model_and_tokenizer(str(args.model_path), str(args.b0_checkpoint), device)
    reference.requires_grad_(False).eval()
    if tokenizer.get_vocab() != reference_tokenizer.get_vocab():
        raise TransferContractError("B0 reference and repair tokenizers differ")
    selected = set_repair_lora_trainable(model)
    frozen_versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
    checkpoint_modules = []
    for name, module in model.named_modules():
        setter = getattr(module, "set_activation_checkpointing", None)
        if callable(setter) and hasattr(module, "transformer"):
            setter("whole_layer")
            checkpoint_modules.append(name)
    require_input_grad = getattr(model, "enable_input_require_grads", None)
    if callable(require_input_grad):
        require_input_grad()
    if not checkpoint_modules:
        raise RuntimeError("native LLaDA activation checkpointing is required for the bounded repair allocation")
    constraints = build_repair_constraints(tokenizer)
    table_equality = _tables_equal(model, reference)
    if not all(table_equality.values()):
        raise TransferContractError("repair initialization did not preserve complete B0 embedding/head tables")
    first = sample_transfer_example(records, tokenizer=tokenizer, constraints=constraints, rng=random.Random(args.seed))
    model.eval()
    with torch.no_grad():
        left = scalar_forward(model, first, tokenizer, constraints, device)
        right = scalar_forward(reference, first, tokenizer, constraints, device)
    finite = (left > torch.finfo(left.dtype).min) & (right > torch.finfo(right.dtype).min)
    initial_delta = float((left[finite].float() - right[finite].float()).abs().max())
    if initial_delta != 0.0:
        raise TransferContractError("initial repair distribution is not exactly the frozen B0 distribution")
    # Eval mode preserves the original deterministic inference dropout behavior;
    # gradients still update the selected LoRA parameters normally.
    model.eval()
    optimizer = torch.optim.AdamW([p for _, p in selected], lr=LEARNING_RATE, weight_decay=0.0)
    config = {"method": TRANSFER_SCHEMA, "role": "repair_only", "construction_checkpoint": b0_identity,
              "repair_initialization": b0_identity, "updates": args.updates, "learning_rate": LEARNING_RATE,
              "effective_batch": args.effective_batch, "seed": args.seed, "optimizer": "fresh_AdamW",
              "reference_KL_weight": REFERENCE_KL_WEIGHT, "MP20_CE_weight": MP20_CE_WEIGHT if retention else 0.0,
              "physics_objective": "uniform-record sample; len(dataset)*dataset_weight times exact supported XYZ conditional CE",
              "reference_objective": "KL(frozen_B0 || repair) on identical physical masked views",
              "MP20_objective": "uniform compatible original train source and first-occurrence anchor XYZ conditional CE",
              "temperature": 0.7, "support_protocol": REPAIR_SUPPORT_PROTOCOL, "repair_view_schema": REPAIR_VIEW_SCHEMA,
              "dropout_mode": "eval_with_gradients", "trainable_names": [name for name, _ in selected],
              "trainable_parameters": sum(p.numel() for _, p in selected), "checkpoint_modules": checkpoint_modules,
              "initial_supported_logit_delta": initial_delta, "initial_tables_equal": table_equality,
              "preparation_sha256": file_sha256(prepared / "PREPARATION_FINAL.json"),
              "new_physical_targets": False, "old_full_path_reference_checks_bypassed": False,
              "old_teacher_guarantee_transferred": False, "engineering_only": args.engineering_only}
    write_json(args.output_dir / "training_config.json", config)
    counts = Counter()
    started = time.monotonic()
    with (args.output_dir / "training_log.jsonl").open("x", encoding="utf-8") as log:
        for step in range(1, args.updates + 1):
            optimizer.zero_grad(set_to_none=True)
            sums = Counter()
            for _ in range(args.effective_batch):
                example = sample_transfer_example(records, tokenizer=tokenizer, constraints=constraints, rng=rng)
                with torch.no_grad():
                    ref = scalar_forward(reference, example, tokenizer, constraints, device)
                current = scalar_forward(model, example, tokenizer, constraints, device)
                physical_loss = weighted_scalar_ce(current, example["target_token"], weight=example["weight"])
                kl = supported_reference_kl(current, ref)
                loss = physical_loss + REFERENCE_KL_WEIGHT * kl
                sums["physical_loss"] += float(physical_loss.detach())
                sums["reference_kl"] += float(kl.detach())
                counts["physics_scalar_states"] += 1
                counts["reference_scalar_states"] += 1
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite physical/retention objective")
                (loss / args.effective_batch).backward()
                if retention:
                    row = retention[rng.randrange(len(retention))]
                    site = row["anchor_sites"][rng.randrange(len(row["anchor_sites"]))]
                    anchor = repair_scalar_example(row["body_token_ids"], site, rng.randrange(3),
                                                     plan_state=row["plan_state"], constraints=constraints)
                    vector = scalar_forward(model, anchor, tokenizer, constraints, device)
                    anchor_ce = weighted_scalar_ce(vector, anchor["target_token"])
                    if not bool(torch.isfinite(anchor_ce)):
                        raise FloatingPointError("nonfinite original-MP20 retention loss")
                    (MP20_CE_WEIGHT * anchor_ce / args.effective_batch).backward()
                    sums["mp20_ce"] += float(anchor_ce.detach())
                    counts["MP20_scalar_states"] += 1
            gradient = torch.nn.utils.clip_grad_norm_([p for _, p in selected], 1.0)
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError("nonfinite repair LoRA gradient")
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for _, p in selected):
                raise FloatingPointError("nonfinite updated repair LoRA")
            event = {"step": step, "gradient_norm": float(gradient), "learning_rate": LEARNING_RATE,
                     **{key: value / args.effective_batch for key, value in sums.items()},
                     "elapsed_seconds": time.monotonic() - started}
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()
            if step == 1 or step % 16 == 0 or step == args.updates:
                print(json.dumps(event), flush=True)
    changed_frozen = [name for name, p in model.named_parameters()
                      if not p.requires_grad and p._version != frozen_versions[name]]
    table_equality = _tables_equal(model, reference)
    if changed_frozen or not all(table_equality.values()):
        raise TransferContractError("a frozen B0 parameter/table changed during repair adaptation")
    checkpoint = args.output_dir / "checkpoints" / f"step-{args.updates}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    # Configured B0 modules_to_save retain the complete trained tables; do not
    # additionally export unused raw embedding copies as new trainable heads.
    model.save_pretrained(checkpoint, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(checkpoint)
    from safetensors import safe_open
    expected_tables = {"wte": tuple(model.get_input_embeddings().weight.shape),
                       "ff_out": tuple(model.get_output_embeddings().weight.shape)}
    saved_tables = {}
    with safe_open(str(checkpoint / "adapter_model.safetensors"), framework="pt", device="cpu") as saved:
        keys = list(saved.keys())
        for name, shape in expected_tables.items():
            matching = [key for key in keys if f".{name}." in key and key.endswith(".weight")
                        and tuple(saved.get_slice(key).get_shape()) == shape]
            if not matching:
                raise TransferContractError(f"saved repair adapter omits the complete frozen B0 {name} table")
            saved_tables[name] = matching
    final = {"method": TRANSFER_SCHEMA, "eligible_policy": not args.engineering_only,
             "policy_path": str(checkpoint.resolve()), "role": "repair_only",
             "construction_checkpoint": b0_identity, "updates": args.updates,
             "learning_rate": LEARNING_RATE, "fresh_optimizer": True, "initial_supported_logit_delta": initial_delta,
             "frozen_parameter_version_changes": changed_frozen, "final_tables_equal_to_B0": table_equality,
             "saved_table_keys": saved_tables, "new_process_reload_verified": False,
             "physical_source_summary": preparation["normalization"], "training_states": dict(counts),
             "new_sampling": False, "new_physics_labels": False, "old_teacher_guarantee_transferred": False,
             "repair_view_schema": REPAIR_VIEW_SCHEMA, "support_protocol": REPAIR_SUPPORT_PROTOCOL,
             "elapsed_seconds": time.monotonic() - started,
             "adapter_sha256": file_sha256(checkpoint / "adapter_model.safetensors"),
             "physical_benefit": "unmeasured_until_registered_G_vs_P_development_and_final_SUN"}
    write_json(checkpoint / "R03_REPAIR_TRANSFER.json", final)
    write_json(args.output_dir / "TRAIN_FINAL.json", final)
    (args.output_dir / "POLICY_PATH").write_text(str(checkpoint.resolve()) + "\n", encoding="utf-8")
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(final, ensure_ascii=False), flush=True)


def main(argv=None):
    args = parse_args(argv)
    b0_identity = verify_b0(args.b0_checkpoint)
    if args.prepare_only:
        destination = prepare(args, args.output_dir, b0_identity)
        print(json.dumps({"status": "prepared", "path": str(destination),
                          "new_model_calls": 0, "new_physics_calls": 0}), flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        prepared = args.prepared_dir or prepare(args, args.output_dir / "data", b0_identity)
        train(args, prepared, b0_identity)
    except BaseException as exc:
        write_json(args.output_dir / "FAILURE.json", {"type": type(exc).__name__, "message": str(exc),
                                                      "eligible_policy": False})
        (args.output_dir / "_FAILED").touch()
        raise


if __name__ == "__main__":
    main()
