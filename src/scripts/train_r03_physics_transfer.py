#!/usr/bin/env python3
"""At most 256 offline repair-only updates from immutable R5-C B0.

Use --prepare-only with --sources-manifest to validate existing K8/K4 native
physics, then --prepared-dir for the one bounded training run. Preparation
loads tokenizers only; neither mode samples or labels any new structure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    write_json(temporary, value)
    os.replace(temporary, path)


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
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--task' in argv:
        i = argv.index('--task')
        if i + 1 >= len(argv) or argv[i + 1] != 'expert-edit':
            raise ValueError('the registered alternate task is expert-edit')
        return expert_main(argv[:i] + argv[i + 2:])
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


def expert_args(argv):
    parser = argparse.ArgumentParser(description='Train and validate an offline-supervised autonomous crystal editor')
    parser.add_argument('--mode', choices=('train', 'verify', 'sample', 'sample-base'), default='train')
    parser.add_argument('--b0-checkpoint', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume-state', type=Path, help='Explicit optimizer/RNG/data-cursor recovery; otherwise checkpoint is a new-stage warmstart')
    parser.add_argument('--data-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=160)
    parser.add_argument('--microbatch', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--module-learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--geometry-aux-fraction', type=float, default=.2)
    parser.add_argument('--content-fraction', type=float, default=.65)
    parser.add_argument('--inspect-fraction', type=float, default=.17)
    parser.add_argument('--student-feedback-fraction', type=float, default=0.)
    parser.add_argument('--healthy-state-fraction', type=float, default=0.)
    parser.add_argument('--content-target-mode', choices=('all_pending','next_token'), default='all_pending')
    parser.add_argument('--evaluation-content-target-mode', choices=('same','all_pending','next_token'), default='same')
    parser.add_argument('--m2t-probability', type=float, default=.7)
    parser.add_argument('--target-representative', choices=('none','old_aligned_grid'), default='none')
    parser.add_argument('--evaluation-target-representative', choices=('same','none','old_aligned_grid'), default='same')
    parser.add_argument('--seed', type=int, default=2026090807)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--smoke-sources', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=40)
    parser.add_argument('--eval-batches', type=int, default=12)
    parser.add_argument('--eval-examples', type=int, default=192,
                        help='Fixed global validation views, independent of GPU count and minibatching')
    parser.add_argument('--checkpoint-every', type=int, default=400)
    parser.add_argument('--early-stopping-patience', type=int, default=0)
    parser.add_argument('--time-budget-hours', type=float, default=1.)
    parser.add_argument('--training-seconds-already-used', type=float, default=0.)
    parser.add_argument('--diagnostic-sources', type=int, default=32)
    parser.add_argument('--split', choices=('train', 'dev'), default='dev')
    parser.add_argument('--source-kind', choices=('positive_edit', 'all_old_states'), default='positive_edit')
    parser.add_argument('--block-size', type=int, choices=(1, 4, 8), default=1)
    parser.add_argument('--sampling-batch-size', type=int, default=8)
    parser.add_argument('--force-mode', choices=('local_xyz', 'all_xyz', 'full_cell'))
    parser.add_argument('--proposal-input', choices=('masked', 'old_values'), default='masked',
                        help='initial content canvas; old_values matches T2T training views')
    parser.add_argument('--accept-threshold', type=float, default=.5)
    args = parser.parse_args(argv)
    if min(args.updates, args.microbatch, args.accumulation, args.eval_every, args.eval_batches) < 1:
        parser.error('positive training and evaluation sizes required')
    if not 0 < args.time_budget_hours <= 16 or args.training_seconds_already_used < 0:
        parser.error('training must obey the cumulative sixteen-hour cap')
    if args.training_seconds_already_used + 3600 * args.time_budget_hours > 16 * 3600:
        parser.error('requested training exceeds the remaining cumulative training cap')
    if args.mode in ('verify', 'sample') and args.checkpoint is None:
        parser.error('verification and sampling require an actual editor checkpoint')
    if args.resume_state is not None and (args.mode != 'train' or args.checkpoint is None):
        parser.error('complete-state recovery requires training and its associated checkpoint')
    if args.checkpoint_every < 1 or args.early_stopping_patience < 0 or args.eval_examples < 6:
        parser.error('invalid checkpoint or early-stopping interval')
    if args.mode == 'sample-base' and (args.checkpoint is not None or args.force_mode not in ('all_xyz','full_cell')
                                      or args.source_kind != 'positive_edit'):
        parser.error('matched B0 controls require the original B0 and a fixed all_xyz/full_cell single-task diagnostic')
    if args.source_kind == 'all_old_states' and args.force_mode is not None:
        parser.error('autonomous old-state evaluation cannot receive a forced teacher scope')
    return args


def verify_saved_tables(model, checkpoint):
    """Compare effective tables with saved trained tables, not a second loader."""
    import torch
    from safetensors import safe_open
    report = {}
    with safe_open(str(Path(checkpoint) / 'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
        for name, table in (('wte', model.get_input_embeddings().weight),
                            ('ff_out', model.get_output_embeddings().weight)):
            preferred = [key for key in saved.keys() if key.endswith(f'.{name}.modules_to_save.weight')]
            alternatives = [key for key in saved.keys() if key.endswith(f'.{name}.weight')]
            keys = preferred or alternatives
            if len(keys) != 1:
                raise ValueError(f'no unambiguous trained B0 table for {name}: {keys}')
            tensor = saved.get_slice(keys[0])
            if list(tensor.get_shape()) != list(table.shape):
                raise ValueError(f'saved {name} shape differs from the effective table')
            for start in range(0, table.shape[0], 2048):
                actual = table[start:start+2048].detach().cpu()
                expected = tensor[start:start+2048]
                if preferred and alternatives:
                    if len(alternatives) != 1 or not torch.equal(expected, saved.get_slice(alternatives[0])[start:start+2048]):
                        raise ValueError(f'ambiguous original trained table aliases for {name}')
                if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                    raise ValueError(f'effective {name} differs from saved trained table at row {start}')
            report[name] = {'key': keys[0], 'shape': list(table.shape), 'dtype': str(table.dtype), 'exact': True}
    return report


def verify_saved_lora(model, checkpoint):
    import torch
    from safetensors import safe_open
    count, keys_seen = 0, set()
    with safe_open(str(Path(checkpoint)/'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
        keys = {name for name in saved.keys() if '.lora_A.' in name or '.lora_B.' in name}
        for name, parameter in model.base_model.named_parameters():
            if '.lora_A.' not in name and '.lora_B.' not in name:
                continue
            key = name.replace('.default.', '.')
            if key not in keys or not torch.equal(parameter.detach().cpu(), saved.get_tensor(key)):
                raise ValueError(f'effective LoRA differs from the registered checkpoint: {name}')
            count += parameter.numel()
            keys_seen.add(key)
        if keys_seen != keys or count != 14_680_064:
            raise ValueError(f'incomplete B0 LoRA partition: {count} parameters, {len(keys_seen)}/{len(keys)} tensors')
    return {'parameters': count, 'tensors': len(keys_seen), 'all_exact': True}


def editor_gradient_audit(model, tokenizer, dataset, objective, device):
    import torch
    from crystal_dlm.expert_edit import make_edit_view, materialize_edit_batch
    selected = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    initial = {name: p.detach().cpu().clone() for name, p in selected}
    audits = {}
    for task in ('G', 'S'):
        if not dataset.content[task]:
            continue
        with torch.no_grad():
            for name, p in selected:
                p.copy_(initial[name])
        temporary = torch.optim.AdamW([p for _, p in selected], lr=1e-4, weight_decay=0.)
        row = dataset.content[task][0]
        groups = {}
        for step in range(2):
            view = make_edit_view(row, 'content', random.Random(123+step), dataset.prefixes[row['prompt']],
                                  content_target_mode=dataset.content_target_mode,
                                  m2t_probability=dataset.m2t_probability)
            batch = materialize_edit_batch([view], tokenizer, device)
            temporary.zero_grad(set_to_none=True)
            result = model(batch['input_ids'], attention_mask=batch['attention_mask'], edit_context=batch['edit_context'])
            loss, _ = objective(result, batch)  # content view has no decision labels
            loss.backward()
            groups = {'lora': sum(float(p.grad.detach().float().square().sum()) for name, p in selected
                                  if 'lora_' in name and p.grad is not None)}
            for name, module in model.content_modules().items():
                groups[name] = sum(float(p.grad.detach().float().square().sum()) for p in module.parameters()
                                   if p.grad is not None)
            if not all(math.isfinite(value) for value in groups.values()):
                raise ValueError(f'nonfinite {task} content gradients')
            temporary.step()
        if not all(value > 0 for value in groups.values()):
            raise ValueError(f'{task} content does not train every required conditioning path: {groups}')
        audits[task] = groups
        del temporary
    with torch.no_grad():
        for name, p in selected:
            p.copy_(initial[name])
    model.zero_grad(set_to_none=True)
    return audits


def editor_eval(model, tokenizer, dataset, objective, device, args, rank, world):
    import torch
    import torch.distributed as dist
    from crystal_dlm.expert_edit import materialize_edit_batch
    model.eval()
    aggregate = Counter()
    local_indices = list(range(rank,args.eval_examples,world))
    mean_fields = ('loss','content_ce','mode_ce','site_bce','quality_bce')
    with torch.no_grad():
        for begin in range(0,len(local_indices),args.microbatch):
            rows = [dataset[index] for index in local_indices[begin:begin+args.microbatch]]
            batch = materialize_edit_batch(rows, tokenizer, device, max_length=args.max_length)
            output = model(batch['input_ids'], attention_mask=batch['attention_mask'], edit_context=batch['edit_context'])
            _, metrics = objective(output, batch)
            for key in mean_fields:
                metrics[key] *= len(rows)
            aggregate.update(metrics)
    keys = sorted(aggregate)
    vector = torch.tensor([aggregate[key] for key in keys], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(vector)
    result = dict(zip(keys, vector.cpu().tolist()))
    for key in mean_fields:
        result[key] /= args.eval_examples
    for task in ('G', 'S', 'G_real', 'G_aux', 'S_real', 'S_aux'):
        result[task+'_content_ce'] = (result[task+'_content_ce_sum']/result[task+'_content_views']
                                      if result[task+'_content_views'] else None)
    for field in ('length', 'angle', 'coord', 'first_lattice'):
        count = result[field+'_content_tokens']
        result[field+'_content_ce'] = result[field+'_content_ce_sum']/count if count else None
    return result


def sample_editor_diagnostics(model, tokenizer, dataset, args, device, rank, world):
    from crystal_dlm.expert_edit import edit_structures
    from crystal_dlm.expert_edit_data import canonical_body, decode_body, certify_geometry, physics_input
    inverse = {int(value): key for key, value in tokenizer.get_vocab().items()}
    selected, seen = [], set()
    # Balanced same-family positives, with independent source groups per split.
    pools = {task: dataset.content[task] for task in ('G', 'S')}
    if args.source_kind == 'all_old_states':
        # Includes healthy, teacher-unavailable and physics-unknown states.
        # Hash order is fixed before inspecting any teacher or student outcome.
        states = {}
        for row in dataset.states:
            if row.get('source_kind') != 'current_B0_full_rich':
                continue
            if row['ancestor_id'] in states:
                raise ValueError('autonomous diagnosis has duplicate original B0 states')
            states[row['ancestor_id']] = row
        if not states:
            raise ValueError('autonomous diagnosis has no registered original B0 states')
        selected = sorted(states.values(), key=lambda row: hashlib.sha256(
                          f'{args.seed}:{row["ancestor_id"]}'.encode()).hexdigest())
    else:
        for i in range(max(map(len, pools.values()))):
            for task in ('G', 'S'):
                if i >= len(pools[task]):
                    continue
                row = pools[task][i]
                key = (row['ancestor_id'], task)
                if key in seen or (args.force_mode and row['action']['mode'] != args.force_mode):
                    continue
                seen.add(key)
                selected.append(row)
            if len(selected) >= args.diagnostic_sources:
                break
    selected = selected[:args.diagnostic_sources]
    results, physics, old_physics = [], [], []
    local_rows = [(index,row) for index,row in enumerate(selected) if index % world == rank]
    requests = [{'prompt': row['prompt'], 'body': row['old_body'], 'num_sites': row['num_atoms'],
                 'tasks': ('G','S') if args.source_kind == 'all_old_states' else (row['task'],),
                 'seed': args.seed+index} for index,row in local_rows]
    def progress(completed, total, batches):
        if completed % 8 == 0 or completed == total:
            print(json.dumps({'sample_rank': rank, 'completed': completed, 'requested': total,
                              'forward_batches': batches}), flush=True)
    sampled = edit_structures(model, tokenizer, requests, allowed_modes=dataset.allowed_modes,
                               block_size=args.block_size, force_mode=args.force_mode,
                               accept_threshold=args.accept_threshold, batch_size=args.sampling_batch_size,
                               progress=progress, proposal_input=args.proposal_input)
    for (index, row), output in zip(local_rows, sampled['results']):
        tasks = ('G', 'S') if args.source_kind == 'all_old_states' else (row['task'],)
        proposal = (output['canonical_body'] if args.source_kind == 'all_old_states' else
                    output['trace'][-1]['proposal_body'] if output['trace'] else row['old_body'])
        arrays = decode_body(proposal, inverse)
        certificate = certify_geometry(arrays)  # evaluation only, never influences the proposal/gate
        result = {'record_id': row['record_id'], 'ancestor_id': row['ancestor_id'], 'task': ''.join(tasks),
                  'num_atoms': row['num_atoms'], 'prompt': row['prompt'], 'old_body': row['old_body'],
                  'source_split': row['source_split'], 'source_row_idx': row['source_row_idx'],
                  'target_body': row.get('target_body'), 'old_physics_id': row.get('old_physics_id'),
                  'target_physics_id': row.get('target_physics_id'), 'proposal_geometry': certificate,
                  'old_geometry': row['old_geometry'], 'output': output, 'selection_basis': args.source_kind,
                  'input_domain': 'declared_compiled_old_states_with_original_failed_ancestors_reported_separately'}
        results.append(result)
        pid = f'expert-student:{args.output_dir.name}:{index}:{row["task"]}'
        text = ''.join(inverse[token] for token in proposal)
        physics.append(physics_input(pid, row['ancestor_id'], row['source_row_idx'], row['source_split'],
                                     'expert_quantized', arrays, text))
        old_arrays = decode_body(row['old_body'],inverse)
        old_text = ''.join(inverse[token] for token in row['old_body'])
        old_physics.append(physics_input(f'expert-initial:{args.output_dir.name}:{index}:{row["task"]}',
            row['ancestor_id'],row['source_row_idx'],row['source_split'],'native',old_arrays,old_text))
    write_jsonl(args.output_dir/f'samples.rank{rank}.jsonl', results)
    write_jsonl(args.output_dir/f'physics.rank{rank}.jsonl', physics)
    write_jsonl(args.output_dir/f'old_physics.rank{rank}.jsonl', old_physics)
    write_json(args.output_dir/f'sampling.rank{rank}.json', {'forward_batches': sampled['forward_batches'],
               'forward_rows': sampled['forward_rows'], 'requests': len(local_rows)})
    return len(selected)


def cpu_state(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key,item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def save_editor_training_checkpoint(model, tokenizer, optimizer, path, probe_examples, *, device, rank, world,
                                    step, example_cursor, cumulative_seconds, contract, best_state, progress_file, grouping):
    import torch
    import torch.distributed as dist
    import numpy as np
    from crystal_dlm.expert_edit import materialize_edit_batch
    local_rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
                 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}
    rngs = [None]*world
    if world > 1:
        dist.all_gather_object(rngs, local_rng)
    else:
        rngs[0] = local_rng
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)
        model.eval()
        model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(path)
        batch = materialize_edit_batch(probe_examples, tokenizer, device)
        with torch.no_grad():
            output = model(batch['input_ids'], attention_mask=batch['attention_mask'], edit_context=batch['edit_context'])
        torch.save({'examples': probe_examples, **{name: getattr(output,name).cpu() for name in
                    ('logits','mode_logits','site_logits','count_logits','quality_logits')}}, path/'roundtrip_probe.pt')
        files = {file.name: file_sha256(file) for file in path.iterdir() if file.is_file() and
                 file.suffix in ('.json','.pt','.safetensors')}
        state = {'schema': 'expert_training_state_v1', 'global_step': step, 'example_cursor': example_cursor,
                 'cumulative_train_seconds': cumulative_seconds, 'world_size': world, 'contract': contract,
                 'grouping': grouping,
                 'model_files_sha256': files, 'optimizer': cpu_state(optimizer.state_dict()), 'rng_states': rngs,
                 'best_state': best_state, 'progress_file': str(progress_file)}
        torch.save(state, path/'training_state.pt')
        write_json(path/'CHECKPOINT_FINAL.json', {key: value for key,value in state.items()
                   if key not in ('optimizer','rng_states')})
        (path/'_CHECKPOINT_SUCCESS').touch()
    if world > 1:
        dist.barrier()


def validate_editor_resume(state, contract, checkpoint, *, verify_files=True):
    if state.get('schema') != 'expert_training_state_v1' or state.get('contract') != contract:
        raise ValueError('resume data, optimizer parameter names or learning schedule changed; use an explicit new-stage warmstart')
    if (type(state.get('global_step')) is not int or state['global_step'] < 0
            or type(state.get('example_cursor')) is not int or state['example_cursor'] < 0
            or not 0 <= state.get('cumulative_train_seconds', -1) <= 16*3600):
        raise ValueError('invalid saved training cursor or cumulative budget')
    if (not isinstance(state.get('grouping'),dict) or set(state['grouping']) != {'world_size','microbatch','accumulation'}
            or any(type(value) is not int or value < 1 for value in state['grouping'].values())):
        raise ValueError('saved optimizer step grouping is incomplete')
    if verify_files:
        if not (checkpoint/'_CHECKPOINT_SUCCESS').is_file():
            raise ValueError('cannot resume an incomplete model/optimizer checkpoint')
        for name, expected in state['model_files_sha256'].items():
            if Path(name).name != name or file_sha256(checkpoint/name) != expected:
                raise ValueError('optimizer state does not belong to the actual saved model files')


def editor_selection_score(metrics):
    dev = metrics.get('dev', metrics)
    values = [dev.get(key) for key in ('G_real_content_ce','S_real_content_ce')]
    values = [value for value in values if value is not None]
    if not values:
        values = [dev.get(key) for key in ('G_content_ce','S_content_ce') if dev.get(key) is not None]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('no finite development content metric for checkpoint selection')
    return sum(values)/len(values)


def expert_main(argv):
    args = expert_args(argv)
    import datetime
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from crystal_dlm.expert_edit import (EDITOR_SCHEMA, ExpertEditDataset, ExpertEditObjective,
                                        load_editor_model, materialize_edit_batch)
    from crystal_dlm.state_training import enable_native_checkpointing
    rank, world, local_rank = (int(os.environ.get(key, default)) for key, default in
                                (('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')))
    if not torch.cuda.is_available() or not 1 <= world <= 6:
        raise RuntimeError('expert training/validation requires 1..6 assigned CUDA devices')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    b0_identity = verify_b0(args.b0_checkpoint) if rank == 0 else None
    if world > 1:
        shared = [b0_identity]
        dist.broadcast_object_list(shared, src=0)
        b0_identity = shared[0]
    model, tokenizer = load_editor_model(args.model_path, args.checkpoint or args.b0_checkpoint,
                                         device, trainable=args.mode == 'train')
    train_data = ExpertEditDataset(args.data_dirs, tokenizer, seed=args.seed, split='train',
                                  smoke_sources=args.smoke_sources, geometry_aux_fraction=args.geometry_aux_fraction,
                                  content_fraction=args.content_fraction,inspect_fraction=args.inspect_fraction,
                                  student_feedback_fraction=args.student_feedback_fraction,healthy_state_fraction=args.healthy_state_fraction,
                                  content_target_mode=args.content_target_mode,m2t_probability=args.m2t_probability,
                                  target_representative=args.target_representative)
    eval_target_mode = args.content_target_mode if args.evaluation_content_target_mode == 'same' else args.evaluation_content_target_mode
    eval_representative = args.target_representative if args.evaluation_target_representative == 'same' else args.evaluation_target_representative
    from copy import copy
    train_eval_data = copy(train_data)
    if eval_representative != args.target_representative:
        train_eval_data = ExpertEditDataset(args.data_dirs, tokenizer, seed=args.seed, split='train',
            smoke_sources=args.smoke_sources, geometry_aux_fraction=args.geometry_aux_fraction,
            content_fraction=args.content_fraction, inspect_fraction=args.inspect_fraction,
            student_feedback_fraction=args.student_feedback_fraction, healthy_state_fraction=args.healthy_state_fraction,
            content_target_mode=eval_target_mode, m2t_probability=args.m2t_probability,
            target_representative=eval_representative)
    train_eval_data.content_target_mode = eval_target_mode
    dev_data = ExpertEditDataset(args.data_dirs, tokenizer, seed=args.seed+10000, split='dev',
                                 geometry_aux_fraction=args.geometry_aux_fraction,content_target_mode=eval_target_mode,
                                 m2t_probability=args.m2t_probability, target_representative=eval_representative)
    train_comps = {row['composition_key'] for row in train_data.records}
    dev_comps = {row['composition_key'] for row in dev_data.records}
    if train_comps & dev_comps:
        raise ValueError('training and development composition groups overlap')
    tables = verify_saved_tables(model, args.b0_checkpoint)
    lora_identity = verify_saved_lora(model, args.checkpoint or args.b0_checkpoint)
    objective = ExpertEditObjective(tokenizer, device)
    if args.mode in ('sample', 'sample-base'):
        model.eval()
        dataset = train_eval_data if args.split == 'train' else dev_data
        # Scope capability is determined by the actual training partition.
        dataset.allowed_modes = (model.training_modes if args.mode == 'sample' else train_data.allowed_modes)
        requested = sample_editor_diagnostics(model, tokenizer, dataset, args, device, rank, world)
        if world > 1:
            dist.barrier()
        if rank == 0:
            for name in ('samples', 'physics', 'old_physics'):
                rows = []
                for worker in range(world):
                    rows.extend(read_jsonl(args.output_dir/f'{name}.rank{worker}.jsonl'))
                if len(rows) != requested:
                    raise ValueError('diagnostic sampling lost a requested source')
                write_jsonl(args.output_dir/(name+'.jsonl'), rows)
            write_json(args.output_dir/'SAMPLE_FINAL.json', {'schema': EDITOR_SCHEMA, 'requested': requested,
                       'split': args.split, 'force_mode': args.force_mode, 'block_size': args.block_size,
                       'sampling_batch_size': args.sampling_batch_size,
                       'proposal_input': args.proposal_input,
                       'sampling_workers': [read_json(args.output_dir/f'sampling.rank{worker}.json') for worker in range(world)],
                       'source_kind': args.source_kind,
                       'checkpoint': str(args.checkpoint or args.b0_checkpoint),
                       'frozen_B0_control': args.mode == 'sample-base', 'source_files': dataset.provenance})
            (args.output_dir/'_SUCCESS').touch()
        if world > 1:
            dist.destroy_process_group()
        return
    if args.mode == 'verify':
        expected = torch.load(args.checkpoint/'roundtrip_probe.pt', map_location='cpu', weights_only=False)
        batch = materialize_edit_batch(expected['examples'], tokenizer, device)
        with torch.no_grad():
            output = model(batch['input_ids'], attention_mask=batch['attention_mask'], edit_context=batch['edit_context'])
        equality = {name: bool(torch.equal(getattr(output, name).cpu(), expected[name])) for name in
                     ('logits', 'mode_logits', 'site_logits', 'count_logits', 'quality_logits')}
        if not all(equality.values()):
            raise ValueError(f'fresh-process editor roundtrip differs: {equality}')
        if rank == 0:
            write_json(args.output_dir/'VERIFY_FINAL.json', {'tables': tables, 'roundtrip_exact': equality})
            (args.output_dir/'_SUCCESS').touch()
        if world > 1:
            dist.destroy_process_group()
        return
    checkpoint_modules = enable_native_checkpointing(model.base_model)
    if not checkpoint_modules:
        raise ValueError('native LLaDA activation checkpointing is required')
    import inspect
    implementation = {}
    checkpoint_options = {}
    for name, module in model.base_model.named_modules():
        callback = getattr(module, '_activation_checkpoint_fn', None)
        if callback is not None:
            checkpoint_options[name] = {key: str(value) if not isinstance(value, (bool, int, float, str, type(None)))
                                         else value for key, value in getattr(callback, 'keywords', {}).items()}
            path = inspect.getsourcefile(type(module))
            if path:
                implementation[str(path)] = file_sha256(path)
    if not checkpoint_options or any(options.get('use_reentrant') is not False for options in checkpoint_options.values()):
        raise ValueError('actual LLaDA checkpoint implementation must establish use_reentrant=False')
    model.base_model.enable_input_require_grads()
    probe_examples = [train_data[0]]
    probe = materialize_edit_batch(probe_examples, tokenizer, device)
    with torch.no_grad():
        initial = model(probe['input_ids'], attention_mask=probe['attention_mask'], edit_context=probe['edit_context'])
        if args.checkpoint is None:
            reference = model.base_model(probe['input_ids'], attention_mask=probe['attention_mask']).logits
            if not torch.equal(initial.logits, reference):
                raise ValueError('zero-initialized editor does not exactly preserve actual B0 logits')
            del reference
    del initial
    model.train()
    audit_started = time.monotonic()
    gradients = editor_gradient_audit(model, tokenizer, train_data, objective, device)
    audit_seconds = time.monotonic() - audit_started
    selected = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    partitions = {'lora': [(n,p) for n,p in selected if 'lora_' in n],
                  'editor': [(n,p) for n,p in selected if 'lora_' not in n]}
    if not all(partitions.values()) or len({id(p) for _,p in selected}) != len(selected):
        raise ValueError('optimizer parameter partition is empty or duplicated')
    optimizer = torch.optim.AdamW([
        {'params': [p for _, p in partitions['lora']], 'lr': args.learning_rate},
        {'params': [p for _, p in partitions['editor']], 'lr': args.module_learning_rate}], weight_decay=args.weight_decay)
    model.training_modes = train_data.allowed_modes
    initialization = {'kind':'original_B0','adapter_sha256':B0_ADAPTER_SHA256}
    if args.checkpoint is not None and args.resume_state is None:
        parent_receipt = args.checkpoint/'CHECKPOINT_FINAL.json'
        if not parent_receipt.is_file() or not (args.checkpoint/'_CHECKPOINT_SUCCESS').is_file():
            raise ValueError('a new editor stage requires the complete initializer checkpoint receipt')
        initialization = {'kind':'editor_checkpoint','path':str(args.checkpoint.resolve()),
                          'receipt_sha256':file_sha256(parent_receipt)}
    contract = {'train_files': train_data.provenance, 'dev_files': dev_data.provenance,
                'initialization': initialization,
                'smoke_sources': args.smoke_sources,
                'train_record_ids_sha256': hashlib.sha256(json.dumps(sorted(row['record_id'] for row in train_data.records)).encode()).hexdigest(),
                'dev_record_ids_sha256': hashlib.sha256(json.dumps(sorted(row['record_id'] for row in dev_data.records)).encode()).hexdigest(),
                'trainer_source_sha256': file_sha256(Path(__file__)),
                'editor_source_sha256': file_sha256(Path(sys.modules['crystal_dlm.expert_edit'].__file__)),
                'seed': args.seed, 'updates': args.updates, 'max_length': args.max_length,
                'learning_rate': args.learning_rate, 'module_learning_rate': args.module_learning_rate,
                'weight_decay': args.weight_decay, 'geometry_aux_fraction': args.geometry_aux_fraction,
                'content_fraction':args.content_fraction,'inspect_fraction':args.inspect_fraction,
                'student_feedback_fraction':args.student_feedback_fraction,
                'healthy_state_fraction':args.healthy_state_fraction,
                'content_target_mode':args.content_target_mode, 'evaluation_content_target_mode':eval_target_mode,
                'm2t_probability':args.m2t_probability,
                'target_representative':args.target_representative, 'evaluation_target_representative':eval_representative,
                'target_representative_source_sha256':file_sha256(Path(__file__).resolve().parents[1]/'crystal_dlm/expert_target_representative.py'),
                'eval_examples': args.eval_examples,
                'parameter_names': [name for name,_ in selected]}
    start_step, example_cursor = 0, 0
    grouping = {'world_size': world, 'microbatch': args.microbatch, 'accumulation': args.accumulation}
    previous_grouping, resume_exact_grouping = None, None
    best_state = {'observed_score': None, 'saved_score': None, 'saved_checkpoint': None, 'bad_evaluations': 0}
    if args.resume_state is not None:
        import numpy as np
        restored = torch.load(args.resume_state, map_location='cpu', weights_only=False)
        contract['initialization'] = restored['contract']['initialization']
        validate_editor_resume(restored, contract, args.checkpoint, verify_files=rank==0)
        if world > 1:
            dist.barrier()
        optimizer.load_state_dict(restored['optimizer'])
        start_step, example_cursor, best_state = restored['global_step'], restored['example_cursor'], restored['best_state']
        explicit_used = args.training_seconds_already_used
        args.training_seconds_already_used = max(explicit_used, restored['cumulative_train_seconds'])
        progress_path = Path(restored.get('progress_file',''))
        if progress_path.is_file():
            try:
                previous = read_json(progress_path)
                previous_used = float(previous['cumulative_train_seconds'])
                if not math.isfinite(previous_used) or previous_used < 0:
                    raise ValueError('nonfinite or negative training progress')
                args.training_seconds_already_used = max(args.training_seconds_already_used, previous_used)
            except (json.JSONDecodeError,KeyError,ValueError,TypeError) as error:
                if explicit_used <= restored['cumulative_train_seconds']:
                    raise ValueError('damaged legacy progress requires an audited conservative training-seconds-already-used override') from error
        previous_grouping = restored['grouping']
        resume_exact_grouping = previous_grouping == grouping
        if resume_exact_grouping:
            rng = restored['rng_states'][rank]
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state(rng['cuda'],device)
        else:
            # The data cursor remains global and exact. A changed allocation is
            # recorded as a new grouping of the remaining examples, not bitwise replay.
            random.seed(args.seed+example_cursor+rank)
            np.random.seed((args.seed+example_cursor+rank)%(2**32))
            torch.manual_seed(args.seed+example_cursor+rank)
        del restored
    if args.training_seconds_already_used+3600*args.time_budget_hours > 16*3600:
        raise ValueError('recovered usage leaves less than the requested training budget')
    learner = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                      static_graph=True) if world > 1 else model
    config = {'schema': EDITOR_SCHEMA, 'world_size': world, 'microbatch': args.microbatch,
              'accumulation': args.accumulation, 'effective_batch': world*args.microbatch*args.accumulation,
              'parameters': {key: sum(p.numel() for _,p in value) for key,value in partitions.items()},
              'total_parameters': sum(p.numel() for p in model.parameters()), 'saved_tables': tables,
              'b0_identity': b0_identity, 'loaded_lora': lora_identity,
              'content_gradient_audit': gradients, 'checkpoint_modules': checkpoint_modules,
              'checkpoint_options': checkpoint_options, 'modeling_source_sha256': implementation,
              'train_sources': len({row['ancestor_id'] for row in train_data.records}),
              'dev_sources': len({row['ancestor_id'] for row in dev_data.records}),
              'positive_edits': {task: len(rows) for task, rows in train_data.content.items()},
              'independent_positive_sources': {task: len({row['ancestor_id'] for row in rows}) for task, rows in train_data.content.items()},
              'initialization_kind': 'complete_state_resume' if args.resume_state else 'new_stage_warmstart' if args.checkpoint else 'original_B0',
              'start_step': start_step, 'start_example_cursor': example_cursor,
              'grouping': grouping, 'previous_grouping': previous_grouping, 'resume_exact_grouping': resume_exact_grouping,
              'checkpoint_selection': 'best_saved_mean_real_G_S_development_content_CE',
              'allowed_modes': train_data.allowed_modes, 'train_files': train_data.provenance,
              'dev_files': dev_data.provenance, 'args': {key: str(value) if isinstance(value, Path) else
              [str(x) for x in value] if key == 'data_dirs' else value for key,value in vars(args).items()}}
    if rank == 0:
        representative_fields = ('record_id','ancestor_id','old_body','target_body','training_target_body',
                                 'training_target_certificate','training_representative_zero_edit')
        representative_files = {}
        for split, dataset in (('train', train_data), ('dev', dev_data)):
            rows = [{k: row[k] for k in representative_fields} for row in dataset.records
                    if 'training_target_certificate' in row]
            if rows:
                destination = args.output_dir / f'{split}_target_representatives.jsonl'
                write_jsonl(destination, rows)
                representative_files[split] = {'path': str(destination), 'sha256': file_sha256(destination),
                    'records': len(rows), 'equivalent_zero_edits': sum(row['training_representative_zero_edit'] for row in rows)}
        config['target_representatives'] = representative_files
        write_json(args.output_dir/'TRAIN_CONFIG.json', config)
        print(json.dumps(config), flush=True)
    started = time.monotonic()
    history, completed = [], start_step
    reserve_seconds = min(120.,max(5.,args.time_budget_hours*180))
    frozen_versions = {name:parameter._version for name,parameter in model.named_parameters() if not parameter.requires_grad}
    progress_file = args.output_dir/'TRAINING_PROGRESS.json'
    for step in range(start_step,args.updates):
        stop = torch.tensor(int(time.monotonic()-started+audit_seconds >= args.time_budget_hours*3600-reserve_seconds), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if int(stop):
            break
        if step == start_step or step % args.eval_every == 0 or step % args.checkpoint_every == 0:
            metrics = {split: editor_eval(model, tokenizer, dataset, objective, device, args, rank, world)
                       for split, dataset in (('train', train_eval_data), ('dev', dev_data))}
            metrics.update(step=step, seconds=time.monotonic()-started)
            history.append(metrics)
            score = editor_selection_score(metrics)
            improved = best_state['observed_score'] is None or score < best_state['observed_score']-1e-4
            if not (args.resume_state is not None and step == start_step):
                best_state['bad_evaluations'] = 0 if improved else best_state['bad_evaluations']+1
            if improved:
                best_state['observed_score'] = score
            if step > start_step and step % args.checkpoint_every == 0:
                checkpoint = args.output_dir/'checkpoints'/f'step-{step:06d}'
                if best_state['saved_score'] is None or score < best_state['saved_score']:
                    best_state.update(saved_score=score,saved_checkpoint=str(checkpoint))
                save_editor_training_checkpoint(model,tokenizer,optimizer,checkpoint,probe_examples,
                    device=device,rank=rank,world=world,step=step,example_cursor=example_cursor,
                    cumulative_seconds=args.training_seconds_already_used+time.monotonic()-started+audit_seconds,
                    contract=contract,best_state=best_state,progress_file=progress_file,grouping=grouping)
            if rank == 0:
                with (args.output_dir/'curves.jsonl').open('a') as stream:
                    stream.write(json.dumps(metrics)+'\n')
                print(json.dumps(metrics), flush=True)
            if args.early_stopping_patience and best_state['bad_evaluations'] >= args.early_stopping_patience:
                break
        learner.train()
        optimizer.zero_grad(set_to_none=True)
        warmup = min(1., (step+1)/max(5, min(50, args.updates//10)))
        decay = .1+.9*.5*(1+math.cos(math.pi*step/args.updates))
        for group, base_lr in zip(optimizer.param_groups, (args.learning_rate, args.module_learning_rate)):
            group['lr'] = base_lr*warmup*decay
        total_loss = 0.
        for micro in range(args.accumulation):
            offset = example_cursor+(micro*world+rank)*args.microbatch
            examples = [train_data[offset+i] for i in range(args.microbatch)]
            batch = materialize_edit_batch(examples, tokenizer, device, max_length=args.max_length)
            output = learner(batch['input_ids'], attention_mask=batch['attention_mask'], edit_context=batch['edit_context'])
            loss, _ = objective(output, batch)
            (loss/args.accumulation).backward()
            total_loss += float(loss.detach())/args.accumulation
        norm = torch.nn.utils.clip_grad_norm_([p for _,p in selected], 1., error_if_nonfinite=True)
        optimizer.step()
        completed = step+1
        example_cursor += args.accumulation*world*args.microbatch
        if rank == 0 and (completed % 10 == 0 or completed == 1):
            print(json.dumps({'step': completed, 'loss': total_loss, 'gradient_norm': float(norm),
                              'seconds': time.monotonic()-started}), flush=True)
            atomic_json(progress_file, {'global_step': completed, 'example_cursor': example_cursor,
                       'cumulative_train_seconds': args.training_seconds_already_used+time.monotonic()-started+audit_seconds})
    final_metrics = {split: editor_eval(model, tokenizer, dataset, objective, device, args, rank, world)
                     for split, dataset in (('train', train_eval_data), ('dev', dev_data))}
    if completed <= start_step:
        raise ValueError('training ended before a new optimizer update')
    for name,parameter in model.named_parameters():
        if name in frozen_versions and parameter._version != frozen_versions[name]:
            raise ValueError('a frozen B0 parameter changed during training')
    checkpoint = args.output_dir/'checkpoint'
    score = editor_selection_score(final_metrics)
    if best_state['saved_score'] is None or score < best_state['saved_score']:
        best_state.update(saved_score=score,saved_checkpoint=str(checkpoint))
    save_editor_training_checkpoint(model,tokenizer,optimizer,checkpoint,probe_examples,
        device=device,rank=rank,world=world,step=completed,example_cursor=example_cursor,
        cumulative_seconds=args.training_seconds_already_used+time.monotonic()-started+audit_seconds,
        contract=contract,best_state=best_state,progress_file=progress_file,grouping=grouping)
    elapsed = time.monotonic()-started+audit_seconds
    if rank == 0:
        final = {'schema': EDITOR_SCHEMA, 'status': 'complete', 'updates': completed, 'eligible_policy': True,
                 'checkpoint': str(checkpoint), 'train_seconds': elapsed,
                 'selected_checkpoint': best_state['saved_checkpoint'], 'best_saved_dev_score': best_state['saved_score'],
                 'new_updates': completed-start_step, 'example_cursor': example_cursor,
                 'cumulative_train_seconds': elapsed+args.training_seconds_already_used,
                 'parameters': config['parameters'], 'metrics': final_metrics,
                 'roundtrip_verified': False, 'mandatory_roundtrip_on_every_load': True, 'free_running_validated': False}
        write_json(args.output_dir/'TRAIN_FINAL.json', final)
        atomic_json(progress_file, {'global_step': completed, 'example_cursor': example_cursor,
                   'cumulative_train_seconds': args.training_seconds_already_used+elapsed})
        (args.output_dir/'POLICY_PATH').write_text(best_state['saved_checkpoint']+'\n')
        (args.output_dir/'_SUCCESS').touch()
        print(json.dumps(final), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
