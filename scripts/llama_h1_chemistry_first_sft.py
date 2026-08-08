#!/usr/bin/env python3
"""Train fixed-endpoint SFT-v2 or SFT-v2-C from the protected P0 adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from crystal_dlm.h1_chemistry_first_sft import (  # noqa: E402
    H1_CHEMISTRY_FIRST_GRAD_ACCUM,
    H1_CHEMISTRY_FIRST_LR,
    H1_CHEMISTRY_FIRST_SFT_SCHEMA,
    H1_CHEMISTRY_FIRST_SFT_SEED,
    H1_CHEMISTRY_FIRST_WEIGHT_DECAY,
    accumulation_group_size,
    accumulation_is_update_boundary,
    optimizer_update_count,
    record_order_sha256,
    warmup_step_count,
)
from crystal_dlm.h1_llm_planner import (  # noqa: E402
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    load_llama3_compatible_config,
)
from crystal_dlm.h1_nocharge_ion_aux import validation_anchor_nll_gate  # noqa: E402
from scripts.build_h1_nocharge_ion_aux_sft_data import sha256_file  # noqa: E402
from scripts.llama_nocharge_ion_aux_sft import (  # noqa: E402
    answer_token_weights,
    collate,
    dual_adapter_identity_report,
    format_prompt,
    save_candidate_adapter,
    sft_nll_by_task,
    tensor_batch_to_device,
    weighted_sft_loss,
)


CANDIDATE_TO_DATA_KEY = {"sft_v2": "sft_v2", "sft_v2_c": "sft_v2_c"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object record")
                yield value


class ChemistryFirstDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.path = Path(path)
        self.rows = list(iter_jsonl(self.path))
        if not self.rows:
            raise ValueError(f"no records found in {self.path}")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        record_ids = []
        for row in self.rows:
            if row.get("schema") != H1_CHEMISTRY_FIRST_SFT_SCHEMA:
                raise ValueError(
                    f"record {row.get('record_id')} has schema {row.get('schema')!r}"
                )
            if row.get("loss_mode") != "sft":
                raise ValueError(f"record {row.get('record_id')} is not pure SFT")
            if float(row.get("sample_weight", -1.0)) != 1.0:
                raise ValueError(f"record {row.get('record_id')} changed sample weight")
            if row.get("generated_charge_field") is not False:
                raise ValueError(f"record {row.get('record_id')} permits generated charge")
            record_ids.append(str(row.get("record_id", "")))
        if any(not value for value in record_ids) or len(record_ids) != len(set(record_ids)):
            raise ValueError(f"{self.path} has missing or duplicate record IDs")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt = format_prompt(self.tokenizer, row)
        prompt_ids = [
            int(value)
            for value in self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ]
        answer = str(row["answer"]).strip()
        answer_ids, answer_weights = answer_token_weights(
            self.tokenizer,
            answer,
            row.get("weighted_answer_spans") or [],
            eos_text=self.tokenizer.eos_token or "",
        )
        if len(prompt_ids) + len(answer_ids) > self.max_length:
            raise ValueError(
                f"record {row['record_id']} would truncate at max_length={self.max_length}: "
                f"prompt={len(prompt_ids)} answer={len(answer_ids)}"
            )
        input_ids = prompt_ids + answer_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(
                [-100] * len(prompt_ids) + answer_ids,
                dtype=torch.long,
            ),
            "token_weight": torch.tensor(
                [0.0] * len(prompt_ids) + answer_weights,
                dtype=torch.float32,
            ),
            "kl_mask": torch.zeros(len(input_ids), dtype=torch.float32),
            "sample_weight": 1.0,
            "record_id": str(row["record_id"]),
            "task": str(row["task"]),
        }


def load_and_verify_data_contract(
    data_dir: Path,
    audit_path: Path,
    *,
    candidate: str,
    allow_fixture: bool,
) -> dict[str, Any]:
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema") != H1_CHEMISTRY_FIRST_SFT_SCHEMA
        or report.get("status") != "pass"
    ):
        raise RuntimeError("chemistry-first data audit identity mismatch")
    if bool(report.get("fixture_only")) and not allow_fixture:
        raise RuntimeError("frozen training cannot consume a fixture data ledger")
    key = CANDIDATE_TO_DATA_KEY[str(candidate)]
    candidate_report = (report.get("outputs") or {}).get(key)
    if not isinstance(candidate_report, Mapping):
        raise RuntimeError(f"data audit omitted candidate {candidate}")
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    expected = {
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
    }
    for field, actual in expected.items():
        if candidate_report.get(field) != actual:
            raise RuntimeError(f"{candidate} {field} mismatch")
    if (report.get("order_pair_audit") or {}).get("record_multiset_equal") is not True:
        raise RuntimeError("SFT-v2/SFT-v2-C record multiset identity did not pass")
    return report


def candidate_sft_loss(model, batch: Mapping[str, Any]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    return weighted_sft_loss(
        outputs.logits,
        batch["labels"],
        batch["token_weight"],
        batch["sample_weight"],
    )


@torch.no_grad()
def validation_anchor_nll(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    adapter_name: str,
) -> dict[str, float]:
    report = sft_nll_by_task(
        model,
        loader,
        device,
        adapter_name=adapter_name,
    )
    if set(report) != {"conditional_structural_anchor"}:
        raise RuntimeError(f"validation ledger is not anchor-only: {sorted(report)}")
    return report["conditional_structural_anchor"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--p0-adapter-path", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATE_TO_DATA_KEY), required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=H1_CHEMISTRY_FIRST_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=H1_CHEMISTRY_FIRST_LR)
    parser.add_argument("--weight-decay", type=float, default=H1_CHEMISTRY_FIRST_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=H1_CHEMISTRY_FIRST_SFT_SEED)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-nonfrozen-fixture", action="store_true")
    args = parser.parse_args()

    report = load_and_verify_data_contract(
        args.data_dir,
        args.data_audit_report,
        candidate=str(args.candidate),
        allow_fixture=bool(args.allow_nonfrozen_fixture),
    )
    if int(args.batch_size) != 1:
        raise RuntimeError("chemistry-first SFT freezes batch size at 1")
    if int(args.grad_accum) != H1_CHEMISTRY_FIRST_GRAD_ACCUM:
        raise RuntimeError("chemistry-first SFT freezes gradient accumulation at 8")
    if float(args.lr) != H1_CHEMISTRY_FIRST_LR:
        raise RuntimeError("chemistry-first SFT freezes LR at 2e-6")
    if float(args.weight_decay) != H1_CHEMISTRY_FIRST_WEIGHT_DECAY:
        raise RuntimeError("chemistry-first SFT freezes weight decay at 0")
    if int(args.seed) != H1_CHEMISTRY_FIRST_SFT_SEED:
        raise RuntimeError("chemistry-first SFT seed role changed")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.preflight_only and device.type != "cuda":
        raise RuntimeError("frozen chemistry-first SFT requires CUDA")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("frozen chemistry-first SFT requires BF16 support")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer_source = (
        args.p0_adapter_path
        if (Path(args.p0_adapter_path) / "tokenizer_config.json").exists()
        else args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds = ChemistryFirstDataset(
        args.data_dir / "train.jsonl",
        tokenizer,
        int(args.max_length),
    )
    val_ds = ChemistryFirstDataset(
        args.data_dir / "val.jsonl",
        tokenizer,
        int(args.max_length),
    )
    data_key = CANDIDATE_TO_DATA_KEY[str(args.candidate)]
    data_identity = report["outputs"][data_key]
    if len(train_ds) != int(data_identity["train_rows"]):
        raise RuntimeError("training record count differs from frozen data audit")
    if len(val_ds) != int(data_identity["val_rows"]):
        raise RuntimeError("validation record count differs from frozen data audit")
    actual_order_sha = record_order_sha256(train_ds.rows)
    if actual_order_sha != data_identity["train_order_sha256"]:
        raise RuntimeError("training record order differs from frozen order ledger")
    total_updates = optimizer_update_count(len(train_ds), int(args.grad_accum))
    warmup_steps = warmup_step_count(total_updates)
    geometry = report.get("optimizer_geometry") or {}
    if not args.allow_nonfrozen_fixture:
        if int(args.max_length) != 512:
            raise RuntimeError("frozen chemistry-first max_length must be 512")
        if len(val_ds) != 9047:
            raise RuntimeError("frozen chemistry-first validation must contain all 9,047 anchors")
        if (
            int(geometry.get("record_count", -1)) != len(train_ds)
            or int(geometry.get("total_updates", -1)) != total_updates
            or int(geometry.get("warmup_steps", -1)) != warmup_steps
        ):
            raise RuntimeError("optimizer geometry differs from the frozen data census")

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate(batch, int(tokenizer.pad_token_id)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate(batch, int(tokenizer.pad_token_id)),
    )

    config = load_llama3_compatible_config(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    ensure_peft_cache_compat()
    from peft import PeftModel

    disable_peft_bnb_autodetect()
    model = PeftModel.from_pretrained(
        base,
        args.p0_adapter_path,
        adapter_name="candidate",
        is_trainable=True,
    )
    model.load_adapter(
        args.p0_adapter_path,
        adapter_name="reference",
        # PEFT 0.16 loads a frozen adapter directly in the BF16 base-model
        # dtype, while the trainable candidate is autocast to FP32.  Loading
        # both through the trainable path preserves byte-identical P0 weights;
        # the reference parameters are frozen immediately below, before any
        # forward or optimizer construction.
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    for name, parameter in model.named_parameters():
        if ".reference." in name:
            parameter.requires_grad_(False)
    model.set_adapter("candidate")
    model.to(device)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    identity = dual_adapter_identity_report(model)
    (args.output_dir / "dual_adapter_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not identity["passed"]:
        raise RuntimeError("candidate/reference protected P0 identity failed")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    config_payload = {
        "schema": H1_CHEMISTRY_FIRST_SFT_SCHEMA,
        "candidate": str(args.candidate),
        "model_path": str(args.model_path),
        "p0_adapter_path": str(args.p0_adapter_path),
        "data_dir": str(args.data_dir),
        "data_audit_report": str(args.data_audit_report),
        "data_audit_summary_sha256": report["summary_sha256"],
        "train_sha256": sha256_file(args.data_dir / "train.jsonl"),
        "val_sha256": sha256_file(args.data_dir / "val.jsonl"),
        "record_multiset_sha256": report["order_pair_audit"]["record_multiset_sha256"],
        "record_order_sha256": actual_order_sha,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "batch_size": 1,
        "gradient_accumulation": int(args.grad_accum),
        "total_updates": total_updates,
        "warmup_steps": warmup_steps,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "shuffle": False,
        "drop_last": False,
        "one_epoch_exact": True,
        "fixed_endpoint_only": True,
        "generated_metric_checkpoint_selection": False,
        "device": str(device),
        "bf16": device.type == "cuda",
        "dual_adapter_identity": identity,
    }

    if args.preflight_only:
        raw_batch = next(iter(train_loader))
        batch = tensor_batch_to_device(raw_batch, device)
        smoke_loss = candidate_sft_loss(model, batch)
        if not torch.isfinite(smoke_loss).item():
            raise RuntimeError("preflight forward loss is non-finite")
        smoke_loss.backward()
        gradient_tensor_count = sum(
            parameter.grad is not None for parameter in trainable
        )
        finite_gradients = all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in trainable
            if parameter.grad is not None
        )
        model.zero_grad(set_to_none=True)
        if gradient_tensor_count == 0 or not finite_gradients:
            raise RuntimeError("preflight gradients are absent or non-finite")
        config_payload["preflight_smoke"] = {
            "loss": float(smoke_loss.item()),
            "gradient_tensor_count": gradient_tensor_count,
            "finite_gradients": finite_gradients,
            "optimizer_step_performed": False,
        }
        (args.output_dir / "train_config.json").write_text(
            json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"status": "preflight_pass", **config_payload},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    reference_anchor = validation_anchor_nll(
        model,
        val_loader,
        device,
        adapter_name="reference",
    )
    config_payload["reference_validation_anchor_nll"] = reference_anchor
    (args.output_dir / "train_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    start = time.time()
    history_path = args.output_dir / "history.jsonl"
    consumed_ids: list[str] = []
    global_step = 0
    running_loss = 0.0
    running_microbatches = 0
    model.zero_grad(set_to_none=True)
    progress = tqdm(total=total_updates, desc=f"H1 chemistry-first {args.candidate}")
    for microbatch_index, raw_batch in enumerate(train_loader):
        consumed_ids.extend(str(value) for value in raw_batch["record_id"])
        batch = tensor_batch_to_device(raw_batch, device)
        loss = candidate_sft_loss(model, batch)
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"non-finite training loss at microbatch {microbatch_index + 1}"
            )
        divisor = accumulation_group_size(
            microbatch_index,
            total_microbatches=len(train_ds),
            grad_accum=int(args.grad_accum),
        )
        (loss / divisor).backward()
        running_loss += float(loss.item())
        running_microbatches += 1
        if not accumulation_is_update_boundary(
            microbatch_index,
            total_microbatches=len(train_ds),
            grad_accum=int(args.grad_accum),
        ):
            continue
        if not all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in trainable
            if parameter.grad is not None
        ):
            raise RuntimeError(f"non-finite gradient before update {global_step + 1}")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        progress.update(1)
        if global_step == 1 or global_step % int(args.logging_steps) == 0 or global_step == total_updates:
            event = {
                "event": "train",
                "step": global_step,
                "loss": running_loss / max(1, running_microbatches),
                "microbatches": running_microbatches,
                "lr": float(scheduler.get_last_lr()[0]),
                "elapsed_sec": time.time() - start,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event), flush=True)
            running_loss = 0.0
            running_microbatches = 0
    progress.close()

    if global_step != total_updates or len(consumed_ids) != len(train_ds):
        raise RuntimeError(
            f"one-epoch consumption mismatch updates={global_step}/{total_updates} "
            f"records={len(consumed_ids)}/{len(train_ds)}"
        )
    if len(consumed_ids) != len(set(consumed_ids)):
        raise RuntimeError("training ledger repeated a record")
    consumed_order_sha = record_order_sha256(
        [{"record_id": value} for value in consumed_ids]
    )
    if consumed_order_sha != actual_order_sha:
        raise RuntimeError("training consumed records out of frozen order")

    endpoint_name = f"checkpoint-{total_updates:06d}"
    endpoint_dir = args.output_dir / endpoint_name
    save_candidate_adapter(model, tokenizer, endpoint_dir)
    adapter_location = json.loads(
        (endpoint_dir / "adapter_location.json").read_text(encoding="utf-8")
    )
    adapter_relative = Path(str(adapter_location["adapter_directory_relative"]))
    if adapter_relative.is_absolute() or ".." in adapter_relative.parts:
        raise RuntimeError("saved candidate adapter location escaped fixed endpoint")
    adapter_dir = endpoint_dir / adapter_relative
    adapter_weight = adapter_dir / str(adapter_location["weight_file"])
    if not adapter_weight.is_file():
        raise RuntimeError("saved candidate adapter weight is missing")
    fixed_adapter_relative = adapter_dir.relative_to(args.output_dir).as_posix()
    fixed_adapter_weight_sha256 = sha256_file(adapter_weight)
    candidate_anchor = validation_anchor_nll(
        model,
        val_loader,
        device,
        adapter_name="candidate",
    )
    anchor_gate = validation_anchor_nll_gate(
        reference_anchor["nll"],
        candidate_anchor["nll"],
        maximum_relative_degradation=0.01,
    )
    terminal = {
        "schema": H1_CHEMISTRY_FIRST_SFT_SCHEMA,
        "status": "pass" if anchor_gate["passed"] else "anchor_nll_noninferiority_failure",
        "candidate": str(args.candidate),
        "global_step": global_step,
        "microbatch_count": len(consumed_ids),
        "consumed_record_order_sha256": consumed_order_sha,
        "fixed_endpoint": endpoint_name,
        "fixed_adapter_relative": fixed_adapter_relative,
        "fixed_adapter_weight_file": adapter_weight.name,
        "fixed_adapter_weight_sha256": fixed_adapter_weight_sha256,
        "reference_validation_anchor_nll": reference_anchor,
        "candidate_validation_anchor_nll": candidate_anchor,
        "conditional_structural_anchor_nll_gate": anchor_gate,
        "generated_metric_checkpoint_selection": False,
        "intermediate_checkpoint_count": 0,
        "elapsed_sec": time.time() - start,
    }
    (args.output_dir / "terminal_report.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker = "_SUCCESS" if anchor_gate["passed"] else "_FAILED"
    (args.output_dir / marker).write_text(
        json.dumps(terminal, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(terminal, ensure_ascii=False, indent=2), flush=True)
    if not anchor_gate["passed"]:
        raise RuntimeError("conditional structural-anchor NLL degraded by more than 1%")


if __name__ == "__main__":
    main()
