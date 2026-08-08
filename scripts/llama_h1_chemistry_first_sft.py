#!/usr/bin/env python3
"""Train fixed-endpoint SFT-v2 or SFT-v2-C from the protected P0 adapter."""

from __future__ import annotations

import argparse
import json
import math
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
from crystal_dlm.peft_adapter_identity import (  # noqa: E402
    PROTECTED_P0_ADAPTER_CONFIG_SHA256,
    PROTECTED_P0_ADAPTER_WEIGHT_SHA256,
    adapter_source_identity_report,
    adapter_value_sha256,
    copy_adapter_state_exact,
)
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
def fixed_adapter_logits(
    model,
    batch: Mapping[str, Any],
    *,
    adapter_name: str,
) -> torch.Tensor:
    """Return one deterministic smoke logit tensor with every adapter frozen."""

    model.set_adapter(adapter_name)
    for name, parameter in model.named_parameters():
        if ".candidate." in name or ".reference." in name:
            parameter.requires_grad_(False)
    model.eval()
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).logits.detach()


def restore_candidate_training_state(model, *, was_training: bool = True) -> None:
    model.set_adapter("candidate")
    for name, parameter in model.named_parameters():
        if ".candidate." in name:
            parameter.requires_grad_(True)
        elif ".reference." in name:
            parameter.requires_grad_(False)
    model.train(was_training)


def optimizer_parameter_identity_report(model, optimizer) -> dict[str, Any]:
    named = list(model.named_parameters())
    candidate = {id(parameter): name for name, parameter in named if ".candidate." in name}
    reference = {id(parameter): name for name, parameter in named if ".reference." in name}
    trainable = {id(parameter): name for name, parameter in named if parameter.requires_grad}
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    optimizer_id_set = set(optimizer_ids)
    passed = bool(
        candidate
        and set(candidate) == set(trainable) == optimizer_id_set
        and len(optimizer_ids) == len(optimizer_id_set)
        and not (optimizer_id_set & set(reference))
        and all(not parameter.requires_grad for name, parameter in named if ".reference." in name)
    )
    return {
        "candidate_parameter_count": len(candidate),
        "reference_parameter_count": len(reference),
        "trainable_parameter_count": len(trainable),
        "optimizer_parameter_count": len(optimizer_ids),
        "optimizer_unique_parameter_count": len(optimizer_id_set),
        "missing_candidate_in_optimizer": sorted(
            candidate[parameter_id] for parameter_id in set(candidate) - optimizer_id_set
        ),
        "unexpected_optimizer_parameters": sorted(
            trainable.get(parameter_id, f"unknown_parameter_id:{parameter_id}")
            for parameter_id in optimizer_id_set - set(candidate)
        ),
        "reference_in_optimizer": sorted(
            reference[parameter_id] for parameter_id in optimizer_id_set & set(reference)
        ),
        "passed": passed,
    }


def candidate_gradient_health_report(model, trainable) -> dict[str, Any]:
    """Fail closed on absent, non-finite, or misrouted candidate gradients."""

    trainable_ids = {id(parameter) for parameter in trainable}
    candidate = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if ".candidate." in name
    }
    gradients = [
        (name, parameter, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    candidate_gradients = [
        (name, gradient)
        for name, parameter, gradient in gradients
        if id(parameter) in candidate
    ]
    unexpected = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and id(parameter) not in candidate
    )
    missing_from_trainable = sorted(
        name for parameter_id, name in candidate.items() if parameter_id not in trainable_ids
    )
    finite = bool(candidate_gradients) and all(
        bool(torch.isfinite(gradient).all().item())
        for _, gradient in candidate_gradients
    )
    return {
        "candidate_parameter_count": len(candidate),
        "candidate_gradient_tensor_count": len(candidate_gradients),
        "all_gradient_tensor_count": len(gradients),
        "finite_candidate_gradients": finite,
        "unexpected_gradient_names": unexpected,
        "candidate_missing_from_trainable": missing_from_trainable,
        "passed_before_clip": bool(
            candidate
            and candidate_gradients
            and finite
            and not unexpected
            and not missing_from_trainable
        ),
    }


def optimizer_state_health_report(
    model,
    optimizer,
    *,
    expected_step: int | None,
) -> dict[str, Any]:
    """Summarize AdamW state without treating a tensor container as one step."""

    candidate_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if ".candidate." in name
    ]
    step_values: list[float] = []
    moment_tensor_count = 0
    nonzero_moment_tensor_count = 0
    finite_moments = True
    state_parameter_count = 0
    for _, parameter in candidate_parameters:
        state = optimizer.state.get(parameter) or {}
        if not state:
            continue
        state_parameter_count += 1
        step = state.get("step")
        if step is not None:
            step_values.append(
                float(step.detach().item()) if torch.is_tensor(step) else float(step)
            )
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state.get(key)
            if moment is None:
                continue
            moment_tensor_count += 1
            finite_moments = bool(
                finite_moments and torch.isfinite(moment).all().item()
            )
            if bool(torch.count_nonzero(moment).item()):
                nonzero_moment_tensor_count += 1
    expected_step_match = bool(
        expected_step is not None
        and state_parameter_count > 0
        and len(step_values) == state_parameter_count
        and all(value == float(expected_step) for value in step_values)
    )
    return {
        "candidate_parameter_count": len(candidate_parameters),
        "state_parameter_count": state_parameter_count,
        "step_values_min": min(step_values) if step_values else None,
        "step_values_max": max(step_values) if step_values else None,
        "expected_step": expected_step,
        "expected_step_match": expected_step_match,
        "moment_tensor_count": moment_tensor_count,
        "nonzero_moment_tensor_count": nonzero_moment_tensor_count,
        "finite_moments": finite_moments,
        "passed": bool(
            expected_step_match
            and moment_tensor_count == 2 * state_parameter_count
            and nonzero_moment_tensor_count > 0
            and finite_moments
        ),
    }


def active_adapter_value(model) -> str:
    value = getattr(model, "active_adapter", None)
    if callable(value):
        value = value()
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def audited_warmup_optimizer_step(
    model,
    trainable,
    optimizer,
    scheduler,
    *,
    step_number: int,
    base_lr: float,
    warmup_steps: int,
    reference_sha_initial: str,
    candidate_sha_initial: str,
    consumed_record_ids: list[str],
    expected_record_order_sha256: str,
) -> dict[str, Any]:
    """Execute one of the first two updates and attest its exact warmup semantics."""

    if step_number not in {1, 2}:
        raise ValueError("warmup optimizer audit is registered only for steps 1 and 2")
    if warmup_steps <= 1:
        raise ValueError("warmup optimizer audit requires at least two warmup steps")
    gradient = candidate_gradient_health_report(model, trainable)
    lr_values = [float(group["lr"]) for group in optimizer.param_groups]
    lr_used = lr_values[0] if lr_values else math.nan
    expected_lr_used = 0.0 if step_number == 1 else float(base_lr) / warmup_steps
    expected_lr_next = float(base_lr) * step_number / warmup_steps
    scheduler_epoch_before = int(scheduler.last_epoch)
    candidate_sha_before = adapter_value_sha256(model, "candidate")
    reference_sha_before = adapter_value_sha256(model, "reference")
    state_before = optimizer_state_health_report(
        model,
        optimizer,
        expected_step=None if step_number == 1 else step_number - 1,
    )
    record_order_sha = record_order_sha256(
        [{"record_id": value} for value in consumed_record_ids]
    )
    failures = []
    if not gradient["passed_before_clip"]:
        failures.append("candidate_gradient_health")
    if not lr_values or not all(
        math.isclose(value, expected_lr_used, rel_tol=1e-12, abs_tol=1e-18)
        for value in lr_values
    ):
        failures.append("lr_used")
    if scheduler_epoch_before != step_number - 1:
        failures.append("scheduler_epoch_before")
    if active_adapter_value(model) != "candidate":
        failures.append("active_adapter")
    if reference_sha_before != reference_sha_initial:
        failures.append("reference_before")
    if candidate_sha_before != candidate_sha_initial:
        failures.append("candidate_before")
    if record_order_sha != expected_record_order_sha256:
        failures.append("record_order")
    if step_number == 1:
        if state_before["state_parameter_count"] != 0:
            failures.append("optimizer_state_before")
    elif not state_before["passed"]:
        failures.append("optimizer_state_before")

    clip_norm = math.nan
    if not failures:
        clip_norm_value = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        clip_norm = float(clip_norm_value.detach().item())
        if not math.isfinite(clip_norm) or clip_norm <= 0.0:
            failures.append("zero_or_nonfinite_gradient_norm")
    gradient["preclip_l2_norm"] = clip_norm
    gradient["nonzero_after_health_check"] = bool(
        math.isfinite(clip_norm) and clip_norm > 0.0
    )

    optimizer_step_performed = False
    if not failures:
        optimizer.step()
        scheduler.step()
        optimizer_step_performed = True
    lr_next_values = [float(group["lr"]) for group in optimizer.param_groups]
    lr_next = lr_next_values[0] if lr_next_values else math.nan
    scheduler_epoch_after = int(scheduler.last_epoch)
    state_after = optimizer_state_health_report(
        model,
        optimizer,
        expected_step=step_number,
    )
    candidate_sha_after = adapter_value_sha256(model, "candidate")
    reference_sha_after = adapter_value_sha256(model, "reference")
    optimizer.zero_grad(set_to_none=True)

    if optimizer_step_performed:
        if not all(
            math.isclose(value, expected_lr_next, rel_tol=1e-12, abs_tol=1e-18)
            for value in lr_next_values
        ):
            failures.append("lr_next")
        if scheduler_epoch_after != step_number:
            failures.append("scheduler_epoch_after")
        if not state_after["passed"]:
            failures.append("optimizer_state_after")
        if reference_sha_after != reference_sha_initial:
            failures.append("reference_after")
        candidate_changed = candidate_sha_after != candidate_sha_before
        if candidate_changed != (step_number == 2):
            failures.append("candidate_change_semantics")
    else:
        candidate_changed = False

    return {
        "schema": "h1_chemistry_first_warmup_optimizer_step_audit_v1",
        "step": step_number,
        "semantic_role": (
            "scheduled_zero_lr_state_update"
            if step_number == 1
            else "first_positive_lr_parameter_update"
        ),
        "active_adapter": active_adapter_value(model),
        "consumed_record_count": len(consumed_record_ids),
        "consumed_record_order_sha256": record_order_sha,
        "expected_record_order_sha256": expected_record_order_sha256,
        "gradient": gradient,
        "clip_grad_norm_calls": 1 if math.isfinite(clip_norm) else 0,
        "lr_used": lr_used,
        "lr_used_values": lr_values,
        "expected_lr_used": expected_lr_used,
        "lr_next": lr_next,
        "lr_next_values": lr_next_values,
        "expected_lr_next": expected_lr_next,
        "scheduler_last_epoch_before": scheduler_epoch_before,
        "scheduler_last_epoch_after": scheduler_epoch_after,
        "optimizer_state_before": state_before,
        "optimizer_state_after": state_after,
        "reference_sha_initial": reference_sha_initial,
        "reference_sha_before": reference_sha_before,
        "reference_sha_after": reference_sha_after,
        "candidate_sha_initial": candidate_sha_initial,
        "candidate_sha_before": candidate_sha_before,
        "candidate_sha_after": candidate_sha_after,
        "candidate_changed": candidate_changed,
        "optimizer_step_performed": optimizer_step_performed,
        "failures": failures,
        "passed": not failures,
    }


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
    parser.add_argument("--optimizer-smoke-updates", type=int, default=0)
    parser.add_argument("--allow-nonfrozen-fixture", action="store_true")
    parser.add_argument(
        "--expected-p0-adapter-weight-sha256",
        default=PROTECTED_P0_ADAPTER_WEIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-p0-adapter-config-sha256",
        default=PROTECTED_P0_ADAPTER_CONFIG_SHA256,
    )
    args = parser.parse_args()

    if args.preflight_only and int(args.optimizer_smoke_updates) != 0:
        raise RuntimeError("preflight-only and optimizer smoke are mutually exclusive")
    if int(args.optimizer_smoke_updates) not in {0, 2}:
        raise RuntimeError("optimizer smoke is frozen at exactly two real updates")

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
        autocast_adapter_dtype=True,
    )
    candidate_source_before_reference = adapter_source_identity_report(
        model,
        "candidate",
        Path(args.p0_adapter_path),
        expected_weight_sha256=str(args.expected_p0_adapter_weight_sha256),
        expected_config_sha256=str(args.expected_p0_adapter_config_sha256),
    )
    (args.output_dir / "candidate_source_identity_before_reference.json").write_text(
        json.dumps(candidate_source_before_reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not candidate_source_before_reference["passed"]:
        raise RuntimeError("candidate adapter is not byte-identical to protected P0")
    model.load_adapter(
        args.p0_adapter_path,
        adapter_name="reference",
        # PEFT 0.16 writes this second adapter while its destination follows
        # the BF16 base-layer dtype and only upcasts after checkpoint loading.
        # The exact source-attested candidate is copied into independent FP32
        # reference storage immediately below, before forward or optimizer.
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    pre_copy_identity = dual_adapter_identity_report(model)
    (args.output_dir / "dual_adapter_identity_before_copy.json").write_text(
        json.dumps(pre_copy_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copy_report = copy_adapter_state_exact(
        model,
        source_adapter="candidate",
        target_adapter="reference",
        expected_dtype=torch.float32,
    )
    model.set_adapter("candidate")
    model.to(device)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    identity = dual_adapter_identity_report(model)
    candidate_source_after_setup = adapter_source_identity_report(
        model,
        "candidate",
        Path(args.p0_adapter_path),
        expected_weight_sha256=str(args.expected_p0_adapter_weight_sha256),
        expected_config_sha256=str(args.expected_p0_adapter_config_sha256),
    )
    reference_source_after_setup = adapter_source_identity_report(
        model,
        "reference",
        Path(args.p0_adapter_path),
        expected_weight_sha256=str(args.expected_p0_adapter_weight_sha256),
        expected_config_sha256=str(args.expected_p0_adapter_config_sha256),
    )
    protected_identity = {
        "schema": "h1_protected_p0_candidate_reference_identity_v2",
        "candidate_source_before_reference": candidate_source_before_reference,
        "pre_copy_pair_diagnostic": pre_copy_identity,
        "copy_report": copy_report,
        "post_setup_pair_identity": identity,
        "candidate_source_after_setup": candidate_source_after_setup,
        "reference_source_after_setup": reference_source_after_setup,
        "passed": bool(
            candidate_source_before_reference["passed"]
            and copy_report["passed"]
            and identity["passed"]
            and candidate_source_after_setup["passed"]
            and reference_source_after_setup["passed"]
        ),
    }
    (args.output_dir / "protected_p0_triplet_identity.json").write_text(
        json.dumps(protected_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dual_adapter_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not protected_identity["passed"]:
        raise RuntimeError("candidate/reference protected P0 identity failed")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    optimizer_identity = optimizer_parameter_identity_report(model, optimizer)
    if not optimizer_identity["passed"]:
        raise RuntimeError("optimizer parameter identity is not candidate-only")
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
        "optimizer_smoke_updates": int(args.optimizer_smoke_updates),
        "device": str(device),
        "bf16": device.type == "cuda",
        "dual_adapter_identity": identity,
        "protected_p0_triplet_identity": protected_identity,
        "optimizer_parameter_identity": optimizer_identity,
    }

    if args.preflight_only:
        reference_sha_before = adapter_value_sha256(model, "reference")
        candidate_sha_before = adapter_value_sha256(model, "candidate")
        val_batch = tensor_batch_to_device(next(iter(val_loader)), device)
        reference_logits = fixed_adapter_logits(
            model,
            val_batch,
            adapter_name="reference",
        )
        candidate_logits = fixed_adapter_logits(
            model,
            val_batch,
            adapter_name="candidate",
        )
        logits_identical = bool(torch.equal(candidate_logits, reference_logits))
        logits_finite = bool(
            torch.isfinite(candidate_logits).all().item()
            and torch.isfinite(reference_logits).all().item()
        )
        del candidate_logits, reference_logits
        restore_candidate_training_state(model)
        if not logits_identical or not logits_finite:
            raise RuntimeError("candidate/reference fixed-record logits differ or are non-finite")
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
        unexpected_gradient_names = sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and ".candidate." not in name
        )
        reference_sha_after_backward = adapter_value_sha256(model, "reference")
        candidate_sha_after_backward = adapter_value_sha256(model, "candidate")
        model.zero_grad(set_to_none=True)
        if (
            gradient_tensor_count == 0
            or not finite_gradients
            or unexpected_gradient_names
            or reference_sha_after_backward != reference_sha_before
            or candidate_sha_after_backward != candidate_sha_before
        ):
            raise RuntimeError("preflight gradients are absent or non-finite")
        config_payload["preflight_smoke"] = {
            "fixed_validation_record_candidate_reference_logits_identical": logits_identical,
            "fixed_validation_record_logits_finite": logits_finite,
            "loss": float(smoke_loss.item()),
            "gradient_tensor_count": gradient_tensor_count,
            "finite_gradients": finite_gradients,
            "unexpected_gradient_names": unexpected_gradient_names,
            "reference_sha_before": reference_sha_before,
            "reference_sha_after_backward": reference_sha_after_backward,
            "candidate_sha_before": candidate_sha_before,
            "candidate_sha_after_backward": candidate_sha_after_backward,
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

    reference_sha_initial = adapter_value_sha256(model, "reference")
    candidate_sha_initial = adapter_value_sha256(model, "candidate")
    reference_anchor = validation_anchor_nll(
        model,
        val_loader,
        device,
        adapter_name="reference",
    )
    reference_sha_after_anchor = adapter_value_sha256(model, "reference")
    candidate_sha_after_reference_anchor = adapter_value_sha256(model, "candidate")
    if (
        reference_sha_after_anchor != reference_sha_initial
        or candidate_sha_after_reference_anchor != candidate_sha_initial
    ):
        raise RuntimeError("reference-anchor evaluation mutated an adapter")
    config_payload["reference_validation_anchor_nll"] = reference_anchor
    config_payload["reference_anchor_identity"] = {
        "reference_sha_before": reference_sha_initial,
        "reference_sha_after": reference_sha_after_anchor,
        "candidate_sha_before": candidate_sha_initial,
        "candidate_sha_after": candidate_sha_after_reference_anchor,
        "passed": True,
    }
    (args.output_dir / "train_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    start = time.time()
    history_path = args.output_dir / "history.jsonl"
    consumed_ids: list[str] = []
    global_step = 0
    first_optimizer_step_identity: dict[str, Any] | None = None
    first_positive_optimizer_step_identity: dict[str, Any] | None = None
    optimizer_step_audits: list[dict[str, Any]] = []
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
        step_number = global_step + 1
        if step_number <= 2:
            expected_prefix_sha = record_order_sha256(
                train_ds.rows[: len(consumed_ids)]
            )
            audit = audited_warmup_optimizer_step(
                model,
                trainable,
                optimizer,
                scheduler,
                step_number=step_number,
                base_lr=float(args.lr),
                warmup_steps=warmup_steps,
                reference_sha_initial=reference_sha_initial,
                candidate_sha_initial=candidate_sha_initial,
                consumed_record_ids=consumed_ids,
                expected_record_order_sha256=expected_prefix_sha,
            )
            optimizer_step_audits.append(audit)
            (args.output_dir / "optimizer_step_audits.json").write_text(
                json.dumps(optimizer_step_audits, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if step_number == 1:
                first_optimizer_step_identity = audit
            else:
                first_positive_optimizer_step_identity = audit
            if not audit["passed"]:
                raise RuntimeError(
                    f"warmup optimizer step {step_number} audit failed: "
                    f"{audit['failures']}"
                )
            lr_used = float(audit["lr_used"])
            lr_next = float(audit["lr_next"])
        else:
            gradient = candidate_gradient_health_report(model, trainable)
            if not gradient["passed_before_clip"]:
                raise RuntimeError(
                    f"gradient routing/finite audit failed before update {step_number}"
                )
            clip_norm_value = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            clip_norm = float(clip_norm_value.detach().item())
            if not math.isfinite(clip_norm) or clip_norm <= 0.0:
                raise RuntimeError(
                    f"zero or non-finite gradient norm before update {step_number}"
                )
            lr_used = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            lr_next = float(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad(set_to_none=True)
        global_step = step_number
        progress.update(1)
        if global_step <= 2 or global_step % int(args.logging_steps) == 0 or global_step == total_updates:
            event = {
                "event": "train",
                "step": global_step,
                "loss": running_loss / max(1, running_microbatches),
                "microbatches": running_microbatches,
                "lr": lr_next,
                "lr_used": lr_used,
                "lr_next": lr_next,
                "elapsed_sec": time.time() - start,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event), flush=True)
            running_loss = 0.0
            running_microbatches = 0
        if (
            int(args.optimizer_smoke_updates) > 0
            and global_step == int(args.optimizer_smoke_updates)
        ):
            break
    progress.close()

    if int(args.optimizer_smoke_updates) > 0:
        expected_microbatches = int(args.optimizer_smoke_updates) * int(args.grad_accum)
        consumed_order_sha = record_order_sha256(
            [{"record_id": value} for value in consumed_ids]
        )
        expected_order_sha = record_order_sha256(
            train_ds.rows[:expected_microbatches]
        )
        reference_sha_smoke = adapter_value_sha256(model, "reference")
        candidate_sha_smoke = adapter_value_sha256(model, "candidate")
        smoke_failures = []
        if global_step != 2:
            smoke_failures.append("optimizer_update_count")
        if len(consumed_ids) != expected_microbatches:
            smoke_failures.append("microbatch_count")
        if consumed_order_sha != expected_order_sha:
            smoke_failures.append("record_order")
        if len(optimizer_step_audits) != 2 or not all(
            report["passed"] for report in optimizer_step_audits
        ):
            smoke_failures.append("optimizer_step_audits")
        if reference_sha_smoke != reference_sha_initial:
            smoke_failures.append("reference_identity")
        if candidate_sha_smoke == candidate_sha_initial:
            smoke_failures.append("candidate_identity")
        smoke_report = {
            "schema": "h1_chemistry_first_two_step_optimizer_smoke_v1",
            "status": "pass" if not smoke_failures else "fail",
            "candidate": str(args.candidate),
            "full_training_total_updates": total_updates,
            "full_training_warmup_steps": warmup_steps,
            "optimizer_updates": global_step,
            "microbatch_count": len(consumed_ids),
            "consumed_record_order_sha256": consumed_order_sha,
            "expected_record_order_sha256": expected_order_sha,
            "optimizer_step_audits": optimizer_step_audits,
            "reference_adapter_sha256_initial": reference_sha_initial,
            "reference_adapter_sha256_terminal": reference_sha_smoke,
            "candidate_adapter_sha256_initial": candidate_sha_initial,
            "candidate_adapter_sha256_terminal": candidate_sha_smoke,
            "scientific_checkpoint_saved": False,
            "generation": False,
            "smact4_executed_on_a800": False,
            "failures": smoke_failures,
            "elapsed_sec": time.time() - start,
        }
        (args.output_dir / "optimizer_smoke_report.json").write_text(
            json.dumps(smoke_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        marker = "_SUCCESS" if not smoke_failures else "_FAILED"
        (args.output_dir / marker).write_text(
            json.dumps(smoke_report, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(smoke_report, ensure_ascii=False, indent=2), flush=True)
        if smoke_failures:
            raise RuntimeError(f"two-step optimizer smoke failed: {smoke_failures}")
        return

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
    reference_sha_terminal = adapter_value_sha256(model, "reference")
    candidate_sha_terminal = adapter_value_sha256(model, "candidate")
    if reference_sha_terminal != reference_sha_initial:
        raise RuntimeError("frozen reference adapter changed during training")
    if candidate_sha_terminal == candidate_sha_initial:
        raise RuntimeError("candidate adapter did not change during training")
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
        "optimizer_parameter_identity": optimizer_identity,
        "first_optimizer_step_identity": first_optimizer_step_identity,
        "first_positive_optimizer_step_identity": first_positive_optimizer_step_identity,
        "optimizer_step_audits": optimizer_step_audits,
        "reference_adapter_sha256_initial": reference_sha_initial,
        "reference_adapter_sha256_terminal": reference_sha_terminal,
        "candidate_adapter_sha256_initial": candidate_sha_initial,
        "candidate_adapter_sha256_terminal": candidate_sha_terminal,
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
