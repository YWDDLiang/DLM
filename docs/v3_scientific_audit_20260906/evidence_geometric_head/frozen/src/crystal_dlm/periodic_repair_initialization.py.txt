"""Fresh periodic DLM initialization from the original, unadapted LLaDA.

The original vocabulary rows stay frozen. Every newly registered crystal token
has trainable input and output row residuals, in addition to the typed numeric
adapter. No legacy crystal checkpoint or generated training data is read here.
New checkpoints have a separate marker and must use this module's loader.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from crystal_dlm.fixed_slot import MASK_TOKEN_ID, build_special_tokens
from crystal_dlm.llada_resize import ensure_llada_vocab_size
from crystal_dlm.periodic_repair_model import (
    FP32Module, PeriodicRepairConfig, PeriodicRepairDLM,
)
from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
from crystal_dlm.r5_dynamic_length import validate_dynamic_tokenizer_contract
from crystal_dlm.transformers_compat import (
    ensure_create_bidirectional_mask, ensure_llada2_rope_parameters,
)


INITIALIZATION_SCHEMA = "raw_pretrained_periodic_dlm_v1"
INITIALIZATION_MARKER = "raw_periodic_initialization.json"
# The fresh canonical training recipe in llada_sft.py and slurm/153.
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "ff_proj", "up_proj")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


class NewCrystalTokenRows(FP32Module):
    """Full-rank residuals for new token rows; pretrained rows have no residual.

Zero initialization reproduces the mean-expanded base exactly. The parameters
remain FP32, while additions are cast to the receiving base tensor's dtype.
Input residuals cover new tokens in both the prompt and the fixed body canvas.
"""

    def __init__(self, vocabulary_size: int, hidden_size: int, token_ids):
        super().__init__()
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        if (ids.ndim != 1 or ids.numel() == 0 or ids.unique().numel() != ids.numel()
                or bool(((ids < 0) | (ids >= vocabulary_size)).any())):
            raise ValueError("new token rows must be unique, nonempty vocabulary IDs")
        lookup = torch.full((vocabulary_size,), -1, dtype=torch.long)
        lookup[ids] = torch.arange(ids.numel(), dtype=torch.long)
        self.register_buffer("token_ids", ids)
        self.register_buffer("token_to_row", lookup)
        self.input_delta = nn.Parameter(torch.zeros(ids.numel(), hidden_size, dtype=torch.float32))
        self.output_delta = nn.Parameter(torch.zeros(ids.numel(), hidden_size, dtype=torch.float32))

    def input_increments(self, input_ids):
        rows = self.token_to_row[input_ids]
        increments = F.embedding(rows.clamp_min(0), self.input_delta)
        return torch.where((rows >= 0)[..., None], increments, 0.)

    def add_output_increments(self, logits, hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            increments = F.linear(hidden.float(), self.output_delta)
        return logits.index_add(-1, self.token_ids, increments.to(logits.dtype))


class FreshPeriodicRepairDLM(PeriodicRepairDLM):
    """The same discrete periodic DLM with learnable freshly added token rows."""

    def __init__(self, base_model, tokenizer, state_config, repair_config,
                 initialization: dict[str, Any]):
        if initialization.get("schema") != INITIALIZATION_SCHEMA:
            raise ValueError("fresh model requires its raw-pretrained initialization record")
        super().__init__(base_model, tokenizer, state_config, repair_config)
        self.raw_initialization = dict(initialization)
        self.new_token_rows = NewCrystalTokenRows(
            self.get_input_embeddings().weight.shape[0], state_config.hidden_size,
            initialization["new_crystal_token_ids"],
        )

    def state_embeddings(self, input_ids, geometry_context):
        embeddings = super().state_embeddings(input_ids, geometry_context)
        return embeddings + self.new_token_rows.input_increments(input_ids).to(embeddings.dtype)

    def forward(self, input_ids, attention_mask=None, *, geometry_context=None, **kwargs):
        requested_hidden = kwargs.pop("output_hidden_states", False)
        output = super().forward(
            input_ids, attention_mask=attention_mask, geometry_context=geometry_context,
            output_hidden_states=True, **kwargs,
        )
        output.logits = self.new_token_rows.add_output_increments(
            output.logits, output.hidden_states[-1],
        )
        if not requested_hidden:
            output.hidden_states = None
        return output

    def repair_modules(self):
        return {**super().repair_modules(), "new_token_rows": self.new_token_rows}

    def save_pretrained(self, output_dir, **kwargs):
        # The frozen expanded means are reconstructed from the raw base. PEFT
        # must not silently export two complete vocabulary tables instead.
        if kwargs.get("save_embedding_layers", False) not in (False, "false"):
            raise ValueError("fresh checkpoints save new row residuals, not full base tables")
        kwargs["save_embedding_layers"] = False
        super().save_pretrained(output_dir, **kwargs)
        (Path(output_dir) / INITIALIZATION_MARKER).write_text(
            json.dumps(self.raw_initialization, indent=2) + "\n", encoding="utf-8",
        )


def extend_raw_crystal_vocabulary(base_model, tokenizer) -> dict[str, Any]:
    """Register the existing ABI without borrowing a crystal-trained tokenizer.

Token IDs, rather than newly allocated matrix rows, define what is new: a raw
model can have unused padded embedding rows that new tokenizer IDs first occupy.
"""
    prior = {str(token): int(index) for token, index in tokenizer.get_vocab().items()}
    tokens = build_special_tokens()
    if any(token in prior for token in tokens):
        raise ValueError("original tokenizer already contains crystal tokens; refusing an adapted source")
    size = len(tokenizer)
    if set(prior.values()) != set(range(size)):
        raise ValueError("raw tokenizer IDs must cover their registered vocabulary contiguously")
    input_layer = base_model.get_input_embeddings()
    output_layer = base_model.get_output_embeddings()
    if not isinstance(input_layer, nn.Embedding) or not isinstance(output_layer, (nn.Linear, nn.Embedding)):
        raise TypeError("fresh row initialization requires ordinary LLaDA embedding/output tables")
    if min(input_layer.weight.shape[0], output_layer.weight.shape[0]) < size:
        raise ValueError("raw pretrained tables do not cover the original tokenizer")
    with torch.no_grad():
        input_mean = input_layer.weight[:size].mean(dim=0, dtype=torch.float32)
        output_mean = output_layer.weight[:size].mean(dim=0, dtype=torch.float32)
        output_bias = getattr(output_layer, "bias", None)
        bias_mean = None if output_bias is None else output_bias[:size].mean(dtype=torch.float32)
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    current = tokenizer.get_vocab()
    if added != len(tokens) or len(tokenizer) != size + len(tokens):
        raise ValueError("crystal vocabulary extension did not add the exact existing schema")
    if any(int(current[token]) != index for token, index in prior.items()):
        raise ValueError("crystal vocabulary extension changed a pretrained token ID")
    ids = [int(current[token]) for token in tokens]
    if set(ids) != set(range(size, len(tokenizer))):
        raise ValueError("new crystal token IDs are not an isolated extension")
    resize_report = ensure_llada_vocab_size(base_model, len(tokenizer))
    input_layer = base_model.get_input_embeddings()
    output_layer = base_model.get_output_embeddings()
    with torch.no_grad():
        input_ids = torch.tensor(ids, dtype=torch.long, device=input_layer.weight.device)
        output_ids = input_ids.to(output_layer.weight.device)
        input_layer.weight.index_copy_(0, input_ids, input_mean.to(input_layer.weight).expand(len(ids), -1))
        output_layer.weight.index_copy_(0, output_ids, output_mean.to(output_layer.weight).expand(len(ids), -1))
        if bias_mean is not None:
            output_layer.bias.index_fill_(0, output_ids, bias_mean.to(output_layer.bias))
    return {
        "source_tokenizer_size": size,
        "source_tokenizer_vocab_sha256": _digest(prior),
        "expanded_tokenizer_size": len(tokenizer),
        "crystal_token_strings_sha256": _digest(tokens),
        "new_crystal_token_ids": ids,
        "new_token_count": len(ids),
        "row_initialization": "pretrained_vocabulary_mean_plus_zero_trainable_residual",
        "resize_report": resize_report,
    }


def set_fresh_repair_trainable(model: FreshPeriodicRepairDLM) -> dict[str, int]:
    """Train fresh LoRA and crystal modules, retaining the registered dropout."""
    if not isinstance(model, FreshPeriodicRepairDLM):
        raise TypeError("fresh trainability applies only to a raw-initialized periodic DLM")
    model.requires_grad_(False)
    lora_parameters = []
    for name, parameter in model.base_model.named_parameters():
        if ".lora_A." in f".{name}" or ".lora_B." in f".{name}":
            parameter.requires_grad_(True)
            lora_parameters.append(parameter)
    if not lora_parameters:
        raise ValueError("the raw backbone has no freshly installed LoRA parameters")
    counts = {"lora": sum(p.numel() for p in lora_parameters)}
    for name, module in {"state_conditioner": model.state_conditioner,
                         **model.repair_modules()}.items():
        module.requires_grad_(True)
        counts[name] = sum(p.numel() for p in module.parameters())
    counts["frozen"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    counts["total_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


def _load_raw_components(model_path, device, *, torch_dtype=None):
    root = Path(model_path).expanduser().resolve(strict=True)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError("model_path must be the local original LLaDA model directory")
    forbidden = ("adapter_config.json", "periodic_state_config.json", "periodic_repair_config.json",
                 INITIALIZATION_MARKER, "TRAIN_FINAL.json")
    if any((root / marker).exists() for marker in forbidden):
        raise ValueError("model_path is an adapted DLM checkpoint, not the original pretrained model")
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    ensure_create_bidirectional_mask()
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("the original tokenizer has neither padding nor EOS")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    if getattr(config, "model_type", None) != "llada":
        raise ValueError("fresh initialization is registered for the original LLaDA backbone")
    ensure_llada2_rope_parameters(config)
    target_device = torch.device(device)
    dtype = torch_dtype or (torch.bfloat16 if target_device.type == "cuda" else torch.float32)
    base = AutoModel.from_pretrained(root, config=config, trust_remote_code=True,
                                    local_files_only=True, torch_dtype=dtype)
    extension = extend_raw_crystal_vocabulary(base, tokenizer)
    advertised_mask = getattr(tokenizer, "mask_token_id", None)
    if advertised_mask is not None and int(advertised_mask) != MASK_TOKEN_ID:
        raise ValueError("the original LLaDA mask ID differs from the retained sampler ABI")
    abi = validate_dynamic_tokenizer_contract(tokenizer, mask_token_id=MASK_TOKEN_ID)
    matches = {
        suffix: [name for name, module in base.named_modules()
                 if name.split(".")[-1] == suffix and isinstance(module, nn.Linear)]
        for suffix in LORA_TARGET_MODULES
    }
    if any(not names for names in matches.values()):
        raise ValueError(f"original LLaDA lacks registered LoRA targets: {matches}")
    report = {
        "schema": INITIALIZATION_SCHEMA, "source_model_path": str(root),
        "source_config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
        "legacy_dlm_checkpoint": None, "pretrained_vocabulary_rows_frozen": True,
        "lora_target_counts": {key: len(value) for key, value in matches.items()},
        "tokenizer_contract": abi, **extension,
    }
    return base, tokenizer, report


def _configs(base, state_config, repair_config):
    hidden = int(base.get_input_embeddings().weight.shape[1])
    state = state_config or PeriodicStateConfig(hidden)
    repair = repair_config or PeriodicRepairConfig(hidden)
    if state.hidden_size != hidden or repair.hidden_size != hidden:
        raise ValueError("periodic modules must match the original backbone hidden size")
    return state, repair


def initialize_fresh_periodic_repair_model(
    model_path, device, *, trainable=True, state_config=None, repair_config=None,
    lora_rank=8, lora_alpha=32, lora_dropout=.05, torch_dtype=None,
):
    """Return ``(model, tokenizer)`` without accepting any checkpoint argument.

    Set the process RNG seed before calling, as with the original SFT loader.
    The serializable provenance is available as ``model.raw_initialization``.
    """
    if lora_rank < 1 or lora_alpha <= 0 or not 0 <= lora_dropout < 1:
        raise ValueError("invalid fresh LoRA hyperparameters")
    base, tokenizer, report = _load_raw_components(model_path, device, torch_dtype=torch_dtype)
    state, repair = _configs(base, state_config, repair_config)
    from peft import LoraConfig, get_peft_model
    lora = {"r": int(lora_rank), "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout), "bias": "none",
            "target_modules": list(LORA_TARGET_MODULES)}
    # modules_to_save is intentionally absent: only our explicit new row
    # residuals are trainable, with no optimizer state for pretrained tables.
    base = get_peft_model(base, LoraConfig(**lora))
    report.update(lora=lora, state_config=asdict(state), repair_config=asdict(repair))
    model = FreshPeriodicRepairDLM(base, tokenizer, state, repair, report).to(device)
    if trainable:
        set_fresh_repair_trainable(model)
        model.train()
    else:
        model.requires_grad_(False).eval()
    return model, tokenizer


def load_fresh_periodic_repair_model(model_path, checkpoint_path, device, *, trainable=False,
                                    torch_dtype=None):
    """Restore only this fresh branch's explicitly marked compact checkpoint."""
    root = Path(checkpoint_path)
    marker = root / INITIALIZATION_MARKER
    if not marker.is_file():
        raise ValueError("checkpoint lacks the raw-pretrained initialization marker; legacy DLM is forbidden")
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    if (recorded.get("schema") != INITIALIZATION_SCHEMA
            or recorded.get("legacy_dlm_checkpoint") is not None
            or recorded.get("pretrained_vocabulary_rows_frozen") is not True):
        raise ValueError("checkpoint is not a registered fresh periodic DLM")
    adapter = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    expected_lora = recorded["lora"]
    if (adapter.get("peft_type") != "LORA" or adapter.get("modules_to_save")
            or set(adapter.get("target_modules") or ()) != set(LORA_TARGET_MODULES)
            or any(adapter.get(key) != expected_lora[key]
                   for key in ("r", "lora_alpha", "lora_dropout", "bias"))):
        raise ValueError("checkpoint LoRA configuration differs from its fresh initialization")
    base, tokenizer, actual = _load_raw_components(model_path, device, torch_dtype=torch_dtype)
    for key in ("schema", "source_config_sha256", "source_tokenizer_size",
                "source_tokenizer_vocab_sha256", "expanded_tokenizer_size",
                "crystal_token_strings_sha256", "new_crystal_token_ids", "lora_target_counts"):
        if actual[key] != recorded[key]:
            raise ValueError(f"raw source reconstruction differs at {key}")
    state = PeriodicStateConfig(**recorded["state_config"])
    repair = PeriodicRepairConfig(**recorded["repair_config"])
    _configs(base, state, repair)
    from peft import PeftModel
    base = PeftModel.from_pretrained(base, str(root), is_trainable=trainable)
    model = FreshPeriodicRepairDLM(base, tokenizer, state, repair, recorded).to(device)
    model.load_state_conditioner(root)
    model.load_repair(root)
    if trainable:
        set_fresh_repair_trainable(model)
        model.train()
    else:
        model.requires_grad_(False).eval()
    return model, tokenizer


__all__ = [
    "INITIALIZATION_MARKER", "INITIALIZATION_SCHEMA", "LORA_TARGET_MODULES",
    "NewCrystalTokenRows", "FreshPeriodicRepairDLM", "extend_raw_crystal_vocabulary",
    "set_fresh_repair_trainable", "initialize_fresh_periodic_repair_model",
    "load_fresh_periodic_repair_model",
]
