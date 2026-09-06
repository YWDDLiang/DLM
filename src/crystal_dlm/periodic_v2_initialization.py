"""Explicit V2 checkpoint ABI; the registered training start is original LLaDA.

V1 model loading and active jobs retain their own marker/implementation. This
module adds the new attention module without importing any adapted checkpoint
as the training initialization.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from crystal_dlm.periodic_repair_initialization import (
    FreshPeriodicRepairDLM, INITIALIZATION_MARKER, INITIALIZATION_SCHEMA,
    LORA_TARGET_MODULES, _configs, _load_raw_components, set_fresh_repair_trainable,
)
from crystal_dlm.periodic_repair_model import PeriodicRepairConfig
from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
from crystal_dlm.periodic_v2_model import PeriodicV2AttentionBias, PeriodicV2Config


V2_MARKER = "periodic_v2_config.json"


def validate_v2_final_checkpoint(root):
    root = Path(root).resolve()
    checkpoint_report = root / "CHECKPOINT_FINAL.json"
    train = root.parent.parent
    checkpoint = json.loads(checkpoint_report.read_text(encoding="utf-8")) if checkpoint_report.is_file() else {}
    if (checkpoint.get("eligible_policy") is not True
            or checkpoint.get("completed_step") != 4524 or checkpoint.get("completed_epochs") != 2
            or not (train / "_SUCCESS").is_file()):
        raise ValueError("V2 loading requires the completed registered final checkpoint")
    report = json.loads((train / "TRAIN_FINAL.json").read_text(encoding="utf-8"))
    if (report.get("eligible_policy") is not True
            or report.get("method") != "periodic_dlm_v2_from_original_llada"
            or Path(report["policy_path"]).resolve() != root
            or report.get("updates") != 4524):
        raise ValueError("V2 checkpoint and completed training endpoint differ")


class PeriodicV2DLM(FreshPeriodicRepairDLM):
    def __init__(self, base_model, tokenizer, state_config, repair_config,
                 initialization, v2_config):
        super().__init__(base_model, tokenizer, state_config, repair_config, initialization)
        if v2_config.hidden_size != state_config.hidden_size:
            raise ValueError("V2 geometry hidden size differs from the backbone")
        advertised_heads = getattr(self.config, "n_heads", None)
        if advertised_heads is not None and int(advertised_heads) != v2_config.heads:
            raise ValueError("V2 bias heads differ from the actual LLaDA attention")
        self.v2_config = v2_config
        self.geometry_attention = PeriodicV2AttentionBias(
            repair_config, v2_config, legacy=self.geometry_attention,
        )

    def save_pretrained(self, output_dir, **kwargs):
        super().save_pretrained(output_dir, **kwargs)
        (Path(output_dir) / V2_MARKER).write_text(
            json.dumps(asdict(self.v2_config), indent=2) + "\n", encoding="utf-8",
        )


def initialize_periodic_v2_model(model_path, device, *, trainable=True, torch_dtype=None):
    """Fresh LoRA/rows/state/GEM; an adapted starting path is explicitly rejected."""
    base, tokenizer, report = _load_raw_components(model_path, device, torch_dtype=torch_dtype)
    state, repair = _configs(base, None, None)
    from peft import LoraConfig, get_peft_model
    lora = {"r": 8, "lora_alpha": 32, "lora_dropout": .05, "bias": "none",
            "target_modules": list(LORA_TARGET_MODULES)}
    base = get_peft_model(base, LoraConfig(**lora))
    v2 = PeriodicV2Config(state.hidden_size, heads=int(base.config.n_heads))
    report.update(lora=lora, state_config=asdict(state), repair_config=asdict(repair))
    model = PeriodicV2DLM(base, tokenizer, state, repair, report, v2).to(device)
    if trainable:
        set_fresh_repair_trainable(model)
        model.train()
    else:
        model.requires_grad_(False).eval()
    return model, tokenizer


def load_periodic_v2_model(model_path, checkpoint_path, device, *, trainable=False,
                           torch_dtype=None):
    """Reconstruct original frozen vocabulary means and restore only V2 deltas."""
    root = Path(checkpoint_path)
    if not (root / V2_MARKER).is_file():
        raise ValueError("checkpoint is not marked as periodic DLM V2")
    validate_v2_final_checkpoint(root)
    recorded = json.loads((root / INITIALIZATION_MARKER).read_text(encoding="utf-8"))
    if (recorded.get("schema") != INITIALIZATION_SCHEMA
            or recorded.get("legacy_dlm_checkpoint") is not None
            or recorded.get("pretrained_vocabulary_rows_frozen") is not True):
        raise ValueError("V2 requires the original-pretrained provenance contract")
    adapter = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    expected = recorded["lora"]
    if (adapter.get("peft_type") != "LORA" or adapter.get("modules_to_save")
            or set(adapter.get("target_modules") or ()) != set(LORA_TARGET_MODULES)
            or any(adapter.get(k) != expected[k] for k in ("r", "lora_alpha", "lora_dropout", "bias"))):
        raise ValueError("V2 LoRA provenance differs from its initialization")
    base, tokenizer, actual = _load_raw_components(model_path, device, torch_dtype=torch_dtype)
    for key in ("schema", "source_config_sha256", "source_tokenizer_size",
                "source_tokenizer_vocab_sha256", "expanded_tokenizer_size",
                "crystal_token_strings_sha256", "new_crystal_token_ids", "lora_target_counts"):
        if actual[key] != recorded[key]:
            raise ValueError(f"V2 raw source reconstruction differs at {key}")
    state = PeriodicStateConfig(**recorded["state_config"])
    repair = PeriodicRepairConfig(**recorded["repair_config"])
    _configs(base, state, repair)
    v2 = PeriodicV2Config(**json.loads((root / V2_MARKER).read_text(encoding="utf-8")))
    from peft import PeftModel
    base = PeftModel.from_pretrained(base, str(root), is_trainable=trainable)
    model = PeriodicV2DLM(base, tokenizer, state, repair, recorded, v2).to(device)
    model.load_state_conditioner(root)
    model.load_repair(root)
    if trainable:
        set_fresh_repair_trainable(model)
        model.train()
    else:
        model.requires_grad_(False).eval()
    return model, tokenizer
