"""Lazy compatibility helpers for the frozen A800 Llama/PEFT environment."""

from __future__ import annotations

from typing import Any


def load_llama3_compatible_config(
    model_path: str,
    *,
    trust_remote_code: bool = True,
) -> Any:
    """Load Llama-3.1 config on the older frozen Transformers build."""

    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.llama.configuration_llama import LlamaConfig

    try:
        return AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        if "rope_scaling" not in str(exc):
            raise

    config_dict, _ = PretrainedConfig.get_config_dict(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    rope_scaling = config_dict.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        config_dict["rope_scaling"] = {
            "type": "dynamic",
            "factor": float(rope_scaling.get("factor", 8.0)),
        }
    return LlamaConfig(**config_dict)


def ensure_peft_cache_compat() -> None:
    """Expose cache symbols expected by newer PEFT on older Transformers."""

    import transformers

    try:
        from transformers.cache_utils import Cache, DynamicCache
    except Exception:
        Cache = object
        DynamicCache = object
    if not hasattr(transformers, "Cache"):
        transformers.Cache = Cache
    if not hasattr(transformers, "DynamicCache"):
        transformers.DynamicCache = DynamicCache
    if not hasattr(transformers, "EncoderDecoderCache"):

        class EncoderDecoderCache:
            """Compatibility placeholder for PEFT imports."""

        transformers.EncoderDecoderCache = EncoderDecoderCache


def disable_peft_bnb_autodetect() -> None:
    """Avoid optional bitsandbytes dispatch paths in the frozen environment."""

    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.model as peft_lora_model

        peft_import_utils.is_bnb_available = lambda: False
        peft_import_utils.is_bnb_4bit_available = lambda: False
        peft_lora_model.is_bnb_available = lambda: False
        peft_lora_model.is_bnb_4bit_available = lambda: False
    except Exception:
        pass
