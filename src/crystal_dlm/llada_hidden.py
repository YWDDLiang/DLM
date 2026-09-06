"""Explicit hidden-only execution of the inspected LLaDA encoder core.

The operations follow the actual LLaDAModel.forward through final ln_f, then
return before vocabulary projection.  The original core owns every block,
LoRA layer, cache and native checkpoint function.  No model method, hook or
downloaded source is changed and no hidden tensor is detached.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def resolve_llada_core(model: nn.Module) -> nn.Module:
    """Resolve a raw LLaDA LM or PEFT wrapper without re-registering modules."""
    candidate = model
    unwrap = getattr(candidate, "get_base_model", None)
    if callable(unwrap):
        candidate = unwrap()
    transformer = getattr(candidate, "transformer", None)
    if not isinstance(transformer, nn.ModuleDict):
        candidate = getattr(candidate, "model", None)
        transformer = getattr(candidate, "transformer", None)
    if (not isinstance(candidate, nn.Module) or not isinstance(transformer, nn.ModuleDict)
            or not all(name in transformer for name in ("wte", "emb_drop", "ln_f"))
            or not ("blocks" in transformer or "block_groups" in transformer)):
        raise TypeError("hidden-only execution requires the inspected LLaDA encoder core")
    return candidate


def _checkpoint_this_layer(strategy, index: int) -> bool:
    value = getattr(strategy, "value", strategy)
    return (value == "whole_layer"
            or (value == "one_in_two" and index % 2 == 0)
            or (value == "one_in_three" and index % 3 == 0)
            or (value == "one_in_four" and index % 4 == 0))


def llada_hidden_forward(model: nn.Module, input_ids: Tensor | None = None, *,
                         inputs_embeds: Tensor | None = None,
                         attention_mask: Tensor | None = None,
                         attention_bias: Tensor | None = None) -> Tensor:
    """Return all final normalized hidden states using the original core modules.

    This encoder route has no KV cache, labels or vocabulary logits.  Bias uses
    the actual runtime's four-dimensional [batch, heads, length, length] ABI;
    singleton batch/head dimensions may broadcast exactly as in core.forward.
    """
    core = resolve_llada_core(model)
    config, transformer = core.config, core.transformer
    if bool(config.alibi) or not bool(config.rope):
        raise ValueError("the inspected LLaDA encoder requires RoPE and no ALiBi")
    if inputs_embeds is None:
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        hidden = transformer.wte(input_ids)
    else:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != config.d_model:
            raise ValueError("inputs_embeds must have shape [batch, length, d_model]")
        hidden = inputs_embeds
    batch, length = hidden.shape[:2]
    if batch < 1 or length < 1:
        raise ValueError("hidden-only execution requires a nonempty batch and sequence")
    if bool(config.input_emb_norm):
        hidden = hidden * math.sqrt(config.d_model)
    hidden = transformer.emb_drop(hidden)

    padding_bias = None
    if attention_mask is not None:
        if attention_mask.shape != (batch, length) or attention_mask.device != hidden.device:
            raise ValueError("attention_mask must match input batch/length/device")
        if bool((attention_mask == 0).any()):
            padding_bias = attention_mask.to(dtype=torch.float32).view(batch, -1)[:, None, None, :]
            padding_bias = (1.0 - padding_bias) * torch.finfo(padding_bias.dtype).min

    bias = attention_bias
    if bias is not None or padding_bias is not None:
        if bias is None:
            bias = core.get_bidirectional_attention_bias(length, hidden.device)
        elif bias.dtype in (torch.int8, torch.bool):
            # Match core.forward: allowed entries remain 1, disallowed become min.
            bias = bias.to(dtype=torch.float32)
            bias = bias.masked_fill(bias == 0.0, torch.finfo(bias.dtype).min)
        if (bias.ndim != 4 or bias.shape[-2] < length or bias.shape[-1] < length
                or bias.device != hidden.device):
            raise ValueError("attention_bias must cover the sequence in the four-dimensional core ABI")
        bias = bias[:, :, :length, :length].to(dtype=torch.float32)
        if padding_bias is not None:
            bias = bias + padding_bias
            # Same negative-infinity repair as ensure_finite_(..., True, False).
            # Functional replacement preserves the geometry-bias autograd path.
            bias = torch.where(torch.isneginf(bias), torch.finfo(bias.dtype).min, bias)

    if config.block_group_size == 1:
        for index, block in enumerate(transformer.blocks):
            if _checkpoint_this_layer(core.activation_checkpointing_strategy, index):
                hidden, _ = core._activation_checkpoint_fn(
                    block, hidden, attention_bias=bias, layer_past=None, use_cache=False)
            else:
                hidden, _ = block(hidden, attention_bias=bias, layer_past=None, use_cache=False)
    else:
        for block_group in transformer.block_groups:
            hidden, _ = block_group(hidden, attention_bias=bias, layers_past=None, use_cache=False)
    return transformer.ln_f(hidden)


__all__ = ["resolve_llada_core", "llada_hidden_forward"]
