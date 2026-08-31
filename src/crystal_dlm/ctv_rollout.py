"""Exact-schedule rollout primitives for CTV-DLM resource canaries.

Only a single currently masked lattice/coordinate token is intervened on.
Composition tokens, schedule order, generator logits and every downstream
component remain frozen.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from crystal_dlm.ctv_branching import (
    CTV_MILESTONES,
    make_branch_layout,
    newly_crossed_milestones,
    select_intervention_from_masked_logits,
    stateless_gumbel_scores,
    visible_free_geometry_fraction,
)
from crystal_dlm.llada_generation import (
    _apply_lightweight_decoding_masks,
    _apply_schema_masks,
    _model_logits,
    _validate_generation_position_groups,
    get_num_transfer_tokens,
)


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _initialize_sequence(
    *,
    model: Any,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gen_length: int,
    mask_id: int,
    prefill_token_ids_by_generation_pos: Mapping[
        int, int | Sequence[int] | torch.Tensor
    ]
    | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if prompt.shape[0] != 1:
        raise ValueError("CTV state collection requires one base trajectory")
    device = _model_device(model)
    x = torch.full(
        (1, prompt.shape[1] + int(gen_length)),
        int(mask_id),
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt.shape[1]] = prompt.to(device=device, dtype=torch.long)
    for raw_position, raw_value in (prefill_token_ids_by_generation_pos or {}).items():
        position = int(raw_position)
        if not 0 <= position < int(gen_length):
            raise ValueError(f"CTV prefill position {position} is outside the suffix")
        if isinstance(raw_value, torch.Tensor):
            value = raw_value.to(device=device, dtype=torch.long).reshape(-1)
        elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
            value = torch.tensor(list(raw_value), device=device, dtype=torch.long).reshape(-1)
        else:
            value = torch.tensor([int(raw_value)], device=device, dtype=torch.long)
        if value.numel() != 1:
            raise ValueError("CTV base prefill requires exactly one token per position")
        x[0, prompt.shape[1] + position] = value[0]
    if attention_mask is None:
        return x, None
    prefix = attention_mask.to(device=device)
    suffix = torch.ones(
        (1, int(gen_length)), dtype=prefix.dtype, device=device
    )
    return x, torch.cat([prefix, suffix], dim=-1)


def _allowed_mask(
    *,
    model: Any,
    gen_length: int,
    allowed_token_ids_by_generation_pos: Sequence[Sequence[int]],
) -> torch.Tensor:
    if len(allowed_token_ids_by_generation_pos) != int(gen_length):
        raise ValueError("CTV schema length changed")
    vocab_size = int(model.get_output_embeddings().weight.shape[0])
    mask = torch.zeros(
        (int(gen_length), vocab_size),
        dtype=torch.bool,
        device=_model_device(model),
    )
    for position, token_ids in enumerate(allowed_token_ids_by_generation_pos):
        values = [int(value) for value in token_ids]
        if not values:
            raise ValueError(f"CTV schema position {position} has no legal tokens")
        mask[position, torch.tensor(values, device=mask.device, dtype=torch.long)] = True
    return mask


def _constrained_logits(
    *,
    model: Any,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_index: torch.Tensor,
    prompt_length: int,
    gen_length: int,
    mask_id: int,
    allowed_mask: torch.Tensor,
    lightweight_decoding_constraints: Mapping[str, Any] | None,
) -> torch.Tensor:
    logits = _model_logits(
        model,
        x,
        attention_mask,
        prompt_index,
        0.0,
        int(mask_id),
    )
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("CTV cfg0 model unexpectedly returned multiple logits tensors")
    _apply_schema_masks(
        logits,
        x,
        int(prompt_length),
        int(gen_length),
        allowed_mask,
        None,
    )
    _apply_lightweight_decoding_masks(
        logits,
        x,
        int(prompt_length),
        int(gen_length),
        dict(lightweight_decoding_constraints or {}),
    )
    return logits


def combine_policy_reference_logits(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    guidance_scale: float,
) -> torch.Tensor:
    """Apply late policy/reference guidance without arithmetic on masked -inf."""

    if policy_logits.shape != reference_logits.shape:
        raise ValueError("policy/reference logits shapes differ")
    scale = float(guidance_scale)
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0.0:
        raise ValueError("late guidance scale must be finite and positive")
    policy_finite = torch.isfinite(policy_logits)
    reference_finite = torch.isfinite(reference_logits)
    if not torch.equal(policy_finite, reference_finite):
        raise RuntimeError("policy/reference legal logit supports differ")
    combined = policy_logits.clone()
    combined[policy_finite] = (
        policy_logits[policy_finite]
        + scale
        * (policy_logits[policy_finite] - reference_logits[policy_finite])
    )
    if bool(torch.isnan(combined).any()):
        raise RuntimeError("late guidance produced NaN logits")
    return combined


def _transfer_step(
    *,
    x: torch.Tensor,
    suffix_logits: torch.Tensor,
    prompt_length: int,
    group_positions: Sequence[int],
    transfer_counts: torch.Tensor,
    step_in_group: int,
    mask_id: int,
    temperature: float,
    noise_groups: Sequence[int | str],
    denoise_step: int,
) -> None:
    suffix = x[:, int(prompt_length) :]
    masked = suffix == int(mask_id)
    scores = stateless_gumbel_scores(
        suffix_logits,
        temperature=float(temperature),
        noise_groups=noise_groups,
        denoise_step=int(denoise_step),
    )
    proposed = torch.argmax(scores, dim=-1)
    probabilities = F.softmax(suffix_logits, dim=-1)
    confidence = torch.gather(
        probabilities, dim=-1, index=proposed.unsqueeze(-1)
    ).squeeze(-1)
    group_allowed = torch.zeros_like(masked, dtype=torch.bool)
    group_allowed[:, torch.tensor(group_positions, device=x.device, dtype=torch.long)] = True
    confidence = torch.where(masked & group_allowed, confidence, -float("inf"))
    transfer = torch.zeros_like(masked, dtype=torch.bool)
    for row_index in range(x.shape[0]):
        count = int(transfer_counts[row_index, int(step_in_group)].detach().item())
        if count <= 0:
            continue
        if count > int((masked[row_index] & group_allowed[row_index]).sum().detach().item()):
            raise RuntimeError("CTV transfer count exceeds the remaining group masks")
        selected = torch.topk(confidence[row_index], k=count).indices
        transfer[row_index, selected] = True
    suffix[transfer] = proposed[transfer]


@torch.no_grad()
def collect_ctv_branch_states(
    model: Any,
    prompt: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    num_atoms: int,
    gen_length: int,
    temperature: float,
    mask_id: int,
    allowed_token_ids_by_generation_pos: Sequence[Sequence[int]],
    prefill_token_ids_by_generation_pos: Mapping[
        int, int | Sequence[int] | torch.Tensor
    ],
    generation_position_groups: Sequence[Sequence[int]],
    lightweight_decoding_constraints: Mapping[str, Any] | None,
    base_noise_group: int | str,
    milestones: Sequence[float] = CTV_MILESTONES,
    reference_model: Any | None = None,
    late_guidance_scale: float = 0.0,
    late_guidance_remaining_mask_threshold: float = 0.0,
    rollout_diagnostics: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Run one frozen base trajectory and capture the two branch states."""

    guidance_scale = float(late_guidance_scale)
    guidance_threshold = float(late_guidance_remaining_mask_threshold)
    if reference_model is None:
        if guidance_scale != 0.0 or guidance_threshold != 0.0:
            raise ValueError("late guidance parameters require a reference model")
    else:
        if _model_device(reference_model) != _model_device(model):
            raise ValueError("policy/reference models must share one device")
        if guidance_scale <= 0.0 or not 0.0 < guidance_threshold < 1.0:
            raise ValueError("late guidance requires positive scale and threshold in (0,1)")

    if hasattr(model, "set_geometry_context"):
        model.set_geometry_context(
            torch.tensor([int(prompt.shape[1])], device=_model_device(model)),
            torch.tensor([int(num_atoms)], device=_model_device(model)),
        )

    x, expanded_attention = _initialize_sequence(
        model=model,
        prompt=prompt,
        attention_mask=attention_mask,
        gen_length=int(gen_length),
        mask_id=int(mask_id),
        prefill_token_ids_by_generation_pos=prefill_token_ids_by_generation_pos,
    )
    prompt_length = int(prompt.shape[1])
    prompt_index = x != int(mask_id)
    allowed_mask = _allowed_mask(
        model=model,
        gen_length=int(gen_length),
        allowed_token_ids_by_generation_pos=allowed_token_ids_by_generation_pos,
    )
    groups = _validate_generation_position_groups(
        [list(group) for group in generation_position_groups], int(gen_length)
    )
    pending = [float(value) for value in milestones]
    captured: list[dict[str, Any]] = []
    fraction = visible_free_geometry_fraction(
        x[0, prompt_length:].detach().cpu().tolist(),
        mask_id=int(mask_id),
        num_atoms=int(num_atoms),
    )
    denoise_step = 0
    guided_denoise_steps = 0
    first_guided_visible_fraction: float | None = None
    for group_index, group in enumerate(groups):
        group_tensor = torch.tensor(group, device=x.device, dtype=torch.long)
        group_mask = x[:, prompt_length:].index_select(1, group_tensor) == int(mask_id)
        group_steps = int(group_mask.sum(dim=1).max().detach().item())
        if group_steps <= 0:
            continue
        full_group_mask = torch.zeros_like(x[:, prompt_length:], dtype=torch.bool)
        full_group_mask[:, group_tensor] = True
        transfers = get_num_transfer_tokens(
            (x[:, prompt_length:] == int(mask_id)) & full_group_mask,
            group_steps,
        )
        for step_in_group in range(group_steps):
            logits = _constrained_logits(
                model=model,
                x=x,
                attention_mask=expanded_attention,
                prompt_index=prompt_index,
                prompt_length=prompt_length,
                gen_length=int(gen_length),
                mask_id=int(mask_id),
                allowed_mask=allowed_mask,
                lightweight_decoding_constraints=lightweight_decoding_constraints,
            )
            remaining_mask_fraction = 1.0 - float(fraction)
            if (
                reference_model is not None
                and remaining_mask_fraction <= guidance_threshold + 1e-12
            ):
                reference_logits = _constrained_logits(
                    model=reference_model,
                    x=x,
                    attention_mask=expanded_attention,
                    prompt_index=prompt_index,
                    prompt_length=prompt_length,
                    gen_length=int(gen_length),
                    mask_id=int(mask_id),
                    allowed_mask=allowed_mask,
                    lightweight_decoding_constraints=lightweight_decoding_constraints,
                )
                logits = combine_policy_reference_logits(
                    logits,
                    reference_logits,
                    guidance_scale=guidance_scale,
                )
                guided_denoise_steps += 1
                if first_guided_visible_fraction is None:
                    first_guided_visible_fraction = float(fraction)
            before = fraction
            _transfer_step(
                x=x,
                suffix_logits=logits[:, prompt_length : prompt_length + int(gen_length)],
                prompt_length=prompt_length,
                group_positions=group,
                transfer_counts=transfers,
                step_in_group=step_in_group,
                mask_id=int(mask_id),
                temperature=float(temperature),
                noise_groups=[base_noise_group],
                denoise_step=denoise_step,
            )
            fraction = visible_free_geometry_fraction(
                x[0, prompt_length:].detach().cpu().tolist(),
                mask_id=int(mask_id),
                num_atoms=int(num_atoms),
            )
            crossed = newly_crossed_milestones(before, fraction, pending=pending)
            if crossed:
                suffix_now = x[0, prompt_length:]
                eligible_positions = [
                    int(position)
                    for position in group
                    if int(suffix_now[int(position)].detach().item()) == int(mask_id)
                ]
                if not eligible_positions:
                    for later_group in groups[group_index + 1 :]:
                        eligible_positions = [
                            int(position)
                            for position in later_group
                            if int(suffix_now[int(position)].detach().item())
                            == int(mask_id)
                        ]
                        if eligible_positions:
                            break
                if not eligible_positions:
                    raise RuntimeError(
                        "CTV milestone has no current/next schedule-group intervention"
                    )
                state_logits = _constrained_logits(
                    model=model,
                    x=x,
                    attention_mask=expanded_attention,
                    prompt_index=prompt_index,
                    prompt_length=prompt_length,
                    gen_length=int(gen_length),
                    mask_id=int(mask_id),
                    allowed_mask=allowed_mask,
                    lightweight_decoding_constraints=lightweight_decoding_constraints,
                )
                for milestone in crossed:
                    selection = select_intervention_from_masked_logits(
                        logits=state_logits[0, prompt_length : prompt_length + int(gen_length)],
                        suffix_token_ids=x[0, prompt_length:],
                        allowed_token_ids_by_generation_pos=allowed_token_ids_by_generation_pos,
                        num_atoms=int(num_atoms),
                        mask_id=int(mask_id),
                        eligible_positions=eligible_positions,
                    )
                    captured.append(
                        {
                            "milestone": float(milestone),
                            "visible_free_geometry_fraction": float(fraction),
                            "tokens": x.clone(),
                            "attention_mask": None
                            if expanded_attention is None
                            else expanded_attention.clone(),
                            "next_denoise_step": int(denoise_step + 1),
                            "group_index": int(group_index),
                            "step_in_group": int(step_in_group),
                            "eligible_schedule_positions": tuple(eligible_positions),
                            **selection,
                        }
                    )
                    pending.remove(float(milestone))
            denoise_step += 1
    if pending or len(captured) != len(tuple(milestones)):
        raise RuntimeError(f"CTV failed to capture frozen milestones: pending={pending}")
    if bool((x[:, prompt_length:] == int(mask_id)).any()):
        raise RuntimeError("CTV frozen base trajectory left masked suffix tokens")
    if rollout_diagnostics is not None:
        rollout_diagnostics.update(
            {
                "first_guided_visible_fraction": first_guided_visible_fraction,
                "guided_denoise_steps": int(guided_denoise_steps),
                "late_guidance_remaining_mask_threshold": guidance_threshold,
                "late_guidance_scale": guidance_scale,
                "total_denoise_steps": int(denoise_step),
            }
        )
    return x, captured


@torch.no_grad()
def complete_ctv_forced_branches(
    model: Any,
    snapshot: Mapping[str, Any],
    *,
    composition_id: str,
    sample_idx: int,
    continuation_seeds: Sequence[int],
    gen_length: int,
    temperature: float,
    mask_id: int,
    allowed_token_ids_by_generation_pos: Sequence[Sequence[int]],
    generation_position_groups: Sequence[Sequence[int]],
    lightweight_decoding_constraints: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Force each legal action and finish all 16 compute-matched branches."""

    source_tokens = snapshot.get("tokens")
    if not isinstance(source_tokens, torch.Tensor) or source_tokens.shape[0] != 1:
        raise ValueError("CTV snapshot must contain one token trajectory")
    prompt_length = int(source_tokens.shape[1]) - int(gen_length)
    position = int(snapshot["position"])
    if position not in {
        int(value) for value in snapshot.get("eligible_schedule_positions", ())
    }:
        raise ValueError("CTV intervention escaped its frozen schedule group")
    if int(source_tokens[0, prompt_length + position].detach().item()) != int(mask_id):
        raise ValueError("CTV intervention position is no longer masked")
    actions = [int(value) for value in snapshot["action_token_ids"]]
    layout = make_branch_layout(
        composition_id=str(composition_id),
        sample_idx=int(sample_idx),
        milestone=float(snapshot["milestone"]),
        intervention_position=position,
        action_token_ids=actions,
        continuation_seeds=continuation_seeds,
    )
    x = source_tokens.repeat(len(layout), 1)
    action_values = torch.tensor(
        [int(row["action_token"]) for row in layout],
        dtype=torch.long,
        device=x.device,
    )
    x[:, prompt_length + position] = action_values
    source_attention = snapshot.get("attention_mask")
    attention = (
        None
        if source_attention is None
        else source_attention.repeat(len(layout), 1)
    )
    prompt_index = source_tokens.repeat(len(layout), 1) != int(mask_id)
    allowed_mask = _allowed_mask(
        model=model,
        gen_length=int(gen_length),
        allowed_token_ids_by_generation_pos=allowed_token_ids_by_generation_pos,
    )
    groups = _validate_generation_position_groups(
        [list(group) for group in generation_position_groups], int(gen_length)
    )
    denoise_step = int(snapshot["next_denoise_step"])
    noise_groups = [int(row["noise_group"]) for row in layout]
    for group in groups:
        group_tensor = torch.tensor(group, device=x.device, dtype=torch.long)
        full_group_mask = torch.zeros_like(x[:, prompt_length:], dtype=torch.bool)
        full_group_mask[:, group_tensor] = True
        group_mask = (x[:, prompt_length:] == int(mask_id)) & full_group_mask
        group_steps = int(group_mask.sum(dim=1).max().detach().item())
        if group_steps <= 0:
            continue
        transfers = get_num_transfer_tokens(group_mask, group_steps)
        for step_in_group in range(group_steps):
            logits = _constrained_logits(
                model=model,
                x=x,
                attention_mask=attention,
                prompt_index=prompt_index,
                prompt_length=prompt_length,
                gen_length=int(gen_length),
                mask_id=int(mask_id),
                allowed_mask=allowed_mask,
                lightweight_decoding_constraints=lightweight_decoding_constraints,
            )
            _transfer_step(
                x=x,
                suffix_logits=logits[:, prompt_length : prompt_length + int(gen_length)],
                prompt_length=prompt_length,
                group_positions=group,
                transfer_counts=transfers,
                step_in_group=step_in_group,
                mask_id=int(mask_id),
                temperature=float(temperature),
                noise_groups=noise_groups,
                denoise_step=denoise_step,
            )
            denoise_step += 1
    if bool((x[:, prompt_length:] == int(mask_id)).any()):
        raise RuntimeError("CTV branch completion left masked suffix tokens")
    return x, layout


__all__ = [
    "collect_ctv_branch_states",
    "combine_policy_reference_logits",
    "complete_ctv_forced_branches",
]
