"""Stateful suffix-visible correction for SPAD crystal decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from crystal_dlm.llada_generation import (
    _apply_lightweight_decoding_masks,
    _apply_schema_masks,
    _candidate_tokens_and_confidence,
    _lattice_matrix_from_token_ids,
    _model_logits,
    _prepare_atom_count_grammar,
)
from crystal_dlm.spad_program import coordinate_positions


@dataclass(frozen=True)
class Model494ResponseConfig:
    """Fixed local trust region for model494 endpoint-response guidance."""

    max_atom_step_A: float = 0.15
    kl_budget_nats: float = 0.05
    max_abs_logit_bias: float = 2.0
    standardized_gain_clip: float = 3.0
    image_radius: int = 2

    def validate(self) -> None:
        if float(self.max_atom_step_A) <= 0.0:
            raise ValueError("max_atom_step_A must be positive")
        if float(self.kl_budget_nats) < 0.0:
            raise ValueError("kl_budget_nats must be non-negative")
        if float(self.max_abs_logit_bias) < 0.0:
            raise ValueError("max_abs_logit_bias must be non-negative")
        if float(self.standardized_gain_clip) <= 0.0:
            raise ValueError("standardized_gain_clip must be positive")
        if int(self.image_radius) not in (1, 2):
            raise ValueError("image_radius must be one (27) or two (125)")


def _minimum_image_vectors(
    fractional_deltas: torch.Tensor,
    lattice: torch.Tensor,
    *,
    image_radius: int,
) -> torch.Tensor:
    """Return shortest Cartesian image vectors for triclinic deltas ``[...,3]``."""

    if fractional_deltas.shape[-1:] != (3,) or lattice.shape != (3, 3):
        raise ValueError("minimum-image inputs must end in [3] with lattice [3, 3]")
    if int(image_radius) not in (1, 2):
        raise ValueError("image_radius must be one (27) or two (125)")
    centered = fractional_deltas - torch.round(fractional_deltas)
    values = torch.arange(
        -int(image_radius),
        int(image_radius) + 1,
        dtype=centered.dtype,
        device=centered.device,
    )
    shifts = torch.cartesian_prod(values, values, values).reshape(-1, 3)
    vectors = (centered.unsqueeze(-2) + shifts) @ lattice
    selected = torch.argmin(vectors.square().sum(dim=-1), dim=-1)
    return torch.gather(
        vectors,
        dim=-2,
        index=selected[..., None, None].expand(*selected.shape, 1, 3),
    ).squeeze(-2)


def _minimum_image_vector(
    fractional_delta: torch.Tensor,
    lattice: torch.Tensor,
    *,
    image_radius: int,
) -> torch.Tensor:
    """Scalar convenience wrapper around :func:`_minimum_image_vectors`."""

    if fractional_delta.shape != (3,):
        raise ValueError("fractional_delta must have shape [3]")
    return _minimum_image_vectors(
        fractional_delta,
        lattice,
        image_radius=int(image_radius),
    )


def _bounded_translation_free_response(
    source_fractional: torch.Tensor,
    target_fractional: torch.Tensor,
    lattice: torch.Tensor,
    config: Model494ResponseConfig,
) -> torch.Tensor:
    """Convert a model494 endpoint into local, translation-free response vectors."""

    if source_fractional.ndim != 2 or source_fractional.shape[-1] != 3:
        raise ValueError("source_fractional must have shape [N, 3]")
    if target_fractional.shape != source_fractional.shape:
        raise ValueError("model494 target coordinates do not match source coordinates")
    if not bool(torch.isfinite(target_fractional).all().item()):
        raise ValueError("model494 target coordinates must be finite")
    vectors = _minimum_image_vectors(
        target_fractional - source_fractional,
        lattice,
        image_radius=int(config.image_radius),
    )
    vectors = vectors - vectors.mean(dim=0, keepdim=True)
    norms = torch.linalg.vector_norm(vectors, dim=1)
    scale = torch.clamp(
        float(config.max_atom_step_A) / norms.clamp_min(1.0e-12),
        max=1.0,
    )
    return vectors * scale.unsqueeze(1)


def _kl_bounded_gain_bias(
    base_logits: torch.Tensor,
    gains: torch.Tensor,
    config: Model494ResponseConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Turn geometric gains into the strongest bias inside a fixed KL budget."""

    if base_logits.ndim != 1 or gains.shape != base_logits.shape:
        raise ValueError("base_logits and gains must be matching vectors")
    if not bool(torch.isfinite(base_logits).all().item()):
        raise ValueError("legal base logits must be finite")
    if not bool(torch.isfinite(gains).all().item()):
        raise ValueError("response gains must be finite")
    work_logits = base_logits.to(dtype=torch.float64)
    work_gains = gains.to(dtype=torch.float64)
    log_p = torch.log_softmax(work_logits, dim=0)
    p = torch.exp(log_p)
    mean = torch.sum(p * work_gains)
    variance = torch.sum(p * (work_gains - mean).square())
    if float(variance.detach().item()) <= 1.0e-16:
        return torch.zeros_like(base_logits), {
            "coefficient": 0.0,
            "kl_nats": 0.0,
            "weighted_gain_std": float(torch.sqrt(variance).detach().item()),
        }
    standardized = (work_gains - mean) / torch.sqrt(variance)
    standardized = standardized.clamp(
        min=-float(config.standardized_gain_clip),
        max=float(config.standardized_gain_clip),
    )
    maximum = float(torch.max(torch.abs(standardized)).detach().item())
    if maximum <= 1.0e-12 or float(config.max_abs_logit_bias) == 0.0:
        return torch.zeros_like(base_logits), {
            "coefficient": 0.0,
            "kl_nats": 0.0,
            "weighted_gain_std": float(torch.sqrt(variance).detach().item()),
        }
    upper = float(config.max_abs_logit_bias) / maximum

    def divergence(coefficient: float) -> torch.Tensor:
        log_q = torch.log_softmax(work_logits + coefficient * standardized, dim=0)
        q = torch.exp(log_q)
        return torch.sum(q * (log_q - log_p))

    budget = float(config.kl_budget_nats)
    if budget <= 0.0:
        coefficient = 0.0
    elif float(divergence(upper).detach().item()) <= budget:
        coefficient = upper
    else:
        lower = 0.0
        for _ in range(32):
            midpoint = 0.5 * (lower + upper)
            if float(divergence(midpoint).detach().item()) <= budget:
                lower = midpoint
            else:
                upper = midpoint
        coefficient = lower
    bias = coefficient * standardized
    kl_value = float(divergence(coefficient).detach().item())
    return bias.to(dtype=base_logits.dtype), {
        "coefficient": float(coefficient),
        "kl_nats": kl_value,
        "weighted_gain_std": float(torch.sqrt(variance).detach().item()),
    }


def _prepare_response_contexts(
    complete_tokens: torch.Tensor,
    *,
    prompt_length: int,
    targets: Sequence[Sequence[Sequence[float]] | None],
    constraints: dict,
    config: Model494ResponseConfig,
) -> list[dict[str, torch.Tensor] | None]:
    if len(targets) != int(complete_tokens.shape[0]):
        raise ValueError("one model494 target is required per batch row")
    count_token_to_n = constraints.get("count_token_to_n", {})
    coordinate_maps = constraints.get("coord_token_to_bin", {})
    period = int(constraints.get("coord_period", 100))
    body_offset = int(constraints.get("body_offset", 0))
    if period <= 0:
        raise ValueError("coordinate period must be positive")
    contexts: list[dict[str, torch.Tensor] | None] = []
    for row_index, raw_target in enumerate(targets):
        if raw_target is None:
            contexts.append(None)
            continue
        count_token = int(
            complete_tokens[row_index, prompt_length + body_offset].detach().item()
        )
        num_atoms = int(count_token_to_n.get(count_token, 0))
        if num_atoms <= 0:
            raise ValueError("response guidance could not decode atom count")
        lattice = _lattice_matrix_from_token_ids(
            complete_tokens[row_index],
            prompt_length=prompt_length,
            constraints=constraints,
        )
        if lattice is None:
            raise ValueError("response guidance requires a valid frozen lattice")
        source_rows: list[list[float]] = []
        for slot in range(num_atoms):
            positions = coordinate_positions(slot)
            bins: list[int] = []
            for axis, position in zip(("X", "Y", "Z"), positions, strict=True):
                token_id = int(
                    complete_tokens[row_index, prompt_length + position]
                    .detach()
                    .item()
                )
                value = coordinate_maps.get(axis, {}).get(token_id)
                if value is None:
                    raise ValueError("response guidance could not decode source coordinate")
                bins.append(int(value))
            source_rows.append([float(value) / float(period) for value in bins])
        source = torch.tensor(
            source_rows,
            dtype=torch.float64,
            device=complete_tokens.device,
        )
        target = torch.as_tensor(
            raw_target,
            dtype=torch.float64,
            device=complete_tokens.device,
        )
        response = _bounded_translation_free_response(source, target, lattice, config)
        contexts.append(
            {
                "source_fractional": source,
                "lattice": lattice,
                "response_cartesian": response,
            }
        )
    return contexts


def _apply_model494_response_bias(
    logits: torch.Tensor,
    x: torch.Tensor,
    *,
    prompt_length: int,
    active: dict[int, tuple[int, int, int]],
    active_generation_mask: torch.Tensor,
    old_by_row: dict[int, tuple[int, int, int]],
    component: int,
    contexts: Sequence[dict[str, torch.Tensor] | None],
    constraints: dict,
    config: Model494ResponseConfig,
) -> dict[int, dict[str, float | int]]:
    """Bias legal coordinate logits toward a bounded model494 response vector."""

    axis = ("X", "Y", "Z")[int(component)]
    token_to_bin = constraints.get("coord_token_to_bin", {}).get(axis, {})
    period = int(constraints.get("coord_period", 100))
    min_value = torch.finfo(logits.dtype).min
    reports: dict[int, dict[str, float | int]] = {}
    for row_index, positions in active.items():
        position = int(positions[component])
        if not bool(active_generation_mask[row_index, position].detach().item()):
            continue
        context = contexts[row_index]
        if context is None:
            continue
        slot_index = (int(positions[0]) - 8) // 4
        source = context["source_fractional"][slot_index]
        current = source.clone()
        old = old_by_row[row_index]
        for prior_component in range(3):
            if prior_component < int(component):
                token_id = int(
                    x[row_index, prompt_length + positions[prior_component]]
                    .detach()
                    .item()
                )
            else:
                token_id = int(old[prior_component])
            prior_axis = ("X", "Y", "Z")[prior_component]
            value = constraints.get("coord_token_to_bin", {}).get(prior_axis, {}).get(
                token_id
            )
            if value is None:
                raise ValueError("response guidance encountered a non-coordinate token")
            current[prior_component] = float(value) / float(period)
        lattice = context["lattice"]
        target_vector = context["response_cartesian"][slot_index]
        previous_vector = _minimum_image_vector(
            current - source,
            lattice,
            image_radius=int(config.image_radius),
        )
        previous_error = (previous_vector - target_vector).square().sum()
        legal_ids: list[int] = []
        legal_bins: list[int] = []
        absolute_position = prompt_length + position
        for token_id, bin_value in token_to_bin.items():
            token_id = int(token_id)
            if not bool(logits[row_index, absolute_position, token_id] > min_value):
                continue
            legal_ids.append(token_id)
            legal_bins.append(int(bin_value))
        if not legal_ids:
            continue
        candidates = current.unsqueeze(0).repeat(len(legal_ids), 1)
        candidates[:, int(component)] = torch.tensor(
            legal_bins,
            dtype=candidates.dtype,
            device=candidates.device,
        ) / float(period)
        candidate_vectors = _minimum_image_vectors(
            candidates - source.unsqueeze(0),
            lattice,
            image_radius=int(config.image_radius),
        )
        gain_tensor = previous_error - (
            candidate_vectors - target_vector.unsqueeze(0)
        ).square().sum(dim=1)
        index = torch.tensor(legal_ids, dtype=torch.long, device=logits.device)
        base = logits[row_index, absolute_position, index]
        gain_tensor = gain_tensor.to(device=logits.device)
        bias, summary = _kl_bounded_gain_bias(base, gain_tensor, config)
        logits[row_index, absolute_position, index] = base + bias
        reports[row_index] = {
            "component": int(component),
            "legal_token_count": len(legal_ids),
            "target_step_A": float(torch.linalg.vector_norm(target_vector).item()),
            "previous_error_A2": float(previous_error.item()),
            "maximum_gain_A2": float(torch.max(gain_tensor).item()),
            "minimum_gain_A2": float(torch.min(gain_tensor).item()),
            **summary,
        }
    return reports


def _full_attention_mask(
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    prompt_length: int,
    gen_length: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.shape[0] != x.shape[0]:
        raise ValueError("attention mask batch size changed")
    if attention_mask.shape[1] == prompt_length:
        return torch.cat(
            [
                attention_mask,
                torch.ones(
                    (x.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=x.device,
                ),
            ],
            dim=1,
        )
    if attention_mask.shape == x.shape:
        return attention_mask.clone()
    raise ValueError("attention mask must cover the prompt or complete canvas")


def _validate_revision_slots(
    revision_slots_by_batch: Sequence[Sequence[int]],
    *,
    batch_size: int,
    gen_length: int,
) -> list[list[int]]:
    if len(revision_slots_by_batch) != int(batch_size):
        raise ValueError("one anchor-revision schedule is required per batch row")
    output: list[list[int]] = []
    for row_index, slots in enumerate(revision_slots_by_batch):
        values = [int(value) for value in slots]
        if len(values) != len(set(values)):
            raise ValueError(f"row {row_index} revisits an anchor more than once")
        for slot in values:
            if slot < 0 or coordinate_positions(slot)[-1] >= int(gen_length):
                raise ValueError(f"row {row_index} anchor slot lies outside canvas")
        output.append(values)
    return output


@torch.no_grad()
def revise_spad_anchors(
    model: Any,
    complete_tokens: torch.Tensor,
    *,
    prompt_length: int,
    gen_length: int,
    revision_slots_by_batch: Sequence[Sequence[int]],
    attention_mask: torch.Tensor | None,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
    allowed_token_ids_by_generation_pos: list[list[int]] | None,
    atom_count_grammar: dict | None,
    lightweight_decoding_constraints: dict | None,
    suffix_visible: bool,
    model494_target_frac_coords_by_batch: Sequence[
        Sequence[Sequence[float]] | None
    ]
    | None = None,
    model494_response_config: Model494ResponseConfig | None = None,
    strict_pbc_no_legal_fallback: bool = False,
) -> tuple[torch.Tensor, list[list[dict[str, Any]]]]:
    """Re-mask registered sites once and fill them with full model context.

    Rows may carry different revision schedules.  Every site is masked as one
    transaction, then X/Y/Z are resolved in order.  Non-active values are
    immutable, and the returned log retains the previous triplet as the
    explicit no-op candidate/provisional geometry.  Optional model494 endpoint
    responses add a deterministic, KL-bounded bias *after* schema/PBC masking;
    they never reopen an illegal token.
    """

    if complete_tokens.ndim != 2:
        raise ValueError("complete_tokens must have shape [batch, sequence]")
    if complete_tokens.shape[1] != int(prompt_length) + int(gen_length):
        raise ValueError("complete token sequence does not match prompt+generation")
    x = complete_tokens.clone()
    suffix = x[:, prompt_length : prompt_length + gen_length]
    if bool((suffix == int(mask_id)).any()):
        raise ValueError("SPAD correction requires a complete predictor canvas")
    schedules = _validate_revision_slots(
        revision_slots_by_batch,
        batch_size=x.shape[0],
        gen_length=gen_length,
    )
    response_enabled = model494_target_frac_coords_by_batch is not None
    response_config = model494_response_config or Model494ResponseConfig()
    response_contexts: list[dict[str, torch.Tensor] | None] | None = None
    if strict_pbc_no_legal_fallback and (
        not lightweight_decoding_constraints
        or not lightweight_decoding_constraints.get("pbc_min_distance_mask")
    ):
        raise ValueError("strict PBC fallback requires PBC hard support")
    if response_enabled:
        response_config.validate()
        if not suffix_visible:
            raise ValueError("model494 response guidance requires suffix visibility")
        if not lightweight_decoding_constraints:
            raise ValueError("model494 response guidance requires coordinate constraints")
        if not lightweight_decoding_constraints.get("pbc_min_distance_mask"):
            raise ValueError("model494 response guidance requires PBC hard support")
        response_contexts = _prepare_response_contexts(
            x,
            prompt_length=prompt_length,
            targets=model494_target_frac_coords_by_batch,
            constraints=lightweight_decoding_constraints,
            config=response_config,
        )
    full_attention = _full_attention_mask(
        x,
        attention_mask,
        prompt_length=prompt_length,
        gen_length=gen_length,
    )
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, :prompt_length] = True
    allowed_mask = None
    prepared_atom_count_grammar = None
    vocab_size = model.get_output_embeddings().weight.shape[0]
    if allowed_token_ids_by_generation_pos is not None:
        if len(allowed_token_ids_by_generation_pos) != int(gen_length):
            raise ValueError("allowed token schema length changed")
        allowed_mask = torch.zeros(
            (gen_length, vocab_size), dtype=torch.bool, device=x.device
        )
        for position, token_ids in enumerate(allowed_token_ids_by_generation_pos):
            if not token_ids:
                raise ValueError(f"generation position {position} has no legal token")
            allowed_mask[
                position,
                torch.tensor(token_ids, dtype=torch.long, device=x.device),
            ] = True
    if atom_count_grammar is not None:
        prepared_atom_count_grammar = _prepare_atom_count_grammar(
            atom_count_grammar, vocab_size, x.device
        )

    logs: list[list[dict[str, Any]]] = [[] for _ in range(x.shape[0])]
    max_revisions = max((len(values) for values in schedules), default=0)
    for revision_index in range(max_revisions):
        active: dict[int, tuple[int, int, int]] = {}
        old_by_row: dict[int, tuple[int, int, int]] = {}
        component_reports: dict[int, list[dict[str, float | int]]] = {}
        skipped_no_legal: set[int] = set()
        before = x.clone()
        stage_attention = None if full_attention is None else full_attention.clone()
        for row_index, slots in enumerate(schedules):
            if revision_index >= len(slots):
                continue
            positions = coordinate_positions(slots[revision_index])
            absolute = tuple(prompt_length + position for position in positions)
            previous = tuple(int(x[row_index, position].item()) for position in absolute)
            if any(value == int(mask_id) for value in previous):
                raise RuntimeError("registered anchor was not committed")
            if response_contexts is not None and response_contexts[row_index] is None:
                logs[row_index].append(
                    {
                        "revision_index": int(revision_index),
                        "slot_index": int(slots[revision_index]),
                        "generation_positions": list(positions),
                        "previous_token_ids": list(previous),
                        "new_token_ids": list(previous),
                        "changed_components": 0,
                        "suffix_visible": bool(suffix_visible),
                        "guidance_status": "guidance_skipped_missing_response",
                        "model494_response_guidance": True,
                    }
                )
                continue
            active[row_index] = positions
            old_by_row[row_index] = previous
            x[row_index, torch.tensor(absolute, device=x.device)] = int(mask_id)
            if stage_attention is not None and not suffix_visible:
                stage_attention[
                    row_index,
                    prompt_length + positions[-1] + 1 : prompt_length + gen_length,
                ] = 0

        for component in range(3):
            group_allowed = torch.zeros_like(x, dtype=torch.bool)
            for row_index, positions in active.items():
                group_allowed[row_index, prompt_length + positions[component]] = True
            if not bool(group_allowed.any()):
                continue
            logits = _model_logits(
                model,
                x,
                stage_attention,
                prompt_index,
                float(cfg_scale),
                int(mask_id),
            )
            if allowed_mask is not None or prepared_atom_count_grammar is not None:
                _apply_schema_masks(
                    logits,
                    x,
                    prompt_length,
                    gen_length,
                    allowed_mask,
                    prepared_atom_count_grammar,
                )
            mask_report = _apply_lightweight_decoding_masks(
                logits,
                x,
                prompt_length,
                gen_length,
                lightweight_decoding_constraints,
                group_allowed[:, prompt_length : prompt_length + gen_length],
                int(mask_id),
            )
            if (
                response_contexts is not None or strict_pbc_no_legal_fallback
            ) and int(component) == 2:
                no_legal = mask_report.get("pbc_no_legal_completion", set())
                for row_index, positions in active.items():
                    if (int(row_index), int(positions[2])) not in no_legal:
                        continue
                    absolute = tuple(
                        prompt_length + position for position in positions
                    )
                    x[row_index, torch.tensor(absolute, device=x.device)] = torch.tensor(
                        old_by_row[row_index], dtype=x.dtype, device=x.device
                    )
                    group_allowed[row_index, prompt_length + positions[2]] = False
                    skipped_no_legal.add(int(row_index))
            if response_contexts is not None:
                reports = _apply_model494_response_bias(
                    logits,
                    x,
                    prompt_length=prompt_length,
                    active=active,
                    active_generation_mask=group_allowed[
                        :, prompt_length : prompt_length + gen_length
                    ],
                    old_by_row=old_by_row,
                    component=int(component),
                    contexts=response_contexts,
                    constraints=lightweight_decoding_constraints,
                    config=response_config,
                )
                for row_index, report in reports.items():
                    component_reports.setdefault(int(row_index), []).append(report)
                x0 = torch.argmax(logits, dim=-1)
            else:
                x0, _confidence = _candidate_tokens_and_confidence(
                    logits, float(temperature), remasking
                )
            x[group_allowed] = x0[group_allowed]

        for row_index, positions in active.items():
            absolute = tuple(prompt_length + position for position in positions)
            if any(int(x[row_index, position].item()) == int(mask_id) for position in absolute):
                raise RuntimeError("anchor correction left a masked coordinate")
            unchanged = torch.ones_like(x[row_index], dtype=torch.bool)
            unchanged[torch.tensor(absolute, device=x.device)] = False
            if not bool(torch.equal(x[row_index][unchanged], before[row_index][unchanged])):
                raise RuntimeError("anchor correction changed a non-active token")
            new = tuple(int(x[row_index, position].item()) for position in absolute)
            old = old_by_row[row_index]
            logs[row_index].append(
                {
                    "revision_index": int(revision_index),
                    "slot_index": int(schedules[row_index][revision_index]),
                    "generation_positions": list(positions),
                    "previous_token_ids": list(old),
                    "new_token_ids": list(new),
                    "changed_components": sum(a != b for a, b in zip(old, new, strict=True)),
                    "suffix_visible": bool(suffix_visible),
                    "guidance_status": (
                        "guidance_skipped_no_legal_completion"
                        if row_index in skipped_no_legal
                        else (
                        "model494_response_applied"
                        if response_contexts is not None
                            else "unguided_spad_revision"
                        )
                    ),
                    "model494_response_guidance": bool(
                        response_contexts is not None
                    ),
                    "model494_response_config": (
                        asdict(response_config)
                        if response_contexts is not None
                        else None
                    ),
                    "response_component_reports": component_reports.get(
                        row_index, []
                    ),
                    "no_op_was_in_schema": bool(
                        allowed_token_ids_by_generation_pos is None
                        or all(
                            old_value
                            in allowed_token_ids_by_generation_pos[generation_position]
                            for old_value, generation_position in zip(old, positions, strict=True)
                        )
                    ),
                }
            )
    if bool((x[:, prompt_length:] == int(mask_id)).any()):
        raise RuntimeError("SPAD correction returned a masked canvas")
    return x, logs


__all__ = ["Model494ResponseConfig", "revise_spad_anchors"]
