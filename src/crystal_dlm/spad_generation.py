"""Stateful suffix-visible correction for SPAD crystal decoding."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from crystal_dlm.llada_generation import (
    _apply_lightweight_decoding_masks,
    _apply_schema_masks,
    _candidate_tokens_and_confidence,
    _model_logits,
    _prepare_atom_count_grammar,
)
from crystal_dlm.spad_program import coordinate_positions


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
) -> tuple[torch.Tensor, list[list[dict[str, Any]]]]:
    """Re-mask each registered anchor once and fill it with full model context.

    Rows may carry different anchor schedules.  Every anchor is masked as one
    transaction, then X/Y/Z are resolved in order.  Non-active values are
    immutable, and the returned log retains the previous triplet as the
    explicit no-op candidate/provisional geometry.
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
            _apply_lightweight_decoding_masks(
                logits,
                x,
                prompt_length,
                gen_length,
                lightweight_decoding_constraints,
            )
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


__all__ = ["revise_spad_anchors"]
