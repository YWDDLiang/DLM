"""Stateful suffix-visible correction for SPAD crystal decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
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
from crystal_dlm.spad_program import LATTICE_POSITIONS, coordinate_positions


SPAD_BASIN_CLOSURE_BLOCK_RADIX = 64
SPAD_BASIN_CLOSURE_SITE_RADIX = 64
SPAD_BASIN_CLOSURE_COMPONENT_RADIX = 3
SPAD_BASIN_CLOSURE_BLOCK_SALT_LIMIT = (
    SPAD_BASIN_CLOSURE_BLOCK_RADIX
    * SPAD_BASIN_CLOSURE_SITE_RADIX
    * SPAD_BASIN_CLOSURE_COMPONENT_RADIX
)


class FixedBatchShapeModelView:
    """Expose one row while preserving the model's deployed batch shape.

    Transformer rows do not attend across the batch, but GPU kernels can make
    borderline sampled logits differ slightly between batch sizes. Repeating
    one state to the original batch size and selecting its original row keeps
    replay/counterfactual continuations numerically aligned with deployment.
    """

    def __init__(self, model: Any, *, batch_size: int, row_index: int) -> None:
        if int(batch_size) <= 0 or not 0 <= int(row_index) < int(batch_size):
            raise ValueError("invalid fixed batch size/row index")
        self.model = model
        self.batch_size = int(batch_size)
        self.row_index = int(row_index)

    def get_output_embeddings(self) -> Any:
        return self.model.get_output_embeddings()

    def __call__(
        self, token_ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None
    ) -> Any:
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1:
            raise ValueError("fixed-batch model view accepts exactly one logical row")
        repeated_ids = token_ids.repeat(self.batch_size, 1)
        repeated_attention = (
            None
            if attention_mask is None
            else attention_mask.repeat(self.batch_size, 1)
        )
        output = self.model(repeated_ids, attention_mask=repeated_attention)
        logits = output.logits
        if int(logits.shape[0]) != self.batch_size:
            raise RuntimeError("underlying model changed fixed replay batch size")
        return SimpleNamespace(logits=logits[self.row_index : self.row_index + 1])


def _spad_basin_closure_block_salt(
    block_index: int,
    site_order_index: int,
    component: int,
) -> int:
    """Encode a closure action in a compact collision-free mixed radix."""

    block = int(block_index)
    site = int(site_order_index)
    axis = int(component)
    if not 0 <= block < SPAD_BASIN_CLOSURE_BLOCK_RADIX:
        raise ValueError("species-block index exceeds closure RNG radix")
    if not 0 <= site < SPAD_BASIN_CLOSURE_SITE_RADIX:
        raise ValueError("site index exceeds closure RNG radix")
    if not 0 <= axis < SPAD_BASIN_CLOSURE_COMPONENT_RADIX:
        raise ValueError("coordinate component exceeds closure RNG radix")
    return (
        (block * SPAD_BASIN_CLOSURE_SITE_RADIX + site)
        * SPAD_BASIN_CLOSURE_COMPONENT_RADIX
        + axis
    )


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


def _transaction_candidate_tokens(
    logits: torch.Tensor,
    *,
    active_absolute_positions: dict[int, int],
    temperature: float,
    remasking: str,
    sampling_seeds_by_batch: Sequence[int] | None,
    salt: int,
) -> torch.Tensor:
    """Sample only active transaction positions with optional row-local RNG.

    The row-local path reproduces the deployed LLaDA Gumbel transform while
    making each proposal independent of batch packing.  The default path is
    unchanged for existing SPAD callers.
    """

    if sampling_seeds_by_batch is None:
        values, _confidence = _candidate_tokens_and_confidence(
            logits,
            float(temperature),
            remasking,
        )
        return values
    if len(sampling_seeds_by_batch) != int(logits.shape[0]):
        raise ValueError("one sampling seed is required per batch row")
    if remasking not in {"low_confidence", "random"}:
        raise NotImplementedError(remasking)
    selected = torch.argmax(logits, dim=-1)
    modulus = 2**63 - 1
    for row_index, absolute_position in active_absolute_positions.items():
        vector = logits[int(row_index), int(absolute_position)]
        if float(temperature) == 0.0:
            token = torch.argmax(vector)
        else:
            generator = torch.Generator(device=vector.device)
            seed = (int(sampling_seeds_by_batch[int(row_index)]) + int(salt)) % modulus
            generator.manual_seed(seed)
            noise = torch.rand(
                vector.shape,
                dtype=torch.float64,
                device=vector.device,
                generator=generator,
            ).clamp_min(torch.finfo(torch.float64).tiny)
            gumbel_denominator = (-torch.log(noise)) ** float(temperature)
            noisy = vector.to(dtype=torch.float64).exp() / gumbel_denominator
            token = torch.argmax(noisy)
        selected[int(row_index), int(absolute_position)] = token.to(
            dtype=selected.dtype
        )
    return selected


def _complete_cell_is_supported(
    row: torch.Tensor,
    *,
    prompt_length: int,
    constraints: dict | None,
) -> bool:
    if not constraints:
        return False
    lattice = _lattice_matrix_from_token_ids(
        row,
        prompt_length=int(prompt_length),
        constraints=constraints,
    )
    if lattice is None or float(torch.det(lattice).detach().item()) <= 1.0e-10:
        return False
    if not constraints.get("pbc_min_distance_mask"):
        return True
    body_offset = int(constraints.get("body_offset", 0))
    count_token = int(row[int(prompt_length) + body_offset].detach().item())
    num_atoms = int(constraints.get("count_token_to_n", {}).get(count_token, 0))
    if num_atoms <= 0:
        return False
    if num_atoms == 1:
        return True
    period = int(constraints.get("coord_period", 100))
    if period <= 0:
        return False
    maps = constraints.get("coord_token_to_bin", {})
    coordinates: list[list[float]] = []
    for slot in range(num_atoms):
        positions = coordinate_positions(slot)
        bins: list[int] = []
        for axis, position in zip(("X", "Y", "Z"), positions, strict=True):
            token = int(row[int(prompt_length) + int(position)].detach().item())
            value = maps.get(axis, {}).get(token)
            if value is None:
                return False
            bins.append(int(value))
        coordinates.append([float(value) / float(period) for value in bins])
    fractional = torch.tensor(
        coordinates,
        dtype=torch.float64,
        device=row.device,
    )
    left, right = torch.triu_indices(num_atoms, num_atoms, offset=1, device=row.device)
    vectors = _minimum_image_vectors(
        fractional[left] - fractional[right],
        lattice,
        image_radius=int(constraints.get("pbc_image_radius", 2)),
    )
    distances = torch.linalg.vector_norm(vectors, dim=1)
    threshold = float(constraints.get("pbc_min_distance_A", 0.5))
    return bool(torch.all(distances >= threshold).detach().item())


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


def _validate_revision_blocks(
    revision_blocks_by_batch: Sequence[Sequence[Sequence[int]]],
    *,
    batch_size: int,
    gen_length: int,
) -> list[list[tuple[int, ...]]]:
    if len(revision_blocks_by_batch) != int(batch_size):
        raise ValueError("one species-block schedule is required per batch row")
    output: list[list[tuple[int, ...]]] = []
    for row_index, raw_blocks in enumerate(revision_blocks_by_batch):
        blocks: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for block_index, raw_slots in enumerate(raw_blocks):
            slots = tuple(int(value) for value in raw_slots)
            if not slots:
                raise ValueError(
                    f"row {row_index} species block {block_index} is empty"
                )
            if len(slots) != len(set(slots)):
                raise ValueError(
                    f"row {row_index} species block {block_index} repeats a site"
                )
            overlap = seen.intersection(slots)
            if overlap:
                raise ValueError(
                    f"row {row_index} revisits sites across species blocks: "
                    f"{sorted(overlap)}"
                )
            for slot in slots:
                if slot < 0 or coordinate_positions(slot)[-1] >= int(gen_length):
                    raise ValueError(
                        f"row {row_index} species-block site lies outside canvas"
                    )
            seen.update(slots)
            blocks.append(slots)
        output.append(blocks)
    return output


@torch.no_grad()
def revise_spad_cell(
    model: Any,
    complete_tokens: torch.Tensor,
    *,
    prompt_length: int,
    gen_length: int,
    attention_mask: torch.Tensor | None,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
    allowed_token_ids_by_generation_pos: list[list[int]] | None,
    atom_count_grammar: dict | None,
    lightweight_decoding_constraints: dict | None,
    strict_geometry_fallback: bool = True,
    sampling_seeds_by_batch: Sequence[int] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Re-mask and regenerate one complete six-token lattice transaction.

    Every site remains visible.  The six lattice values are resolved in native
    order and are committed together.  If the resulting cell makes the full
    crystal leave the existing lattice/PBC support, the complete old cell is
    restored rather than committing a partial or invalid action.
    """

    if complete_tokens.ndim != 2:
        raise ValueError("complete_tokens must have shape [batch, sequence]")
    if complete_tokens.shape[1] != int(prompt_length) + int(gen_length):
        raise ValueError("complete token sequence does not match prompt+generation")
    if sampling_seeds_by_batch is not None and len(sampling_seeds_by_batch) != int(
        complete_tokens.shape[0]
    ):
        raise ValueError("one sampling seed is required per batch row")
    if strict_geometry_fallback and (
        not lightweight_decoding_constraints
        or not lightweight_decoding_constraints.get("pbc_min_distance_mask")
    ):
        raise ValueError("strict cell fallback requires PBC geometry support")
    x = complete_tokens.clone()
    if bool((x[:, prompt_length:] == int(mask_id)).any()):
        raise ValueError("SPAD cell closure requires a complete predictor canvas")
    positions = tuple(int(value) for value in LATTICE_POSITIONS)
    if positions[-1] >= int(gen_length):
        raise ValueError("lattice transaction lies outside generation canvas")
    absolute = torch.tensor(
        [int(prompt_length) + value for value in positions],
        dtype=torch.long,
        device=x.device,
    )
    before = x.clone()
    previous = x.index_select(1, absolute).clone()
    x[:, absolute] = int(mask_id)
    full_attention = _full_attention_mask(
        x,
        attention_mask,
        prompt_length=int(prompt_length),
        gen_length=int(gen_length),
    )
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, : int(prompt_length)] = True
    vocab_size = model.get_output_embeddings().weight.shape[0]
    allowed_mask = None
    if allowed_token_ids_by_generation_pos is not None:
        if len(allowed_token_ids_by_generation_pos) != int(gen_length):
            raise ValueError("allowed token schema length changed")
        allowed_mask = torch.zeros(
            (int(gen_length), int(vocab_size)),
            dtype=torch.bool,
            device=x.device,
        )
        for position, token_ids in enumerate(allowed_token_ids_by_generation_pos):
            if not token_ids:
                raise ValueError(f"generation position {position} has no legal token")
            allowed_mask[
                int(position),
                torch.tensor(token_ids, dtype=torch.long, device=x.device),
            ] = True
    prepared_atom_count_grammar = None
    if atom_count_grammar is not None:
        prepared_atom_count_grammar = _prepare_atom_count_grammar(
            atom_count_grammar,
            int(vocab_size),
            x.device,
        )

    for component, generation_position in enumerate(positions):
        absolute_position = int(prompt_length) + int(generation_position)
        group_allowed = torch.zeros_like(x, dtype=torch.bool)
        group_allowed[:, absolute_position] = True
        logits = _model_logits(
            model,
            x,
            full_attention,
            prompt_index,
            float(cfg_scale),
            int(mask_id),
        )
        if allowed_mask is not None or prepared_atom_count_grammar is not None:
            _apply_schema_masks(
                logits,
                x,
                int(prompt_length),
                int(gen_length),
                allowed_mask,
                prepared_atom_count_grammar,
            )
        _apply_lightweight_decoding_masks(
            logits,
            x,
            int(prompt_length),
            int(gen_length),
            lightweight_decoding_constraints,
            group_allowed[:, int(prompt_length) : int(prompt_length) + int(gen_length)],
            int(mask_id),
        )
        selected = _transaction_candidate_tokens(
            logits,
            active_absolute_positions={
                row_index: absolute_position for row_index in range(int(x.shape[0]))
            },
            temperature=float(temperature),
            remasking=remasking,
            sampling_seeds_by_batch=sampling_seeds_by_batch,
            salt=10_007 * int(component),
        )
        x[group_allowed] = selected[group_allowed]

    logs: list[dict[str, Any]] = []
    for row_index in range(int(x.shape[0])):
        if bool((x[row_index, absolute] == int(mask_id)).any()):
            raise RuntimeError("cell closure left a masked lattice token")
        unchanged = torch.ones_like(x[row_index], dtype=torch.bool)
        unchanged[absolute] = False
        if not bool(torch.equal(x[row_index][unchanged], before[row_index][unchanged])):
            raise RuntimeError("cell closure changed a non-lattice token")
        proposed = tuple(int(value) for value in x[row_index, absolute].tolist())
        old = tuple(int(value) for value in previous[row_index].tolist())
        supported = _complete_cell_is_supported(
            x[row_index],
            prompt_length=int(prompt_length),
            constraints=lightweight_decoding_constraints,
        )
        restored = bool(strict_geometry_fallback and not supported)
        if restored:
            x[row_index, absolute] = previous[row_index]
        logs.append(
            {
                "generation_positions": list(positions),
                "previous_token_ids": list(old),
                "proposed_token_ids": list(proposed),
                "new_token_ids": [
                    int(value) for value in x[row_index, absolute].tolist()
                ],
                "changed_components": sum(
                    left != right
                    for left, right in zip(
                        old,
                        x[row_index, absolute].tolist(),
                        strict=True,
                    )
                ),
                "all_sites_visible": True,
                "geometry_supported_before_restore": bool(supported),
                "restored_complete_noop": bool(restored),
                "guidance_status": (
                    "cell_restored_outside_geometry_support"
                    if restored
                    else "unguided_spad_cell_revision"
                ),
                "no_op_was_in_schema": bool(
                    allowed_token_ids_by_generation_pos is None
                    or all(
                        old_value
                        in allowed_token_ids_by_generation_pos[generation_position]
                        for old_value, generation_position in zip(
                            old,
                            positions,
                            strict=True,
                        )
                    )
                ),
            }
        )
    if bool((x[:, prompt_length:] == int(mask_id)).any()):
        raise RuntimeError("SPAD cell closure returned a masked canvas")
    return x, logs


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
    sampling_seeds_by_batch: Sequence[int] | None = None,
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
                x0 = _transaction_candidate_tokens(
                    logits,
                    active_absolute_positions={
                        int(row_index): int(prompt_length + positions[component])
                        for row_index, positions in active.items()
                        if int(row_index) not in skipped_no_legal
                    },
                    temperature=float(temperature),
                    remasking=remasking,
                    sampling_seeds_by_batch=sampling_seeds_by_batch,
                    salt=1_000_003 * int(revision_index) + 1_009 * int(component),
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


@torch.no_grad()
def revise_spad_species_blocks(
    model: Any,
    complete_tokens: torch.Tensor,
    *,
    prompt_length: int,
    gen_length: int,
    revision_blocks_by_batch: Sequence[Sequence[Sequence[int]]],
    attention_mask: torch.Tensor | None,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
    allowed_token_ids_by_generation_pos: list[list[int]] | None,
    atom_count_grammar: dict | None,
    lightweight_decoding_constraints: dict | None,
    sampling_seeds_by_batch: Sequence[int] | None = None,
    block_index_offset: int = 0,
) -> tuple[torch.Tensor, list[list[dict[str, Any]]]]:
    """Close complete species blocks while preserving full future context.

    Each block starts with every XYZ coordinate in that block masked.  Sites
    are then committed in the supplied order, with X, Y and Z resolved
    sequentially.  The lattice and all non-active species remain immutable and
    visible.  A site whose Z component has no legal PBC completion is restored
    atomically to its previous XYZ without rolling back other sites.
    """

    if complete_tokens.ndim != 2:
        raise ValueError("complete_tokens must have shape [batch, sequence]")
    if complete_tokens.shape[1] != int(prompt_length) + int(gen_length):
        raise ValueError("complete token sequence does not match prompt+generation")
    if sampling_seeds_by_batch is not None and len(sampling_seeds_by_batch) != int(
        complete_tokens.shape[0]
    ):
        raise ValueError("one sampling seed is required per batch row")
    if int(block_index_offset) < 0:
        raise ValueError("block_index_offset must be non-negative")
    x = complete_tokens.clone()
    if bool((x[:, int(prompt_length) :] == int(mask_id)).any()):
        raise ValueError("SPAD block closure requires a complete predictor canvas")
    schedules = _validate_revision_blocks(
        revision_blocks_by_batch,
        batch_size=int(x.shape[0]),
        gen_length=int(gen_length),
    )
    full_attention = _full_attention_mask(
        x,
        attention_mask,
        prompt_length=int(prompt_length),
        gen_length=int(gen_length),
    )
    if full_attention is not None:
        full_attention[
            :, int(prompt_length) : int(prompt_length) + int(gen_length)
        ] = 1
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, : int(prompt_length)] = True
    vocab_size = model.get_output_embeddings().weight.shape[0]
    allowed_mask = None
    if allowed_token_ids_by_generation_pos is not None:
        if len(allowed_token_ids_by_generation_pos) != int(gen_length):
            raise ValueError("allowed token schema length changed")
        allowed_mask = torch.zeros(
            (int(gen_length), int(vocab_size)),
            dtype=torch.bool,
            device=x.device,
        )
        for position, token_ids in enumerate(allowed_token_ids_by_generation_pos):
            if not token_ids:
                raise ValueError(f"generation position {position} has no legal token")
            allowed_mask[
                int(position),
                torch.tensor(token_ids, dtype=torch.long, device=x.device),
            ] = True
    prepared_atom_count_grammar = None
    if atom_count_grammar is not None:
        prepared_atom_count_grammar = _prepare_atom_count_grammar(
            atom_count_grammar,
            int(vocab_size),
            x.device,
        )

    logs: list[list[dict[str, Any]]] = [[] for _ in range(int(x.shape[0]))]
    max_blocks = max((len(blocks) for blocks in schedules), default=0)
    for block_index in range(max_blocks):
        global_block_index = int(block_index_offset) + int(block_index)
        before = x.clone()
        active_blocks: dict[int, tuple[int, ...]] = {}
        previous_by_row: dict[int, dict[int, tuple[int, int, int]]] = {}
        block_absolute_by_row: dict[int, tuple[int, ...]] = {}
        for row_index, blocks in enumerate(schedules):
            if block_index >= len(blocks):
                continue
            slots = blocks[block_index]
            active_blocks[row_index] = slots
            previous_by_row[row_index] = {}
            block_absolute: list[int] = []
            for slot in slots:
                positions = coordinate_positions(slot)
                absolute = tuple(
                    int(prompt_length) + int(position) for position in positions
                )
                previous = tuple(
                    int(x[row_index, position].detach().item()) for position in absolute
                )
                if any(value == int(mask_id) for value in previous):
                    raise RuntimeError("species-block site was not committed")
                previous_by_row[row_index][slot] = previous
                block_absolute.extend(absolute)
            absolute_tuple = tuple(block_absolute)
            block_absolute_by_row[row_index] = absolute_tuple
            x[
                row_index,
                torch.tensor(absolute_tuple, dtype=torch.long, device=x.device),
            ] = int(mask_id)

        initially_masked_by_row = {
            row_index: bool(
                torch.all(
                    x[
                        row_index,
                        torch.tensor(absolute, dtype=torch.long, device=x.device),
                    ]
                    == int(mask_id)
                ).detach().item()
            )
            for row_index, absolute in block_absolute_by_row.items()
        }
        site_logs_by_row: dict[int, list[dict[str, Any]]] = {
            row_index: [] for row_index in active_blocks
        }
        max_sites = max((len(slots) for slots in active_blocks.values()), default=0)
        for site_order_index in range(max_sites):
            active_sites = {
                row_index: slots[site_order_index]
                for row_index, slots in active_blocks.items()
                if site_order_index < len(slots)
            }
            restored_rows: set[int] = set()
            for component in range(3):
                group_allowed = torch.zeros_like(x, dtype=torch.bool)
                active_absolute_positions: dict[int, int] = {}
                for row_index, slot in active_sites.items():
                    position = coordinate_positions(slot)[component]
                    absolute_position = int(prompt_length) + int(position)
                    group_allowed[row_index, absolute_position] = True
                    active_absolute_positions[row_index] = absolute_position
                if not active_absolute_positions:
                    continue
                logits = _model_logits(
                    model,
                    x,
                    full_attention,
                    prompt_index,
                    float(cfg_scale),
                    int(mask_id),
                )
                if allowed_mask is not None or prepared_atom_count_grammar is not None:
                    _apply_schema_masks(
                        logits,
                        x,
                        int(prompt_length),
                        int(gen_length),
                        allowed_mask,
                        prepared_atom_count_grammar,
                    )
                mask_report = _apply_lightweight_decoding_masks(
                    logits,
                    x,
                    int(prompt_length),
                    int(gen_length),
                    lightweight_decoding_constraints,
                    group_allowed[
                        :, int(prompt_length) : int(prompt_length) + int(gen_length)
                    ],
                    int(mask_id),
                )
                if int(component) == 2:
                    no_legal = mask_report.get("pbc_no_legal_completion", set())
                    for row_index, slot in active_sites.items():
                        z_position = coordinate_positions(slot)[2]
                        if (int(row_index), int(z_position)) not in no_legal:
                            continue
                        positions = coordinate_positions(slot)
                        absolute = tuple(
                            int(prompt_length) + int(position) for position in positions
                        )
                        x[
                            row_index,
                            torch.tensor(absolute, dtype=torch.long, device=x.device),
                        ] = torch.tensor(
                            previous_by_row[row_index][slot],
                            dtype=x.dtype,
                            device=x.device,
                        )
                        group_allowed[row_index, int(prompt_length) + z_position] = False
                        active_absolute_positions.pop(row_index, None)
                        restored_rows.add(int(row_index))
                selected = _transaction_candidate_tokens(
                    logits,
                    active_absolute_positions=active_absolute_positions,
                    temperature=float(temperature),
                    remasking=remasking,
                    sampling_seeds_by_batch=sampling_seeds_by_batch,
                    salt=_spad_basin_closure_block_salt(
                        global_block_index,
                        int(site_order_index),
                        int(component),
                    ),
                )
                x[group_allowed] = selected[group_allowed]

            for row_index, slot in active_sites.items():
                positions = coordinate_positions(slot)
                absolute = tuple(
                    int(prompt_length) + int(position) for position in positions
                )
                if any(
                    int(x[row_index, position].detach().item()) == int(mask_id)
                    for position in absolute
                ):
                    raise RuntimeError("species-block closure left a masked coordinate")
                old = previous_by_row[row_index][slot]
                new = tuple(
                    int(x[row_index, position].detach().item()) for position in absolute
                )
                site_logs_by_row[row_index].append(
                    {
                        "block_index": global_block_index,
                        "site_order_index": int(site_order_index),
                        "slot_index": int(slot),
                        "generation_positions": list(positions),
                        "previous_token_ids": list(old),
                        "new_token_ids": list(new),
                        "changed_components": sum(
                            left != right
                            for left, right in zip(old, new, strict=True)
                        ),
                        "restored_site_no_legal_z": bool(
                            int(row_index) in restored_rows
                        ),
                        "suffix_visible": True,
                        "no_op_was_in_schema": bool(
                            allowed_token_ids_by_generation_pos is None
                            or all(
                                old_value
                                in allowed_token_ids_by_generation_pos[position]
                                for old_value, position in zip(
                                    old, positions, strict=True
                                )
                            )
                        ),
                    }
                )

        for row_index, slots in active_blocks.items():
            absolute = block_absolute_by_row[row_index]
            absolute_tensor = torch.tensor(
                absolute, dtype=torch.long, device=x.device
            )
            if bool((x[row_index, absolute_tensor] == int(mask_id)).any()):
                raise RuntimeError("species-block closure returned a masked block")
            proposed_flat = [
                int(value) for value in x[row_index, absolute_tensor].tolist()
            ]
            geometry_check_enabled = bool(
                lightweight_decoding_constraints
                and lightweight_decoding_constraints.get("pbc_min_distance_mask")
            )
            geometry_supported_before_restore: bool | None = None
            restored_complete_block = False
            if geometry_check_enabled:
                geometry_supported_before_restore = _complete_cell_is_supported(
                    x[row_index],
                    prompt_length=int(prompt_length),
                    constraints=lightweight_decoding_constraints,
                )
                restored_complete_block = not geometry_supported_before_restore
                if restored_complete_block:
                    x[row_index, absolute_tensor] = before[row_index, absolute_tensor]
            unchanged = torch.ones_like(x[row_index], dtype=torch.bool)
            unchanged[absolute_tensor] = False
            if not bool(
                torch.equal(x[row_index][unchanged], before[row_index][unchanged])
            ):
                raise RuntimeError("species-block closure changed a non-active token")
            previous_flat = [
                token
                for slot in slots
                for token in previous_by_row[row_index][slot]
            ]
            new_flat = [int(value) for value in x[row_index, absolute_tensor].tolist()]
            logs[row_index].append(
                {
                    "block_index": global_block_index,
                    "slot_indices": list(slots),
                    "generation_positions": [
                        position
                        for slot in slots
                        for position in coordinate_positions(slot)
                    ],
                    "previous_token_ids": previous_flat,
                    "proposed_token_ids": proposed_flat,
                    "new_token_ids": new_flat,
                    "changed_components": sum(
                        left != right
                        for left, right in zip(
                            previous_flat, new_flat, strict=True
                        )
                    ),
                    "all_block_sites_masked_initially": bool(
                        initially_masked_by_row[row_index]
                    ),
                    "suffix_visible": True,
                    "non_active_tokens_unchanged": True,
                    "geometry_supported_before_restore": (
                        geometry_supported_before_restore
                    ),
                    "restored_complete_block": bool(restored_complete_block),
                    "restored_site_count": sum(
                        bool(site["restored_site_no_legal_z"])
                        for site in site_logs_by_row[row_index]
                    ),
                    "site_revisions": site_logs_by_row[row_index],
                }
            )
    if bool((x[:, int(prompt_length) :] == int(mask_id)).any()):
        raise RuntimeError("SPAD species-block closure returned a masked canvas")
    for row_index in range(int(x.shape[0])):
        final_geometry_supported = _complete_cell_is_supported(
            x[row_index],
            prompt_length=int(prompt_length),
            constraints=lightweight_decoding_constraints,
        )
        if not logs[row_index]:
            raise RuntimeError("species-block closure requires at least one block")
        logs[row_index][-1]["final_geometry_supported"] = bool(
            final_geometry_supported
        )
    return x, logs


@torch.no_grad()
def continue_spad_species_blocks_from_cursor(
    model: Any,
    state_tokens: torch.Tensor,
    *,
    block_entry_tokens: torch.Tensor,
    prompt_length: int,
    gen_length: int,
    revision_blocks: Sequence[Sequence[int]],
    block_index: int,
    site_order_index: int,
    action_token_ids: Sequence[int],
    attention_mask: torch.Tensor | None,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
    allowed_token_ids_by_generation_pos: list[list[int]] | None,
    atom_count_grammar: dict | None,
    lightweight_decoding_constraints: dict | None,
    sampling_seed: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Resume one real closure trajectory after injecting an XYZ action.

    ``state_tokens`` is the suffix-visible state at the start of the selected
    site: earlier sites in the current species block are committed, while the
    active site and the block remainder are masked. ``block_entry_tokens`` is
    the complete canvas immediately before that whole block was masked. The
    active XYZ transaction is supplied atomically; later sites use the exact
    deployed component-wise sampler and original global RNG salts. If the
    completed current block leaves geometric support, the *whole* block is
    restored to its entry snapshot before subsequent blocks execute.

    This API intentionally handles one row. Preflight candidates have
    different cursors, and keeping each continuation isolated makes state and
    RNG identity explicit while avoiding padded cross-row control flow.
    """

    if state_tokens.ndim != 2 or tuple(state_tokens.shape[:1]) != (1,):
        raise ValueError("cursor continuation requires state_tokens [1, sequence]")
    if block_entry_tokens.shape != state_tokens.shape:
        raise ValueError("block_entry_tokens must match state_tokens")
    if state_tokens.shape[1] != int(prompt_length) + int(gen_length):
        raise ValueError("cursor state does not match prompt+generation length")
    schedules = _validate_revision_blocks(
        [revision_blocks], batch_size=1, gen_length=int(gen_length)
    )[0]
    block_id = int(block_index)
    site_id = int(site_order_index)
    if not 0 <= block_id < len(schedules):
        raise IndexError("block cursor lies outside revision schedule")
    slots = schedules[block_id]
    if not 0 <= site_id < len(slots):
        raise IndexError("site cursor lies outside species block")
    action = tuple(int(value) for value in action_token_ids)
    if len(action) != 3 or any(value == int(mask_id) for value in action):
        raise ValueError("cursor action must contain three committed token ids")

    x = state_tokens.clone()
    entry = block_entry_tokens.clone()
    active_slot = int(slots[site_id])
    active_positions = coordinate_positions(active_slot)
    active_absolute = tuple(int(prompt_length) + value for value in active_positions)
    if not bool(
        (x[0, torch.tensor(active_absolute, device=x.device)] == int(mask_id)).all()
    ):
        raise ValueError("active XYZ transaction must be masked in cursor state")
    x[0, torch.tensor(active_absolute, device=x.device)] = torch.tensor(
        action, dtype=x.dtype, device=x.device
    )

    future_slots = tuple(int(value) for value in slots[site_id + 1 :])
    future_absolute = tuple(
        int(prompt_length) + position
        for slot in future_slots
        for position in coordinate_positions(slot)
    )
    if future_absolute and not bool(
        (state_tokens[0, torch.tensor(future_absolute, device=x.device)] == int(mask_id)).all()
    ):
        raise ValueError("all future sites in the current block must stay masked")

    full_attention = _full_attention_mask(
        x,
        attention_mask,
        prompt_length=int(prompt_length),
        gen_length=int(gen_length),
    )
    if full_attention is not None:
        full_attention[:, int(prompt_length) : int(prompt_length) + int(gen_length)] = 1
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, : int(prompt_length)] = True
    vocab_size = int(model.get_output_embeddings().weight.shape[0])
    allowed_mask = None
    if allowed_token_ids_by_generation_pos is not None:
        if len(allowed_token_ids_by_generation_pos) != int(gen_length):
            raise ValueError("allowed token schema length changed")
        allowed_mask = torch.zeros(
            (int(gen_length), vocab_size), dtype=torch.bool, device=x.device
        )
        for position, token_ids in enumerate(allowed_token_ids_by_generation_pos):
            if not token_ids:
                raise ValueError(f"generation position {position} has no legal token")
            allowed_mask[position, torch.tensor(token_ids, device=x.device)] = True
    prepared_atom_count_grammar = None
    if atom_count_grammar is not None:
        prepared_atom_count_grammar = _prepare_atom_count_grammar(
            atom_count_grammar, vocab_size, x.device
        )

    future_logs: list[dict[str, Any]] = []
    for original_site_order, slot in enumerate(
        future_slots, start=site_id + 1
    ):
        positions = coordinate_positions(slot)
        absolute = tuple(int(prompt_length) + value for value in positions)
        previous = tuple(int(entry[0, value].item()) for value in absolute)
        restored = False
        for component in range(3):
            generation_position = int(positions[component])
            absolute_position = int(prompt_length) + generation_position
            group_allowed = torch.zeros_like(x, dtype=torch.bool)
            group_allowed[0, absolute_position] = True
            logits = _model_logits(
                model, x, full_attention, prompt_index, float(cfg_scale), int(mask_id)
            )
            if allowed_mask is not None or prepared_atom_count_grammar is not None:
                _apply_schema_masks(
                    logits,
                    x,
                    int(prompt_length),
                    int(gen_length),
                    allowed_mask,
                    prepared_atom_count_grammar,
                )
            mask_report = _apply_lightweight_decoding_masks(
                logits,
                x,
                int(prompt_length),
                int(gen_length),
                lightweight_decoding_constraints,
                group_allowed[:, int(prompt_length) : int(prompt_length) + int(gen_length)],
                int(mask_id),
            )
            if component == 2 and (0, generation_position) in mask_report.get(
                "pbc_no_legal_completion", set()
            ):
                x[0, torch.tensor(absolute, device=x.device)] = torch.tensor(
                    previous, dtype=x.dtype, device=x.device
                )
                restored = True
                break
            selected = _transaction_candidate_tokens(
                logits,
                active_absolute_positions={0: absolute_position},
                temperature=float(temperature),
                remasking=remasking,
                sampling_seeds_by_batch=(
                    None if sampling_seed is None else [int(sampling_seed)]
                ),
                salt=_spad_basin_closure_block_salt(
                    block_id, int(original_site_order), int(component)
                ),
            )
            x[group_allowed] = selected[group_allowed]
        if bool((x[0, torch.tensor(absolute, device=x.device)] == int(mask_id)).any()):
            raise RuntimeError("cursor continuation left a masked future site")
        future_logs.append(
            {
                "block_index": block_id,
                "site_order_index": int(original_site_order),
                "slot_index": int(slot),
                "generation_positions": list(positions),
                "previous_token_ids": list(previous),
                "new_token_ids": [int(x[0, value].item()) for value in absolute],
                "restored_site_no_legal_z": bool(restored),
            }
        )

    block_positions = tuple(
        int(prompt_length) + position
        for slot in slots
        for position in coordinate_positions(slot)
    )
    geometry_check_enabled = bool(
        lightweight_decoding_constraints
        and lightweight_decoding_constraints.get("pbc_min_distance_mask")
    )
    supported_before_restore: bool | None = None
    restored_block = False
    if geometry_check_enabled:
        supported_before_restore = _complete_cell_is_supported(
            x[0],
            prompt_length=int(prompt_length),
            constraints=lightweight_decoding_constraints,
        )
        restored_block = not supported_before_restore
        if restored_block:
            block_tensor = torch.tensor(block_positions, device=x.device)
            x[0, block_tensor] = entry[0, block_tensor]

    later_logs: list[dict[str, Any]] = []
    if block_id + 1 < len(schedules):
        x, batched_logs = revise_spad_species_blocks(
            model,
            x,
            prompt_length=int(prompt_length),
            gen_length=int(gen_length),
            revision_blocks_by_batch=[[list(value) for value in schedules[block_id + 1 :]]],
            attention_mask=attention_mask,
            temperature=float(temperature),
            cfg_scale=float(cfg_scale),
            remasking=remasking,
            mask_id=int(mask_id),
            allowed_token_ids_by_generation_pos=allowed_token_ids_by_generation_pos,
            atom_count_grammar=atom_count_grammar,
            lightweight_decoding_constraints=lightweight_decoding_constraints,
            sampling_seeds_by_batch=(
                None if sampling_seed is None else [int(sampling_seed)]
            ),
            block_index_offset=block_id + 1,
        )
        later_logs = batched_logs[0]
    if bool((x[:, int(prompt_length) :] == int(mask_id)).any()):
        raise RuntimeError("cursor continuation returned a masked canvas")
    return x, {
        "schema": "spad_basin_cursor_continuation_v1",
        "block_index": block_id,
        "site_order_index": site_id,
        "slot_index": active_slot,
        "action_token_ids": list(action),
        "future_site_revisions": future_logs,
        "geometry_supported_before_restore": supported_before_restore,
        "restored_complete_block": bool(restored_block),
        "later_block_revisions": later_logs,
    }


__all__ = [
    "FixedBatchShapeModelView",
    "Model494ResponseConfig",
    "SPAD_BASIN_CLOSURE_BLOCK_SALT_LIMIT",
    "_spad_basin_closure_block_salt",
    "continue_spad_species_blocks_from_cursor",
    "revise_spad_anchors",
    "revise_spad_cell",
    "revise_spad_species_blocks",
]
