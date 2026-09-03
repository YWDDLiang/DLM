"""Minimal LLaDA generation utilities adapted for crystal sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from crystal_dlm.generation_schedule import n_elements_coords_lattice_schedule
from crystal_dlm.lattice_geometry import lattice_angle_rad
from crystal_dlm.periodic_geometry_ops import minimum_image_distances


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def _validate_generation_position_groups(
    generation_position_groups: list[list[int]],
    gen_length: int,
) -> list[list[int]]:
    normalized: list[list[int]] = []
    seen: set[int] = set()
    for group_idx, group in enumerate(generation_position_groups):
        if not group:
            raise ValueError(f"Generation schedule group {group_idx} is empty")
        normalized_group: list[int] = []
        for position in group:
            position = int(position)
            if not 0 <= position < gen_length:
                raise ValueError(f"Generation position {position} outside 0..{gen_length - 1}")
            if position in seen:
                raise ValueError(f"Generation position {position} appears in multiple schedule groups")
            seen.add(position)
            normalized_group.append(position)
        normalized.append(normalized_group)
    missing = sorted(set(range(int(gen_length))) - seen)
    if missing:
        preview = ",".join(str(value) for value in missing[:12])
        raise ValueError(
            "Generation schedule does not cover every position; "
            f"missing {len(missing)} positions ({preview})"
        )
    return normalized


def _validate_generation_position_groups_by_batch(
    generation_position_groups_by_batch: list[list[list[int]]],
    *,
    batch_size: int,
    gen_length: int,
) -> list[list[list[int]]]:
    """Validate one exact position schedule per batch row.

    SPAD programs can activate different native sites for equal-length samples.
    Keeping the schedules row-local preserves batching without physically
    permuting the canonical ``7 + 4N`` canvas.
    """

    if len(generation_position_groups_by_batch) != int(batch_size):
        raise ValueError(
            "generation_position_groups_by_batch must contain one schedule "
            "per batch row"
        )
    return [
        _validate_generation_position_groups(groups, int(gen_length))
        for groups in generation_position_groups_by_batch
    ]


def _build_token_mask(token_ids: list[int], vocab_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((vocab_size,), dtype=torch.bool, device=device)
    if token_ids:
        mask[torch.tensor(token_ids, dtype=torch.long, device=device)] = True
    return mask


def _prepare_atom_count_grammar(
    atom_count_grammar: dict | None,
    vocab_size: int,
    device: torch.device,
) -> dict | None:
    if atom_count_grammar is None:
        return None
    representation = str(atom_count_grammar.get("representation", "fixed_slot"))
    coord_token_ids = atom_count_grammar["coord_token_ids"]
    prepared = {
        "representation": representation,
        "body_offset": int(atom_count_grammar.get("body_offset", 0)),
        "max_atoms": int(atom_count_grammar.get("max_atoms", 20)),
        "count_token_to_n": {
            int(token_id): int(num_atoms)
            for token_id, num_atoms in atom_count_grammar["count_token_to_n"].items()
        },
        "element_mask": _build_token_mask(atom_count_grammar["element_token_ids"], vocab_size, device),
        "coord_masks": {
            axis: _build_token_mask([int(item) for item in ids], vocab_size, device)
            for axis, ids in coord_token_ids.items()
        },
    }
    if representation == "dynamic_v1":
        prepared["eos_mask"] = _build_token_mask(
            [int(atom_count_grammar["eos_token_id"])],
            vocab_size,
            device,
        )
    else:
        pad_coord_token_ids = atom_count_grammar.get("pad_coord_token_ids", {})
        prepared["empty_mask"] = _build_token_mask(
            [int(atom_count_grammar["empty_token_id"])],
            vocab_size,
            device,
        )
        prepared["pad_coord_masks"] = {
            axis: _build_token_mask([int(token_id)], vocab_size, device)
            for axis, token_id in pad_coord_token_ids.items()
        }
    return prepared


def _apply_atom_count_grammar_mask(
    generation_logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    grammar: dict | None,
) -> None:
    if grammar is None:
        return
    min_value = torch.finfo(generation_logits.dtype).min
    max_atoms = int(grammar["max_atoms"])
    body_offset = int(grammar.get("body_offset", 0))
    count_token_to_n = grammar["count_token_to_n"]
    axes = ("X", "Y", "Z")
    if grammar.get("representation") == "dynamic_v1":
        for batch_idx in range(generation_logits.shape[0]):
            count_token_id = int(x[batch_idx, prompt_length + body_offset].detach().item())
            num_atoms = count_token_to_n.get(count_token_id)
            if num_atoms is None:
                continue
            for slot_index in range(max_atoms):
                base = body_offset + 7 + slot_index * 4
                if base + 3 >= generation_logits.shape[1]:
                    break
                if slot_index < num_atoms:
                    generation_logits[batch_idx, base].masked_fill_(~grammar["element_mask"], min_value)
                    for offset, axis in enumerate(axes, start=1):
                        generation_logits[batch_idx, base + offset].masked_fill_(
                            ~grammar["coord_masks"][axis],
                            min_value,
                        )
                else:
                    for offset in range(4):
                        generation_logits[batch_idx, base + offset].masked_fill_(
                            ~grammar["eos_mask"],
                            min_value,
                        )
        return
    for batch_idx in range(generation_logits.shape[0]):
        count_token_id = int(x[batch_idx, prompt_length + body_offset].detach().item())
        num_atoms = count_token_to_n.get(count_token_id)
        if num_atoms is None:
            continue
        for slot_index in range(max_atoms):
            base = body_offset + 7 + slot_index * 5
            if base + 4 >= generation_logits.shape[1]:
                break
            if slot_index < num_atoms:
                generation_logits[batch_idx, base + 1].masked_fill_(~grammar["element_mask"], min_value)
                for offset, axis in enumerate(axes, start=2):
                    generation_logits[batch_idx, base + offset].masked_fill_(
                        ~grammar["coord_masks"][axis],
                        min_value,
                    )
            else:
                generation_logits[batch_idx, base + 1].masked_fill_(~grammar["empty_mask"], min_value)
                for offset, axis in enumerate(axes, start=2):
                    generation_logits[batch_idx, base + offset].masked_fill_(
                        ~grammar["pad_coord_masks"][axis],
                        min_value,
                    )


def _model_logits(
    model,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_index: torch.Tensor,
    cfg_scale: float,
    mask_id: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        model_input = torch.cat([x, un_x], dim=0)
        if attention_mask is None:
            attention_mask_input = None
        else:
            attention_mask_input = torch.cat([attention_mask, attention_mask], dim=0)
        logits = model(model_input, attention_mask=attention_mask_input).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        return un_logits + (cfg_scale + 1.0) * (logits - un_logits)
    return model(x, attention_mask=attention_mask).logits


def _apply_schema_masks(
    logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    gen_length: int,
    allowed_mask: torch.Tensor | None,
    prepared_atom_count_grammar: dict | None,
) -> None:
    generation_logits = logits[:, prompt_length : prompt_length + gen_length, :]
    if allowed_mask is not None:
        generation_logits.masked_fill_(~allowed_mask.unsqueeze(0), torch.finfo(logits.dtype).min)
    _apply_atom_count_grammar_mask(
        generation_logits,
        x,
        prompt_length,
        prepared_atom_count_grammar,
    )


def _mask_zero_lengths(
    generation_logits: torch.Tensor,
    constraints: dict,
    min_value: float,
) -> None:
    zero_length_token_ids_by_position = constraints.get("zero_length_token_ids_by_position", {})
    for rel_pos, token_id in zero_length_token_ids_by_position.items():
        rel_pos = int(rel_pos)
        if 0 <= rel_pos < generation_logits.shape[1]:
            generation_logits[:, rel_pos, int(token_id)] = min_value


def _apply_lattice_volume_mask(
    generation_logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    constraints: dict,
    min_value: float,
) -> None:
    angle_token_to_bin = constraints.get("angle_token_to_bin", {})
    gamma_bin_to_token_id = constraints.get("gamma_bin_to_token_id", {})
    min_lattice_rad = float(constraints.get("min_lattice_rad", 1e-4))
    body_offset = int(constraints.get("body_offset", 0))
    if body_offset + 6 >= generation_logits.shape[1]:
        return
    alpha_map = angle_token_to_bin.get("AA", {})
    beta_map = angle_token_to_bin.get("AB", {})
    for batch_idx in range(generation_logits.shape[0]):
        alpha_token = int(x[batch_idx, prompt_length + body_offset + 4].detach().item())
        beta_token = int(x[batch_idx, prompt_length + body_offset + 5].detach().item())
        alpha = alpha_map.get(alpha_token)
        beta = beta_map.get(beta_token)
        if alpha is None or beta is None:
            continue
        legal_count = 0
        for gamma, token_id in gamma_bin_to_token_id.items():
            if lattice_angle_rad(int(alpha), int(beta), int(gamma)) <= min_lattice_rad:
                generation_logits[batch_idx, body_offset + 6, int(token_id)] = min_value
            else:
                legal_count += 1
        if legal_count == 0:
            for token_id in gamma_bin_to_token_id.values():
                generation_logits[batch_idx, body_offset + 6, int(token_id)] = min_value


def _apply_duplicate_coordinate_mask(
    generation_logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    constraints: dict,
    min_value: float,
) -> None:
    coord_token_to_bin = constraints.get("coord_token_to_bin", {})
    z_bin_to_token_id = constraints.get("z_bin_to_token_id", {})
    count_token_to_n = constraints.get("count_token_to_n", {})
    x_token_to_bin = coord_token_to_bin.get("X", {})
    y_token_to_bin = coord_token_to_bin.get("Y", {})
    z_token_to_bin = coord_token_to_bin.get("Z", {})
    max_atoms = int(constraints.get("max_atoms", 20))
    coord_period = constraints.get("coord_period")
    body_offset = int(constraints.get("body_offset", 0))

    def coord_key(value: int) -> int:
        if coord_period is None:
            return int(value)
        period = int(coord_period)
        if period <= 0:
            return int(value)
        return int(value) % period

    representation = str(constraints.get("representation", "fixed_slot"))
    for batch_idx in range(generation_logits.shape[0]):
        count_token_id = int(x[batch_idx, prompt_length + body_offset].detach().item())
        num_atoms = int(count_token_to_n.get(count_token_id, 0))
        if num_atoms <= 0:
            continue
        num_atoms = min(num_atoms, max_atoms)
        slots: list[tuple[int, tuple[int, int] | None, int | None]] = []
        for slot_index in range(num_atoms):
            if representation == "dynamic_v1":
                base = body_offset + 7 + slot_index * 4
                x_offset, y_offset, z_offset = 1, 2, 3
            else:
                base = body_offset + 7 + slot_index * 5
                x_offset, y_offset, z_offset = 2, 3, 4
            if base + z_offset >= generation_logits.shape[1]:
                break
            x_token = int(x[batch_idx, prompt_length + base + x_offset].detach().item())
            y_token = int(x[batch_idx, prompt_length + base + y_offset].detach().item())
            z_token = int(x[batch_idx, prompt_length + base + z_offset].detach().item())
            x_bin = x_token_to_bin.get(x_token)
            y_bin = y_token_to_bin.get(y_token)
            if x_bin is None or y_bin is None:
                slots.append((base + z_offset, None, None))
                continue
            xy = (coord_key(int(x_bin)), coord_key(int(y_bin)))
            z_bin = z_token_to_bin.get(z_token)
            slots.append((base + z_offset, xy, None if z_bin is None else coord_key(int(z_bin))))

        seen_by_xy: dict[tuple[int, int], set[int]] = {}
        for _z_position, xy, z_bin in slots:
            if xy is None or z_bin is None:
                continue
            seen_by_xy.setdefault(xy, set()).add(z_bin)

        for z_position, xy, z_bin in slots:
            if xy is None:
                continue
            banned_z_bins = set(seen_by_xy.get(xy, set()))
            if z_bin is not None:
                banned_z_bins.discard(z_bin)
            if not banned_z_bins:
                continue
            for candidate_z_bin, token_id in z_bin_to_token_id.items():
                if coord_key(int(candidate_z_bin)) in banned_z_bins:
                    generation_logits[batch_idx, z_position, int(token_id)] = min_value


def _lattice_matrix_from_token_ids(
    x: torch.Tensor,
    *,
    prompt_length: int,
    constraints: dict,
) -> torch.Tensor | None:
    body_offset = int(constraints.get("body_offset", 0))
    length_maps = constraints.get("length_token_to_bin", {})
    angle_maps = constraints.get("angle_token_to_bin", {})
    length_step = float(constraints.get("length_step", 0.1))
    length_values: list[float] = []
    for offset, prefix in zip((1, 2, 3), ("LA", "LB", "LC"), strict=True):
        token = int(x[prompt_length + body_offset + offset].detach().item())
        value = length_maps.get(prefix, {}).get(token)
        if value is None:
            return None
        length_values.append(float(value) * length_step)
    angle_values: list[float] = []
    for offset, prefix in zip((4, 5, 6), ("AA", "AB", "AG"), strict=True):
        token = int(x[prompt_length + body_offset + offset].detach().item())
        value = angle_maps.get(prefix, {}).get(token)
        if value is None:
            return None
        angle_values.append(float(value))
    a, b, c = length_values
    if min(a, b, c) <= 0.0:
        return None
    alpha, beta, gamma = [
        torch.deg2rad(torch.tensor(value, dtype=torch.float64, device=x.device))
        for value in angle_values
    ]
    sin_gamma = torch.sin(gamma)
    if float(torch.abs(sin_gamma).detach().item()) < 1.0e-8:
        return None
    c_x = c * torch.cos(beta)
    c_y = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / sin_gamma
    c_z_squared = c * c - c_x * c_x - c_y * c_y
    if float(c_z_squared.detach().item()) <= 1.0e-10:
        return None
    zero = torch.zeros((), dtype=torch.float64, device=x.device)
    return torch.stack(
        [
            torch.stack((torch.as_tensor(a, dtype=torch.float64, device=x.device), zero, zero)),
            torch.stack((torch.as_tensor(b, dtype=torch.float64, device=x.device) * torch.cos(gamma), torch.as_tensor(b, dtype=torch.float64, device=x.device) * sin_gamma, zero)),
            torch.stack((c_x, c_y, torch.sqrt(c_z_squared))),
        ]
    )


def _aggregate_periodic_coordinate_alias_logits(
    generation_logits: torch.Tensor,
    constraints: dict,
    min_value: float,
    active_generation_mask: torch.Tensor | None,
) -> None:
    aliases = constraints.get("coordinate_alias_token_ids", {})
    body_offset = int(constraints.get("body_offset", 0))
    max_atoms = int(constraints.get("max_atoms", 20))
    for slot in range(max_atoms):
        base = body_offset + 7 + 4 * slot
        for component, axis in enumerate(("X", "Y", "Z"), start=1):
            position = base + component
            if position >= generation_logits.shape[1] or axis not in aliases:
                continue
            canonical_id, alias_id = (int(value) for value in aliases[axis])
            for row in range(generation_logits.shape[0]):
                if active_generation_mask is not None and not bool(
                    active_generation_mask[row, position].detach().item()
                ):
                    continue
                canonical = generation_logits[row, position, canonical_id].clone()
                alias = generation_logits[row, position, alias_id].clone()
                generation_logits[row, position, canonical_id] = torch.logaddexp(
                    canonical, alias
                )
                generation_logits[row, position, alias_id] = min_value


def _apply_pbc_min_distance_mask(
    generation_logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    constraints: dict,
    min_value: float,
    active_generation_mask: torch.Tensor | None,
    mask_id: int,
) -> set[tuple[int, int]]:
    no_legal_completion: set[tuple[int, int]] = set()
    coord_maps = constraints.get("coord_token_to_bin", {})
    coord_ids = constraints.get("coord_bin_to_token_id", {})
    count_token_to_n = constraints.get("count_token_to_n", {})
    threshold = float(constraints.get("pbc_min_distance_A", 0.5))
    image_radius = int(constraints.get("pbc_image_radius", 2))
    body_offset = int(constraints.get("body_offset", 0))
    period = int(constraints.get("coord_period", 100))
    if period <= 0:
        raise ValueError("coordinate period must be positive")
    for row in range(generation_logits.shape[0]):
        count_token = int(x[row, prompt_length + body_offset].detach().item())
        num_atoms = int(count_token_to_n.get(count_token, 0))
        if num_atoms <= 1:
            continue
        lattice = _lattice_matrix_from_token_ids(
            x[row], prompt_length=prompt_length, constraints=constraints
        )
        if lattice is None:
            continue
        for slot in range(num_atoms):
            z_position = body_offset + 10 + 4 * slot
            if z_position >= generation_logits.shape[1]:
                break
            if active_generation_mask is not None and not bool(
                active_generation_mask[row, z_position].detach().item()
            ):
                continue
            if int(x[row, prompt_length + z_position].detach().item()) != int(mask_id):
                continue
            x_position = body_offset + 8 + 4 * slot
            y_position = body_offset + 9 + 4 * slot
            x_bin = coord_maps.get("X", {}).get(
                int(x[row, prompt_length + x_position].detach().item())
            )
            y_bin = coord_maps.get("Y", {}).get(
                int(x[row, prompt_length + y_position].detach().item())
            )
            if x_bin is None or y_bin is None:
                continue
            other: list[list[float]] = []
            for prior in range(num_atoms):
                if prior == slot:
                    continue
                base = body_offset + 8 + 4 * prior
                bins = [
                    coord_maps[axis].get(
                        int(x[row, prompt_length + base + axis_index].detach().item())
                    )
                    for axis_index, axis in enumerate(("X", "Y", "Z"))
                ]
                if any(value is None for value in bins):
                    continue
                other.append([float(value) / period for value in bins])
            if not other:
                continue
            candidate_bins = sorted(int(value) for value in coord_ids.get("Z", {}))
            candidates = torch.tensor(
                [
                    [float(x_bin) / period, float(y_bin) / period, float(value) / period]
                    for value in candidate_bins
                ],
                dtype=torch.float64,
                device=x.device,
            )
            existing = torch.tensor(other, dtype=torch.float64, device=x.device)
            deltas = candidates.unsqueeze(1) - existing.unsqueeze(0)
            distances = minimum_image_distances(
                deltas, lattice, image_radius=image_radius
            )
            legal = distances.min(dim=1).values >= threshold
            if not bool(legal.any().item()):
                # Do not create an empty action set; the caller can record the
                # unresolved risk and retain the model distribution.
                no_legal_completion.add((int(row), int(z_position)))
                continue
            for candidate_bin, allowed in zip(candidate_bins, legal, strict=True):
                if not bool(allowed.detach().item()):
                    token_id = int(coord_ids["Z"][candidate_bin])
                    generation_logits[row, z_position, token_id] = min_value
    return no_legal_completion


def _apply_lightweight_decoding_masks(
    logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    gen_length: int,
    constraints: dict | None,
    active_generation_mask: torch.Tensor | None = None,
    mask_id: int = 126336,
) -> dict[str, set[tuple[int, int]]]:
    report: dict[str, set[tuple[int, int]]] = {
        "pbc_no_legal_completion": set()
    }
    if not constraints:
        return report
    generation_logits = logits[:, prompt_length : prompt_length + gen_length, :]
    min_value = torch.finfo(generation_logits.dtype).min
    if active_generation_mask is not None and active_generation_mask.shape != generation_logits.shape[:2]:
        raise ValueError("active_generation_mask must match [batch, generation length]")
    if constraints.get("canonicalize_periodic_alias"):
        _aggregate_periodic_coordinate_alias_logits(
            generation_logits,
            constraints,
            min_value,
            active_generation_mask,
        )
    if constraints.get("lattice_volume_mask"):
        _mask_zero_lengths(generation_logits, constraints, min_value)
        _apply_lattice_volume_mask(generation_logits, x, prompt_length, constraints, min_value)
    if constraints.get("duplicate_coordinate_mask"):
        _apply_duplicate_coordinate_mask(generation_logits, x, prompt_length, constraints, min_value)
    if constraints.get("pbc_min_distance_mask"):
        report["pbc_no_legal_completion"] = _apply_pbc_min_distance_mask(
            generation_logits,
            x,
            prompt_length,
            constraints,
            min_value,
            active_generation_mask,
            int(mask_id),
        )
    return report


def _candidate_tokens_and_confidence(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    return x0, x0_p


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    steps: int = 107,
    gen_length: int = 107,
    block_length: int = 107,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    allowed_token_ids_by_generation_pos: list[list[int]] | None = None,
    prefill_token_ids_by_generation_pos: dict[int, int | list[int] | torch.Tensor] | None = None,
    atom_count_grammar: dict | None = None,
    generation_position_groups: list[list[int]] | None = None,
    generation_position_groups_by_batch: list[list[list[int]]] | None = None,
    lightweight_decoding_constraints: dict | None = None,
    return_step_map: bool = False,
) -> torch.Tensor:
    """Generate suffix tokens with LLaDA-style masked diffusion."""

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    if prefill_token_ids_by_generation_pos:
        for generation_pos, token_id in prefill_token_ids_by_generation_pos.items():
            if 0 <= generation_pos < gen_length:
                if isinstance(token_id, torch.Tensor):
                    values = token_id.to(device=model.device, dtype=torch.long)
                elif isinstance(token_id, list):
                    values = torch.tensor(token_id, dtype=torch.long, device=model.device)
                else:
                    values = torch.full((prompt.shape[0],), int(token_id), dtype=torch.long, device=model.device)
                if values.numel() != prompt.shape[0]:
                    raise ValueError(
                        f"Prefill position {generation_pos} has {values.numel()} values for batch size {prompt.shape[0]}"
                    )
                x[:, prompt.shape[1] + generation_pos] = values.view(-1)
    step_map = None
    if return_step_map:
        step_map = torch.full(
            (prompt.shape[0], gen_length),
            -1,
            dtype=torch.long,
            device=model.device,
        )

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != mask_id
    allowed_mask = None
    prepared_atom_count_grammar = None
    if allowed_token_ids_by_generation_pos is not None:
        if len(allowed_token_ids_by_generation_pos) != gen_length:
            raise ValueError("allowed_token_ids_by_generation_pos length must match gen_length")
        vocab_size = model.get_output_embeddings().weight.shape[0]
        allowed_mask = torch.zeros((gen_length, vocab_size), dtype=torch.bool, device=model.device)
        for generation_pos, token_ids in enumerate(allowed_token_ids_by_generation_pos):
            if not token_ids:
                raise ValueError(f"No allowed token ids for generation position {generation_pos}")
            allowed_mask[generation_pos, torch.tensor(token_ids, dtype=torch.long, device=model.device)] = True
        prepared_atom_count_grammar = _prepare_atom_count_grammar(
            atom_count_grammar,
            vocab_size,
            model.device,
        )
    elif atom_count_grammar is not None:
        vocab_size = model.get_output_embeddings().weight.shape[0]
        prepared_atom_count_grammar = _prepare_atom_count_grammar(
            atom_count_grammar,
            vocab_size,
            model.device,
        )

    if (
        generation_position_groups is not None
        and generation_position_groups_by_batch is not None
    ):
        raise ValueError(
            "Pass either a shared generation schedule or row-local schedules, not both"
        )
    position_schedules = None
    if generation_position_groups_by_batch is not None:
        position_schedules = _validate_generation_position_groups_by_batch(
            generation_position_groups_by_batch,
            batch_size=prompt.shape[0],
            gen_length=gen_length,
        )
    elif generation_position_groups is not None:
        shared = _validate_generation_position_groups(
            generation_position_groups, gen_length
        )
        position_schedules = [shared for _ in range(prompt.shape[0])]

    if position_schedules is not None:
        denoise_step = 0
        max_groups = max(len(groups) for groups in position_schedules)
        for group_index in range(max_groups):
            group_allowed = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            for row_index, groups in enumerate(position_schedules):
                if group_index >= len(groups):
                    continue
                absolute_positions = torch.tensor(
                    [prompt.shape[1] + position for position in groups[group_index]],
                    dtype=torch.long,
                    device=x.device,
                )
                group_allowed[row_index, absolute_positions] = True
            group_mask_index = (x == mask_id) & group_allowed
            group_steps = int(group_mask_index.sum(dim=1).max().detach().item())
            if group_steps <= 0:
                continue
            num_transfer_tokens = get_num_transfer_tokens(group_mask_index, group_steps)
            for i in range(group_steps):
                mask_index = x == mask_id
                logits = _model_logits(model, x, attention_mask, prompt_index, cfg_scale, mask_id)
                if allowed_mask is not None or prepared_atom_count_grammar is not None:
                    _apply_schema_masks(
                        logits,
                        x,
                        prompt.shape[1],
                        gen_length,
                        allowed_mask,
                        prepared_atom_count_grammar,
                    )
                _apply_lightweight_decoding_masks(
                    logits,
                    x,
                    prompt.shape[1],
                    gen_length,
                    lightweight_decoding_constraints,
                    group_allowed[
                        :, prompt.shape[1] : prompt.shape[1] + gen_length
                    ],
                    mask_id,
                )
                x0, x0_p = _candidate_tokens_and_confidence(logits, temperature, remasking)
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index & group_allowed, x0_p, -float("inf"))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i].detach().item())
                    if k <= 0:
                        continue
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
                if step_map is not None:
                    suffix_transfer = transfer_index[:, prompt.shape[1] : prompt.shape[1] + gen_length]
                    step_map[suffix_transfer & (step_map < 0)] = denoise_step
                x[transfer_index] = x0[transfer_index]
                denoise_step += 1
        return (x, step_map) if return_step_map else x

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by number of blocks")
    block_steps = steps // num_blocks

    denoise_step = 0
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, block_steps)

        for i in range(block_steps):
            mask_index = x == mask_id
            logits = _model_logits(model, x, attention_mask, prompt_index, cfg_scale, mask_id)
            if allowed_mask is not None or prepared_atom_count_grammar is not None:
                _apply_schema_masks(
                    logits,
                    x,
                    prompt.shape[1],
                    gen_length,
                    allowed_mask,
                    prepared_atom_count_grammar,
                )
            _apply_lightweight_decoding_masks(
                logits,
                x,
                prompt.shape[1],
                gen_length,
                lightweight_decoding_constraints,
                mask_index[
                    :, prompt.shape[1] : prompt.shape[1] + gen_length
                ],
                mask_id,
            )
            x0, x0_p = _candidate_tokens_and_confidence(logits, temperature, remasking)

            x0_p[:, block_end:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float("inf"))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            if step_map is not None:
                suffix_transfer = transfer_index[:, prompt.shape[1] : prompt.shape[1] + gen_length]
                step_map[suffix_transfer & (step_map < 0)] = denoise_step
            x[transfer_index] = x0[transfer_index]
            denoise_step += 1

    return (x, step_map) if return_step_map else x
