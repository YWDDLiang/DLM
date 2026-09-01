"""Differentiable periodic-geometry losses for dynamic ``7+4N`` DLM bodies."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.periodic_geometry_ops import (
    element_radius,
    minimum_image_distances,
)


_FAMILY_PATTERNS = {
    "length": re.compile(r"^<L([ABC])_(\d{3})>$"),
    "angle": re.compile(r"^<A([ABG])_(\d{3})>$"),
    "coord": re.compile(r"^<([XYZ])_(\d{3})>$"),
}
_ELEMENT_PATTERN = re.compile(r"^<E_([A-Z][a-z]?)>$")


def build_geometry_token_support(tokenizer: Any) -> dict[str, dict[str, dict[str, list[float] | list[int]]]]:
    """Build a JSON-serializable legal-token/value table from a tokenizer."""

    support: dict[str, dict[str, dict[str, list[Any]]]] = {
        "length": {axis: {"ids": [], "values": []} for axis in "ABC"},
        "angle": {axis: {"ids": [], "values": []} for axis in "ABG"},
        "coord": {axis: {"ids": [], "values": []} for axis in "XYZ"},
    }
    for token, token_id in tokenizer.get_vocab().items():
        for family, pattern in _FAMILY_PATTERNS.items():
            match = pattern.fullmatch(str(token))
            if match is None:
                continue
            axis, raw_value = match.groups()
            value = int(raw_value)
            if family == "length":
                value = value * 0.1
            elif family == "coord":
                value = value / 100.0
            support[family][axis]["ids"].append(int(token_id))
            support[family][axis]["values"].append(float(value))
            break
    for family in support.values():
        for axis, table in family.items():
            pairs = sorted(zip(table["ids"], table["values"]))
            if not pairs:
                raise ValueError(f"tokenizer has no geometry support for axis {axis}")
            table["ids"] = [pair[0] for pair in pairs]
            table["values"] = [pair[1] for pair in pairs]
    return support


def build_species_radius_support(tokenizer: Any) -> dict[str, list[float] | list[int]]:
    """Map committed element-token ids to the frozen in-repo radius table."""

    pairs: list[tuple[int, float]] = []
    for token, token_id in tokenizer.get_vocab().items():
        match = _ELEMENT_PATTERN.fullmatch(str(token))
        if match is None:
            continue
        symbol = match.group(1)
        if symbol not in SYMBOL_TO_Z:
            continue
        radius = element_radius(SYMBOL_TO_Z[symbol])
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"invalid radius for element token {token}")
        pairs.append((int(token_id), radius))
    pairs.sort()
    if not pairs:
        raise ValueError("tokenizer has no element tokens for species-aware margins")
    return {
        "ids": [token_id for token_id, _radius in pairs],
        "values": [radius for _token_id, radius in pairs],
    }


def _family_expectation(
    logits: torch.Tensor,
    target_id: torch.Tensor,
    is_masked: torch.Tensor,
    table: Mapping[str, list[float] | list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(table["ids"], dtype=torch.long, device=logits.device)
    values = torch.tensor(table["values"], dtype=logits.dtype, device=logits.device)
    probabilities = F.softmax(logits.index_select(-1, ids), dim=-1)
    predicted = (probabilities * values).sum(dim=-1)
    matches = ids.unsqueeze(0) == target_id.reshape(-1, 1)
    if not bool(matches.any(dim=1).all().item()):
        raise ValueError("target token is outside its registered geometry family")
    target = (matches.to(values.dtype) * values.unsqueeze(0)).sum(dim=-1)
    value = torch.where(is_masked.reshape(-1), predicted, target)
    return value, target


def _lattice_matrix(lengths: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    angles = torch.deg2rad(angles_deg.clamp(1.0, 179.0))
    alpha, beta, gamma = angles.unbind()
    a, b, c = lengths.clamp_min(0.1).unbind()
    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_gamma = torch.sin(gamma).abs().clamp_min(1e-4)
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = torch.sqrt((c.square() - cx.square() - cy.square()).clamp_min(1e-8))
    zero = torch.zeros((), dtype=lengths.dtype, device=lengths.device)
    return torch.stack(
        (
            torch.stack((a, zero, zero)),
            torch.stack((b * cos_gamma, b * sin_gamma, zero)),
            torch.stack((cx, cy, cz)),
        )
    )


def lattice_matrix_from_parameters(
    lengths: torch.Tensor, angles_deg: torch.Tensor
) -> torch.Tensor:
    """Public differentiable row-vector lattice constructor."""

    return _lattice_matrix(lengths, angles_deg)


def _pair_distances(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    *,
    exact_triclinic: bool = False,
    image_radius: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(frac_coords.shape[0])
    if count < 2:
        empty = frac_coords.new_zeros((0,))
        return empty, torch.empty((2, 0), dtype=torch.long, device=frac_coords.device)
    pairs = torch.triu_indices(count, count, offset=1, device=frac_coords.device)
    delta = frac_coords.index_select(0, pairs[0]) - frac_coords.index_select(0, pairs[1])
    centered = delta - torch.round(delta)
    if exact_triclinic:
        distances = minimum_image_distances(
            centered,
            lattice,
            image_radius=int(image_radius),
        )
    else:
        cartesian = centered @ lattice
        distances = torch.linalg.vector_norm(cartesian, dim=-1)
    return distances, pairs


def _species_aware_margins(
    species_token_ids: torch.Tensor,
    pairs: torch.Tensor,
    radius_support: Mapping[str, list[float] | list[int]],
    *,
    scale: float,
    floor: float,
    ceiling: float,
) -> torch.Tensor:
    ids = torch.tensor(
        radius_support["ids"],
        dtype=torch.long,
        device=species_token_ids.device,
    )
    values = torch.tensor(
        radius_support["values"],
        dtype=torch.float32,
        device=species_token_ids.device,
    )
    matches = species_token_ids.reshape(-1, 1) == ids.reshape(1, -1)
    radii = (matches.to(values.dtype) * values.reshape(1, -1)).sum(dim=1)
    margins = float(scale) * (
        radii.index_select(0, pairs[0]) + radii.index_select(0, pairs[1])
    )
    return margins.clamp(min=float(floor), max=float(ceiling))


def _smooth_rdf(distances: torch.Tensor, centers: torch.Tensor, sigma: float = 0.35) -> torch.Tensor:
    if distances.numel() == 0:
        return centers.new_zeros(centers.shape)
    values = torch.exp(-0.5 * ((distances.unsqueeze(-1) - centers) / sigma).square())
    return values.mean(dim=0)


def _sample_objective(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    masked: torch.Tensor,
    prompt_length: int,
    num_atoms: int,
    support: Mapping[str, Mapping[str, Mapping[str, list[Any]]]],
    exact_triclinic_pbc: bool,
    periodic_image_radius: int,
    species_margin_scale: float,
    species_margin_floor: float,
    species_margin_ceiling: float,
) -> dict[str, torch.Tensor]:
    # CUDA determinant/linear-algebra kernels do not support bfloat16.  Keep
    # the language model in its native dtype but evaluate the small geometry
    # graph in float32; the cast remains differentiable back to the logits.
    geometry_logits = logits.to(torch.float32)
    device = geometry_logits.device
    dtype = geometry_logits.dtype
    lattice_positions = [prompt_length + offset for offset in range(1, 7)]
    if max(lattice_positions) >= input_ids.shape[0]:
        raise ValueError("dynamic lattice positions exceed sequence length")
    length_values, target_lengths = [], []
    angle_values, target_angles = [], []
    for position, axis in zip(lattice_positions[:3], "ABC"):
        value, target = _family_expectation(
            geometry_logits[position].unsqueeze(0), input_ids[position].unsqueeze(0),
            masked[position].unsqueeze(0), support["length"][axis],
        )
        length_values.append(value[0]); target_lengths.append(target[0])
    for position, axis in zip(lattice_positions[3:], "ABG"):
        value, target = _family_expectation(
            geometry_logits[position].unsqueeze(0), input_ids[position].unsqueeze(0),
            masked[position].unsqueeze(0), support["angle"][axis],
        )
        angle_values.append(value[0]); target_angles.append(target[0])

    frac_rows, target_frac_rows, species_ids = [], [], []
    for site in range(num_atoms):
        base = prompt_length + 7 + 4 * site
        if base + 3 >= input_ids.shape[0]:
            raise ValueError("dynamic site positions exceed sequence length")
        species_ids.append(input_ids[base])
        values, targets = [], []
        for offset, axis in enumerate("XYZ", start=1):
            value, target = _family_expectation(
                geometry_logits[base + offset].unsqueeze(0), input_ids[base + offset].unsqueeze(0),
                masked[base + offset].unsqueeze(0), support["coord"][axis],
            )
            values.append(value[0]); targets.append(target[0])
        frac_rows.append(torch.stack(values)); target_frac_rows.append(torch.stack(targets))

    lengths = torch.stack(length_values)
    angles = torch.stack(angle_values)
    target_lengths_tensor = torch.stack(target_lengths)
    target_angles_tensor = torch.stack(target_angles)
    frac = torch.stack(frac_rows)
    target_frac = torch.stack(target_frac_rows)
    lattice = _lattice_matrix(lengths, angles)
    target_lattice = _lattice_matrix(target_lengths_tensor, target_angles_tensor)

    metric = lattice @ lattice.transpose(0, 1)
    target_metric = target_lattice @ target_lattice.transpose(0, 1)
    metric_scale = target_metric.diagonal().mean().detach().clamp_min(1.0)
    metric_loss = ((metric - target_metric) / metric_scale).square().mean()
    volume = torch.linalg.det(lattice).abs().clamp_min(1e-6)
    target_volume = torch.linalg.det(target_lattice).abs().clamp_min(1e-6)
    metric_loss = metric_loss + (
        torch.log(volume / max(num_atoms, 1)) - torch.log(target_volume / max(num_atoms, 1))
    ).square()

    distances, pairs = _pair_distances(
        frac,
        lattice,
        exact_triclinic=bool(exact_triclinic_pbc),
        image_radius=int(periodic_image_radius),
    )
    target_distances, _ = _pair_distances(
        target_frac,
        target_lattice,
        exact_triclinic=bool(exact_triclinic_pbc),
        image_radius=int(periodic_image_radius),
    )
    zero = geometry_logits.sum() * 0.0
    if distances.numel() == 0:
        return {"metric": metric_loss, "pair_rdf": zero, "overlap": zero, "coordination": zero}

    species = torch.stack(species_ids)
    same_species = species.index_select(0, pairs[0]) == species.index_select(0, pairs[1])
    centers = torch.linspace(0.5, 6.0, 12, dtype=dtype, device=device)
    rdf_losses = []
    for selector in (same_species, ~same_species):
        if bool(selector.any().item()):
            rdf_losses.append(
                F.mse_loss(
                    _smooth_rdf(distances[selector], centers),
                    _smooth_rdf(target_distances[selector], centers),
                )
            )
    pair_rdf_loss = torch.stack(rdf_losses).mean() if rdf_losses else zero
    if float(species_margin_scale) > 0.0:
        radius_support = support.get("species_radius_by_token_id")
        if not isinstance(radius_support, Mapping):
            raise ValueError("species-aware margin requested without radius support")
        overlap_margins = _species_aware_margins(
            species,
            pairs,
            radius_support,
            scale=float(species_margin_scale),
            floor=float(species_margin_floor),
            ceiling=float(species_margin_ceiling),
        )
        overlap_loss = (
            F.relu(
                (overlap_margins - distances)
                / overlap_margins.clamp_min(1.0e-6)
            )
            .square()
            .mean()
        )
    else:
        overlap_margins = torch.full_like(distances, 0.75)
        overlap_loss = F.relu(overlap_margins - distances).square().mean()

    predicted_neighbors = torch.sigmoid((3.0 - distances) / 0.25)
    target_neighbors = torch.sigmoid((3.0 - target_distances) / 0.25)
    predicted_coordination = torch.zeros((num_atoms,), dtype=dtype, device=device)
    target_coordination = torch.zeros((num_atoms,), dtype=dtype, device=device)
    predicted_coordination.index_add_(0, pairs[0], predicted_neighbors)
    predicted_coordination.index_add_(0, pairs[1], predicted_neighbors)
    target_coordination.index_add_(0, pairs[0], target_neighbors)
    target_coordination.index_add_(0, pairs[1], target_neighbors)
    coordination_loss = F.mse_loss(
        predicted_coordination / max(num_atoms - 1, 1),
        target_coordination / max(num_atoms - 1, 1),
    )
    return {
        "metric": metric_loss,
        "pair_rdf": pair_rdf_loss,
        "overlap": overlap_loss,
        "coordination": coordination_loss,
    }


def periodic_geometry_objective(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    masked_indices: torch.Tensor,
    prompt_lengths: torch.Tensor,
    num_atoms: torch.Tensor,
    support: Mapping[str, Mapping[str, Mapping[str, list[Any]]]],
    exact_triclinic_pbc: bool = False,
    periodic_image_radius: int = 1,
    species_margin_scale: float = 0.0,
    species_margin_floor: float = 0.6,
    species_margin_ceiling: float = 1.4,
) -> dict[str, torch.Tensor | int]:
    """Compute normalized coupled geometry losses for a dynamic-v1 batch."""

    totals: dict[str, list[torch.Tensor]] = {name: [] for name in ("metric", "pair_rdf", "overlap", "coordination")}
    for sample in range(input_ids.shape[0]):
        count = int(num_atoms[sample].detach().cpu())
        if not 1 <= count <= 20:
            raise ValueError(f"num_atoms {count} outside dynamic schema")
        components = _sample_objective(
            logits[sample], input_ids[sample], masked_indices[sample],
            int(prompt_lengths[sample].detach().cpu()), count, support,
            bool(exact_triclinic_pbc),
            int(periodic_image_radius),
            float(species_margin_scale),
            float(species_margin_floor),
            float(species_margin_ceiling),
        )
        for name, value in components.items():
            totals[name].append(value)
    reduced = {name: torch.stack(values).mean() for name, values in totals.items()}
    return {**reduced, "samples": int(input_ids.shape[0])}


__all__ = [
    "build_geometry_token_support",
    "build_species_radius_support",
    "lattice_matrix_from_parameters",
    "periodic_geometry_objective",
]
