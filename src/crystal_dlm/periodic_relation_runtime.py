"""Runtime bridge for the periodic residual adapter and LLaDA logits."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from crystal_dlm.fixed_slot import SYMBOL_TO_Z
from crystal_dlm.periodic_geometry_objective import (
    build_geometry_token_support,
    lattice_matrix_from_parameters,
)
from crystal_dlm.periodic_relation_adapter import (
    PeriodicRelationAdapter,
    PeriodicRelationConfig,
    SoftCrystalGeometry,
)


ADAPTER_CONFIG_NAME = "periodic_relation_config.json"
ADAPTER_STATE_NAME = "periodic_relation_adapter.pt"
_ELEMENT_RE = re.compile(r"^<E_([A-Z][a-z]?)>$")


def build_periodic_relation_support(tokenizer: Any) -> dict[str, Any]:
    elements: dict[int, int] = {}
    for token, token_id in tokenizer.get_vocab().items():
        match = _ELEMENT_RE.fullmatch(str(token))
        if match and match.group(1) in SYMBOL_TO_Z:
            elements[int(token_id)] = int(SYMBOL_TO_Z[match.group(1)])
    if not elements:
        raise ValueError("tokenizer has no dynamic element tokens")
    return {"geometry": build_geometry_token_support(tokenizer), "elements": elements}


def _soft_or_committed(
    logits: torch.Tensor,
    current_id: int,
    table: Mapping[str, list[Any]],
    *,
    periodic: bool = False,
    circular_mean_min_resultant: float = 0.25,
) -> torch.Tensor:
    ids = torch.tensor(table["ids"], dtype=torch.long, device=logits.device)
    values = torch.tensor(table["values"], dtype=torch.float32, device=logits.device)
    matches = ids == int(current_id)
    if bool(matches.any().item()):
        return values[matches][0]
    probabilities = F.softmax(logits.float().index_select(-1, ids), dim=-1)
    linear_mean = (probabilities * values).sum(dim=-1)
    if not periodic:
        return linear_mean
    phase = 2.0 * math.pi * values
    cosine = (probabilities * torch.cos(phase)).sum()
    sine = (probabilities * torch.sin(phase)).sum()
    resultant = torch.sqrt(cosine.square() + sine.square())
    circular_mean = torch.remainder(torch.atan2(sine, cosine) / (2.0 * math.pi), 1.0)
    return torch.where(
        resultant >= float(circular_mean_min_resultant),
        circular_mean,
        linear_mean,
    )


def _soft_or_committed_with_confidence(
    logits: torch.Tensor,
    current_id: int,
    table: Mapping[str, list[Any]],
    *,
    periodic: bool,
    floor: float,
    circular_mean_min_resultant: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(table["ids"], dtype=torch.long, device=logits.device)
    values = torch.tensor(table["values"], dtype=torch.float32, device=logits.device)
    matches = ids == int(current_id)
    if bool(matches.any().item()):
        return values[matches][0], values.new_tensor(1.0)
    probabilities = F.softmax(logits.float().index_select(-1, ids), dim=-1)
    linear_mean = (probabilities * values).sum(dim=-1)
    mean = linear_mean
    if int(values.numel()) <= 1:
        entropy = mean.new_zeros(())
    else:
        entropy = -(
            probabilities * probabilities.clamp_min(1.0e-12).log()
        ).sum() / math.log(int(values.numel()))
    if periodic:
        phase = 2.0 * math.pi * values
        resultant = torch.sqrt(
            (probabilities * torch.cos(phase)).sum().square()
            + (probabilities * torch.sin(phase)).sum().square()
        )
        circular_mean = torch.remainder(
            torch.atan2(
                (probabilities * torch.sin(phase)).sum(),
                (probabilities * torch.cos(phase)).sum(),
            )
            / (2.0 * math.pi),
            1.0,
        )
        mean = torch.where(
            resultant >= float(circular_mean_min_resultant),
            circular_mean,
            linear_mean,
        )
        spread = 1.0 - resultant.clamp(0.0, 1.0)
    else:
        value_range = (values.max() - values.min()).clamp_min(1.0e-6)
        variance = (probabilities * (values - mean).square()).sum()
        spread = (4.0 * variance / value_range.square()).clamp(0.0, 1.0)
    uncertainty = (0.5 * entropy + 0.5 * spread).clamp(0.0, 1.0)
    confidence = float(floor) + (1.0 - float(floor)) * (1.0 - uncertainty)
    return mean, confidence.clamp(float(floor), 1.0)


def soft_geometry_from_q0(
    *,
    q0: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    num_sites: torch.Tensor,
    support: Mapping[str, Any],
    max_sites: int = 20,
    uncertainty_gate: bool = False,
    uncertainty_gate_floor: float = 0.25,
    circular_mean_min_resultant: float = 0.25,
) -> SoftCrystalGeometry:
    """Decode committed tokens and q0 expectations without target leakage."""

    if q0.ndim != 3 or input_ids.shape != q0.shape[:2]:
        raise ValueError("q0/input_ids shape mismatch")
    batch = int(q0.shape[0])
    if prompt_lengths.shape != (batch,) or num_sites.shape != (batch,):
        raise ValueError("prompt_lengths/num_sites must match q0 batch")
    if max_sites > 20:
        raise ValueError("max_sites exceeds dynamic schema")
    geometry_support = support["geometry"]
    element_support = {int(key): int(value) for key, value in support["elements"].items()}
    lattices = []
    coordinates = q0.new_zeros((batch, max_sites, 3), dtype=torch.float32)
    species = torch.zeros((batch, max_sites), dtype=torch.long, device=q0.device)
    lattice_confidence = q0.new_ones((batch,), dtype=torch.float32)
    site_confidence = q0.new_ones((batch, max_sites), dtype=torch.float32)
    for sample in range(batch):
        prompt = int(prompt_lengths[sample].detach().cpu())
        count = int(num_sites[sample].detach().cpu())
        if not 1 <= count <= max_sites:
            raise ValueError(f"num_sites {count} outside 1..{max_sites}")
        lattice_values: list[torch.Tensor] = []
        lattice_confidences: list[torch.Tensor] = []
        for family, positions, axes in (
            ("length", range(1, 4), "ABC"),
            ("angle", range(4, 7), "ABG"),
        ):
            for position, axis in zip(positions, axes):
                if uncertainty_gate:
                    value, confidence = _soft_or_committed_with_confidence(
                        q0[sample, prompt + position],
                        int(input_ids[sample, prompt + position].detach().item()),
                        geometry_support[family][axis],
                        periodic=False,
                        floor=float(uncertainty_gate_floor),
                        circular_mean_min_resultant=float(circular_mean_min_resultant),
                    )
                    lattice_confidences.append(confidence)
                else:
                    value = _soft_or_committed(
                        q0[sample, prompt + position],
                        int(input_ids[sample, prompt + position].detach().item()),
                        geometry_support[family][axis],
                        periodic=False,
                        circular_mean_min_resultant=float(circular_mean_min_resultant),
                    )
                lattice_values.append(value)
        lengths = lattice_values[:3]
        angles = lattice_values[3:]
        if uncertainty_gate:
            lattice_confidence[sample] = torch.stack(lattice_confidences).clamp_min(
                float(uncertainty_gate_floor)
            ).log().mean().exp()
        lattices.append(
            lattice_matrix_from_parameters(torch.stack(lengths), torch.stack(angles))
        )
        for site in range(count):
            base = prompt + 7 + 4 * site
            element_id = int(input_ids[sample, base].detach().item())
            if element_id not in element_support:
                raise ValueError("element token must be committed before periodic relation")
            species[sample, site] = element_support[element_id]
            coordinate_confidences: list[torch.Tensor] = []
            for offset, axis in enumerate("XYZ", start=1):
                if uncertainty_gate:
                    value, confidence = _soft_or_committed_with_confidence(
                        q0[sample, base + offset],
                        int(input_ids[sample, base + offset].detach().item()),
                        geometry_support["coord"][axis],
                        periodic=True,
                        floor=float(uncertainty_gate_floor),
                        circular_mean_min_resultant=float(circular_mean_min_resultant),
                    )
                    coordinate_confidences.append(confidence)
                else:
                    value = _soft_or_committed(
                        q0[sample, base + offset],
                        int(input_ids[sample, base + offset].detach().item()),
                        geometry_support["coord"][axis],
                        periodic=True,
                        circular_mean_min_resultant=float(circular_mean_min_resultant),
                    )
                coordinates[sample, site, offset - 1] = value
            if uncertainty_gate:
                coordinate_confidence = torch.stack(coordinate_confidences).clamp_min(
                    float(uncertainty_gate_floor)
                ).log().mean().exp()
                site_confidence[sample, site] = torch.sqrt(
                    lattice_confidence[sample] * coordinate_confidence
                )
    return SoftCrystalGeometry(
        lattice=torch.stack(lattices),
        fractional_coordinates=coordinates,
        species=species,
        prompt_lengths=prompt_lengths,
        num_sites=num_sites,
        lattice_confidence=lattice_confidence if uncertainty_gate else None,
        site_confidence=site_confidence if uncertainty_gate else None,
    )


class PeriodicRelationLogitsModel(nn.Module):
    """Apply one q0-derived residual before the unchanged output head."""

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        config: PeriodicRelationConfig,
        *,
        adapter_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.periodic_relation_adapter = PeriodicRelationAdapter(config)
        if adapter_state is not None:
            self.periodic_relation_adapter.load_state_dict(dict(adapter_state), strict=True)
        reference_weight = base_model.get_output_embeddings().weight
        self.periodic_relation_adapter.to(
            device=reference_weight.device,
            dtype=torch.float32,
        )
        self.periodic_relation_support = build_periodic_relation_support(tokenizer)
        self._prompt_lengths: torch.Tensor | None = None
        self._num_sites: torch.Tensor | None = None
        self.step0_checked = adapter_state is not None
        self.step0_max_logit_delta: float | None = 0.0 if adapter_state is not None else None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def config(self) -> Any:
        return self.base_model.config

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_geometry_context(
        self, prompt_lengths: torch.Tensor, num_sites: torch.Tensor
    ) -> None:
        self._prompt_lengths = prompt_lengths
        self._num_sites = num_sites

    @staticmethod
    def _expand_context(values: torch.Tensor, batch: int) -> torch.Tensor:
        if values.shape[0] == batch:
            return values
        if batch % values.shape[0] != 0:
            raise ValueError("geometry context cannot expand to model batch")
        return values.repeat(batch // values.shape[0])

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("PeriodicRelationLogitsModel requires input_ids")
        if self._prompt_lengths is None or self._num_sites is None:
            raise RuntimeError("set_geometry_context must be called before forward")
        output_head = self.get_output_embeddings()
        captured: list[torch.Tensor] = []

        def capture_hidden(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("output-head pre-hook did not receive hidden states")
            captured.append(inputs[0])

        hook = output_head.register_forward_pre_hook(capture_hidden)
        try:
            outputs = self.base_model(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("expected exactly one output-head hidden-state capture")
        q0 = outputs.logits
        prompt_lengths = self._expand_context(self._prompt_lengths, q0.shape[0]).to(q0.device)
        num_sites = self._expand_context(self._num_sites, q0.shape[0]).to(q0.device)
        geometry = soft_geometry_from_q0(
            q0=q0,
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            num_sites=num_sites,
            support=self.periodic_relation_support,
            max_sites=self.periodic_relation_adapter.config.max_sites,
            uncertainty_gate=bool(self.periodic_relation_adapter.config.uncertainty_gate),
            uncertainty_gate_floor=float(
                self.periodic_relation_adapter.config.uncertainty_gate_floor
            ),
            circular_mean_min_resultant=float(
                self.periodic_relation_adapter.config.circular_mean_min_resultant
            ),
        )
        relation = self.periodic_relation_adapter(captured[0], geometry)
        q1 = output_head(relation.hidden_states)
        if not self.step0_checked:
            delta = float((q1 - q0).abs().max().detach().float().cpu())
            if delta != 0.0:
                raise RuntimeError(f"periodic relation step-0 equality failed: {delta}")
            self.step0_checked = True
            self.step0_max_logit_delta = delta
        outputs.logits = q1
        return outputs

    def save_pretrained(self, output_dir: str | Path, **kwargs) -> None:
        output = Path(output_dir)
        try:
            self.base_model.save_pretrained(output, **kwargs)
        except TypeError:
            self.base_model.save_pretrained(output)
        torch.save(
            {key: value.detach().cpu() for key, value in self.periodic_relation_adapter.state_dict().items()},
            output / ADAPTER_STATE_NAME,
        )
        payload = {
            "schema": "periodic_relation_adapter_v1",
            "config": asdict(self.periodic_relation_adapter.config),
            "step0_checked": bool(self.step0_checked),
            "step0_max_logit_delta": self.step0_max_logit_delta,
        }
        (output / ADAPTER_CONFIG_NAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


def set_periodic_relation_only_trainable(
    model: PeriodicRelationLogitsModel,
) -> dict[str, int]:
    """Freeze the base policy and leave only the periodic residual trainable."""

    if not isinstance(model, PeriodicRelationLogitsModel):
        raise TypeError("periodic-relation-only training requires the wrapped model")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.periodic_relation_adapter.parameters():
        parameter.requires_grad_(True)
    trainable = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if trainable <= 0 or frozen <= 0:
        raise RuntimeError("periodic-relation-only parameter partition is empty")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def wrap_with_periodic_relation(
    base_model: nn.Module,
    tokenizer: Any,
    *,
    rank: int,
    checkpoint: str | Path | None = None,
    uncertainty_gate: bool = False,
    uncertainty_gate_floor: float = 0.25,
    image_radius: int = 1,
    circular_mean_min_resultant: float = 0.25,
) -> PeriodicRelationLogitsModel:
    hidden_size = int(base_model.get_output_embeddings().weight.shape[1])
    if checkpoint is None:
        config = PeriodicRelationConfig(
            hidden_size=hidden_size,
            rank=int(rank),
            uncertainty_gate=bool(uncertainty_gate),
            uncertainty_gate_floor=float(uncertainty_gate_floor),
            image_radius=int(image_radius),
            circular_mean_min_resultant=float(circular_mean_min_resultant),
        )
        state = None
    else:
        root = Path(checkpoint)
        payload = json.loads((root / ADAPTER_CONFIG_NAME).read_text())
        if payload.get("step0_checked") is not True or payload.get("step0_max_logit_delta") != 0.0:
            raise ValueError("periodic relation checkpoint lacks a valid step-0 equality record")
        config = PeriodicRelationConfig(**payload["config"])
        if config.hidden_size != hidden_size or config.rank != int(rank):
            raise ValueError("periodic relation checkpoint/config mismatch")
        state = torch.load(root / ADAPTER_STATE_NAME, map_location="cpu", weights_only=True)
    return PeriodicRelationLogitsModel(base_model, tokenizer, config, adapter_state=state)


__all__ = [
    "ADAPTER_CONFIG_NAME",
    "ADAPTER_STATE_NAME",
    "PeriodicRelationLogitsModel",
    "build_periodic_relation_support",
    "soft_geometry_from_q0",
    "set_periodic_relation_only_trainable",
    "wrap_with_periodic_relation",
]
