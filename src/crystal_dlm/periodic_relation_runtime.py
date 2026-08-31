"""Runtime bridge for the periodic residual adapter and LLaDA logits."""

from __future__ import annotations

from dataclasses import asdict
import json
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
) -> torch.Tensor:
    ids = torch.tensor(table["ids"], dtype=torch.long, device=logits.device)
    values = torch.tensor(table["values"], dtype=torch.float32, device=logits.device)
    matches = ids == int(current_id)
    if bool(matches.any().item()):
        return values[matches][0]
    probabilities = F.softmax(logits.float().index_select(-1, ids), dim=-1)
    return (probabilities * values).sum(dim=-1)


def soft_geometry_from_q0(
    *,
    q0: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    num_sites: torch.Tensor,
    support: Mapping[str, Any],
    max_sites: int = 20,
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
    for sample in range(batch):
        prompt = int(prompt_lengths[sample].detach().cpu())
        count = int(num_sites[sample].detach().cpu())
        if not 1 <= count <= max_sites:
            raise ValueError(f"num_sites {count} outside 1..{max_sites}")
        lengths = [
            _soft_or_committed(
                q0[sample, prompt + position],
                int(input_ids[sample, prompt + position].detach().item()),
                geometry_support["length"][axis],
            )
            for position, axis in zip(range(1, 4), "ABC")
        ]
        angles = [
            _soft_or_committed(
                q0[sample, prompt + position],
                int(input_ids[sample, prompt + position].detach().item()),
                geometry_support["angle"][axis],
            )
            for position, axis in zip(range(4, 7), "ABG")
        ]
        lattices.append(
            lattice_matrix_from_parameters(torch.stack(lengths), torch.stack(angles))
        )
        for site in range(count):
            base = prompt + 7 + 4 * site
            element_id = int(input_ids[sample, base].detach().item())
            if element_id not in element_support:
                raise ValueError("element token must be committed before periodic relation")
            species[sample, site] = element_support[element_id]
            for offset, axis in enumerate("XYZ", start=1):
                coordinates[sample, site, offset - 1] = _soft_or_committed(
                    q0[sample, base + offset],
                    int(input_ids[sample, base + offset].detach().item()),
                    geometry_support["coord"][axis],
                )
    return SoftCrystalGeometry(
        lattice=torch.stack(lattices),
        fractional_coordinates=coordinates,
        species=species,
        prompt_lengths=prompt_lengths,
        num_sites=num_sites,
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
        self.step0_checked = False
        self.step0_max_logit_delta: float | None = None

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


def wrap_with_periodic_relation(
    base_model: nn.Module,
    tokenizer: Any,
    *,
    rank: int,
    checkpoint: str | Path | None = None,
) -> PeriodicRelationLogitsModel:
    hidden_size = int(base_model.get_output_embeddings().weight.shape[1])
    if checkpoint is None:
        config = PeriodicRelationConfig(hidden_size=hidden_size, rank=int(rank))
        state = None
    else:
        root = Path(checkpoint)
        payload = json.loads((root / ADAPTER_CONFIG_NAME).read_text())
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
    "wrap_with_periodic_relation",
]
