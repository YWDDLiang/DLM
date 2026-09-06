"""Shared V2 token path and explicit full-float geometry path for H-P33.

Geometry uses the same LoRA/block/conditioner/bias objects as token construction.
Its input state is data; its final hidden and readout remain differentiable.
Checkpoint reconstruction and method-specific final-policy selection are separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, NamedTuple

import torch
from torch import Tensor, nn

from crystal_dlm.fixed_slot import COUNT_RE
from crystal_dlm.llada_hidden import llada_hidden_forward, resolve_llada_core
from crystal_dlm.mixed_geometry_diffusion import GeometryState, LatticeNormalizer, MixedGeometryConfig
from crystal_dlm.periodic_repair_initialization import INITIALIZATION_MARKER, set_fresh_repair_trainable
from crystal_dlm.periodic_repair_model import FP32Module
from crystal_dlm.periodic_v2_initialization import (
    V2_MARKER, PeriodicV2DLM, load_periodic_v2_architecture, load_periodic_v2_model,
)
from crystal_dlm.state_conditioned_model import CrystalStateContext


MIXED_SCHEMA = "shared_llada_mixed_geometry_v1"
MIXED_MARKER = "mixed_geometry_config.json"
MIXED_HEADS_FILE = "mixed_geometry_heads.pt"
MIXED_NORMALIZER_FILE = "mixed_geometry_normalizer.json"
_TOKEN_COMPONENTS = (
    "adapter_config.json", INITIALIZATION_MARKER, V2_MARKER,
    "periodic_state_config.json", "periodic_state.pt",
    "periodic_repair_config.json", "periodic_repair.pt",
)


@dataclass(frozen=True)
class MixedGeometryModelConfig:
    hidden_size: int
    time_features: int = 64
    time_width: int = 128
    time_frequency_base: float = 10000.0
    time_input_scale: float = 1000.0
    schema: str = MIXED_SCHEMA

    def __post_init__(self) -> None:
        for name in ("hidden_size", "time_features", "time_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.time_features < 4 or self.time_features % 2:
            raise ValueError("time_features must be even and >= 4")
        if (not math.isfinite(self.time_frequency_base) or self.time_frequency_base <= 1
                or not math.isfinite(self.time_input_scale) or self.time_input_scale <= 0):
            raise ValueError("time frequency base/scale are invalid")
        if self.schema != MIXED_SCHEMA:
            raise ValueError("unsupported mixed geometry model schema")


class GeometryModelOutput(NamedTuple):
    v_prediction: Tensor
    u_prediction: Tensor
    hidden_states: tuple[Tensor, ...] | None = None


class MixedGeometryHeads(FP32Module):
    def __init__(self, config: MixedGeometryModelConfig):
        super().__init__()
        self.config = config
        frequencies = torch.exp(-math.log(config.time_frequency_base)
                                * torch.arange(config.time_features // 2, dtype=torch.float32)
                                / (config.time_features // 2 - 1))
        self.register_buffer("time_frequencies", frequencies)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_features, config.time_width, dtype=torch.float32), nn.SiLU(),
            nn.Linear(config.time_width, config.hidden_size, dtype=torch.float32),
        )
        self.geometry_mode = nn.Parameter(torch.zeros(config.hidden_size, dtype=torch.float32))
        self.v_head = nn.Linear(config.hidden_size, 6, dtype=torch.float32)
        self.u_head = nn.Linear(config.hidden_size, 3, dtype=torch.float32)
        for layer in (self.v_head, self.u_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def time_embedding(self, time: Tensor) -> Tensor:
        with torch.autocast(device_type=time.device.type, enabled=False):
            phase = self.config.time_input_scale * time.detach().float()[:, None] * self.time_frequencies
            features = torch.cat((phase.sin(), phase.cos()), dim=-1)
            return self.time_mlp(features) + self.geometry_mode

    def forward(self, hidden: Tensor, cell_positions: Tensor, site_positions: Tensor,
                atom_mask: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden_fp32 = hidden.float()  # Cast retains the backbone autograd path.
            rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            cell_hidden = hidden_fp32[rows, cell_positions].mean(dim=1)
            site_hidden = hidden_fp32[rows, site_positions]
            v = self.v_head(cell_hidden)
            u = torch.where(atom_mask[..., None], self.u_head(site_hidden), 0.0)
        return v, u


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_component_hashes(root: Path) -> dict[str, str]:
    paths = [root / name for name in _TOKEN_COMPONENTS]
    weights = sorted({*root.glob("adapter_model*.safetensors"), *root.glob("adapter_model*.bin")})
    paths.extend(weights)
    paths.extend(sorted(root.glob("adapter_model*.index.json")))
    if not weights or any(not path.is_file() for path in paths):
        raise ValueError("checkpoint lacks a complete compact token architecture")
    return {path.name: _sha256(path) for path in paths}


def _validate_parent_provenance(value: Mapping) -> dict:
    if (not isinstance(value, Mapping) or value.get("schema") != "mixed_geometry_parent_v1"
            or value.get("parent_kind") != "completed_periodic_v2"
            or not isinstance(value.get("checkpoint_path"), str)):
        raise ValueError("mixed geometry requires an explicit completed-V2 parent provenance record")
    files = value.get("files_sha256")
    required = {"CHECKPOINT_FINAL.json", V2_MARKER, INITIALIZATION_MARKER, "adapter_config.json"}
    if (not isinstance(files, dict) or not required.issubset(files)
            or any(not isinstance(digest, str) or len(digest) != 64
                   or any(character not in "0123456789abcdef" for character in digest)
                   for digest in files.values())):
        raise ValueError("parent provenance must retain checkpoint component hashes")
    return json.loads(json.dumps(dict(value), allow_nan=False))


def _capture_parent_provenance(checkpoint_path: str | Path) -> dict:
    # The initializer calls the strict public V2 loader before reaching here.
    root = Path(checkpoint_path).resolve()
    files = _token_component_hashes(root)
    files["CHECKPOINT_FINAL.json"] = _sha256(root / "CHECKPOINT_FINAL.json")
    record = {"schema": "mixed_geometry_parent_v1", "parent_kind": "completed_periodic_v2",
              "checkpoint_path": str(root), "files_sha256": files}
    training_final = root.parent.parent / "TRAIN_FINAL.json"
    if training_final.is_file():
        record["training_final_sha256"] = _sha256(training_final)
    return _validate_parent_provenance(record)


class MixedGeometryDLM(nn.Module):
    def __init__(self, token_model: PeriodicV2DLM, tokenizer, normalizer: LatticeNormalizer, *,
                 parent_provenance: Mapping, model_config: MixedGeometryModelConfig | None = None,
                 diffusion_config: MixedGeometryConfig | None = None):
        super().__init__()
        if not isinstance(token_model, PeriodicV2DLM) or not isinstance(normalizer, LatticeNormalizer):
            raise TypeError("mixed geometry requires a reconstructed PeriodicV2DLM and LatticeNormalizer")
        self.token_model = token_model
        self.normalizer = normalizer
        self.parent_provenance = _validate_parent_provenance(parent_provenance)
        self.model_config = model_config or MixedGeometryModelConfig(token_model.state_config.hidden_size)
        if self.model_config.hidden_size != token_model.state_config.hidden_size:
            raise ValueError("mixed readout hidden size differs from the shared V2 backbone")
        self.diffusion_config = diffusion_config or MixedGeometryConfig(
            reference_length=normalizer.reference_length, normalizer_std_floor=normalizer.std_floor)
        if (self.diffusion_config.reference_length != normalizer.reference_length
                or self.diffusion_config.normalizer_std_floor != normalizer.std_floor):
            raise ValueError("diffusion and fitted normalizer unit/floor contracts differ")
        resolve_llada_core(token_model.base_model)
        self.geometry_heads = MixedGeometryHeads(self.model_config)
        counts = torch.zeros(token_model.get_input_embeddings().weight.shape[0], dtype=torch.long)
        for token, token_id in tokenizer.get_vocab().items():
            match = COUNT_RE.fullmatch(str(token))
            if match:
                counts[int(token_id)] = int(match.group(1))
        self.register_buffer("atom_count_by_token", counts, persistent=False)
        self.train(token_model.training)

    @property
    def base_model(self):
        return self.token_model.base_model

    @property
    def config(self):
        return self.token_model.config

    @property
    def device(self):
        return self.token_model.device

    @property
    def raw_initialization(self):
        return self.token_model.raw_initialization

    @property
    def state_config(self):
        return self.token_model.state_config

    @property
    def repair_config(self):
        return self.token_model.repair_config

    @property
    def v2_config(self):
        return self.token_model.v2_config

    @property
    def mask_id(self):
        return self.token_model.mask_id

    @property
    def state_conditioner(self):
        return self.token_model.state_conditioner

    @property
    def geometry_attention(self):
        return self.token_model.geometry_attention

    @property
    def new_token_rows(self):
        return self.token_model.new_token_rows

    def get_input_embeddings(self):
        return self.token_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.token_model.get_output_embeddings()

    def geometry_inputs(self, input_ids: Tensor, context: CrystalStateContext, state: GeometryState,
                        species: Tensor, attention_mask: Tensor | None = None):
        """Build floating geometry and a position-only legacy layout context.

        No numeric value in context.old_token_ids is read.  N/E are checked in
        the current scaffold; complete geometry comes only from z/fractional.
        """
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("geometry scaffold must be integer [batch,length]")
        batch, length = input_ids.shape
        device = input_ids.device
        if not isinstance(context, CrystalStateContext) or not isinstance(state, GeometryState):
            raise TypeError("geometry mode requires CrystalStateContext layout and explicit GeometryState")
        if context.old_token_ids.shape != input_ids.shape:
            raise ValueError("layout scaffold shape differs from input_ids")
        if (context.prompt_lengths.shape != (batch,) or context.num_sites.shape != (batch,)
                or context.prompt_lengths.dtype not in (torch.int32, torch.int64)
                or context.num_sites.dtype not in (torch.int32, torch.int64)):
            raise ValueError("layout needs integer prompt_lengths/num_sites per crystal")
        prompt = context.prompt_lengths.to(device=device, dtype=torch.long)
        counts = context.num_sites.to(device=device, dtype=torch.long)
        if (state.z.shape != (batch, 6) or state.fractional.ndim != 3
                or state.fractional.shape[0] != batch or state.fractional.shape[-1] != 3):
            raise ValueError("geometry state shapes must be [batch,6] and [batch,sites,3]")
        sites = state.fractional.shape[1]
        if (state.atom_mask.shape != (batch, sites) or state.atom_mask.dtype != torch.bool
                or species.shape != (batch, sites) or species.dtype not in (torch.int32, torch.int64)
                or context.program_rank.shape != (batch, sites)):
            raise ValueError("mask, atomic-Z species and program ranks must cover the padded sites")
        if any(tensor.device != device for tensor in (state.z, state.fractional, state.t, state.atom_mask, species)):
            raise ValueError("continuous state/species must share the scaffold device")
        if (sites < 1 or sites > self.token_model.state_config.max_sites
                or bool(((counts < 1) | (counts > sites) | (prompt < 0)
                         | (prompt + 7 + 4 * counts > length)).any())):
            raise ValueError("geometry layout exceeds the retained body or padded site dimensions")
        slot = torch.arange(sites, device=device)[None]
        present = slot < counts[:, None]
        if not torch.equal(present, state.atom_mask):
            raise ValueError("body sites must use contiguous canonical padding matching num_sites")
        rows = torch.arange(batch, device=device)[:, None]
        cell_positions = prompt[:, None] + torch.arange(1, 7, device=device)[None]
        site_positions = (prompt[:, None] + 7 + 4 * slot).clamp_max(length - 1)
        rank = context.program_rank.to(device=device)
        if rank.dtype not in (torch.int32, torch.int64) or bool((rank[present] < 0).any()):
            raise ValueError("program ranks must be nonnegative integers on present sites")
        rank = torch.where(present, rank.long(), 0)
        atomic_z = torch.where(present, species.long(), 0)
        if bool(((atomic_z < 1) | (atomic_z > 118)).logical_and(present).any()):
            raise ValueError("present sites require supported atomic numbers")
        scaffold_species = self.token_model.species_by_token[input_ids[rows, site_positions]]
        if bool(((scaffold_species != atomic_z) & present).any()):
            raise ValueError("continuous-state species differ from the hard E scaffold")
        if not torch.equal(self.atom_count_by_token[input_ids[torch.arange(batch, device=device), prompt]], counts):
            raise ValueError("continuous-state atom counts differ from the hard N scaffold")
        relative = torch.arange(length, device=device)[None] - prompt[:, None]
        body_mask = (relative >= 0) & (relative < 7 + 4 * counts[:, None])
        numeric_mask = ((relative >= 1) & (relative <= 6)) | (
            (relative >= 8) & (relative < 7 + 4 * counts[:, None]) & ((relative - 8).remainder(4) < 3))
        if bool((input_ids[numeric_mask] != self.token_model.mask_id).any()):
            raise ValueError("geometry-mode numeric scaffold slots must all be MASK")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape or attention_mask.device != device:
                raise ValueError("attention mask must match the input scaffold")
            if bool((attention_mask[body_mask] == 0).any()):
                raise ValueError("all real geometry body slots must be visible to attention")
        time = torch.as_tensor(state.t, dtype=torch.float32, device=device).detach()
        try:
            time = torch.broadcast_to(time, (batch,))
        except RuntimeError as error:
            raise ValueError("one time is required per crystal") from error
        if not bool((torch.isfinite(time) & (time > 0) & (time <= 1)).all()):
            raise ValueError("geometry time must be in (0,1]")
        with torch.no_grad():
            lattice = self.normalizer.decode(state.z, counts).float()
            fractional = torch.where(present[..., None], state.fractional.detach().float(), 0.0).remainder(1)
            if not bool(torch.isfinite(lattice).all() and torch.isfinite(fractional).all()):
                raise FloatingPointError("continuous geometry is nonfinite after its FP32 input conversion")
        geometry = {"lattice": lattice, "fractional": fractional, "species": atomic_z,
                    "site_known": present, "lattice_known": torch.ones(batch, dtype=torch.bool, device=device),
                    "program_rank": rank, "active_sites": present}
        layout = CrystalStateContext(input_ids, prompt, counts, rank, numeric_mask,
                                     task_ids=None, numeric_noise_level=None, numeric_noise_components=None)
        return geometry, layout, body_mask, cell_positions, site_positions, time

    def forward(self, input_ids, attention_mask=None, *, mode="token", geometry_context=None,
                geometry_state=None, species=None, output_hidden_states=False, **kwargs):
        if mode == "token":
            if geometry_state is not None or species is not None:
                raise ValueError("token mode must not receive a continuous geometry state/species")
            return self.token_model(input_ids, attention_mask=attention_mask, geometry_context=geometry_context,
                                    output_hidden_states=output_hidden_states, **kwargs)
        if mode != "geometry":
            raise ValueError("mode must be 'token' or 'geometry'")
        if species is None or geometry_state is None:
            raise ValueError("geometry mode requires geometry_state and atomic-Z species")
        if kwargs.pop("use_cache", False) not in (None, False) or kwargs.pop("past_key_values", None) is not None:
            raise ValueError("geometry mode does not use a KV cache")
        if kwargs.pop("return_dict", True) not in (None, True) or kwargs:
            raise TypeError("geometry mode accepts only its explicit state and hidden-output arguments")
        geometry, layout, body_mask, cell_positions, site_positions, time = self.geometry_inputs(
            input_ids, geometry_context, geometry_state, species, attention_mask)
        encoded = self.state_conditioner(**geometry)
        embeddings = self.get_input_embeddings()(input_ids)
        embeddings = embeddings + self.new_token_rows.input_increments(input_ids).to(embeddings.dtype)
        residual = torch.zeros_like(embeddings)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)[:, None]
        residual[rows, cell_positions] = encoded["cell_embedding"][:, None].to(embeddings.dtype)
        site_residual = encoded["site_embeddings"].to(embeddings.dtype) * geometry["site_known"][..., None]
        for offset in range(4):
            positions = (site_positions + offset).clamp_max(input_ids.shape[1] - 1)
            residual = residual.scatter_add(1, positions[..., None].expand(-1, -1, embeddings.shape[-1]), site_residual)
        time_residual = self.geometry_heads.time_embedding(time)
        embeddings = embeddings + residual + time_residual[:, None].to(embeddings.dtype) * body_mask[..., None]
        tasks = torch.zeros(input_ids.shape[0], 9, dtype=torch.float32, device=input_ids.device)
        bias = self.geometry_attention(geometry, layout, input_ids.shape[1], tasks)
        hidden = llada_hidden_forward(self.base_model, inputs_embeds=embeddings,
                                      attention_mask=attention_mask, attention_bias=bias)
        v, u = self.geometry_heads(hidden, cell_positions, site_positions, geometry["site_known"])
        return GeometryModelOutput(v, u, (hidden,) if output_hidden_states else None)

    def save_pretrained(self, output_dir, **kwargs) -> None:
        root = Path(output_dir)
        self.token_model.save_pretrained(root, **kwargs)
        torch.save(self.geometry_heads.state_dict(), root / MIXED_HEADS_FILE)
        self.normalizer.save(root / MIXED_NORMALIZER_FILE)
        record = {"schema": MIXED_SCHEMA, "model_config": asdict(self.model_config),
                  "diffusion_config": asdict(self.diffusion_config), "parent_provenance": self.parent_provenance,
                  "policy_selection": "caller_owned_not_inferred_from_architecture",
                  "token_components_sha256": _token_component_hashes(root),
                  "heads_sha256": _sha256(root / MIXED_HEADS_FILE),
                  "normalizer_sha256": _sha256(root / MIXED_NORMALIZER_FILE)}
        # Marker is last: a partially written directory is not loadable as mixed.
        (root / MIXED_MARKER).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def set_mixed_geometry_trainable(model: MixedGeometryDLM) -> dict[str, int]:
    if not isinstance(model, MixedGeometryDLM):
        raise TypeError("mixed trainability requires MixedGeometryDLM")
    counts = dict(set_fresh_repair_trainable(model.token_model))
    model.geometry_heads.requires_grad_(True)
    counts["geometry_heads"] = sum(parameter.numel() for parameter in model.geometry_heads.parameters())
    counts["total_trainable"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    counts["frozen"] = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return counts


def initialize_mixed_geometry_model(model_path, v2_checkpoint_path, normalizer: LatticeNormalizer, device, *,
                                     trainable=True, model_config=None, diffusion_config=None, torch_dtype=None):
    """Warm-start through the unchanged strict V2 final-policy loader."""
    token_model, tokenizer = load_periodic_v2_model(model_path, v2_checkpoint_path, device,
                                                   trainable=trainable, torch_dtype=torch_dtype)
    model = MixedGeometryDLM(token_model, tokenizer, normalizer,
                             parent_provenance=_capture_parent_provenance(v2_checkpoint_path),
                             model_config=model_config, diffusion_config=diffusion_config).to(device)
    if trainable:
        set_mixed_geometry_trainable(model)
        model.train()
    else:
        model.requires_grad_(False).eval()
    return model, tokenizer


def load_mixed_geometry_model(model_path, checkpoint_path, device, *, trainable=False, torch_dtype=None):
    """Reconstruct a marked mixed checkpoint; caller selects final/resume policy.

    This does not impose V2's completed-step number on the derived method.  The
    caller must check its own training completion/eligibility before sampling.
    """
    root = Path(checkpoint_path)
    marker = root / MIXED_MARKER
    if not marker.is_file():
        raise ValueError("checkpoint lacks the mixed geometry marker")
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("schema") != MIXED_SCHEMA:
        raise ValueError("unsupported mixed geometry checkpoint schema")
    parent = _validate_parent_provenance(record["parent_provenance"])
    if _token_component_hashes(root) != record.get("token_components_sha256"):
        raise ValueError("mixed token architecture component hashes differ")
    if (_sha256(root / MIXED_HEADS_FILE) != record.get("heads_sha256")
            or _sha256(root / MIXED_NORMALIZER_FILE) != record.get("normalizer_sha256")):
        raise ValueError("mixed heads or normalizer hash differs")
    normalizer = LatticeNormalizer.load(root / MIXED_NORMALIZER_FILE)
    model_config = MixedGeometryModelConfig(**record["model_config"])
    diffusion_config = MixedGeometryConfig(**record["diffusion_config"])
    token_model, tokenizer = load_periodic_v2_architecture(model_path, root, device,
                                                          trainable=trainable, torch_dtype=torch_dtype)
    model = MixedGeometryDLM(token_model, tokenizer, normalizer, parent_provenance=parent,
                             model_config=model_config, diffusion_config=diffusion_config).to(device)
    heads = torch.load(root / MIXED_HEADS_FILE, map_location="cpu", weights_only=True)
    model.geometry_heads.load_state_dict(heads, strict=True)
    if trainable:
        set_mixed_geometry_trainable(model)
        model.train()
    else:
        model.requires_grad_(False).eval()
    return model, tokenizer


__all__ = [
    "MIXED_SCHEMA", "MIXED_MARKER", "MIXED_HEADS_FILE", "MIXED_NORMALIZER_FILE",
    "MixedGeometryModelConfig", "GeometryModelOutput", "MixedGeometryHeads", "MixedGeometryDLM",
    "set_mixed_geometry_trainable", "initialize_mixed_geometry_model", "load_mixed_geometry_model",
]
