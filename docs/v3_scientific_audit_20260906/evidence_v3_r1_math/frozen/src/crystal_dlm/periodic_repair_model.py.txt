"""Discrete DLM repair with actual periodic pair bias and numeric output rows.

All geometry is decoded from the explicitly supplied old integer state. Unknown
sites are masked before pair arithmetic. No probability-averaged coordinates,
continuous output, force call, or mutable attention hook is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn

from crystal_dlm.periodic_geometry_objective import build_geometry_token_support
from crystal_dlm.state_conditioned_model import StateConditionedDLM, set_state_lora_trainable


@dataclass(frozen=True)
class PeriodicRepairConfig:
    hidden_size: int
    width: int = 64
    radial_bins: int = 16
    fourier_modes: int = 8
    image_radius: int = 2
    radial_cutoff_A: float = 6.0
    schema: str = "periodic_discrete_self_repair_v1"

    def __post_init__(self):
        if min(self.hidden_size, self.width, self.radial_bins, self.fourier_modes) < 1:
            raise ValueError("repair dimensions must be positive")
        if self.image_radius not in (1, 2) or self.radial_cutoff_A <= 0:
            raise ValueError("unsupported periodic shell")


class FP32Module(nn.Module):
    def _apply(self, fn, recurse=True):
        def keep(tensor):
            result = fn(tensor)
            if result.is_floating_point() and result.dtype != torch.float32:
                return tensor.to(device=result.device, dtype=torch.float32)
            return result
        return super()._apply(keep, recurse=recurse)


class RepairTaskProjection(FP32Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(9, hidden_size, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.linear.weight)

    def forward(self, value):
        return self.linear(value)


def token_sites(context, length):
    positions = torch.arange(length, device=context.old_token_ids.device)[None]
    relative = positions - context.prompt_lengths[:, None]
    sites = torch.div(relative - 7, 4, rounding_mode="floor")
    valid = (relative >= 7) & (relative < 7 + 4 * context.num_sites[:, None])
    return sites.clamp(0, context.program_rank.shape[1] - 1), valid


def task_features(context, input_ids, geometry, mask_id):
    """Explicit task and noise metadata; -1 denotes unknown error magnitude.

Visibility never determines the noise level. A completely visible but erroneous
structure can have mask ratio zero and an unknown (or supplied positive) noise.
    """
    if context.task_ids is None or context.numeric_noise_level is None:
        raise ValueError("repair model requires explicit task ids and numeric noise metadata")
    if context.task_ids.shape != context.num_sites.shape or context.numeric_noise_level.shape != context.num_sites.shape:
        raise ValueError("one task and noise value is required per crystal")
    if bool(((context.task_ids < 0) | (context.task_ids > 4)).any()):
        raise ValueError("unknown repair task id")
    noise = context.numeric_noise_level.float()
    if not bool(torch.isfinite(noise).all()) or bool(((noise < 0) & (noise != -1)).any()):
        raise ValueError("noise must be nonnegative or explicitly unknown (-1)")
    relative = torch.arange(input_ids.shape[1], device=input_ids.device)[None] - context.prompt_lengths[:, None]
    numeric = ((relative >= 1) & (relative <= 6)) | (
        (relative >= 8) & (relative < 7 + 4 * context.num_sites[:, None])
        & ((relative - 8).remainder(4) < 3)
    )
    ratio = ((input_ids == mask_id) & numeric).sum(-1) / numeric.sum(-1).clamp_min(1)
    complete = geometry["lattice_known"] & (
        geometry["site_known"].sum(-1) == context.num_sites
    )
    scalars = torch.stack((ratio.float(), complete.float(), (noise >= 0).float(),
                           noise.clamp_min(0)), -1)
    return torch.cat((scalars, nn.functional.one_hot(context.task_ids.long(), 5).float()), -1)


class PeriodicAttentionBias(FP32Module):
    """One shared additive bias passed explicitly through all Transformer layers.

Distances minimize over a centered, finite image shell. This is not a claim of
exact nearest-image enumeration for every possible unreduced triclinic basis.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.species = nn.Embedding(119, 16, padding_idx=0)
        self.edge = nn.Sequential(
            nn.Linear(config.radial_bins + 6 + 32 + 7 + 2 + 9, config.width),
            nn.SiLU(), nn.Linear(config.width, 1, bias=False),
        )
        nn.init.zeros_(self.edge[-1].weight)
        r = config.image_radius
        axis = torch.arange(-r, r + 1, dtype=torch.float32)
        self.register_buffer("shifts", torch.cartesian_prod(axis, axis, axis))
        self.register_buffer("centers", torch.linspace(0, config.radial_cutoff_A, config.radial_bins))

    def pair_features(self, geometry, tasks):
        lattice = geometry["lattice"].float()
        known = geometry["site_known"] & geometry["lattice_known"][:, None]
        frac = torch.where(known[..., None], geometry["fractional"].float(), 0.)
        lattice = torch.where(geometry["lattice_known"][:, None, None], lattice,
                              torch.eye(3, device=lattice.device)[None])
        delta = frac[:, :, None] - frac[:, None, :]
        delta = delta - delta.round()
        cart = torch.einsum("bijsk,bkd->bijsd", delta[..., None, :] + self.shifts, lattice)
        distances = cart.square().sum(-1).clamp_min(1e-12).sqrt().amin(-1)
        width = self.config.radial_cutoff_A / max(self.config.radial_bins - 1, 1)
        radial = torch.exp(-((distances[..., None] - self.centers) / width).square())
        phase = 2 * math.pi * delta
        periodic = torch.cat((phase.sin(), phase.cos()), -1)
        batch, sites = known.shape
        z = self.species(geometry["species"])
        gram = lattice @ lattice.transpose(-1, -2)
        scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-10)
        cell = torch.stack((gram[:, 0, 0], gram[:, 1, 1], gram[:, 2, 2],
                            gram[:, 0, 1], gram[:, 0, 2], gram[:, 1, 2]), -1) / scale[:, None]
        cell = torch.cat((cell, scale.log()[:, None]), -1)
        active = geometry["active_sites"].float()
        features = torch.cat((
            radial, periodic,
            z[:, :, None].expand(-1, -1, sites, -1),
            z[:, None, :].expand(-1, sites, -1, -1),
            cell[:, None, None].expand(-1, sites, sites, -1),
            active[:, :, None, None].expand(-1, -1, sites, -1),
            active[:, None, :, None].expand(-1, sites, -1, -1),
            tasks[:, None, None].expand(-1, sites, sites, -1),
        ), -1)
        pair_known = known[:, :, None] & known[:, None, :]
        pair_known &= ~torch.eye(sites, device=known.device, dtype=torch.bool)[None]
        return torch.where(pair_known[..., None], features, 0.), pair_known

    def forward(self, geometry, context, length, tasks):
        with torch.autocast(device_type=context.old_token_ids.device.type, enabled=False):
            features, known = self.pair_features(geometry, tasks.float())
            pair_bias = self.edge(features).squeeze(-1) * known
            index, native = token_sites(context, length)
            rows = torch.arange(index.shape[0], device=index.device)[:, None, None]
            bias = pair_bias[rows, index[:, :, None], index[:, None, :]]
            bias = bias * (native[:, :, None] & native[:, None, :])
            return bias[:, None]


def numeric_features(values, family, modes=8):
    values = torch.as_tensor(values, dtype=torch.float32)
    if family == "coord":
        phase = 2 * math.pi * values.remainder(1)[:, None] * torch.arange(1, modes + 1)
        return torch.cat((torch.ones_like(values[:, None]), phase.sin(), phase.cos()), -1)
    if family == "length":
        value = values.clamp_min(.1).log()
        centers = torch.linspace(math.log(.1), math.log(50.), 16)
        width = (centers[-1] - centers[0]) / 15
    else:
        value = values / 180.
        centers = torch.linspace(0., 1., 16)
        width = 1 / 15
    return torch.cat((torch.ones_like(value[:, None]), value[:, None],
                      torch.exp(-((value[:, None] - centers) / width).square())), -1)


class CrystalNumericAdapter(FP32Module):
    """Zero-initialized residual logits only for the native typed numeric rows."""
    def __init__(self, tokenizer, config):
        super().__init__()
        self.tables = []
        self.projections = nn.ModuleList()
        supports = build_geometry_token_support(tokenizer)
        axes = [("length", x) for x in "ABC"] + [("angle", x) for x in "ABG"]
        axes += [("coord", x) for x in "XYZ"]
        for index, (family, axis) in enumerate(axes):
            table = supports[family][axis]
            features = numeric_features(table["values"], family, config.fourier_modes)
            self.register_buffer(f"ids_{index}", torch.tensor(table["ids"], dtype=torch.long))
            self.register_buffer(f"features_{index}", features)
            projection = nn.Linear(config.hidden_size, features.shape[1], bias=False)
            nn.init.zeros_(projection.weight)
            self.projections.append(projection)
            self.tables.append((family, axis))

    def forward(self, logits, hidden, context):
        result = logits.clone()
        device = hidden.device
        with torch.autocast(device_type=device.type, enabled=False):
            for family_index, projection in enumerate(self.projections):
                if family_index < 6:
                    row = torch.arange(hidden.shape[0], device=device)
                    pos = context.prompt_lengths + family_index + 1
                else:
                    site = torch.arange(context.program_rank.shape[1], device=device)[None]
                    row, slot = torch.where(site < context.num_sites[:, None])
                    pos = context.prompt_lengths[row] + 8 + 4 * slot + family_index - 6
                ids = getattr(self, f"ids_{family_index}")
                features = getattr(self, f"features_{family_index}")
                increments = projection(hidden[row, pos].float()) @ features.T
                result.index_put_((row[:, None], pos[:, None], ids[None]),
                                  increments.to(result.dtype), accumulate=True)
        return result


class PeriodicRepairDLM(StateConditionedDLM):
    def __init__(self, base_model, tokenizer, state_config, repair_config):
        super().__init__(base_model, tokenizer, state_config)
        self.repair_config = repair_config
        self.geometry_attention = PeriodicAttentionBias(repair_config)
        self.numeric_adapter = CrystalNumericAdapter(tokenizer, repair_config)
        self.repair_task_projection = RepairTaskProjection(state_config.hidden_size)
        from crystal_dlm.fixed_slot import MASK_TOKEN_ID
        self.mask_id = int(getattr(tokenizer, "mask_token_id", None) or MASK_TOKEN_ID)

    def forward(self, input_ids, attention_mask=None, *, geometry_context=None, **kwargs):
        if geometry_context is None:
            raise ValueError("periodic repair requires explicit old-state context")
        geometry = self.geometry_inputs(geometry_context)
        tasks = task_features(geometry_context, input_ids, geometry, self.mask_id)
        embeddings = self.state_embeddings(input_ids, geometry_context)
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            task_delta = self.repair_task_projection(tasks.float())
        embeddings = embeddings + (
            task_delta[:, None].to(embeddings.dtype) * geometry_context.active_token_mask[..., None]
        )
        bias = self.geometry_attention(geometry, geometry_context, input_ids.shape[1], tasks)
        requested_hidden = kwargs.pop("output_hidden_states", False)
        if kwargs.get("attention_bias") is not None:
            raise ValueError("caller must not replace the registered geometry attention bias")
        kwargs.pop("attention_bias", None)
        output = self.base_model(input_ids=None, inputs_embeds=embeddings,
                                 attention_mask=attention_mask, attention_bias=bias,
                                 output_hidden_states=True, **kwargs)
        if not output.hidden_states:
            raise RuntimeError("base model did not return final normalized hidden states")
        output.logits = self.numeric_adapter(output.logits, output.hidden_states[-1], geometry_context)
        if not requested_hidden:
            output.hidden_states = None
        return output

    def save_pretrained(self, output_dir, **kwargs):
        super().save_pretrained(output_dir, **kwargs)
        root = Path(output_dir)
        (root / "periodic_repair_config.json").write_text(
            json.dumps(asdict(self.repair_config), indent=2) + "\n", encoding="utf-8")
        torch.save({key: module.state_dict() for key, module in self.repair_modules().items()},
                   root / "periodic_repair.pt")

    def repair_modules(self):
        return {name: getattr(self, name) for name in
                ("geometry_attention", "numeric_adapter", "repair_task_projection")}

    def load_repair(self, path):
        root = Path(path)
        if json.loads((root / "periodic_repair_config.json").read_text()) != asdict(self.repair_config):
            raise ValueError("repair checkpoint configuration differs")
        state = torch.load(root / "periodic_repair.pt", map_location="cpu", weights_only=True)
        if set(state) != set(self.repair_modules()):
            raise ValueError("repair checkpoint module partition differs")
        for name, module in self.repair_modules().items():
            module.load_state_dict(state[name], strict=True)


def set_repair_trainable(model):
    counts = set_state_lora_trainable(model)
    for name, module in model.repair_modules().items():
        module.requires_grad_(True)
        counts[name] = sum(p.numel() for p in module.parameters())
    counts["frozen"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return counts


def load_repair_model(model_path, checkpoint_path, device, *, trainable=False):
    from crystal_dlm.periodic_state_conditioning import PeriodicStateConfig
    from scripts.sample_llada_dynamic_crystals import load_model_and_tokenizer
    root = Path(checkpoint_path)
    base, tokenizer = load_model_and_tokenizer(str(model_path), str(root), device)
    state_config = PeriodicStateConfig(**json.loads((root / "periodic_state_config.json").read_text()))
    saved = root / "periodic_repair_config.json"
    config = (PeriodicRepairConfig(**json.loads(saved.read_text())) if saved.exists()
              else PeriodicRepairConfig(state_config.hidden_size))
    model = PeriodicRepairDLM(base, tokenizer, state_config, config).to(device)
    model.load_state_conditioner(root)
    if saved.exists():
        model.load_repair(root)
    if trainable:
        set_repair_trainable(model)
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    return model, tokenizer
