"""Attempt-accounted unconditional sampling on the stratified WQ state space."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .bridge import TargetStratumBridge
from .charts import LatticeChartCodec, PyXtalChartCatalog
from .contracts import (
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
    write_json_exclusive,
)
from .events import TopologyEvent, TopologyEventType
from .kernel import TopologyEventKernel, TransitionError
from .model import WQCoDenoiser, WQModelConfig, WQModelOutput, WQVariant
from .protocol import RegisteredProtocol, load_protocol
from .revision import (
    FieldRef,
    REVISION_THRESHOLDS,
    RevisionBudget,
    load_revision_threshold_lock,
)
from .runtime import (
    StateExpansionError,
    concatenate_tensor_batches,
    compute_geometry_evidence,
    expand_state,
    project_atom_scores,
    split_model_output,
    tensorize_state,
)
from .state import GeometryEvidence, OrbitState, StratifiedState
from .vocabulary import (
    ATOMIC_NUMBER_TO_CLASS,
    MP20_ATOMIC_NUMBERS,
    crystal_system_from_space_group,
    target_to_atomic_number,
)


REVISION_CONTROLS = (
    "auto",
    "none",
    "confidence",
    "geometry",
    "random-count",
    "shuffled-geometry",
    "extra-call",
)
CALL_GRID = (16, 32, 64, 128)
EVENT_CLASS = {
    TopologyEventType.NONE: 0,
    TopologyEventType.BIRTH: 1,
    TopologyEventType.DEATH: 2,
    TopologyEventType.WYCKOFF_CHANGE: 3,
    TopologyEventType.SPECIES_CHANGE: 4,
}


@dataclasses.dataclass(frozen=True, slots=True)
class SamplingConfig:
    checkpoint: str
    output_jsonl: str
    attempt_ledger: str
    experiment_id: str
    variant: WQVariant
    training_seed: int
    sampling_seed: int
    attempts: int
    pairing_id: str | None = None
    start_ordinal: int = 0
    backbone_calls: int = 64
    revision_control: str = "auto"
    revision_threshold: float = 0.7
    revision_lock: str | None = None
    temperature: float = 1.0
    disc_once_tau: float = 0.5
    inference_batch_size: int = 64
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.training_seed not in {11, 23, 47}:
            raise ValueError("registered training seeds are 11/23/47")
        if self.attempts <= 0 or self.start_ordinal < 0:
            raise ValueError("attempt count must be positive and ordinal non-negative")
        if self.backbone_calls not in CALL_GRID:
            raise ValueError(f"backbone_calls must be in {CALL_GRID}")
        if self.revision_control not in REVISION_CONTROLS:
            raise ValueError(f"unknown revision control: {self.revision_control}")
        if self.revision_threshold not in REVISION_THRESHOLDS:
            raise ValueError("revision threshold is outside the frozen grid")
        if not 0.0 < self.temperature <= 5.0:
            raise ValueError("temperature must be in (0,5]")
        if self.disc_once_tau not in {0.25, 0.5, 0.75, 1.0}:
            raise ValueError("DISC-ONCE tau/T is outside the registered grid")
        if self.inference_batch_size not in {16, 32, 64, 128}:
            raise ValueError("inference batch size must be one of 16/32/64/128")


@dataclasses.dataclass(slots=True)
class _AttemptContext:
    attempt_id: str
    python_rng: random.Random
    torch_generator: torch.Generator
    calls: dict[str, int]
    trace: list[dict[str, Any]]
    last_state: StratifiedState | None = None
    reverse_step: int = -1
    topology_event_steps: set[int] = dataclasses.field(default_factory=set)


def _reserve_topology_event_slot(context: _AttemptContext) -> None:
    """Enforce one explicit event-kernel transition per reverse step."""

    if context.reverse_step < 0:
        raise TransitionError("topology event occurred outside a reverse step")
    if context.reverse_step in context.topology_event_steps:
        raise TransitionError(
            f"more than one topology event at reverse step {context.reverse_step}"
        )
    context.topology_event_steps.add(context.reverse_step)


def _resolved_revision_control(config: SamplingConfig) -> str:
    if config.revision_control != "auto":
        return config.revision_control
    if config.variant is WQVariant.STRAT_GEO:
        return "geometry"
    if config.variant is WQVariant.STRAT_CONF:
        return "confidence"
    return "none"


def _autocast(device: torch.device) -> Any:
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _log_softmax(values: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.log_softmax(values.float() / temperature, dim=-1)


def _draw_class(
    logits: torch.Tensor,
    generator: torch.Generator,
    *,
    legal: Sequence[int] | None = None,
    temperature: float = 1.0,
) -> tuple[int, float]:
    values = logits.float() / temperature
    if legal is not None:
        legal_values = tuple(sorted(set(int(value) for value in legal)))
        if not legal_values:
            raise ValueError("categorical draw has empty legal support")
        mask = torch.full_like(values, -torch.inf)
        indices = torch.tensor(legal_values, device=values.device, dtype=torch.long)
        mask[indices] = values[indices]
        values = mask
    probabilities = torch.softmax(values, dim=-1)
    if not bool(torch.all(torch.isfinite(probabilities))) or float(probabilities.sum()) <= 0:
        raise ValueError("categorical probabilities are not finite")
    selected = int(torch.multinomial(probabilities, 1, generator=generator).item())
    return selected, float(probabilities[selected].item())


def _draw_gaussian(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    dimension: int,
    generator: torch.Generator,
) -> tuple[float, ...]:
    if dimension == 0:
        return ()
    noise = torch.randn(
        dimension,
        device=mean.device,
        dtype=torch.float32,
        generator=generator,
    )
    values = mean[:dimension].float() + log_scale[:dimension].float().exp() * noise
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("Gaussian bridge produced non-finite values")
    return tuple(float(value % 1.0) for value in values.detach().cpu())


def _alpha(time_value: float) -> float:
    return max(math.cos(0.5 * math.pi * float(time_value)) ** 2, 1.0e-8)


def _sigma(time_value: float) -> float:
    return math.exp((1.0 - time_value) * math.log(0.005) + time_value * math.log(0.5))


def _entropy_uncertainty(logits: torch.Tensor) -> np.ndarray:
    probabilities = torch.softmax(logits.float(), dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=-1)
    maximum = math.log(logits.shape[-1])
    return (entropy / maximum).detach().cpu().numpy()


def _load_model(
    checkpoint_path: str | Path,
    *,
    config: SamplingConfig,
    protocol: RegisteredProtocol,
    device: torch.device,
) -> tuple[WQCoDenoiser, dict[str, Any]]:
    checkpoint = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "wqcodiff_ema_model_v1":
        raise ValueError("sampling requires a final EMA checkpoint")
    if payload.get("protocol_name") != protocol.name or payload.get("protocol_sha256") != protocol.sha256:
        raise ValueError("checkpoint/protocol hash mismatch")
    training = payload.get("training_config", {})
    if str(training.get("variant")) != config.variant.value:
        raise ValueError("checkpoint variant does not match sampling variant")
    if int(training.get("training_seed", -1)) != config.training_seed:
        raise ValueError("checkpoint training seed does not match sampling config")
    source_bundle_sha256 = str(payload.get("source_bundle_sha256", ""))
    if len(source_bundle_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in source_bundle_sha256
    ):
        raise ValueError("checkpoint lacks a valid frozen source bundle hash")
    dataset_files = payload.get("dataset_files")
    if not isinstance(dataset_files, list) or not dataset_files:
        raise ValueError("checkpoint lacks training dataset file identities")
    resolved_control = _resolved_revision_control(config)
    if config.revision_lock:
        lock = load_revision_threshold_lock(
            config.revision_lock,
            protocol_name=protocol.name,
            protocol_sha256=protocol.sha256,
        )
        if float(lock["selected_threshold"]) != config.revision_threshold:
            raise ValueError("sampling threshold differs from the frozen Day-7 lock")
    elif bool(payload.get("paper_eligible")) and resolved_control != "none":
        raise ValueError(
            "paper-eligible revision sampling requires the frozen Day-7 threshold lock"
        )
    model_config = WQModelConfig(**payload["model_config"])
    model = WQCoDenoiser(model_config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    checkpoint_digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_digest.update(block)
    return model, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest.hexdigest(),
        "source_bundle_sha256": source_bundle_sha256,
        "training_dataset_files": dataset_files,
    }


def _initial_state(
    model: WQCoDenoiser,
    catalog: PyXtalChartCatalog,
    context: _AttemptContext,
    config: SamplingConfig,
    device: torch.device,
) -> tuple[StratifiedState, set[str], set[str]]:
    start_time = 1.0 - 1.0 / (config.backbone_calls + 1)
    with torch.no_grad(), _autocast(device):
        time_tensor = torch.tensor([start_time], device=device)
        masked = model.forward_prior(
            time_tensor,
            torch.zeros(1, dtype=torch.long, device=device),
        )
        if config.variant is WQVariant.ATOM_JOINT:
            space_group = 1
        else:
            sg_target, _ = _draw_class(
                masked.space_group_logits[0],
                context.torch_generator,
                temperature=config.temperature,
            )
            space_group = sg_target + 1
        conditioned = model.forward_prior(
            time_tensor,
            torch.tensor([space_group], dtype=torch.long, device=device),
        )
    context.calls["prior"] += 2
    legal_wyckoff = [
        value
        for value in catalog.types(space_group)
        if int(catalog.get(space_group, value).primitive_multiplicity) <= 20
    ]
    if config.variant is WQVariant.ATOM_JOINT:
        wyckoff_type = 0
    else:
        wyckoff_type, _ = _draw_class(
            conditioned.first_wyckoff_logits[0],
            context.torch_generator,
            legal=legal_wyckoff,
            temperature=config.temperature,
        )
    species_target, _ = _draw_class(
        conditioned.first_species_logits[0],
        context.torch_generator,
        temperature=config.temperature,
    )
    if config.variant is WQVariant.D3PM:
        # The D3PM lane starts its categorical chain from the registered
        # uniform terminal distribution, while the learned prior still fixes
        # the space group and initializes continuous charts.
        wyckoff_type = legal_wyckoff[
            int(
                torch.randint(
                    len(legal_wyckoff),
                    (1,),
                    generator=context.torch_generator,
                    device=device,
                ).item()
            )
        ]
        species_target = int(
            torch.randint(
                len(MP20_ATOMIC_NUMBERS),
                (1,),
                generator=context.torch_generator,
                device=device,
            ).item()
        )
    spec = catalog.get(space_group, wyckoff_type)
    clean_coordinate = _draw_gaussian(
        conditioned.first_coordinate_mean[0],
        conditioned.first_coordinate_log_scale[0],
        spec.dimension,
        context.torch_generator,
    )
    if spec.dimension:
        q_noise = torch.randn(
            spec.dimension,
            generator=context.torch_generator,
            device=device,
        ).cpu().numpy()
        coordinate = tuple(
            float((value + _sigma(start_time) * noise) % 1.0)
            for value, noise in zip(clean_coordinate, q_noise)
        )
    else:
        coordinate = ()
    lattice_system = crystal_system_from_space_group(space_group)
    lattice_dimension = LatticeChartCodec.dimension(lattice_system)
    clean_lattice_tensor = (
        conditioned.lattice_chart_mean[0, :lattice_dimension].float()
        + conditioned.lattice_chart_log_scale[0, :lattice_dimension].float().exp()
        * torch.randn(
            lattice_dimension,
            generator=context.torch_generator,
            device=device,
        )
    )
    epsilon = torch.randn(
        lattice_dimension,
        generator=context.torch_generator,
        device=device,
    )
    alpha = _alpha(start_time)
    noisy_lattice = math.sqrt(alpha) * clean_lattice_tensor + math.sqrt(1.0 - alpha) * epsilon
    if not bool(torch.all(torch.isfinite(noisy_lattice))):
        raise ValueError("initial lattice prior produced non-finite values")
    orbit = OrbitState(
        orbit_id="o0",
        wyckoff_type=wyckoff_type,
        species=target_to_atomic_number(species_target),
        multiplicity=spec.multiplicity,
        primitive_multiplicity=spec.primitive_multiplicity,
        chart_dimension=spec.dimension,
        free_coordinate=coordinate,
    )
    state = StratifiedState(
        space_group=space_group,
        lattice_system=lattice_system,
        lattice_chart=tuple(float(value) for value in noisy_lattice.detach().cpu()),
        orbits=(orbit,),
        attempt_id=context.attempt_id,
        timestep=start_time,
        space_group_committed=True,
    )
    context.trace.append(
        {
            "step": -1,
            "reverse_step": -1,
            "action": "space_group_commit_and_first_orbit_prior",
            "space_group": space_group,
            "wyckoff_type": wyckoff_type,
            "species": orbit.species,
            "space_group_rollback_allowed": False,
            "discrete_terminal_distribution": (
                "uniform_categorical" if config.variant is WQVariant.D3PM else "mask"
            ),
            "representation": (
                "dynamic_atom_set_p1"
                if config.variant is WQVariant.ATOM_JOINT
                else "wyckoff_quotient"
            ),
        }
    )
    if config.variant is WQVariant.D3PM:
        return state, set(), set()
    return state, {orbit.orbit_id}, {orbit.orbit_id}


def _d3pm_posterior_draw(
    logits: torch.Tensor,
    current: int,
    *,
    current_time: float,
    next_time: float,
    generator: torch.Generator,
    temperature: float,
    legal: Sequence[int] | None = None,
) -> int:
    if legal is None:
        candidate_count = int(logits.numel())
        if candidate_count == 0:
            raise ValueError("D3PM posterior has empty legal support")
        clean = torch.softmax(logits.float() / temperature, dim=-1)
        alpha_t = _alpha(current_time)
        alpha_s = _alpha(next_time)
        transition = min(max(alpha_t / max(alpha_s, 1.0e-12), 0.0), 1.0)
        uniform = 1.0 / candidate_count
        prior_s = alpha_s * clean + (1.0 - alpha_s) * uniform
        likelihood = torch.full_like(prior_s, (1.0 - transition) * uniform)
        if 0 <= current < candidate_count:
            likelihood[current] += transition
        posterior = prior_s * likelihood
        posterior = posterior / posterior.sum().clamp_min(1.0e-12)
        return int(torch.multinomial(posterior, 1, generator=generator).item())

    candidates = list(legal)
    if not candidates:
        raise ValueError("D3PM posterior has empty legal support")
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=logits.device)
    clean = torch.softmax(logits.float()[candidate_tensor] / temperature, dim=-1)
    alpha_t = _alpha(current_time)
    alpha_s = _alpha(next_time)
    transition = min(max(alpha_t / max(alpha_s, 1.0e-12), 0.0), 1.0)
    uniform = 1.0 / len(candidates)
    prior_s = alpha_s * clean + (1.0 - alpha_s) * uniform
    likelihood = torch.full_like(prior_s, (1.0 - transition) * uniform)
    if current in candidates:
        likelihood[candidates.index(current)] += transition
    posterior = prior_s * likelihood
    posterior = posterior / posterior.sum().clamp_min(1.0e-12)
    selected = int(torch.multinomial(posterior, 1, generator=generator).item())
    return int(candidates[selected])


def _d3pm_reverse_fields(
    state: StratifiedState,
    output: WQModelOutput,
    catalog: PyXtalChartCatalog,
    context: _AttemptContext,
    *,
    current_time: float,
    next_time: float,
    temperature: float,
) -> StratifiedState:
    """One exact uniform-kernel posterior step using predicted clean categories."""

    orbits = list(state.orbits)
    current_count = sum(orbit.primitive_multiplicity for orbit in orbits)
    wyckoff_specs = tuple(
        (candidate, catalog.get(state.space_group, candidate))
        for candidate in catalog.types(state.space_group)
    )
    wyckoff_changed = False
    for index, current in enumerate(tuple(orbits)):
        species_class = ATOMIC_NUMBER_TO_CLASS[int(current.species)]
        next_species_class = _d3pm_posterior_draw(
            output.species_logits[index],
            species_class,
            current_time=current_time,
            next_time=next_time,
            generator=context.torch_generator,
            temperature=temperature,
        )
        next_species = target_to_atomic_number(next_species_class)
        legal_wyckoff: list[int] = []
        for candidate, spec in wyckoff_specs:
            target_count = (
                current_count
                - current.primitive_multiplicity
                + spec.primitive_multiplicity
            )
            if 1 <= target_count <= 20:
                legal_wyckoff.append(candidate)
        next_wyckoff = _d3pm_posterior_draw(
            output.wyckoff_logits[index],
            current.wyckoff_type,
            current_time=current_time,
            next_time=next_time,
            generator=context.torch_generator,
            temperature=temperature,
            legal=legal_wyckoff,
        )
        replacement = dataclasses.replace(current, species=next_species)
        # A Wyckoff type transition changes strata.  Limit it to one orbit per
        # reverse step and instantiate the target chart through the learned
        # one-shot bridge; no rejected draw is retried.
        if next_wyckoff != current.wyckoff_type and not wyckoff_changed:
            spec = catalog.get(state.space_group, next_wyckoff)
            coordinate = _draw_gaussian(
                output.bridge_mean[index],
                output.bridge_log_scale[index],
                spec.dimension,
                context.torch_generator,
            )
            replacement = OrbitState(
                orbit_id=current.orbit_id,
                wyckoff_type=next_wyckoff,
                species=next_species,
                multiplicity=spec.multiplicity,
                primitive_multiplicity=spec.primitive_multiplicity,
                chart_dimension=spec.dimension,
                free_coordinate=coordinate,
            )
            context.calls["bridge"] += 1
            wyckoff_changed = True
            current_count += (
                spec.primitive_multiplicity - current.primitive_multiplicity
            )
        if replacement.species != current.species or replacement.wyckoff_type != current.wyckoff_type:
            context.trace.append(
                {
                    "step": len(context.trace),
                    "reverse_step": context.reverse_step,
                    "action": "d3pm_categorical_transition",
                    "orbit_id": current.orbit_id,
                    "species_before": current.species,
                    "species_after": replacement.species,
                    "wyckoff_before": current.wyckoff_type,
                    "wyckoff_after": replacement.wyckoff_type,
                    "dimension_before": current.chart_dimension,
                    "dimension_after": replacement.chart_dimension,
                }
            )
        orbits[index] = replacement
    return state.replace_orbits(orbits)


def _replace_masked_fields(
    state: StratifiedState,
    output: WQModelOutput,
    catalog: PyXtalChartCatalog,
    context: _AttemptContext,
    *,
    masked_species: set[str],
    masked_wyckoff: set[str],
    pending_revision: set[FieldRef],
    time_value: float,
    final_step: bool,
    temperature: float,
) -> StratifiedState:
    orbits = list(state.orbits)
    for index, orbit in enumerate(tuple(orbits)):
        if orbit.orbit_id in masked_species:
            target, confidence = _draw_class(
                output.species_logits[index],
                context.torch_generator,
                temperature=temperature,
            )
            if final_step or confidence >= time_value:
                replacement = dataclasses.replace(
                    orbits[index], species=target_to_atomic_number(target)
                )
                was_revision = FieldRef(orbit.orbit_id, "species") in pending_revision
                context.trace.append(
                    {
                        "step": len(context.trace),
                        "reverse_step": context.reverse_step,
                        "action": "species_commit",
                        "orbit_id": orbit.orbit_id,
                        "old": orbits[index].species,
                        "new": replacement.species,
                        "confidence": confidence,
                        "revision_fill": was_revision,
                    }
                )
                orbits[index] = replacement
                masked_species.remove(orbit.orbit_id)
                pending_revision.discard(FieldRef(orbit.orbit_id, "species"))
        if orbit.orbit_id in masked_wyckoff:
            current = orbits[index]
            legal = []
            current_atom_count = sum(
                int(value.primitive_multiplicity) for value in orbits
            )
            for candidate in catalog.types(state.space_group):
                spec = catalog.get(state.space_group, candidate)
                new_count = (
                    current_atom_count
                    - int(current.primitive_multiplicity)
                    + int(spec.primitive_multiplicity)
                )
                if 1 <= new_count <= 20:
                    legal.append(candidate)
            target, confidence = _draw_class(
                output.wyckoff_logits[index],
                context.torch_generator,
                legal=legal,
                temperature=temperature,
            )
            if final_step or confidence >= time_value:
                spec = catalog.get(state.space_group, target)
                coordinate = _draw_gaussian(
                    output.bridge_mean[index],
                    output.bridge_log_scale[index],
                    spec.dimension,
                    context.torch_generator,
                )
                replacement = OrbitState(
                    orbit_id=current.orbit_id,
                    wyckoff_type=target,
                    species=current.species,
                    multiplicity=spec.multiplicity,
                    primitive_multiplicity=spec.primitive_multiplicity,
                    chart_dimension=spec.dimension,
                    free_coordinate=coordinate,
                )
                was_revision = FieldRef(orbit.orbit_id, "wyckoff_type") in pending_revision
                context.trace.append(
                    {
                        "step": len(context.trace),
                        "reverse_step": context.reverse_step,
                        "action": "wyckoff_commit",
                        "orbit_id": orbit.orbit_id,
                        "old": current.wyckoff_type,
                        "new": target,
                        "confidence": confidence,
                        "revision_fill": was_revision,
                        "dimension_before": current.chart_dimension,
                        "dimension_after": spec.dimension,
                    }
                )
                orbits[index] = replacement
                masked_wyckoff.remove(orbit.orbit_id)
                pending_revision.discard(FieldRef(orbit.orbit_id, "wyckoff_type"))
    return state.replace_orbits(orbits)


def _continuous_step(
    state: StratifiedState,
    expanded: Any,
    output: WQModelOutput,
    *,
    current_time: float,
    next_time: float,
) -> tuple[StratifiedState, tuple[float, ...]]:
    atom_scores = output.atom_coordinate_score.float().detach().cpu().numpy()
    tangent_scores = project_atom_scores(expanded, atom_scores)
    coefficient = 0.5 * (_sigma(current_time) ** 2 - _sigma(next_time) ** 2)
    orbits: list[OrbitState] = []
    score_norms: list[float] = []
    for orbit, score in zip(state.orbits, tangent_scores):
        if not np.all(np.isfinite(score)):
            raise StateExpansionError("non-finite tangent score")
        coordinate = tuple(
            float((value + coefficient * delta) % 1.0)
            for value, delta in zip(orbit.free_coordinate, score)
        )
        orbits.append(dataclasses.replace(orbit, free_coordinate=coordinate))
        score_norms.append(float(np.linalg.norm(score)))
    chart = np.asarray(state.lattice_chart, dtype=np.float64)
    epsilon = output.lattice_score[0, : len(chart)].float().detach().cpu().numpy()
    alpha_current = _alpha(current_time)
    alpha_next = _alpha(next_time)
    clean = (chart - math.sqrt(1.0 - alpha_current) * epsilon) / math.sqrt(alpha_current)
    next_chart = math.sqrt(alpha_next) * clean + math.sqrt(1.0 - alpha_next) * epsilon
    if not np.all(np.isfinite(next_chart)):
        raise StateExpansionError("non-finite lattice reverse step")
    return (
        dataclasses.replace(
            state,
            lattice_chart=tuple(float(value) for value in next_chart),
            orbits=tuple(orbits),
            timestep=float(next_time),
        ),
        tuple(score_norms),
    )


def _event_logits(
    state: StratifiedState,
    output: WQModelOutput,
    kernel: TopologyEventKernel,
    config: SamplingConfig,
    *,
    step: int,
    midpoint: int,
    recovery: bool = False,
) -> list[tuple[TopologyEvent, float]]:
    # Event support can contain thousands of legal birth/species/Wyckoff
    # candidates.  Indexing CUDA tensors inside that Python loop used to
    # trigger one device synchronization per candidate when the scalar was
    # converted below (more than one million ``Tensor.cpu`` calls for a
    # 64-structure/16-step recovery batch).  Keep the registered GPU
    # log-softmax calculation unchanged, then transfer each compact head once
    # before enumerating the exact same ordered support on CPU.
    type_log = _log_softmax(output.event_logits[0], config.temperature).detach().cpu()
    pointer_log = _log_softmax(
        output.event_orbit_logits, config.temperature
    ).detach().cpu()
    birth_species = _log_softmax(
        output.birth_species_logits[0], config.temperature
    ).detach().cpu()
    birth_wyckoff = _log_softmax(
        output.birth_wyckoff_logits[0], config.temperature
    ).detach().cpu()
    orbit_species = _log_softmax(
        output.species_logits, config.temperature
    ).detach().cpu()
    orbit_wyckoff = _log_softmax(
        output.wyckoff_logits, config.temperature
    ).detach().cpu()
    orbit_index = {orbit.orbit_id: index for index, orbit in enumerate(state.orbits)}
    result: list[tuple[TopologyEvent, float]] = []
    cutoff = max(1, int(round(config.disc_once_tau * config.backbone_calls)))
    for event in kernel.legal_events(state):
        kind = event.event_type
        if not recovery and config.variant in {WQVariant.AR, WQVariant.DLM_MONO} and kind not in {
            TopologyEventType.NONE,
            TopologyEventType.BIRTH,
        }:
            continue
        if (
            not recovery
            and config.variant is WQVariant.DISC_ONCE
            and step >= cutoff
            and kind is not TopologyEventType.NONE
        ):
            continue
        if not recovery and config.variant in {WQVariant.STRAT_CONF, WQVariant.STRAT_GEO} and step >= midpoint and kind in {
            TopologyEventType.WYCKOFF_CHANGE,
            TopologyEventType.SPECIES_CHANGE,
        }:
            # After the revision phase starts these changes must arise from a
            # logged true remask, not an unaccounted event-head shortcut.
            continue
        value = type_log[EVENT_CLASS[kind]]
        if kind is TopologyEventType.BIRTH:
            value = (
                value
                + birth_species[ATOMIC_NUMBER_TO_CLASS[int(event.target_species)]]
                + birth_wyckoff[int(event.target_wyckoff_type)]
            )
        elif kind is TopologyEventType.DEATH:
            value = value + pointer_log[orbit_index[str(event.orbit_id)]]
        elif kind is TopologyEventType.WYCKOFF_CHANGE:
            index = orbit_index[str(event.orbit_id)]
            value = value + pointer_log[index] + orbit_wyckoff[index, int(event.target_wyckoff_type)]
        elif kind is TopologyEventType.SPECIES_CHANGE:
            index = orbit_index[str(event.orbit_id)]
            target_class = ATOMIC_NUMBER_TO_CLASS[int(event.target_species)]
            value = value + pointer_log[index] + orbit_species[index, target_class]
        result.append((event, float(value)))
    return result


def _draw_event(
    weighted_logits: Sequence[tuple[TopologyEvent, float]],
    context: _AttemptContext,
    *,
    time_value: float,
) -> TopologyEvent:
    if not weighted_logits:
        return TopologyEvent(TopologyEventType.NONE)
    maximum = max(value for _, value in weighted_logits)
    weights = [math.exp(value - maximum) for _, value in weighted_logits]
    total = math.fsum(weights)
    probabilities = [value / total for value in weights]
    confidence = max(probabilities)
    # A deterministic confidence gate prevents an untrained event head from
    # churning one topology event at every reverse step.
    if confidence < 0.5 + 0.45 * time_value:
        return TopologyEvent(TopologyEventType.NONE)
    threshold = context.python_rng.random()
    cumulative = 0.0
    for (event, _), probability in zip(weighted_logits, probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return event
    return weighted_logits[-1][0]


def _apply_event(
    state: StratifiedState,
    event: TopologyEvent,
    output: WQModelOutput,
    catalog: PyXtalChartCatalog,
    context: _AttemptContext,
) -> StratifiedState:
    if event.event_type is TopologyEventType.NONE:
        return state
    _reserve_topology_event_slot(context)
    coordinate: tuple[float, ...] | None = None
    if event.event_type is TopologyEventType.BIRTH:
        spec = catalog.get(state.space_group, int(event.target_wyckoff_type))
        coordinate = _draw_gaussian(
            output.birth_coordinate_mean[0],
            output.birth_coordinate_log_scale[0],
            spec.dimension,
            context.torch_generator,
        )
    elif event.event_type is TopologyEventType.WYCKOFF_CHANGE:
        spec = catalog.get(state.space_group, int(event.target_wyckoff_type))
        index = next(
            index for index, orbit in enumerate(state.orbits) if orbit.orbit_id == event.orbit_id
        )
        coordinate = _draw_gaussian(
            output.bridge_mean[index],
            output.bridge_log_scale[index],
            spec.dimension,
            context.torch_generator,
        )

    def residual(_state: StratifiedState, spec: Any, _species: int, base: tuple[float, ...]) -> Sequence[float]:
        if coordinate is None or len(coordinate) != spec.dimension:
            raise ValueError("learned bridge coordinate missing or dimension-mismatched")
        return tuple(target - source for target, source in zip(coordinate, base))

    bridge = TargetStratumBridge(
        catalog,
        residual_model=residual if coordinate is not None else None,
    )
    kernel = TopologyEventKernel(
        catalog=catalog,
        bridge=bridge,
        species=MP20_ATOMIC_NUMBERS,
    )
    before = state.continuous_dimension
    target = kernel.apply(state, event, context.python_rng)
    if coordinate is not None:
        context.calls["bridge"] += 1
    context.trace.append(
        {
            "step": len(context.trace),
            "reverse_step": context.reverse_step,
            "action": "topology_event",
            **event.to_dict(),
            "dimension_before": before,
            "dimension_after": target.continuous_dimension,
            "atom_count_before": state.atom_count,
            "atom_count_after": target.atom_count,
        }
    )
    return target


def _revision_scores(
    state: StratifiedState,
    output: WQModelOutput,
    control: str,
) -> dict[FieldRef, float]:
    values: dict[FieldRef, float] = {}
    if control in {"geometry", "shuffled-geometry", "random-count"}:
        scores = torch.sigmoid(output.revision_logits.float()).detach().cpu().numpy()
        for index, orbit in enumerate(state.orbits):
            values[FieldRef(orbit.orbit_id, "existence")] = float(scores[index, 0])
            values[FieldRef(orbit.orbit_id, "wyckoff_type")] = float(scores[index, 1])
            values[FieldRef(orbit.orbit_id, "species")] = float(scores[index, 2])
        return values
    if control == "confidence":
        species_conf = torch.softmax(output.species_logits.float(), dim=-1).max(dim=-1).values
        wyckoff_conf = torch.softmax(output.wyckoff_logits.float(), dim=-1).max(dim=-1).values
        event_conf = torch.softmax(output.event_logits[0].float(), dim=-1).max()
        for index, orbit in enumerate(state.orbits):
            values[FieldRef(orbit.orbit_id, "existence")] = float(1.0 - event_conf)
            values[FieldRef(orbit.orbit_id, "wyckoff_type")] = float(1.0 - wyckoff_conf[index])
            values[FieldRef(orbit.orbit_id, "species")] = float(1.0 - species_conf[index])
    return values


def _select_revisions(
    state: StratifiedState,
    output: WQModelOutput,
    budget: RevisionBudget,
    config: SamplingConfig,
    context: _AttemptContext,
    control: str,
) -> tuple[FieldRef, ...]:
    if control in {"none", "extra-call"}:
        return ()
    scores = _revision_scores(state, output, control)
    if control == "random-count":
        preview = budget.preview(
            scores,
            threshold=config.revision_threshold,
            current_field_count=state.field_count,
        )
        candidates = [
            field
            for field in scores
            if budget.count(field) < 2
        ]
        context.python_rng.shuffle(candidates)
        selected_random = candidates[: len(preview.selected)]
        scores = {
            field: (1.0 if field in selected_random else 0.0)
            for field in candidates
        }
    decision = budget.select(
        scores,
        threshold=config.revision_threshold,
        current_field_count=state.field_count,
    )
    context.trace.append(
        {
            "step": len(context.trace),
            "reverse_step": context.reverse_step,
            "action": "revision_decision",
            "control": control,
            "threshold": config.revision_threshold,
            "selected": [dataclasses.asdict(field) for field in decision.selected],
            "eligible": decision.eligible,
            "remaining_total_budget": decision.remaining_total_budget,
        }
    )
    return decision.selected


def _sample_one(
    model: WQCoDenoiser,
    catalog: PyXtalChartCatalog,
    context: _AttemptContext,
    config: SamplingConfig,
    device: torch.device,
) -> dict[str, Any]:
    state, masked_species, masked_wyckoff = _initial_state(
        model, catalog, context, config, device
    )
    context.last_state = state
    control = _resolved_revision_control(config)
    pending_revision: set[FieldRef] = set()
    pending_existence: set[str] = set()
    midpoint = config.backbone_calls // 2
    disc_once_cutoff = max(
        1, int(round(config.disc_once_tau * config.backbone_calls))
    )
    revision_budget: RevisionBudget | None = None
    score_norms: tuple[float, ...] = (0.0,) * len(state.orbits)
    uncertainties: tuple[float, ...] = (0.0,) * len(state.orbits)
    start_time = state.timestep
    times = np.linspace(start_time, 0.0, config.backbone_calls + 1)

    for step in range(config.backbone_calls):
        context.reverse_step = step
        current_time = float(times[step])
        next_time = float(times[step + 1])
        expanded = expand_state(state, catalog)
        context.calls["projection"] += 1
        evidence = list(
            compute_geometry_evidence(
                state,
                expanded,
                score_norms=score_norms,
                basin_uncertainties=uncertainties,
            )
        )
        if control == "shuffled-geometry" and len(evidence) > 1:
            context.python_rng.shuffle(evidence)
        batch = tensorize_state(
            state,
            expanded,
            evidence,
            time=current_time,
            masked_species=frozenset(masked_species),
            masked_wyckoff=frozenset(masked_wyckoff),
        ).to(device)
        with torch.no_grad(), _autocast(device):
            output = model(batch, variant=config.variant)
            context.calls["joint"] += 1
            if control == "extra-call":
                # The second result is deliberately used, not discarded.  In
                # eval mode it is normally identical, making this a clean
                # control for "more network calls" without topology rollback.
                output = model(batch, variant=config.variant)
                context.calls["joint"] += 1

        state, score_norms = _continuous_step(
            state,
            expanded,
            output,
            current_time=current_time,
            next_time=next_time,
        )
        context.last_state = state
        species_uncertainty = _entropy_uncertainty(output.species_logits)
        wyckoff_uncertainty = _entropy_uncertainty(output.wyckoff_logits)
        uncertainties = tuple(
            float(max(species_uncertainty[index], wyckoff_uncertainty[index]))
            for index in range(len(state.orbits))
        )
        state = _replace_masked_fields(
            state,
            output,
            catalog,
            context,
            masked_species=masked_species,
            masked_wyckoff=masked_wyckoff,
            pending_revision=pending_revision,
            time_value=current_time,
            final_step=(
                step == config.backbone_calls - 1
                or (
                    config.variant is WQVariant.DISC_ONCE
                    and step == disc_once_cutoff - 1
                )
            ),
            temperature=config.temperature,
        )
        context.last_state = state
        if config.variant is WQVariant.D3PM:
            state = _d3pm_reverse_fields(
                state,
                output,
                catalog,
                context,
                current_time=current_time,
                next_time=next_time,
                temperature=config.temperature,
            )
            context.last_state = state

        if step == midpoint:
            revision_budget = RevisionBudget(state.field_count)
        if (
            revision_budget is not None
            and step >= midpoint
            and step < config.backbone_calls - 1
        ):
            selected = _select_revisions(
                state,
                output,
                revision_budget,
                config,
                context,
                control,
            )
            for field in selected:
                pending_revision.add(field)
                if field.field == "species":
                    masked_species.add(field.orbit_id)
                elif field.field == "wyckoff_type":
                    masked_wyckoff.add(field.orbit_id)
                else:
                    pending_existence.add(field.orbit_id)

        kernel = TopologyEventKernel(
            catalog=catalog,
            bridge=TargetStratumBridge(catalog),
            species=MP20_ATOMIC_NUMBERS,
        )
        if pending_existence:
            orbit_id = sorted(pending_existence)[0]
            pending_existence.remove(orbit_id)
            pending_revision.discard(FieldRef(orbit_id, "existence"))
            death = next(
                (
                    event
                    for event in kernel.legal_events(state)
                    if event.event_type is TopologyEventType.DEATH
                    and event.orbit_id == orbit_id
                ),
                None,
            )
            if death is not None:
                index = next(
                    index for index, orbit in enumerate(state.orbits) if orbit.orbit_id == orbit_id
                )
                death_logit = float(output.event_logits[0, EVENT_CLASS[TopologyEventType.DEATH]])
                death_logit += float(output.event_orbit_logits[index])
                none_logit = float(output.event_logits[0, EVENT_CLASS[TopologyEventType.NONE]])
                probability = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, none_logit - death_logit))))
                event = death if context.python_rng.random() < probability else TopologyEvent(TopologyEventType.NONE)
            else:
                event = TopologyEvent(TopologyEventType.NONE)
        else:
            event = _draw_event(
                _event_logits(
                    state,
                    output,
                    kernel,
                    config,
                    step=step,
                    midpoint=midpoint,
                ),
                context,
                time_value=current_time,
            )
        before_ids = {orbit.orbit_id for orbit in state.orbits}
        state = _apply_event(state, event, output, catalog, context)
        context.last_state = state
        after_ids = {orbit.orbit_id for orbit in state.orbits}
        for removed in before_ids - after_ids:
            masked_species.discard(removed)
            masked_wyckoff.discard(removed)
            pending_existence.discard(removed)
            pending_revision = {
                field for field in pending_revision if field.orbit_id != removed
            }
        if len(score_norms) != len(state.orbits):
            score_norms = (0.0,) * len(state.orbits)
            uncertainties = (0.0,) * len(state.orbits)

    if masked_species or masked_wyckoff or pending_existence:
        raise StateExpansionError("reverse process ended with uncommitted fields")
    final_expanded = expand_state(state, catalog)
    context.calls["projection"] += 1
    structure = final_expanded.pymatgen_structure()
    cif = structure.to(fmt="cif")
    structure_hash = hashlib.sha256(cif.encode("utf-8")).hexdigest()
    return {
        "schema": "wqcodiff_generation_attempt_v1",
        "attempt_id": context.attempt_id,
        "experiment_id": config.experiment_id,
        "pairing_id": config.pairing_id or config.experiment_id,
        "method": config.variant.value,
        "training_seed": config.training_seed,
        "sampling_seed": config.sampling_seed,
        "status": AttemptStatus.SUCCEEDED.value,
        "state": state.to_dict(),
        "structure": structure.as_dict(),
        "structure_cif_sha256": structure_hash,
        "intended_space_group": (
            None if config.variant is WQVariant.ATOM_JOINT else state.space_group
        ),
        "redetected_space_group": final_expanded.redetected_space_group,
        "atom_count": state.atom_count,
        "orbit_count": len(state.orbits),
        "revision_control": control,
        "revision_threshold": config.revision_threshold,
        "disc_once_tau": (
            config.disc_once_tau
            if config.variant is WQVariant.DISC_ONCE
            else None
        ),
        "temperature": config.temperature,
        "revision_initial_field_count": (
            None
            if revision_budget is None
            else revision_budget.initial_field_count
        ),
        "revision_churn": (
            None if revision_budget is None else revision_budget.churn
        ),
        "backbone_calls": config.backbone_calls,
        "calls": dict(context.calls),
        "trace": context.trace,
    }


@dataclasses.dataclass(slots=True)
class _SampleWork:
    context: _AttemptContext
    state: StratifiedState | None = None
    masked_species: set[str] = dataclasses.field(default_factory=set)
    masked_wyckoff: set[str] = dataclasses.field(default_factory=set)
    pending_revision: set[FieldRef] = dataclasses.field(default_factory=set)
    pending_existence: set[str] = dataclasses.field(default_factory=set)
    revision_budget: RevisionBudget | None = None
    score_norms: tuple[float, ...] = ()
    uncertainties: tuple[float, ...] = ()
    artifact: dict[str, Any] | None = None
    error: Exception | None = None


def _advance_sample_work(
    work: _SampleWork,
    *,
    expanded: Any,
    output: WQModelOutput,
    catalog: PyXtalChartCatalog,
    kernel: TopologyEventKernel,
    config: SamplingConfig,
    step: int,
    current_time: float,
    next_time: float,
    control: str,
    midpoint: int,
    disc_once_cutoff: int,
) -> None:
    state = work.state
    if state is None:
        raise RuntimeError("sample work has no active state")
    context = work.context
    context.reverse_step = step
    state, score_norms = _continuous_step(
        state,
        expanded,
        output,
        current_time=current_time,
        next_time=next_time,
    )
    context.last_state = state
    species_uncertainty = _entropy_uncertainty(output.species_logits)
    wyckoff_uncertainty = _entropy_uncertainty(output.wyckoff_logits)
    uncertainties = tuple(
        float(max(species_uncertainty[index], wyckoff_uncertainty[index]))
        for index in range(len(state.orbits))
    )
    state = _replace_masked_fields(
        state,
        output,
        catalog,
        context,
        masked_species=work.masked_species,
        masked_wyckoff=work.masked_wyckoff,
        pending_revision=work.pending_revision,
        time_value=current_time,
        final_step=(
            step == config.backbone_calls - 1
            or (
                config.variant is WQVariant.DISC_ONCE
                and step == disc_once_cutoff - 1
            )
        ),
        temperature=config.temperature,
    )
    context.last_state = state
    if config.variant is WQVariant.D3PM:
        state = _d3pm_reverse_fields(
            state,
            output,
            catalog,
            context,
            current_time=current_time,
            next_time=next_time,
            temperature=config.temperature,
        )
        context.last_state = state

    if step == midpoint:
        work.revision_budget = RevisionBudget(state.field_count)
    if (
        work.revision_budget is not None
        and step >= midpoint
        and step < config.backbone_calls - 1
    ):
        selected = _select_revisions(
            state,
            output,
            work.revision_budget,
            config,
            context,
            control,
        )
        for field in selected:
            work.pending_revision.add(field)
            if field.field == "species":
                work.masked_species.add(field.orbit_id)
            elif field.field == "wyckoff_type":
                work.masked_wyckoff.add(field.orbit_id)
            else:
                work.pending_existence.add(field.orbit_id)

    if work.pending_existence:
        orbit_id = sorted(work.pending_existence)[0]
        work.pending_existence.remove(orbit_id)
        work.pending_revision.discard(FieldRef(orbit_id, "existence"))
        death = next(
            (
                event
                for event in kernel.legal_events(state)
                if event.event_type is TopologyEventType.DEATH
                and event.orbit_id == orbit_id
            ),
            None,
        )
        if death is not None:
            index = next(
                index
                for index, orbit in enumerate(state.orbits)
                if orbit.orbit_id == orbit_id
            )
            death_logit = float(
                output.event_logits[0, EVENT_CLASS[TopologyEventType.DEATH]]
            ) + float(output.event_orbit_logits[index])
            none_logit = float(
                output.event_logits[0, EVENT_CLASS[TopologyEventType.NONE]]
            )
            probability = 1.0 / (
                1.0
                + math.exp(
                    max(-60.0, min(60.0, none_logit - death_logit))
                )
            )
            event = (
                death
                if context.python_rng.random() < probability
                else TopologyEvent(TopologyEventType.NONE)
            )
        else:
            event = TopologyEvent(TopologyEventType.NONE)
    else:
        event = _draw_event(
            _event_logits(
                state,
                output,
                kernel,
                config,
                step=step,
                midpoint=midpoint,
            ),
            context,
            time_value=current_time,
        )
    before_ids = {orbit.orbit_id for orbit in state.orbits}
    state = _apply_event(state, event, output, catalog, context)
    context.last_state = state
    after_ids = {orbit.orbit_id for orbit in state.orbits}
    for removed in before_ids - after_ids:
        work.masked_species.discard(removed)
        work.masked_wyckoff.discard(removed)
        work.pending_existence.discard(removed)
        work.pending_revision = {
            field
            for field in work.pending_revision
            if field.orbit_id != removed
        }
    if len(score_norms) != len(state.orbits):
        score_norms = (0.0,) * len(state.orbits)
        uncertainties = (0.0,) * len(state.orbits)
    work.state = state
    work.score_norms = score_norms
    work.uncertainties = uncertainties


def _sample_batch(
    model: WQCoDenoiser,
    catalog: PyXtalChartCatalog,
    contexts: Sequence[_AttemptContext],
    config: SamplingConfig,
    device: torch.device,
) -> tuple[_SampleWork, ...]:
    """Run synchronized ragged reverse processes with per-attempt RNG/state."""

    works = [_SampleWork(context=context) for context in contexts]
    control = _resolved_revision_control(config)
    for work in works:
        try:
            state, masked_species, masked_wyckoff = _initial_state(
                model, catalog, work.context, config, device
            )
            work.state = state
            work.masked_species = masked_species
            work.masked_wyckoff = masked_wyckoff
            work.score_norms = (0.0,) * len(state.orbits)
            work.uncertainties = (0.0,) * len(state.orbits)
            work.context.last_state = state
        except Exception as exc:
            work.error = exc

    midpoint = config.backbone_calls // 2
    disc_once_cutoff = max(
        1, int(round(config.disc_once_tau * config.backbone_calls))
    )
    start_time = 1.0 - 1.0 / (config.backbone_calls + 1)
    times = np.linspace(start_time, 0.0, config.backbone_calls + 1)
    kernel = TopologyEventKernel(
        catalog=catalog,
        bridge=TargetStratumBridge(catalog),
        species=MP20_ATOMIC_NUMBERS,
    )
    for step in range(config.backbone_calls):
        current_time = float(times[step])
        next_time = float(times[step + 1])
        prepared: list[tuple[_SampleWork, Any, Any]] = []
        for work in works:
            if work.error is not None or work.state is None:
                continue
            try:
                expanded = expand_state(work.state, catalog)
                work.context.calls["projection"] += 1
                evidence = list(
                    compute_geometry_evidence(
                        work.state,
                        expanded,
                        score_norms=work.score_norms,
                        basin_uncertainties=work.uncertainties,
                    )
                )
                if control == "shuffled-geometry" and len(evidence) > 1:
                    work.context.python_rng.shuffle(evidence)
                tensor = tensorize_state(
                    work.state,
                    expanded,
                    evidence,
                    time=current_time,
                    masked_species=frozenset(work.masked_species),
                    masked_wyckoff=frozenset(work.masked_wyckoff),
                )
                prepared.append((work, expanded, tensor))
            except Exception as exc:
                work.error = exc
        if not prepared:
            continue
        input_batches = tuple(item[2] for item in prepared)
        try:
            joined = concatenate_tensor_batches(input_batches).to(device)
            with torch.no_grad(), _autocast(device):
                batched_output = model(joined, variant=config.variant)
                if control == "extra-call":
                    batched_output = model(joined, variant=config.variant)
            outputs = split_model_output(batched_output, input_batches)
        except Exception as exc:
            for work, _, _ in prepared:
                work.error = exc
            continue
        for (work, expanded, _), output in zip(prepared, outputs):
            work.context.calls["joint"] += 2 if control == "extra-call" else 1
            try:
                _advance_sample_work(
                    work,
                    expanded=expanded,
                    output=output,
                    catalog=catalog,
                    kernel=kernel,
                    config=config,
                    step=step,
                    current_time=current_time,
                    next_time=next_time,
                    control=control,
                    midpoint=midpoint,
                    disc_once_cutoff=disc_once_cutoff,
                )
            except Exception as exc:
                work.error = exc

    for work in works:
        if work.error is not None or work.state is None:
            continue
        try:
            if (
                work.masked_species
                or work.masked_wyckoff
                or work.pending_existence
            ):
                raise StateExpansionError(
                    "reverse process ended with uncommitted fields"
                )
            final_expanded = expand_state(work.state, catalog)
            work.context.calls["projection"] += 1
            structure = final_expanded.pymatgen_structure()
            cif = structure.to(fmt="cif")
            work.artifact = {
                "schema": "wqcodiff_generation_attempt_v1",
                "attempt_id": work.context.attempt_id,
                "experiment_id": config.experiment_id,
                "pairing_id": config.pairing_id or config.experiment_id,
                "method": config.variant.value,
                "training_seed": config.training_seed,
                "sampling_seed": config.sampling_seed,
                "status": AttemptStatus.SUCCEEDED.value,
                "state": work.state.to_dict(),
                "structure": structure.as_dict(),
                "structure_cif_sha256": hashlib.sha256(
                    cif.encode("utf-8")
                ).hexdigest(),
                "intended_space_group": (
                    None
                    if config.variant is WQVariant.ATOM_JOINT
                    else work.state.space_group
                ),
                "redetected_space_group": final_expanded.redetected_space_group,
                "atom_count": work.state.atom_count,
                "orbit_count": len(work.state.orbits),
                "revision_control": control,
                "revision_threshold": config.revision_threshold,
                "disc_once_tau": (
                    config.disc_once_tau
                    if config.variant is WQVariant.DISC_ONCE
                    else None
                ),
                "temperature": config.temperature,
                "revision_initial_field_count": (
                    None
                    if work.revision_budget is None
                    else work.revision_budget.initial_field_count
                ),
                "revision_churn": (
                    None
                    if work.revision_budget is None
                    else work.revision_budget.churn
                ),
                "backbone_calls": config.backbone_calls,
                "calls": dict(work.context.calls),
                "trace": work.context.trace,
            }
        except Exception as exc:
            work.error = exc
    return tuple(works)


def _failure_status(exc: Exception) -> AttemptStatus:
    message = str(exc).lower()
    if isinstance(exc, TransitionError) and "bridge_failure" in message:
        return AttemptStatus.BRIDGE_FAILURE
    if isinstance(exc, StateExpansionError):
        return AttemptStatus.PROJECTION_FAILURE
    if "invalid" in message or "multiplicity" in message or "atom-count" in message:
        return AttemptStatus.INVALID_TOPOLOGY
    return AttemptStatus.FAILED


def sample(config: SamplingConfig, *, protocol_path: str | Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered GPU sampling requires CUDA inside Slurm")
    model, model_provenance = _load_model(
        config.checkpoint,
        config=config,
        protocol=protocol,
        device=device,
    )
    revision_lock_sha256 = (
        hashlib.sha256(Path(config.revision_lock).read_bytes()).hexdigest()
        if config.revision_lock
        else None
    )
    catalog = PyXtalChartCatalog()
    deriver = SeedDeriver(protocol.name, config.experiment_id)
    pair_deriver = SeedDeriver(
        protocol.name,
        config.pairing_id or config.experiment_id,
    )
    attempt_ledger = AttemptLedger(config.attempt_ledger)
    artifact_ledger = ArtifactLedger(config.output_jsonl)
    existing = {record.attempt_id for record in attempt_ledger.records()}
    ordinals = range(config.start_ordinal, config.start_ordinal + config.attempts)
    identities = [
        (
            ordinal,
            deriver.attempt_id(
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                ordinal=ordinal,
                method=config.variant.value,
            ),
        )
        for ordinal in ordinals
    ]
    overlap = sorted({attempt_id for _, attempt_id in identities} & existing)
    if overlap:
        raise ValueError(
            f"sampling would retry/replace {len(overlap)} existing attempt IDs; first={overlap[0]}"
        )

    succeeded = 0
    failed = 0
    started_all = time.monotonic()
    for chunk_start in range(0, len(identities), config.inference_batch_size):
        chunk = identities[
            chunk_start : chunk_start + config.inference_batch_size
        ]
        prepared: list[dict[str, Any]] = []
        for ordinal, attempt_id in chunk:
            seed = deriver.derive(
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                attempt_id=attempt_id,
                stage="graph",
            )
            pair_id = pair_deriver.pair_id(
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                ordinal=ordinal,
            )
            paired_seed = pair_deriver.paired_derive(
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                ordinal=ordinal,
                stage="initial_noise_and_reverse_process",
            )
            attempt_ledger.append(
                AttemptRecord(
                    attempt_id=attempt_id,
                    method=config.variant.value,
                    training_seed=config.training_seed,
                    sampling_seed=config.sampling_seed,
                    stage="graph",
                    status=AttemptStatus.SUBMITTED,
                    seed=seed,
                )
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(paired_seed)
            prepared.append(
                {
                    "ordinal": ordinal,
                    "attempt_id": attempt_id,
                    "seed": seed,
                    "pair_id": pair_id,
                    "paired_seed": paired_seed,
                    "context": _AttemptContext(
                        attempt_id=attempt_id,
                        python_rng=random.Random(paired_seed),
                        torch_generator=generator,
                        calls={
                            "prior": 0,
                            "joint": 0,
                            "bridge": 0,
                            "projection": 0,
                        },
                        trace=[],
                    ),
                }
            )
        chunk_started = time.monotonic()
        try:
            works = _sample_batch(
                model,
                catalog,
                [item["context"] for item in prepared],
                config,
                device,
            )
        except Exception as exc:  # defensive batch-level terminal accounting
            works = tuple(
                _SampleWork(context=item["context"], error=exc)
                for item in prepared
            )
        chunk_elapsed = time.monotonic() - chunk_started
        amortized_elapsed = chunk_elapsed / len(prepared)
        for item, work in zip(prepared, works):
            ordinal = int(item["ordinal"])
            attempt_id = str(item["attempt_id"])
            seed = int(item["seed"])
            pair_id = str(item["pair_id"])
            paired_seed = int(item["paired_seed"])
            context = work.context
            flops_lower_bound = float(
                2 * model.parameter_count() * context.calls["joint"]
            )
            common_metadata = {
                "flops_estimator": "2x_parameter_count_per_joint_call_lower_bound",
                "experiment_id": config.experiment_id,
                "pairing_id": config.pairing_id or config.experiment_id,
                "revision_control": _resolved_revision_control(config),
                "revision_threshold": config.revision_threshold,
                "disc_once_tau": (
                    config.disc_once_tau
                    if config.variant is WQVariant.DISC_ONCE
                    else None
                ),
                "temperature": config.temperature,
                "backbone_calls": config.backbone_calls,
                "ordinal": ordinal,
                "pair_id": pair_id,
                "paired_seed": paired_seed,
                "revision_lock_sha256": revision_lock_sha256,
                "checkpoint_sha256": model_provenance["checkpoint_sha256"],
                "source_bundle_sha256": model_provenance[
                    "source_bundle_sha256"
                ],
                "inference_batch_size": len(prepared),
                "walltime_allocation": "equal_amortized_within_inference_batch",
            }
            if work.error is None and work.artifact is not None:
                artifact = work.artifact
                artifact.update(
                    {
                        "ordinal": ordinal,
                        "pair_id": pair_id,
                        "paired_seed": paired_seed,
                        "revision_lock_sha256": revision_lock_sha256,
                        "checkpoint_sha256": model_provenance[
                            "checkpoint_sha256"
                        ],
                        "source_bundle_sha256": model_provenance[
                            "source_bundle_sha256"
                        ],
                        "walltime_s": amortized_elapsed,
                        "inference_batch_size": len(prepared),
                        "inference_batch_elapsed_s": chunk_elapsed,
                        "walltime_allocation": common_metadata[
                            "walltime_allocation"
                        ],
                        "generation_flops_lower_bound": flops_lower_bound,
                        "generation_flops_estimator": (
                            "2x_parameter_count_per_joint_call_lower_bound_not_actual_flops"
                        ),
                    }
                )
                digest = artifact_ledger.append(artifact)
                attempt_ledger.append(
                    AttemptRecord(
                        attempt_id=attempt_id,
                        method=config.variant.value,
                        training_seed=config.training_seed,
                        sampling_seed=config.sampling_seed,
                        stage="graph",
                        status=AttemptStatus.SUCCEEDED,
                        artifact_hash=digest,
                        seed=seed,
                        calls=context.calls,
                        flops=flops_lower_bound,
                        walltime_s=amortized_elapsed,
                        metadata=common_metadata,
                    )
                )
                succeeded += 1
                continue

            exc = work.error or RuntimeError("batched sampler produced no artifact")
            status = _failure_status(exc)
            reason = f"{type(exc).__name__}:{exc}"
            digest = artifact_ledger.append(
                {
                    "schema": "wqcodiff_generation_attempt_v1",
                    "attempt_id": attempt_id,
                    "experiment_id": config.experiment_id,
                    "pairing_id": config.pairing_id or config.experiment_id,
                    "method": config.variant.value,
                    "training_seed": config.training_seed,
                    "sampling_seed": config.sampling_seed,
                    "ordinal": ordinal,
                    "pair_id": pair_id,
                    "paired_seed": paired_seed,
                    "status": status.value,
                    "reason": reason,
                    "state": (
                        None
                        if context.last_state is None
                        else context.last_state.to_dict()
                    ),
                    "intended_space_group": (
                        None
                        if context.last_state is None
                        or config.variant is WQVariant.ATOM_JOINT
                        else context.last_state.space_group
                    ),
                    "atom_count": (
                        None
                        if context.last_state is None
                        else context.last_state.atom_count
                    ),
                    "orbit_count": (
                        None
                        if context.last_state is None
                        else len(context.last_state.orbits)
                    ),
                    "revision_control": _resolved_revision_control(config),
                    "revision_threshold": config.revision_threshold,
                    "disc_once_tau": (
                        config.disc_once_tau
                        if config.variant is WQVariant.DISC_ONCE
                        else None
                    ),
                    "temperature": config.temperature,
                    "revision_initial_field_count": (
                        None
                        if work.revision_budget is None
                        else work.revision_budget.initial_field_count
                    ),
                    "revision_churn": (
                        None
                        if work.revision_budget is None
                        else work.revision_budget.churn
                    ),
                    "revision_lock_sha256": revision_lock_sha256,
                    "checkpoint_sha256": model_provenance[
                        "checkpoint_sha256"
                    ],
                    "source_bundle_sha256": model_provenance[
                        "source_bundle_sha256"
                    ],
                    "backbone_calls": config.backbone_calls,
                    "calls": context.calls,
                    "trace": context.trace,
                    "walltime_s": amortized_elapsed,
                    "inference_batch_size": len(prepared),
                    "inference_batch_elapsed_s": chunk_elapsed,
                    "walltime_allocation": common_metadata[
                        "walltime_allocation"
                    ],
                    "generation_flops_lower_bound": flops_lower_bound,
                    "generation_flops_estimator": (
                        "2x_parameter_count_per_joint_call_lower_bound_not_actual_flops"
                    ),
                }
            )
            attempt_ledger.append(
                AttemptRecord(
                    attempt_id=attempt_id,
                    method=config.variant.value,
                    training_seed=config.training_seed,
                    sampling_seed=config.sampling_seed,
                    stage="graph",
                    status=status,
                    reason=reason,
                    artifact_hash=digest,
                    seed=seed,
                    calls=context.calls,
                    flops=flops_lower_bound,
                    walltime_s=amortized_elapsed,
                    metadata=common_metadata,
                )
            )
            failed += 1

    result = {
        "ok": succeeded + failed == config.attempts,
        "schema": "wqcodiff_sampling_summary_v1",
        "protocol_name": protocol.name,
        "protocol_sha256": protocol.sha256,
        "variant": config.variant.value,
        "experiment_id": config.experiment_id,
        "training_seed": config.training_seed,
        "sampling_seed": config.sampling_seed,
        "pairing_id": config.pairing_id or config.experiment_id,
        "revision_threshold": config.revision_threshold,
        "disc_once_tau": (
            config.disc_once_tau
            if config.variant is WQVariant.DISC_ONCE
            else None
        ),
        "temperature": config.temperature,
        "backbone_calls": config.backbone_calls,
        "revision_lock_sha256": revision_lock_sha256,
        "model_provenance": model_provenance,
        "inference_batch_size": config.inference_batch_size,
        "attempts": config.attempts,
        "succeeded": succeeded,
        "failed": failed,
        "all_attempts_terminal": succeeded + failed == config.attempts,
        "all_attempts_succeeded": failed == 0,
        "success_rate": succeeded / config.attempts,
        "output_jsonl": str(Path(config.output_jsonl).resolve()),
        "attempt_ledger": str(Path(config.attempt_ledger).resolve()),
        "elapsed_s": time.monotonic() - started_all,
    }
    summary_path = Path(config.output_jsonl).with_suffix(".summary.json")
    write_json_exclusive(summary_path, result)
    return result
