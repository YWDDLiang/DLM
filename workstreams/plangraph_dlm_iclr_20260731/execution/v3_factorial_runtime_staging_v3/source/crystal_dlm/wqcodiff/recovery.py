"""Hash-fixed Day-7 corruption recovery and DLM falsification cells."""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import hashlib
import json
import random
import re
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .bridge import TargetStratumBridge
from .charts import PyXtalChartCatalog
from .contracts import (
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
    write_json_exclusive,
)
from .events import TopologyEvent, TopologyEventType
from .kernel import TopologyEventKernel
from .model import WQCoDenoiser, WQVariant
from .protocol import load_protocol
from .revision import FieldRef, RevisionBudget
from .runtime import (
    concatenate_tensor_batches,
    compute_geometry_evidence,
    expand_state,
    split_model_output,
    tensorize_state,
)
from .sampling import (
    SamplingConfig,
    _AttemptContext,
    _apply_event,
    _autocast,
    _continuous_step,
    _d3pm_reverse_fields,
    _draw_event,
    _event_logits,
    _load_model,
    _replace_masked_fields,
    _select_revisions,
)
from .state import GeometryEvidence, StratifiedState
from .training_data import JsonlRecordIndex
from .vocabulary import MP20_ATOMIC_NUMBERS


CORRUPTION_OPERATORS = (
    "none",
    "deletion",
    "false-insertion",
    "wrong-wyckoff",
    "wrong-species",
    "joint",
)
GEOMETRY_CONDITIONS = ("clean", "noisy", "shuffled", "absent")
RECOVERY_SCHEDULES = (
    "fixed",
    "discrete-first",
    "continuous-first",
    "confidence-adaptive",
    "geometry-adaptive",
)
RECOVERY_CONTROLS = ("none", "random-count", "shuffled-geometry", "extra-call")


class CorruptionNotApplicable(RuntimeError):
    pass


_RUNTIME_WORKER_LOCAL = threading.local()


def _runtime_worker_catalog(catalog: PyXtalChartCatalog) -> PyXtalChartCatalog:
    """Use one immutable PyXtal metadata cache per preparation thread."""

    if threading.current_thread() is threading.main_thread():
        return catalog
    cached = getattr(_RUNTIME_WORKER_LOCAL, "catalog", None)
    if cached is None:
        cached = PyXtalChartCatalog(hall_style=catalog.hall_style)
        _RUNTIME_WORKER_LOCAL.catalog = cached
    return cached


@dataclasses.dataclass(frozen=True, slots=True)
class RecoveryConfig:
    checkpoint: str
    dataset_paths: tuple[str, ...]
    output_jsonl: str
    attempt_ledger: str
    experiment_id: str
    runtime_source_bundle_sha256: str
    variant: WQVariant
    training_seed: int
    corruption_seed: int
    structures: int
    corruption_level: float
    operator: str
    geometry_condition: str
    schedule: str
    pairing_id: str | None = None
    control: str = "none"
    calls: int = 16
    revision_threshold: float = 0.7
    temperature: float = 1.0
    inference_batch_size: int = 64
    runtime_workers: int = 1
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.dataset_paths:
            raise ValueError("recovery dataset paths are required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.runtime_source_bundle_sha256):
            raise ValueError("recovery requires a lowercase runtime source-bundle SHA256")
        if self.structures <= 0:
            raise ValueError("recovery structures must be positive")
        if self.corruption_level not in {0.3, 0.5, 0.7, 0.9}:
            raise ValueError("corruption level is outside the frozen grid")
        if self.operator not in CORRUPTION_OPERATORS:
            raise ValueError("unknown corruption operator")
        if self.operator == "none" and self.geometry_condition != "clean":
            raise ValueError("the clean calibration operator requires clean geometry")
        if self.geometry_condition not in GEOMETRY_CONDITIONS:
            raise ValueError("unknown geometry condition")
        if self.schedule not in RECOVERY_SCHEDULES:
            raise ValueError("unknown recovery schedule")
        if self.control not in RECOVERY_CONTROLS:
            raise ValueError("unknown recovery control")
        if self.control != "none" and self.schedule != "geometry-adaptive":
            raise ValueError("recovery controls are only defined for geometry-adaptive schedule")
        if self.calls not in {16, 32, 64, 128}:
            raise ValueError("recovery calls must use the frozen call grid")
        if self.inference_batch_size not in {16, 32, 64, 128}:
            raise ValueError("inference batch size must be one of 16/32/64/128")
        if self.runtime_workers not in {1, 2, 4, 8, 12}:
            raise ValueError("runtime workers must be one of 1/2/4/8/12")


def _primary(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record["decompositions"]["symprec_1e-02"]


def _hash_fixed_records(index: JsonlRecordIndex, count: int) -> list[Mapping[str, Any]]:
    records = [index[value] for value in range(len(index))]
    records.sort(
        key=lambda record: hashlib.sha256(
            str(record["material_id"]).encode("utf-8")
        ).hexdigest()
    )
    if count > len(records):
        raise ValueError(f"requested {count} recovery structures, dataset has {len(records)}")
    return records[:count]


def _kernel(catalog: PyXtalChartCatalog) -> TopologyEventKernel:
    return TopologyEventKernel(
        catalog=catalog,
        bridge=TargetStratumBridge(catalog),
        species=MP20_ATOMIC_NUMBERS,
    )


def _random_non_none_event(
    state: StratifiedState,
    kernel: TopologyEventKernel,
    rng: random.Random,
    allowed: set[TopologyEventType],
) -> TopologyEvent:
    events = kernel.legal_events(state, event_types=allowed)
    if not events:
        raise CorruptionNotApplicable(
            "no legal corruption event for the fixed structure/operator"
        )
    return events[rng.randrange(len(events))]


def _apply_random_event(
    state: StratifiedState,
    event: TopologyEvent,
    kernel: TopologyEventKernel,
    rng: random.Random,
) -> StratifiedState:
    return kernel.apply(state, event, rng)


def _corrupt_discrete(
    source: StratifiedState,
    catalog: PyXtalChartCatalog,
    *,
    operator: str,
    level: float,
    rng: random.Random,
) -> tuple[StratifiedState, list[dict[str, Any]]]:
    state = source
    kernel = _kernel(catalog)
    trace: list[dict[str, Any]] = []
    requested = max(1, int(round(level * len(source.orbits))))
    operator_types = {
        "deletion": {TopologyEventType.DEATH},
        "false-insertion": {TopologyEventType.BIRTH},
        "wrong-wyckoff": {TopologyEventType.WYCKOFF_CHANGE},
        "wrong-species": {TopologyEventType.SPECIES_CHANGE},
    }
    for step in range(requested):
        if operator == "joint":
            kinds = {
                TopologyEventType.BIRTH,
                TopologyEventType.DEATH,
                TopologyEventType.WYCKOFF_CHANGE,
                TopologyEventType.SPECIES_CHANGE,
            }
        else:
            kinds = operator_types[operator]
        try:
            event = _random_non_none_event(state, kernel, rng, kinds)
        except CorruptionNotApplicable:
            if step == 0:
                raise
            break
        before = state.topology_hash()
        candidate = _apply_random_event(state, event, kernel, rng)
        if candidate.topology_hash() == before:
            raise RuntimeError("corruption event did not alter topology")
        state = candidate
        trace.append(event.to_dict())
    if state.topology_hash() == source.topology_hash():
        raise CorruptionNotApplicable("corruption cancelled back to the source topology")
    return state, trace


def _corrupt_geometry(
    state: StratifiedState,
    *,
    condition: str,
    level: float,
    joint: bool,
    rng: random.Random,
) -> StratifiedState:
    effective = "noisy" if joint and condition == "clean" else condition
    if effective == "clean":
        return state
    orbits = list(state.orbits)
    chart = np.asarray(state.lattice_chart, dtype=np.float64)
    if effective == "noisy":
        chart += np.asarray(
            [rng.gauss(0.0, 0.35 * level) for _ in chart], dtype=np.float64
        )
        orbits = [
            dataclasses.replace(
                orbit,
                free_coordinate=tuple(
                    (value + rng.gauss(0.0, 0.5 * level)) % 1.0
                    for value in orbit.free_coordinate
                ),
            )
            for orbit in orbits
        ]
    elif effective == "shuffled":
        by_dimension: dict[int, list[tuple[float, ...]]] = collections.defaultdict(list)
        for orbit in orbits:
            by_dimension[orbit.chart_dimension].append(orbit.free_coordinate)
        for values in by_dimension.values():
            rng.shuffle(values)
        counters: dict[int, int] = collections.defaultdict(int)
        replaced = []
        for orbit in orbits:
            value = by_dimension[orbit.chart_dimension][counters[orbit.chart_dimension]]
            counters[orbit.chart_dimension] += 1
            replaced.append(dataclasses.replace(orbit, free_coordinate=value))
        orbits = replaced
        chart = chart[::-1].copy()
    elif effective == "absent":
        chart = np.asarray([rng.gauss(0.0, 1.0) for _ in chart], dtype=np.float64)
        orbits = [
            dataclasses.replace(
                orbit,
                free_coordinate=tuple(rng.random() for _ in orbit.free_coordinate),
            )
            for orbit in orbits
        ]
    else:  # pragma: no cover - guarded by config
        raise ValueError(effective)
    return dataclasses.replace(
        state,
        lattice_chart=tuple(float(value) for value in chart),
        orbits=tuple(orbits),
    )


def corrupt_state(
    source: StratifiedState,
    catalog: PyXtalChartCatalog,
    config: RecoveryConfig,
    rng: random.Random,
) -> tuple[StratifiedState, list[dict[str, Any]]]:
    if config.operator == "none":
        discrete, trace = source, []
    else:
        discrete, trace = _corrupt_discrete(
            source,
            catalog,
            operator=config.operator,
            level=config.corruption_level,
            rng=rng,
        )
    result = _corrupt_geometry(
        discrete,
        condition=config.geometry_condition,
        level=config.corruption_level,
        joint=config.operator == "joint",
        rng=rng,
    )
    return dataclasses.replace(result, timestep=config.corruption_level), trace


def topology_edit_distance(first: StratifiedState, second: StratifiedState) -> int:
    def counter(state: StratifiedState) -> collections.Counter[tuple[int, int, int]]:
        return collections.Counter(
            (
                orbit.wyckoff_type,
                orbit.species,
                int(orbit.primitive_multiplicity),
            )
            for orbit in state.orbits
        )

    left = counter(first)
    right = counter(second)
    exact = sum((left & right).values())
    unmatched_left = sum(left.values()) - exact
    unmatched_right = sum(right.values()) - exact
    return int(first.space_group != second.space_group) + max(
        unmatched_left, unmatched_right
    )


def _coordinate_error(source: StratifiedState, recovered: StratifiedState) -> float | None:
    remaining = list(recovered.orbits)
    errors: list[float] = []
    for orbit in source.orbits:
        candidates = [
            (index, value)
            for index, value in enumerate(remaining)
            if value.wyckoff_type == orbit.wyckoff_type
            and value.species == orbit.species
            and value.chart_dimension == orbit.chart_dimension
        ]
        if not candidates:
            continue
        index, match = min(
            candidates,
            key=lambda item: sum(
                min(abs(a - b), 1.0 - abs(a - b)) ** 2
                for a, b in zip(orbit.free_coordinate, item[1].free_coordinate)
            ),
        )
        if orbit.chart_dimension:
            errors.append(
                float(
                    np.linalg.norm(
                        [
                            min(abs(a - b), 1.0 - abs(a - b))
                            for a, b in zip(
                                orbit.free_coordinate, match.free_coordinate
                            )
                        ]
                    )
                )
            )
        remaining.pop(index)
    return None if not errors else float(np.mean(errors))


def _run_recovery(
    source: StratifiedState,
    corrupt: StratifiedState,
    model: WQCoDenoiser,
    catalog: PyXtalChartCatalog,
    context: _AttemptContext,
    config: RecoveryConfig,
    device: torch.device,
) -> tuple[StratifiedState, Mapping[str, Any]]:
    sampling_config = SamplingConfig(
        checkpoint=config.checkpoint,
        output_jsonl=config.output_jsonl,
        attempt_ledger=config.attempt_ledger,
        experiment_id=config.experiment_id,
        pairing_id=config.pairing_id,
        variant=config.variant,
        training_seed=config.training_seed,
        sampling_seed=config.corruption_seed,
        attempts=1,
        backbone_calls=config.calls,
        revision_control=(
            config.control
            if config.schedule == "geometry-adaptive" and config.control != "none"
            else "geometry"
            if config.schedule == "geometry-adaptive"
            else "confidence"
            if config.schedule == "confidence-adaptive"
            else "none"
        ),
        revision_threshold=config.revision_threshold,
        temperature=config.temperature,
        device=config.device,
    )
    state = corrupt
    masked_species: set[str] = set()
    masked_wyckoff: set[str] = set()
    pending_revision: set[FieldRef] = set()
    pending_existence: set[str] = set()
    budget = RevisionBudget(state.field_count)
    wrong_to_right = 0
    right_to_wrong = 0
    initial_distance = topology_edit_distance(source, state)
    previous_distance = initial_distance
    revision_resolved_actions = 0
    revision_wrong_to_right = 0
    revision_right_to_wrong = 0
    event_counts: collections.Counter[str] = collections.Counter()
    dimension_change_count = 0

    def record_distance_change(
        before: int,
        after: int,
        *,
        revision_actions: int = 0,
    ) -> None:
        nonlocal wrong_to_right, right_to_wrong
        nonlocal revision_resolved_actions, revision_wrong_to_right, revision_right_to_wrong
        if after < before:
            wrong_to_right += before - after
            if revision_actions:
                revision_wrong_to_right += before - after
        elif after > before:
            right_to_wrong += after - before
            if revision_actions:
                revision_right_to_wrong += after - before
        revision_resolved_actions += revision_actions
    times = np.linspace(config.corruption_level, 0.0, config.calls + 1)
    for step in range(config.calls):
        context.reverse_step = step
        current_time = float(times[step])
        next_time = float(times[step + 1])
        midpoint = config.calls // 2
        do_discrete = (
            config.schedule not in {"continuous-first", "discrete-first"}
            or (config.schedule == "discrete-first" and step < midpoint)
            or (config.schedule == "continuous-first" and step >= midpoint)
        )
        do_continuous = (
            config.schedule not in {"continuous-first", "discrete-first"}
            or (config.schedule == "continuous-first" and step < midpoint)
            or (config.schedule == "discrete-first" and step >= midpoint)
        )
        expanded = expand_state(state, catalog)
        context.calls["projection"] += 1
        if config.geometry_condition == "absent":
            evidence = [GeometryEvidence(0, 0, 0, 0, 0, 0) for _ in state.orbits]
        else:
            evidence = list(compute_geometry_evidence(state, expanded))
            if (
                config.geometry_condition == "shuffled"
                or config.control == "shuffled-geometry"
            ) and len(evidence) > 1:
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
        if config.control == "extra-call":
            with torch.no_grad(), _autocast(device):
                output = model(batch, variant=config.variant)
            context.calls["joint"] += 1
        if do_continuous:
            state, _ = _continuous_step(
                state,
                expanded,
                output,
                current_time=current_time,
                next_time=next_time,
            )
        else:
            state = dataclasses.replace(state, timestep=next_time)

        if do_discrete:
            trace_start = len(context.trace)
            state = _replace_masked_fields(
                state,
                output,
                catalog,
                context,
                masked_species=masked_species,
                masked_wyckoff=masked_wyckoff,
                pending_revision=pending_revision,
                time_value=current_time,
                final_step=step == config.calls - 1,
                temperature=config.temperature,
            )
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
            replacement_trace = context.trace[trace_start:]
            revision_actions = sum(
                bool(item.get("revision_fill")) for item in replacement_trace
            )
            dimension_change_count += sum(
                item.get("action") == "wyckoff_commit"
                and item.get("dimension_before") != item.get("dimension_after")
                for item in replacement_trace
            )
            current_distance = topology_edit_distance(source, state)
            record_distance_change(
                previous_distance,
                current_distance,
                revision_actions=revision_actions,
            )
            previous_distance = current_distance
            if config.schedule in {"confidence-adaptive", "geometry-adaptive"} and step < config.calls - 1:
                selected = _select_revisions(
                    state,
                    output,
                    budget,
                    sampling_config,
                    context,
                    "confidence"
                    if config.schedule == "confidence-adaptive"
                    else "random-count"
                    if config.control == "random-count"
                    else "geometry",
                )
                for field in selected:
                    pending_revision.add(field)
                    if field.field == "species":
                        masked_species.add(field.orbit_id)
                    elif field.field == "wyckoff_type":
                        masked_wyckoff.add(field.orbit_id)
                    else:
                        pending_existence.add(field.orbit_id)

            kernel = _kernel(catalog)
            if pending_existence:
                orbit_id = sorted(pending_existence)[0]
                pending_existence.remove(orbit_id)
                pending_revision.discard(FieldRef(orbit_id, "existence"))
                candidates = [
                    event
                    for event in kernel.legal_events(state)
                    if event.event_type is TopologyEventType.DEATH
                    and event.orbit_id == orbit_id
                ]
                event = candidates[0] if candidates else TopologyEvent(TopologyEventType.NONE)
            else:
                event = _draw_event(
                    _event_logits(
                        state,
                        output,
                        kernel,
                        sampling_config,
                        step=step,
                        midpoint=midpoint,
                        recovery=True,
                    ),
                    context,
                    time_value=current_time,
                )
            state = _apply_event(state, event, output, catalog, context)
            if event.event_type is not TopologyEventType.NONE:
                event_counts[event.event_type.value] += 1
                if context.trace[-1].get("dimension_before") != context.trace[-1].get(
                    "dimension_after"
                ):
                    dimension_change_count += 1
            current_distance = topology_edit_distance(source, state)
            record_distance_change(previous_distance, current_distance)
            previous_distance = current_distance
            valid_ids = {orbit.orbit_id for orbit in state.orbits}
            masked_species.intersection_update(valid_ids)
            masked_wyckoff.intersection_update(valid_ids)
            pending_existence.intersection_update(valid_ids)
            pending_revision = {
                field for field in pending_revision if field.orbit_id in valid_ids
            }
    precision_denominator = revision_resolved_actions
    return state, {
        "initial_topology_edit_distance": initial_distance,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "net_correction": wrong_to_right - right_to_wrong,
        "revision_resolved_actions": revision_resolved_actions,
        "revision_wrong_to_right": revision_wrong_to_right,
        "revision_right_to_wrong": revision_right_to_wrong,
        "revision_precision": (
            None
            if precision_denominator == 0
            else min(1.0, revision_wrong_to_right / precision_denominator)
        ),
        "revision_recall": (
            None
            if initial_distance == 0
            else min(1.0, revision_wrong_to_right / initial_distance)
        ),
        "unresolved_remasks": len(masked_species)
        + len(masked_wyckoff)
        + len(pending_existence),
        "revision_churn": budget.churn,
        "revision_selected_actions": budget.total,
        "initial_revisable_field_count": budget.initial_field_count,
        "event_counts": dict(sorted(event_counts.items())),
        "dimension_change_count": dimension_change_count,
    }


@dataclasses.dataclass(slots=True)
class _RecoveryWork:
    context: _AttemptContext
    source: StratifiedState | None = None
    state: StratifiedState | None = None
    corrupt_topology_hash: str | None = None
    corruption_trace: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    masked_species: set[str] = dataclasses.field(default_factory=set)
    masked_wyckoff: set[str] = dataclasses.field(default_factory=set)
    pending_revision: set[FieldRef] = dataclasses.field(default_factory=set)
    pending_existence: set[str] = dataclasses.field(default_factory=set)
    budget: RevisionBudget | None = None
    initial_distance: int = 0
    previous_distance: int = 0
    wrong_to_right: int = 0
    right_to_wrong: int = 0
    revision_resolved_actions: int = 0
    revision_wrong_to_right: int = 0
    revision_right_to_wrong: int = 0
    event_counts: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    dimension_change_count: int = 0
    runtime_profile_s: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    mechanism: dict[str, Any] | None = None
    error: Exception | None = None

    def initialize(self, source: StratifiedState, corrupt: StratifiedState) -> None:
        self.source = source
        self.state = corrupt
        self.corrupt_topology_hash = corrupt.topology_hash()
        self.context.last_state = corrupt
        self.budget = RevisionBudget(corrupt.field_count)
        self.initial_distance = topology_edit_distance(source, corrupt)
        self.previous_distance = self.initial_distance


def _record_recovery_distance(
    work: _RecoveryWork,
    before: int,
    after: int,
    *,
    revision_actions: int = 0,
) -> None:
    if after < before:
        work.wrong_to_right += before - after
        if revision_actions:
            work.revision_wrong_to_right += before - after
    elif after > before:
        work.right_to_wrong += after - before
        if revision_actions:
            work.revision_right_to_wrong += after - before
    work.revision_resolved_actions += revision_actions


def _advance_recovery_work(
    work: _RecoveryWork,
    *,
    expanded: Any,
    output: Any,
    catalog: PyXtalChartCatalog,
    kernel: TopologyEventKernel,
    config: RecoveryConfig,
    sampling_config: SamplingConfig,
    step: int,
    current_time: float,
    next_time: float,
    midpoint: int,
    do_discrete: bool,
    do_continuous: bool,
) -> None:
    state = work.state
    source = work.source
    if state is None or source is None or work.budget is None:
        raise RuntimeError("recovery work was not initialized")
    context = work.context
    context.reverse_step = step
    if do_continuous:
        state, _ = _continuous_step(
            state,
            expanded,
            output,
            current_time=current_time,
            next_time=next_time,
        )
    else:
        state = dataclasses.replace(state, timestep=next_time)
    context.last_state = state
    if do_discrete:
        trace_start = len(context.trace)
        state = _replace_masked_fields(
            state,
            output,
            catalog,
            context,
            masked_species=work.masked_species,
            masked_wyckoff=work.masked_wyckoff,
            pending_revision=work.pending_revision,
            time_value=current_time,
            final_step=step == config.calls - 1,
            temperature=config.temperature,
        )
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
        replacement_trace = context.trace[trace_start:]
        revision_actions = sum(
            bool(item.get("revision_fill")) for item in replacement_trace
        )
        work.dimension_change_count += sum(
            item.get("action") == "wyckoff_commit"
            and item.get("dimension_before") != item.get("dimension_after")
            for item in replacement_trace
        )
        current_distance = topology_edit_distance(source, state)
        _record_recovery_distance(
            work,
            work.previous_distance,
            current_distance,
            revision_actions=revision_actions,
        )
        work.previous_distance = current_distance
        if (
            config.schedule in {"confidence-adaptive", "geometry-adaptive"}
            and step < config.calls - 1
        ):
            selected = _select_revisions(
                state,
                output,
                work.budget,
                sampling_config,
                context,
                "confidence"
                if config.schedule == "confidence-adaptive"
                else "random-count"
                if config.control == "random-count"
                else "geometry",
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
            candidates = [
                event
                for event in kernel.legal_events(state)
                if event.event_type is TopologyEventType.DEATH
                and event.orbit_id == orbit_id
            ]
            event = (
                candidates[0]
                if candidates
                else TopologyEvent(TopologyEventType.NONE)
            )
        else:
            event = _draw_event(
                _event_logits(
                    state,
                    output,
                    kernel,
                    sampling_config,
                    step=step,
                    midpoint=midpoint,
                    recovery=True,
                ),
                context,
                time_value=current_time,
            )
        state = _apply_event(state, event, output, catalog, context)
        if event.event_type is not TopologyEventType.NONE:
            work.event_counts[event.event_type.value] += 1
            if context.trace[-1].get("dimension_before") != context.trace[-1].get(
                "dimension_after"
            ):
                work.dimension_change_count += 1
        current_distance = topology_edit_distance(source, state)
        _record_recovery_distance(work, work.previous_distance, current_distance)
        work.previous_distance = current_distance
        valid_ids = {orbit.orbit_id for orbit in state.orbits}
        work.masked_species.intersection_update(valid_ids)
        work.masked_wyckoff.intersection_update(valid_ids)
        work.pending_existence.intersection_update(valid_ids)
        work.pending_revision = {
            field
            for field in work.pending_revision
            if field.orbit_id in valid_ids
        }
    work.state = state
    context.last_state = state


def _prepare_recovery_work(
    work: _RecoveryWork,
    *,
    catalog: PyXtalChartCatalog,
    config: RecoveryConfig,
    current_time: float,
) -> tuple[_RecoveryWork, Any | None, Any | None, Exception | None]:
    """Prepare one CPU graph without touching another attempt's RNG/state."""

    if work.error is not None or work.state is None:
        return work, None, None, work.error
    try:
        phase_started = time.perf_counter()
        runtime_catalog = (
            _runtime_worker_catalog(catalog)
            if config.runtime_workers > 1
            and isinstance(catalog, PyXtalChartCatalog)
            else catalog
        )
        expanded = expand_state(
            work.state,
            runtime_catalog,
            redetect_space_group=config.geometry_condition != "absent",
        )
        work.runtime_profile_s["prepare_expand"] += (
            time.perf_counter() - phase_started
        )
        work.context.calls["projection"] += 1
        phase_started = time.perf_counter()
        if config.geometry_condition == "absent":
            evidence = [
                GeometryEvidence(0, 0, 0, 0, 0, 0)
                for _ in work.state.orbits
            ]
        else:
            evidence = list(compute_geometry_evidence(work.state, expanded))
            if (
                config.geometry_condition == "shuffled"
                or config.control == "shuffled-geometry"
            ) and len(evidence) > 1:
                work.context.python_rng.shuffle(evidence)
        work.runtime_profile_s["prepare_evidence"] += (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        tensor = tensorize_state(
            work.state,
            expanded,
            evidence,
            time=current_time,
            masked_species=frozenset(work.masked_species),
            masked_wyckoff=frozenset(work.masked_wyckoff),
        )
        work.runtime_profile_s["prepare_tensorize"] += (
            time.perf_counter() - phase_started
        )
        return work, expanded, tensor, None
    except Exception as exc:
        return work, None, None, exc


def _run_recovery_batch(
    works: Sequence[_RecoveryWork],
    model: WQCoDenoiser,
    catalog: PyXtalChartCatalog,
    config: RecoveryConfig,
    device: torch.device,
) -> tuple[_RecoveryWork, ...]:
    """Batched Day-7 reverse process with isolated per-attempt failures."""

    sampling_config = SamplingConfig(
        checkpoint=config.checkpoint,
        output_jsonl=config.output_jsonl,
        attempt_ledger=config.attempt_ledger,
        experiment_id=config.experiment_id,
        pairing_id=config.pairing_id,
        variant=config.variant,
        training_seed=config.training_seed,
        sampling_seed=config.corruption_seed,
        attempts=1,
        backbone_calls=config.calls,
        revision_control=(
            config.control
            if config.schedule == "geometry-adaptive" and config.control != "none"
            else "geometry"
            if config.schedule == "geometry-adaptive"
            else "confidence"
            if config.schedule == "confidence-adaptive"
            else "none"
        ),
        revision_threshold=config.revision_threshold,
        temperature=config.temperature,
        inference_batch_size=config.inference_batch_size,
        device=config.device,
    )
    kernel = _kernel(catalog)
    midpoint = config.calls // 2
    times = np.linspace(config.corruption_level, 0.0, config.calls + 1)
    pool_context = (
        concurrent.futures.ThreadPoolExecutor(
            max_workers=config.runtime_workers,
            thread_name_prefix="wq-runtime",
        )
        if config.runtime_workers > 1
        else nullcontext(None)
    )
    with pool_context as executor:
        for step in range(config.calls):
            current_time = float(times[step])
            next_time = float(times[step + 1])
            do_discrete = (
                config.schedule not in {"continuous-first", "discrete-first"}
                or (config.schedule == "discrete-first" and step < midpoint)
                or (config.schedule == "continuous-first" and step >= midpoint)
            )
            do_continuous = (
                config.schedule not in {"continuous-first", "discrete-first"}
                or (config.schedule == "continuous-first" and step < midpoint)
                or (config.schedule == "discrete-first" and step >= midpoint)
            )
            active = [
                work
                for work in works
                if work.error is None and work.state is not None
            ]
            prepare = lambda work: _prepare_recovery_work(
                work,
                catalog=catalog,
                config=config,
                current_time=current_time,
            )
            results = (
                map(prepare, active)
                if executor is None
                else executor.map(prepare, active)
            )
            prepared: list[tuple[_RecoveryWork, Any, Any]] = []
            for work, expanded, tensor, error in results:
                if error is not None:
                    work.error = error
                elif expanded is not None and tensor is not None:
                    prepared.append((work, expanded, tensor))
            if not prepared:
                continue
            input_batches = tuple(item[2] for item in prepared)
            model_started = time.perf_counter()
            try:
                joined = concatenate_tensor_batches(input_batches).to(device)
                with torch.no_grad(), _autocast(device):
                    batched_output = model(joined, variant=config.variant)
                    if config.control == "extra-call":
                        batched_output = model(joined, variant=config.variant)
                outputs = split_model_output(batched_output, input_batches)
            except Exception as exc:
                for work, _, _ in prepared:
                    work.error = exc
                continue
            model_amortized = (
                time.perf_counter() - model_started
            ) / len(prepared)
            for (work, expanded, _), output in zip(prepared, outputs):
                work.runtime_profile_s["model_amortized"] += model_amortized
                work.context.calls["joint"] += (
                    2 if config.control == "extra-call" else 1
                )
                reverse_started = time.perf_counter()
                try:
                    _advance_recovery_work(
                        work,
                        expanded=expanded,
                        output=output,
                        catalog=catalog,
                        kernel=kernel,
                        config=config,
                        sampling_config=sampling_config,
                        step=step,
                        current_time=current_time,
                        next_time=next_time,
                        midpoint=midpoint,
                        do_discrete=do_discrete,
                        do_continuous=do_continuous,
                    )
                except Exception as exc:
                    work.error = exc
                finally:
                    work.runtime_profile_s["reverse_update"] += (
                        time.perf_counter() - reverse_started
                    )

    for work in works:
        if work.error is not None or work.source is None or work.state is None:
            continue
        assert work.budget is not None
        precision_denominator = work.revision_resolved_actions
        work.mechanism = {
            "initial_topology_edit_distance": work.initial_distance,
            "wrong_to_right": work.wrong_to_right,
            "right_to_wrong": work.right_to_wrong,
            "net_correction": work.wrong_to_right - work.right_to_wrong,
            "revision_resolved_actions": work.revision_resolved_actions,
            "revision_wrong_to_right": work.revision_wrong_to_right,
            "revision_right_to_wrong": work.revision_right_to_wrong,
            "revision_precision": (
                None
                if precision_denominator == 0
                else min(
                    1.0,
                    work.revision_wrong_to_right / precision_denominator,
                )
            ),
            "revision_recall": (
                None
                if work.initial_distance == 0
                else min(
                    1.0,
                    work.revision_wrong_to_right / work.initial_distance,
                )
            ),
            "unresolved_remasks": len(work.masked_species)
            + len(work.masked_wyckoff)
            + len(work.pending_existence),
            "revision_churn": work.budget.churn,
            "revision_selected_actions": work.budget.total,
            "initial_revisable_field_count": work.budget.initial_field_count,
            "event_counts": dict(sorted(work.event_counts.items())),
            "dimension_change_count": work.dimension_change_count,
        }
    return tuple(works)


def run_recovery_cell(
    config: RecoveryConfig,
    *,
    protocol_path: str | Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered recovery experiments require CUDA inside Slurm")
    sampling_config = SamplingConfig(
        checkpoint=config.checkpoint,
        output_jsonl=config.output_jsonl,
        attempt_ledger=config.attempt_ledger,
        experiment_id=config.experiment_id,
        pairing_id=config.pairing_id,
        variant=config.variant,
        training_seed=config.training_seed,
        sampling_seed=config.corruption_seed,
        attempts=1,
        backbone_calls=config.calls,
        revision_control="none",
        revision_threshold=config.revision_threshold,
        temperature=config.temperature,
        device=config.device,
    )
    model, model_provenance = _load_model(
        config.checkpoint,
        config=sampling_config,
        protocol=protocol,
        device=device,
    )
    catalog = PyXtalChartCatalog()
    index = JsonlRecordIndex(config.dataset_paths)
    records = _hash_fixed_records(index, config.structures)
    index.close()
    subset_hash = hashlib.sha256(
        "\n".join(str(record["material_id"]) for record in records).encode("utf-8")
    ).hexdigest()
    deriver = SeedDeriver(protocol.name, config.experiment_id)
    pair_deriver = SeedDeriver(
        protocol.name,
        config.pairing_id or config.experiment_id,
    )
    attempts = AttemptLedger(config.attempt_ledger)
    artifacts = ArtifactLedger(config.output_jsonl)
    existing = {record.attempt_id for record in attempts.records()}
    succeeded = 0
    exact = 0
    edit_before: list[int] = []
    edit_after: list[int] = []
    not_applicable = 0
    runtime_profile_s: collections.Counter[str] = collections.Counter()
    started_all = time.monotonic()
    indexed_records = list(enumerate(records))
    for chunk_start in range(
        0, len(indexed_records), config.inference_batch_size
    ):
        chunk = indexed_records[
            chunk_start : chunk_start + config.inference_batch_size
        ]
        prepared: list[dict[str, Any]] = []
        works: list[_RecoveryWork] = []
        for ordinal, record in chunk:
            attempt_id = deriver.attempt_id(
                training_seed=config.training_seed,
                sampling_seed=config.corruption_seed,
                ordinal=ordinal,
                method=config.variant.value,
            )
            if attempt_id in existing:
                raise ValueError(
                    f"recovery would retry immutable attempt {attempt_id}"
                )
            seed = deriver.derive(
                training_seed=config.training_seed,
                sampling_seed=config.corruption_seed,
                attempt_id=attempt_id,
                stage="recovery",
            )
            pair_id = pair_deriver.pair_id(
                training_seed=config.training_seed,
                sampling_seed=config.corruption_seed,
                ordinal=ordinal,
            )
            paired_seed = pair_deriver.paired_derive(
                training_seed=config.training_seed,
                sampling_seed=config.corruption_seed,
                ordinal=ordinal,
                stage="recovery_corruption_and_reverse",
            )
            attempts.append(
                AttemptRecord(
                    attempt_id=attempt_id,
                    method=config.variant.value,
                    training_seed=config.training_seed,
                    sampling_seed=config.corruption_seed,
                    stage="recovery",
                    status=AttemptStatus.SUBMITTED,
                    seed=seed,
                )
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(paired_seed)
            context = _AttemptContext(
                attempt_id,
                random.Random(paired_seed),
                generator,
                {"joint": 0, "bridge": 0, "projection": 0},
                [],
            )
            work = _RecoveryWork(context=context)
            try:
                source = StratifiedState.from_dict(
                    dict(_primary(record)["state"])
                )
                corrupt, corruption_trace = corrupt_state(
                    source,
                    catalog,
                    config,
                    context.python_rng,
                )
                work.corruption_trace = corruption_trace
                work.initialize(source, corrupt)
            except Exception as exc:
                work.error = exc
            works.append(work)
            prepared.append(
                {
                    "ordinal": ordinal,
                    "record": record,
                    "attempt_id": attempt_id,
                    "seed": seed,
                    "pair_id": pair_id,
                    "paired_seed": paired_seed,
                }
            )

        chunk_started = time.monotonic()
        try:
            completed = _run_recovery_batch(
                works, model, catalog, config, device
            )
        except Exception as exc:  # defensive batch-level terminal accounting
            for work in works:
                if work.error is None:
                    work.error = exc
            completed = tuple(works)
        chunk_elapsed = time.monotonic() - chunk_started
        amortized_elapsed = chunk_elapsed / len(prepared)
        for item, work in zip(prepared, completed):
            runtime_profile_s.update(work.runtime_profile_s)
            ordinal = int(item["ordinal"])
            record = item["record"]
            attempt_id = str(item["attempt_id"])
            seed = int(item["seed"])
            pair_id = str(item["pair_id"])
            paired_seed = int(item["paired_seed"])
            context = work.context
            initial_field_count = (
                None if work.source is None else work.source.field_count
            )
            flops = float(
                2 * model.parameter_count() * context.calls["joint"]
            )
            metadata = {
                "ordinal": ordinal,
                "pair_id": pair_id,
                "paired_seed": paired_seed,
                "subset_hash": subset_hash,
                "checkpoint_sha256": model_provenance["checkpoint_sha256"],
                "source_bundle_sha256": model_provenance[
                    "source_bundle_sha256"
                ],
                "runtime_source_bundle_sha256": config.runtime_source_bundle_sha256,
                "inference_batch_size": len(prepared),
                "runtime_workers": config.runtime_workers,
                "runtime_profile_s": dict(sorted(work.runtime_profile_s.items())),
                "walltime_allocation": (
                    "equal_amortized_within_inference_batch"
                ),
            }
            if (
                work.error is None
                and work.source is not None
                and work.state is not None
                and work.mechanism is not None
            ):
                source = work.source
                recovered = work.state
                before = work.initial_distance
                after = topology_edit_distance(source, recovered)
                is_exact = recovered.topology_hash() == source.topology_hash()
                artifact = {
                    "schema": "wqcodiff_recovery_attempt_v1",
                    "attempt_id": attempt_id,
                    "material_id": record["material_id"],
                    "method": config.variant.value,
                    "training_seed": config.training_seed,
                    "corruption_seed": config.corruption_seed,
                    "ordinal": ordinal,
                    "pair_id": pair_id,
                    "paired_seed": paired_seed,
                    "corruption_level": config.corruption_level,
                    "operator": config.operator,
                    "geometry_condition": config.geometry_condition,
                    "schedule": config.schedule,
                    "control": config.control,
                    "revision_threshold": config.revision_threshold,
                    "subset_hash": subset_hash,
                    "checkpoint_sha256": model_provenance[
                        "checkpoint_sha256"
                    ],
                    "source_bundle_sha256": model_provenance[
                        "source_bundle_sha256"
                    ],
                    "runtime_source_bundle_sha256": config.runtime_source_bundle_sha256,
                    "status": AttemptStatus.SUCCEEDED.value,
                    "applicable": True,
                    "source_topology_hash": source.topology_hash(),
                    "corrupt_topology_hash": work.corrupt_topology_hash,
                    "recovered_topology_hash": recovered.topology_hash(),
                    "exact_full_protostructure_recovery": is_exact,
                    "space_group_recovery": (
                        recovered.space_group == source.space_group
                    ),
                    "topology_edit_distance_before": before,
                    "topology_edit_distance_after": after,
                    "orbit_count_error": abs(
                        len(recovered.orbits) - len(source.orbits)
                    ),
                    "tangent_coordinate_error": _coordinate_error(
                        source, recovered
                    ),
                    # StructureMatcher is not a registered Day-7 gate.  It is
                    # intentionally deferred to the bounded final-evaluation
                    # workflow: a v24 infrastructure pilot demonstrated that
                    # one adversarial recovered structure can keep
                    # StructureMatcher.fit on a CPU core for longer than the
                    # entire recovery batch, turning unrelated planned
                    # attempts into job-level timeouts.
                    "structure_match": None,
                    "structure_match_status": "deferred_not_registered_day7_gate",
                    "initial_revisable_field_count": initial_field_count,
                    "corruption_trace": work.corruption_trace,
                    "recovery_trace": context.trace,
                    "mechanism": work.mechanism,
                    "calls": context.calls,
                    "walltime_s": amortized_elapsed,
                    "inference_batch_size": len(prepared),
                    "runtime_workers": config.runtime_workers,
                    "runtime_profile_s": dict(
                        sorted(work.runtime_profile_s.items())
                    ),
                    "inference_batch_elapsed_s": chunk_elapsed,
                    "walltime_allocation": metadata["walltime_allocation"],
                }
                digest = artifacts.append(artifact)
                attempts.append(
                    AttemptRecord(
                        attempt_id=attempt_id,
                        method=config.variant.value,
                        training_seed=config.training_seed,
                        sampling_seed=config.corruption_seed,
                        stage="recovery",
                        status=AttemptStatus.SUCCEEDED,
                        artifact_hash=digest,
                        seed=seed,
                        calls=context.calls,
                        flops=flops,
                        walltime_s=amortized_elapsed,
                        metadata=metadata,
                    )
                )
                succeeded += 1
                exact += int(is_exact)
                edit_before.append(before)
                edit_after.append(after)
                continue

            exc = work.error or RuntimeError(
                "batched recovery produced no terminal state"
            )
            reason = f"{type(exc).__name__}:{exc}"
            not_applicable += int(isinstance(exc, CorruptionNotApplicable))
            digest = artifacts.append(
                {
                    "schema": "wqcodiff_recovery_attempt_v1",
                    "attempt_id": attempt_id,
                    "material_id": record["material_id"],
                    "method": config.variant.value,
                    "training_seed": config.training_seed,
                    "corruption_seed": config.corruption_seed,
                    "ordinal": ordinal,
                    "pair_id": pair_id,
                    "paired_seed": paired_seed,
                    "corruption_level": config.corruption_level,
                    "operator": config.operator,
                    "geometry_condition": config.geometry_condition,
                    "schedule": config.schedule,
                    "control": config.control,
                    "revision_threshold": config.revision_threshold,
                    "subset_hash": subset_hash,
                    "checkpoint_sha256": model_provenance[
                        "checkpoint_sha256"
                    ],
                    "source_bundle_sha256": model_provenance[
                        "source_bundle_sha256"
                    ],
                    "runtime_source_bundle_sha256": config.runtime_source_bundle_sha256,
                    "status": AttemptStatus.FAILED.value,
                    "applicable": not isinstance(
                        exc, CorruptionNotApplicable
                    ),
                    "reason": reason,
                    "initial_revisable_field_count": initial_field_count,
                    "calls": context.calls,
                    "walltime_s": amortized_elapsed,
                    "inference_batch_size": len(prepared),
                    "runtime_workers": config.runtime_workers,
                    "runtime_profile_s": dict(
                        sorted(work.runtime_profile_s.items())
                    ),
                    "inference_batch_elapsed_s": chunk_elapsed,
                    "walltime_allocation": metadata["walltime_allocation"],
                }
            )
            attempts.append(
                AttemptRecord(
                    attempt_id=attempt_id,
                    method=config.variant.value,
                    training_seed=config.training_seed,
                    sampling_seed=config.corruption_seed,
                    stage="recovery",
                    status=AttemptStatus.FAILED,
                    reason=reason,
                    artifact_hash=digest,
                    seed=seed,
                    calls=context.calls,
                    flops=flops,
                    walltime_s=amortized_elapsed,
                    metadata=metadata,
                )
            )
    summary = {
        "ok": True,
        "schema": "wqcodiff_recovery_cell_v1",
        "protocol_name": protocol.name,
        "protocol_sha256": protocol.sha256,
        "method": config.variant.value,
        "pairing_id": config.pairing_id or config.experiment_id,
        "structures": config.structures,
        "succeeded": succeeded,
        "failed": config.structures - succeeded,
        "all_attempts_terminal": True,
        "all_attempts_succeeded": succeeded == config.structures,
        "not_applicable": not_applicable,
        "subset_hash": subset_hash,
        "model_provenance": model_provenance,
        "runtime_source_bundle_sha256": config.runtime_source_bundle_sha256,
        "exact_recovery_attempt_rate": exact / config.structures,
        "exact_recovery_success_rate": exact / max(succeeded, 1),
        "mean_edit_distance_before": None
        if not edit_before
        else float(np.mean(edit_before)),
        "mean_edit_distance_after": None
        if not edit_after
        else float(np.mean(edit_after)),
        "elapsed_s": time.monotonic() - started_all,
        "runtime_profile_s": dict(sorted(runtime_profile_s.items())),
        "cell": {
            "corruption_seed": config.corruption_seed,
            "level": config.corruption_level,
            "operator": config.operator,
            "geometry_condition": config.geometry_condition,
            "schedule": config.schedule,
            "control": config.control,
            "revision_threshold": config.revision_threshold,
            "inference_batch_size": config.inference_batch_size,
            "runtime_workers": config.runtime_workers,
            "calls": config.calls,
        },
    }
    write_json_exclusive(
        Path(config.output_jsonl).with_suffix(".summary.json"), summary
    )
    return summary
