"""Attempt-accounted Llama-to-Wyckoff-to-CSP closed-loop generation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..charts import PyXtalChartCatalog
from ..contracts import (
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
    write_json_exclusive,
)
from ..events import TopologyEvent, TopologyEventType
from ..revision import FieldRef, REVISION_THRESHOLDS, RevisionBudget
from ..runtime import compute_geometry_evidence, expand_state, tensorize_state
from ..sampling import _AttemptContext, _apply_event, _continuous_step
from ..state import GeometryEvidence, StratifiedState
from .gate import GateALock, sha256_file
from .inference import WQLlamaEngine
from .lora import validate_trained_adapter
from .protocol import load_protocol_v4
from .wq_refiner import CrysLLMGenWQRefiner, load_registered_csp_refiner
from .wq_text import TopologyEdit


METHOD_CONTROLS = {
    "C-WQ-HANDOFF": "none",
    "C-WQ-CONFEDIT": "confidence",
    "C-WQ-GEOREV": "geometry",
    "C-WQ-BIRTH-DEATH-ONLY": "geometry_birth_death_only",
    "C-WQ-RANDOM-MATCHED-COUNT": "random_count",
    "C-WQ-SHUFFLED-GEOMETRY": "shuffled_geometry",
    "C-WQ-EXTRA-CALL-IGNORED": "extra_call",
}


class WQAttemptFailure(RuntimeError):
    """Preserve compute accounting when one immutable attempt fails."""

    def __init__(
        self,
        cause: Exception,
        calls: Mapping[str, int],
        *,
        trace: list[dict[str, Any]] | None = None,
        edit_usages: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(f"{type(cause).__name__}:{cause}")
        self.cause = cause
        self.calls = dict(calls)
        self.trace = list(trace or ())
        self.edit_usages = list(edit_usages or ())


@dataclasses.dataclass(frozen=True, slots=True)
class CrysLLMGenWQSamplingConfig:
    protocol_path: str
    gate_a_lock: str
    refiner_checkpoint: str
    llama_root: str
    llama_adapter: str
    output_jsonl: str
    attempt_ledger: str
    report_path: str
    experiment_id: str
    pairing_id: str
    method: str
    training_seed: int
    sampling_seed: int
    attempts: int
    execution_patch_sha256: str
    adapter_training_execution_patch_sha256: str | None = None
    refiner_training_execution_patch_sha256: str | None = None
    start_ordinal: int = 0
    reverse_steps: int = 32
    handoff_tau: float = 1.0
    revision_threshold: float = 0.7
    revision_calibration_lock: str | None = None
    reference_generation_jsonl: str | None = None
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.method not in METHOD_CONTROLS:
            raise ValueError("method is outside the registered WQ configuration set")
        if self.training_seed not in {11, 23, 47}:
            raise ValueError("training seed is outside 11/23/47")
        if self.attempts <= 0 or self.start_ordinal < 0:
            raise ValueError("attempt denominator/ordinal is invalid")
        if self.reverse_steps != 32:
            raise ValueError("matched CrysLLMGen WQ methods use 32 reverse steps")
        if self.handoff_tau not in {0.25, 0.5, 0.75, 1.0}:
            raise ValueError("handoff tau is outside the frozen grid")
        if self.revision_threshold not in REVISION_THRESHOLDS:
            raise ValueError("revision threshold is outside the frozen grid")
        if not self.pairing_id:
            raise ValueError("method-independent pairing ID is required")
        for label, value in (
            ("evaluation execution patch", self.execution_patch_sha256),
            (
                "adapter training execution patch",
                self.adapter_training_execution_patch_sha256
                or self.execution_patch_sha256,
            ),
            (
                "refiner training execution patch",
                self.refiner_training_execution_patch_sha256
                or self.execution_patch_sha256,
            ),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{label} must be one lowercase SHA256")
        requires_reference = self.method in {
            "C-WQ-RANDOM-MATCHED-COUNT",
            "C-WQ-SHUFFLED-GEOMETRY",
            "C-WQ-EXTRA-CALL-IGNORED",
        }
        if requires_reference != (self.reference_generation_jsonl is not None):
            raise ValueError(
                "matched-count controls require, and primary methods forbid, a GEOREV reference"
            )
        requires_calibration = self.method != "C-WQ-HANDOFF"
        if requires_calibration != (self.revision_calibration_lock is not None):
            raise ValueError(
                "all editing/control methods require, and handoff forbids, a revision calibration lock"
            )


def _autocast(device: torch.device) -> Any:
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _load_refiner(
    checkpoint_path: Path,
    *,
    project_root: Path,
    device: torch.device,
    expected_execution_patch_sha256: str,
) -> tuple[CrysLLMGenWQRefiner, Mapping[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "crysllmgen_wq_refiner_ema_v1":
        raise ValueError("unsupported WQ refiner checkpoint")
    if payload.get("execution_patch_sha256") != expected_execution_patch_sha256:
        raise ValueError("refiner checkpoint identity mismatch: execution patch")
    mapping = payload.get("mapping", {})
    model, observed = load_registered_csp_refiner(
        snapshot_root=project_root / "crystal_dlm/wqcodiff/crysllmgen/upstream",
        checkpoint=mapping["checkpoint"],
    )
    if observed["checkpoint_sha256"] != mapping.get("checkpoint_sha256"):
        raise ValueError("parent CSP checkpoint changed after refiner training")
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


def _field_scores(
    state: StratifiedState,
    output: Any,
    *,
    control: str,
) -> dict[FieldRef, float]:
    result: dict[FieldRef, float] = {}
    if control in {
        "geometry",
        "geometry_birth_death_only",
        "random_count",
        "shuffled_geometry",
    }:
        revisions = torch.sigmoid(output.revision_logits.float()).detach().cpu().numpy()
        event = torch.softmax(output.event_logits[0].float(), dim=-1).detach().cpu().numpy()
        global_topology = float(1.0 - event[0])
        for index, orbit in enumerate(state.orbits):
            result[FieldRef(orbit.orbit_id, "existence")] = max(
                float(revisions[index, 0]), global_topology
            )
            result[FieldRef(orbit.orbit_id, "wyckoff_type")] = float(
                revisions[index, 1]
            )
            result[FieldRef(orbit.orbit_id, "species")] = float(revisions[index, 2])
    elif control == "confidence":
        species = 1.0 - torch.softmax(output.species_logits.float(), -1).max(-1).values
        wyckoff = 1.0 - torch.softmax(output.wyckoff_logits.float(), -1).max(-1).values
        event = 1.0 - torch.softmax(output.event_logits[0].float(), -1)[0]
        for index, orbit in enumerate(state.orbits):
            result[FieldRef(orbit.orbit_id, "existence")] = float(event)
            result[FieldRef(orbit.orbit_id, "wyckoff_type")] = float(wyckoff[index])
            result[FieldRef(orbit.orbit_id, "species")] = float(species[index])
    return result


def _select_one_trigger(
    state: StratifiedState,
    output: Any,
    budget: RevisionBudget,
    *,
    threshold: float,
    control: str,
    rng: random.Random,
) -> tuple[FieldRef, float] | None:
    scores = _field_scores(state, output, control=control)
    if not scores:
        return None
    best = max(scores, key=lambda field: (scores[field], field.orbit_id, field.field))
    if control == "random_count" and scores[best] >= threshold:
        candidates = [field for field in scores if budget.count(field) < 2]
        if candidates:
            random_field = candidates[rng.randrange(len(candidates))]
            scores = {random_field: 1.0}
            best = random_field
    else:
        scores = {best: scores[best]}
    decision = budget.select(
        scores,
        threshold=threshold,
        current_field_count=state.field_count,
    )
    if not decision.selected:
        return None
    selected = decision.selected[0]
    return selected, float(scores[selected])


def _select_forced_trigger(
    state: StratifiedState,
    output: Any,
    budget: RevisionBudget,
    *,
    control: str,
    rng: random.Random,
) -> tuple[FieldRef, float] | None:
    scores = _field_scores(state, output, control=control)
    candidates = [field for field in scores if budget.count(field) < 2]
    if not candidates or budget.remaining <= 0:
        return None
    if control == "random_count":
        field = candidates[rng.randrange(len(candidates))]
    else:
        field = max(candidates, key=lambda value: (scores[value], value.orbit_id, value.field))
    decision = budget.select(
        {field: 1.0},
        threshold=0.5,
        current_field_count=state.field_count,
    )
    if not decision.selected:
        return None
    return decision.selected[0], float(scores[field])


def _reference_revision_steps(
    path: str | Path,
    *,
    expected_handoff_tau: float,
) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("schema") != "wqcodiff_generation_attempt_v1"
                or row.get("method") != "C-WQ-GEOREV"
            ):
                raise ValueError(
                    f"GEOREV reference line {line_number} has incompatible identity"
                )
            if float(row.get("handoff_tau", -1.0)) != expected_handoff_tau:
                raise ValueError("GEOREV reference uses a different frozen handoff tau")
            pair_id = str(row.get("pair_id", ""))
            if not pair_id or pair_id in result:
                raise ValueError("GEOREV reference has missing/duplicate pair IDs")
            steps = tuple(
                int(value["reverse_step"])
                for value in row.get("edit_usages", ())
            )
            if tuple(sorted(set(steps))) != steps or any(
                not 16 <= value < 32 for value in steps
            ):
                raise ValueError("GEOREV reference revision schedule is invalid")
            result[pair_id] = steps
    if not result:
        raise ValueError("GEOREV reference is empty")
    return result


def _edit_event(
    edit: TopologyEdit,
    state: StratifiedState,
) -> TopologyEvent:
    if edit.kind == "noop":
        return TopologyEvent(TopologyEventType.NONE)
    if edit.kind == "birth":
        return TopologyEvent(
            TopologyEventType.BIRTH,
            target_wyckoff_type=edit.wyckoff_type,
            target_species=edit.species,
            new_orbit_id=(
                f"llama-{state.topology_hash()[:12]}-{edit.wyckoff_type}-{edit.species}"
            ),
        )
    if edit.orbit_index is None:
        raise ValueError("direct edit has no orbit pointer")
    orbit = state.orbits[edit.orbit_index]
    if edit.kind == "death":
        return TopologyEvent(TopologyEventType.DEATH, orbit_id=orbit.orbit_id)
    if edit.kind == "type_change":
        return TopologyEvent(
            TopologyEventType.WYCKOFF_CHANGE,
            orbit_id=orbit.orbit_id,
            target_wyckoff_type=edit.wyckoff_type,
            new_orbit_id=orbit.orbit_id,
        )
    if edit.kind == "species_change":
        return TopologyEvent(
            TopologyEventType.SPECIES_CHANGE,
            orbit_id=orbit.orbit_id,
            target_species=edit.species,
        )
    raise ValueError(f"unsupported direct edit: {edit.kind}")


def _derived_subseed(seed: int, label: str, step: int) -> int:
    raw = f"{seed}:{label}:{step}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _sample_one(
    *,
    model: CrysLLMGenWQRefiner,
    llama: WQLlamaEngine,
    catalog: PyXtalChartCatalog,
    config: CrysLLMGenWQSamplingConfig,
    attempt_id: str,
    seed: int,
    device: torch.device,
    reference_revision_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    rng = random.Random(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    context = _AttemptContext(
        attempt_id=attempt_id,
        python_rng=rng,
        torch_generator=generator,
        calls={
            "llama": 0,
            "llama_tokens": 0,
            "joint": 0,
            "projection": 0,
            "bridge": 0,
        },
        trace=[],
    )
    edit_usages: list[dict[str, Any]] = []
    try:
        state, proposal_text, proposal_usage = llama.propose(
            catalog=catalog,
            seed=_derived_subseed(seed, "proposal", -1),
            attempt_id=attempt_id,
        )
        context.calls["llama"] += 1
        context.calls["llama_tokens"] += int(proposal_usage["generated_tokens"])
        context.last_state = state
        control = METHOD_CONTROLS[config.method]
        budget = RevisionBudget(state.field_count)
        score_norms: tuple[float, ...] = (0.0,) * len(state.orbits)
        uncertainties: tuple[float, ...] = (0.0,) * len(state.orbits)
        # Direct CrysLLMGen-style injection treats the Llama proposal as the
        # state at tau and runs the same 64-call refiner budget from tau to 0.
        # The proposal is deliberately not forward-noised in the headline
        # comparison; correctly noised injection is a validation diagnostic.
        times = np.linspace(config.handoff_tau, 0.0, config.reverse_steps + 1)
        for step in range(config.reverse_steps):
            context.reverse_step = step
            current = float(times[step])
            following = float(times[step + 1])
            midpoint = 0.5 * (current + following)
            output = None
            last_evidence: list[GeometryEvidence] = []
            for half_current, half_next in ((current, midpoint), (midpoint, following)):
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
                if control == "shuffled_geometry" and len(evidence) > 1:
                    rng.shuffle(evidence)
                last_evidence = evidence
                batch = tensorize_state(
                    state,
                    expanded,
                    evidence,
                    time=half_current,
                ).to(device)
                with torch.no_grad(), _autocast(device):
                    output = model(
                        batch,
                        use_geometry_evidence=control not in {"confidence", "none"},
                    )
                    context.calls["joint"] += 1
                state, score_norms = _continuous_step(
                    state,
                    expanded,
                    output,
                    current_time=half_current,
                    next_time=half_next,
                )
                species_prob = torch.softmax(output.species_logits.float(), -1)
                wyckoff_prob = torch.softmax(output.wyckoff_logits.float(), -1)
                uncertainties = tuple(
                    float(
                        max(
                            1.0 - species_prob[index].max(),
                            1.0 - wyckoff_prob[index].max(),
                        )
                    )
                    for index in range(len(state.orbits))
                )
            assert output is not None
            context.last_state = state
            trigger = None
            if control in {"random_count", "shuffled_geometry"} and step in reference_revision_steps:
                trigger = _select_forced_trigger(
                    state,
                    output,
                    budget,
                    control=control,
                    rng=rng,
                )
                if trigger is None:
                    raise RuntimeError("matched-count control could not realize a reference trigger")
            elif control not in {"none", "extra_call"} and step >= config.reverse_steps // 2:
                trigger = _select_one_trigger(
                    state,
                    output,
                    budget,
                    threshold=config.revision_threshold,
                    control=control,
                    rng=rng,
                )
            if control == "extra_call" and step in reference_revision_steps:
                with torch.no_grad(), _autocast(device):
                    _ = model(batch, use_geometry_evidence=False)
                context.calls["joint"] += 1
                context.trace.append(
                    {
                        "step": len(context.trace),
                        "reverse_step": step,
                        "action": "extra_call_ignored",
                    }
                )
            if trigger is not None:
                field, trigger_score = trigger
                edit_evidence = last_evidence
                if control == "random_count":
                    edit_evidence = [
                        GeometryEvidence(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        for _ in state.orbits
                    ]
                edit, edit_text, usage = llama.edit(
                    state,
                    edit_evidence,
                    catalog=catalog,
                    seed=_derived_subseed(seed, "edit", step),
                )
                context.calls["llama"] += 1
                context.calls["llama_tokens"] += int(usage["generated_tokens"])
                usage = {
                    **usage,
                    "reverse_step": step,
                    "trigger_field": dataclasses.asdict(field),
                    "trigger_score": trigger_score,
                    "edit_text": edit_text,
                    "edit_kind": edit.kind,
                }
                edit_usages.append(usage)
                if control == "geometry_birth_death_only" and edit.kind not in {
                    "noop",
                    "birth",
                    "death",
                }:
                    edit = TopologyEdit("noop")
                event = _edit_event(edit, state)
                before = state.topology_hash(include_geometry=True)
                state = _apply_event(state, event, output, catalog, context)
                context.last_state = state
                context.trace.append(
                    {
                        "step": len(context.trace),
                        "reverse_step": step,
                        "action": "llama_direct_edit",
                        "trigger_field": dataclasses.asdict(field),
                        "trigger_score": trigger_score,
                        "command": edit_text,
                        "applied_kind": edit.kind,
                        "topology_before": before,
                        "topology_after": state.topology_hash(include_geometry=True),
                    }
                )
                if len(score_norms) != len(state.orbits):
                    score_norms = (0.0,) * len(state.orbits)
                    uncertainties = (0.0,) * len(state.orbits)
        final = expand_state(state, catalog)
        context.calls["projection"] += 1
        structure = final.pymatgen_structure()
        cif = structure.to(fmt="cif")
    except Exception as exc:
        raise WQAttemptFailure(
            exc,
            context.calls,
            trace=context.trace,
            edit_usages=edit_usages,
        ) from exc
    return {
        "schema": "wqcodiff_generation_attempt_v1",
        "producer_schema": "crysllmgen_wq_generation_attempt_v1",
        "attempt_id": attempt_id,
        "method": config.method,
        "training_seed": config.training_seed,
        "sampling_seed": config.sampling_seed,
        "status": AttemptStatus.SUCCEEDED.value,
        "reason": "",
        "state": state.to_dict(),
        "structure": structure.as_dict(),
        "structure_cif_sha256": hashlib.sha256(cif.encode("utf-8")).hexdigest(),
        "intended_space_group": state.space_group,
        "redetected_space_group": final.redetected_space_group,
        "atom_count": state.atom_count,
        "orbit_count": len(state.orbits),
        "proposal_text": proposal_text,
        "proposal_usage": proposal_usage,
        "edit_usages": edit_usages,
        "revision_control": control,
        "handoff_tau": config.handoff_tau,
        "revision_threshold": config.revision_threshold,
        "reference_revision_steps": list(reference_revision_steps),
        "revision_initial_field_count": budget.initial_field_count,
        "revision_total": budget.total,
        "revision_churn": budget.churn,
        "reverse_steps": config.reverse_steps,
        "calls": context.calls,
        "trace": context.trace,
    }


def sample(config: CrysLLMGenWQSamplingConfig) -> dict[str, Any]:
    protocol = load_protocol_v4(config.protocol_path)
    project_root = Path(config.protocol_path).resolve().parents[3]
    gate = GateALock.load(
        config.gate_a_lock,
        project_root=project_root,
        protocol_path=config.protocol_path,
        execution_patch_manifest_sha256=config.execution_patch_sha256,
    )
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered sampling requires CUDA")
    checkpoint_path = Path(config.refiner_checkpoint).resolve()
    adapter_training_execution_patch_sha256 = (
        config.adapter_training_execution_patch_sha256
        or config.execution_patch_sha256
    )
    refiner_training_execution_patch_sha256 = (
        config.refiner_training_execution_patch_sha256
        or config.execution_patch_sha256
    )
    model, checkpoint = _load_refiner(
        checkpoint_path,
        project_root=project_root,
        device=device,
        expected_execution_patch_sha256=(
            refiner_training_execution_patch_sha256
        ),
    )
    if checkpoint.get("source_bundle_sha256") != gate.source_bundle_sha256:
        raise ValueError("sampling checkpoint/Gate A source mismatch")
    if int(checkpoint.get("training_seed", -1)) != config.training_seed:
        raise ValueError("sampling training seed/checkpoint mismatch")
    adapter_training_identity = validate_trained_adapter(
        adapter_root=config.llama_adapter,
        gate_a_lock_sha256=gate.sha256,
        source_bundle_sha256=gate.source_bundle_sha256,
        representation="wyckoff",
        training_stage="mixed_edit",
        training_seed=config.training_seed,
        execution_patch_sha256=adapter_training_execution_patch_sha256,
    )
    llama = WQLlamaEngine.load(
        base_root=config.llama_root,
        adapter_root=config.llama_adapter,
    )
    catalog = PyXtalChartCatalog(hall_style="spglib")
    output_path = Path(config.output_jsonl).resolve()
    ledger_path = Path(config.attempt_ledger).resolve()
    report_path = Path(config.report_path).resolve()
    for path in (output_path, ledger_path, report_path):
        if path.exists():
            raise FileExistsError(f"sampling evidence is immutable: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactLedger(output_path)
    attempts = AttemptLedger(ledger_path)
    deriver = SeedDeriver(protocol.name, config.experiment_id)
    pairing_deriver = SeedDeriver(protocol.name, config.pairing_id)
    refiner_sha256 = sha256_file(checkpoint_path)
    parameter_count = model.parameter_count()
    model_identity = {
        "refiner_checkpoint_sha256": refiner_sha256,
        "refiner_training_execution_patch_sha256": (
            refiner_training_execution_patch_sha256
        ),
        "evaluation_execution_patch_sha256": config.execution_patch_sha256,
        "adapter_training": adapter_training_identity,
        **llama.identity,
    }
    revision_lock_identity = None
    if config.revision_calibration_lock is not None:
        from .revision_calibration import RevisionCalibrationLock

        revision_lock = RevisionCalibrationLock.load(
            config.revision_calibration_lock,
            gate_a_lock_sha256=gate.sha256,
            refiner_checkpoint_sha256=refiner_sha256,
            llama_adapter_sha256=str(llama.identity["adapter_model_sha256"]),
        )
        if revision_lock.threshold != config.revision_threshold:
            raise ValueError("sampling threshold differs from validation calibration")
        revision_lock_identity = {
            "path": str(revision_lock.path),
            "sha256": revision_lock.sha256,
            "threshold": revision_lock.threshold,
        }
    reference_by_pair = (
        {}
        if config.reference_generation_jsonl is None
        else _reference_revision_steps(
            config.reference_generation_jsonl,
            expected_handoff_tau=config.handoff_tau,
        )
    )
    reference_identity = (
        None
        if config.reference_generation_jsonl is None
        else {
            "path": str(Path(config.reference_generation_jsonl).resolve()),
            "sha256": sha256_file(config.reference_generation_jsonl),
            "pairs": len(reference_by_pair),
        }
    )
    expected_ids: list[str] = []
    for ordinal in range(config.start_ordinal, config.start_ordinal + config.attempts):
        attempt_id = deriver.attempt_id(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            ordinal=ordinal,
            method=config.method,
        )
        expected_ids.append(attempt_id)
        seed = deriver.derive(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            attempt_id=attempt_id,
            stage="generation",
        )
        attempts.append(
            AttemptRecord(
                attempt_id=attempt_id,
                method=config.method,
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                stage="generation",
                status=AttemptStatus.SUBMITTED,
                seed=seed,
                metadata={
                    "ordinal": ordinal,
                    "pair_id": pairing_deriver.pair_id(
                        training_seed=config.training_seed,
                        sampling_seed=config.sampling_seed,
                        ordinal=ordinal,
                    ),
                },
            )
        )
    succeeded = 0
    failed = 0
    started_all = time.monotonic()
    for ordinal, attempt_id in zip(
        range(config.start_ordinal, config.start_ordinal + config.attempts),
        expected_ids,
    ):
        ledger_seed = deriver.derive(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            attempt_id=attempt_id,
            stage="generation",
        )
        sampling_seed = pairing_deriver.paired_derive(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            ordinal=ordinal,
            stage="generation_sampling",
        )
        started = time.monotonic()
        calls = {
            "llama": 0,
            "llama_tokens": 0,
            "joint": 0,
            "projection": 0,
            "bridge": 0,
        }
        try:
            pair_id = pairing_deriver.pair_id(
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                ordinal=ordinal,
            )
            reference_steps = reference_by_pair.get(pair_id, ())
            if reference_by_pair and pair_id not in reference_by_pair:
                raise ValueError("matched-count control has no paired GEOREV reference")
            row = _sample_one(
                model=model,
                llama=llama,
                catalog=catalog,
                config=config,
                attempt_id=attempt_id,
                seed=sampling_seed,
                device=device,
                reference_revision_steps=reference_steps,
            )
            calls = dict(row["calls"])
            status = AttemptStatus.SUCCEEDED
            reason = ""
            succeeded += 1
        except WQAttemptFailure as exc:
            calls = dict(exc.calls)
            status = AttemptStatus.FAILED
            reason = str(exc)
            row = {
                "schema": "wqcodiff_generation_attempt_v1",
                "producer_schema": "crysllmgen_wq_generation_attempt_v1",
                "attempt_id": attempt_id,
                "method": config.method,
                "training_seed": config.training_seed,
                "sampling_seed": config.sampling_seed,
                "status": status.value,
                "reason": reason,
                "calls": calls,
                "trace": exc.trace,
                "edit_usages": exc.edit_usages,
                "reference_revision_steps": list(reference_steps),
            }
            failed += 1
        elapsed = time.monotonic() - started
        row.update(
            {
                "ordinal": ordinal,
                "pair_id": pairing_deriver.pair_id(
                    training_seed=config.training_seed,
                    sampling_seed=config.sampling_seed,
                    ordinal=ordinal,
                ),
                "ledger_seed": ledger_seed,
                "sampling_seed_derived": sampling_seed,
                "paired_seed": sampling_seed,
                "experiment_id": config.experiment_id,
                "pairing_id": config.pairing_id,
                "handoff_tau": config.handoff_tau,
                "stage": "raw",
                "backbone_calls": calls.get("joint", 0),
                "generation_flops_lower_bound": float(
                    2 * parameter_count * calls.get("joint", 0)
                    + 2
                    * int(llama.identity["parameter_count_with_adapter"])
                    * calls.get("llama_tokens", 0)
                ),
                "generation_flops_estimator": "2_parameter_count_per_forward_or_decode_token",
                "walltime_s": elapsed,
                "retry_or_replacement_used": False,
                "adapter_training_execution_patch_sha256": (
                    adapter_training_execution_patch_sha256
                ),
                "refiner_training_execution_patch_sha256": (
                    refiner_training_execution_patch_sha256
                ),
                "evaluation_execution_patch_sha256": (
                    config.execution_patch_sha256
                ),
                "model_identity": model_identity,
                "matched_count_reference": reference_identity,
                "revision_calibration_lock": revision_lock_identity,
            }
        )
        digest = artifacts.append(row)
        flops = float(
            2 * parameter_count * calls.get("joint", 0)
            + 2
            * int(llama.identity["parameter_count_with_adapter"])
            * calls.get("llama_tokens", 0)
        )
        attempts.append(
            AttemptRecord(
                attempt_id=attempt_id,
                method=config.method,
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                stage="generation",
                status=status,
                reason=reason,
                artifact_hash=digest,
                seed=ledger_seed,
                calls=calls,
                flops=flops,
                walltime_s=elapsed,
                metadata={
                    "ordinal": ordinal,
                    "pairing_id": config.pairing_id,
                    "pair_id": pairing_deriver.pair_id(
                        training_seed=config.training_seed,
                        sampling_seed=config.sampling_seed,
                        ordinal=ordinal,
                    ),
                    "sampling_seed_derived": sampling_seed,
                },
            )
        )
    audit = attempts.audit(
        seed_deriver=deriver,
        terminal_stage="generation",
        expected_attempt_ids=expected_ids,
    )
    report = {
        "schema": "crysllmgen_wq_sampling_report_v1",
        "ok": audit.ok and succeeded + failed == config.attempts,
        "method": config.method,
        "handoff_tau": config.handoff_tau,
        "training_seed": config.training_seed,
        "sampling_seed": config.sampling_seed,
        "submitted": config.attempts,
        "terminal": succeeded + failed,
        "succeeded": succeeded,
        "failed": failed,
        "retry_or_replacement_used": False,
        "audit": dataclasses.asdict(audit),
        "output_jsonl": str(output_path),
        "output_sha256": sha256_file(output_path),
        "attempt_ledger": str(ledger_path),
        "attempt_ledger_sha256": sha256_file(ledger_path),
        "model_identity": {
            "refiner_checkpoint": str(checkpoint_path),
            "refiner_checkpoint_sha256": refiner_sha256,
            "refiner_training_execution_patch_sha256": (
                refiner_training_execution_patch_sha256
            ),
            "adapter_training": adapter_training_identity,
            "adapter_training_execution_patch_sha256": (
                adapter_training_execution_patch_sha256
            ),
            "evaluation_execution_patch_sha256": (
                config.execution_patch_sha256
            ),
            **llama.identity,
        },
        "matched_count_reference": reference_identity,
        "revision_calibration_lock": revision_lock_identity,
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": gate.sha256,
        "walltime_s": time.monotonic() - started_all,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
        "numerical_threads": int(os.environ.get("OMP_NUM_THREADS", "0")),
    }
    write_json_exclusive(report_path, report)
    if not report["ok"]:
        raise RuntimeError("WQ sampling attempt audit failed")
    return report
