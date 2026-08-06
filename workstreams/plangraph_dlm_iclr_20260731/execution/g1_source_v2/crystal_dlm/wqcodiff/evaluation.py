"""Attempt-level raw/common-refiner/relaxed MLIP-SUN evaluation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
    write_json_exclusive,
)
from .charts import wyckoff_type_to_letter
from .dataset import material_family_from_symbols
from .hull import load_frozen_hull
from .metrics import compute_relational_metrics, matcher_contract_hash
from .mlip import EvaluatorLock, MLIPCalculator
from .novelty_reference import load_novelty_reference
from .protocol import load_protocol


def _load_evaluation_protocol(path: str | Path) -> Any:
    """Accept the historical v3 evaluator contract or active CrysLLMGen v4."""

    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("protocol", {}).get("schema_version", -1)) == 4:
        from .crysllmgen.protocol import load_protocol_v4

        return load_protocol_v4(path)
    return load_protocol(path)


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationConfig:
    input_jsonl: str
    output_jsonl: str
    energy_cache_jsonl: str
    attempt_ledger: str
    experiment_id: str
    evaluator: str
    stage: str
    asset_lock: str
    model_root: str
    hull_path: str
    novelty_reference: str
    device: str = "cuda"
    subset_size: int | None = None

    def __post_init__(self) -> None:
        if self.evaluator not in {"chgnet", "mattersim", "mace"}:
            raise ValueError("unknown evaluator")
        if self.stage not in {"raw", "common-refiner", "relaxed"}:
            raise ValueError("unknown evaluation stage")
        if self.subset_size is not None and self.subset_size <= 0:
            raise ValueError("subset_size must be positive")


def _read_generation(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    pairs: set[str] = set()
    methods: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("schema") != "wqcodiff_generation_attempt_v1":
                raise ValueError(f"generation line {line_number} has invalid schema")
            attempt_id = str(payload.get("attempt_id", ""))
            if not attempt_id:
                raise ValueError(f"generation line {line_number} has no attempt_id")
            if attempt_id in ids:
                raise ValueError(f"duplicate generation attempt {attempt_id}")
            pair_id = str(payload.get("pair_id") or "")
            if not pair_id:
                raise ValueError(f"generation line {line_number} has no pair_id")
            if pair_id in pairs:
                raise ValueError(f"duplicate generation pair {pair_id}")
            try:
                status = AttemptStatus(str(payload.get("status")))
            except ValueError as exc:
                raise ValueError(
                    f"generation line {line_number} has unknown status"
                ) from exc
            if not status.terminal:
                raise ValueError(f"generation line {line_number} is not terminal")
            method = str(payload.get("method") or "")
            if not method:
                raise ValueError(f"generation line {line_number} has no method")
            ids.add(attempt_id)
            pairs.add(pair_id)
            methods.add(method)
            result.append(payload)
    if not result:
        raise ValueError("generation artifact is empty")
    if len(methods) != 1:
        raise ValueError(f"generation artifact mixes methods: {sorted(methods)}")
    return result


def _subset(records: Sequence[dict[str, Any]], size: int | None) -> list[dict[str, Any]]:
    if size is None:
        return list(records)
    if size > len(records):
        raise ValueError(f"requested subset {size}, only {len(records)} attempts exist")
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(
            str(record.get("pair_id") or record["attempt_id"]).encode("utf-8")
        ).hexdigest(),
    )
    return ordered[:size]


def _trace_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    events = [
        item
        for item in record.get("trace", ())
        if item.get("action") == "topology_event"
    ]
    event_counts = {
        event_type: sum(item.get("event_type") == event_type for item in events)
        for event_type in (
            "orbit_birth",
            "orbit_death",
            "wyckoff_type_change",
            "species_change",
        )
    }
    revision_decisions = [
        item
        for item in record.get("trace", ())
        if item.get("action") == "revision_decision"
    ]
    selected = sum(len(item.get("selected", ())) for item in revision_decisions)
    selected += sum(
        item.get("action") == "llama_direct_edit"
        for item in record.get("trace", ())
    )
    revision_fills = sum(
        bool(item.get("revision_fill"))
        for item in record.get("trace", ())
        if item.get("action") in {"species_commit", "wyckoff_commit"}
    )
    return {
        "topology_event_counts": event_counts,
        "topology_changed": any(event_counts.values()),
        "orbit_count_changed": bool(
            event_counts["orbit_birth"] or event_counts["orbit_death"]
        ),
        "dimension_changed": any(
            item.get("dimension_before") != item.get("dimension_after")
            for item in events
            if "dimension_before" in item and "dimension_after" in item
        ),
        "revision_events": selected,
        "revision_fills": revision_fills,
        "occupied_wyckoff_dof": (
            sum(
                len(orbit.get("free_coordinate", ()))
                for orbit in record.get("state", {}).get("orbits", ())
            )
            if record.get("state")
            else None
        ),
    }


def _material_family(structure: Any) -> str:
    return material_family_from_symbols(
        str(element.symbol) for element in structure.composition.elements
    )


def _material_family_from_state(state: Any) -> str:
    if not isinstance(state, Mapping):
        return "unknown"
    try:
        from pymatgen.core.periodic_table import Element

        symbols = {
            str(Element.from_Z(int(orbit["species"])).symbol)
            for orbit in state["orbits"]
        }
    except (KeyError, TypeError, ValueError):
        return "unknown"
    return material_family_from_symbols(symbols) if symbols else "unknown"


def _symmetry_diagnostics(
    structure: Any,
    state_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from pymatgen.core.periodic_table import Element
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5.0)
        conventional = analyzer.get_conventional_standard_structure(
            international_monoclinic=True
        )
        detected = SpacegroupAnalyzer(
            conventional, symprec=0.01, angle_tolerance=5.0
        ).get_symmetrized_structure()
        redetected_multiset = sorted(
            (
                str(symbol),
                str(conventional[indices[0]].specie.symbol),
                len(indices),
            )
            for indices, symbol in zip(
                detected.equivalent_indices,
                detected.wyckoff_symbols,
            )
        )
        intended_multiset = None
        if state_payload is not None:
            intended_multiset = sorted(
                (
                    f"{int(orbit['multiplicity'])}{wyckoff_type_to_letter(int(orbit['wyckoff_type']))}",
                    Element.from_Z(int(orbit["species"])).symbol,
                    int(orbit["multiplicity"]),
                )
                for orbit in state_payload.get("orbits", ())
            )
        return {
            "redetected_space_group": int(analyzer.get_space_group_number()),
            "redetected_wyckoff_multiset": redetected_multiset,
            "intended_wyckoff_multiset": intended_multiset,
            "wyckoff_multiset_match": (
                None
                if intended_multiset is None
                else intended_multiset == redetected_multiset
            ),
            "symmetry_detection_error": None,
        }
    except Exception as exc:
        return {
            "redetected_space_group": None,
            "redetected_wyckoff_multiset": None,
            "intended_wyckoff_multiset": None,
            "wyckoff_multiset_match": None,
            "symmetry_detection_error": f"{type(exc).__name__}:{exc}",
        }


def _structure_hash(structure: Any) -> str:
    return hashlib.sha256(structure.to(fmt="cif").encode("utf-8")).hexdigest()


def _failure_status(reason: str) -> AttemptStatus:
    lowered = reason.lower()
    if "unsupported_element" in lowered or "unsupported_elements" in lowered:
        return AttemptStatus.UNSUPPORTED_ELEMENT
    if "nonconverged" in lowered:
        return AttemptStatus.NONCONVERGED
    if "hull" in lowered:
        return AttemptStatus.MISSING_HULL
    if "cache" in lowered:
        return AttemptStatus.CACHE_MISMATCH
    return AttemptStatus.FAILED


class _EnergyCache:
    def __init__(self, path: str | Path) -> None:
        self.ledger = ArtifactLedger(
            path,
            key_fields=("structure_hash", "evaluator", "stage", "contract_hash"),
        )
        self.records: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
        for record in self.ledger.records():
            key = (
                str(record["structure_hash"]),
                str(record["evaluator"]),
                str(record["stage"]),
                str(record["contract_hash"]),
            )
            if key in self.records:
                raise ValueError(f"duplicate energy-cache key: {key}")
            self.records[key] = record

    def get(
        self, structure_hash: str, evaluator: str, stage: str, contract_hash: str
    ) -> Mapping[str, Any] | None:
        return self.records.get((structure_hash, evaluator, stage, contract_hash))

    def append(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        self.ledger.append(record)
        key = (
            str(record["structure_hash"]),
            str(record["evaluator"]),
            str(record["stage"]),
            str(record["contract_hash"]),
        )
        self.records[key] = dict(record)
        return record


def _evaluate_energy(
    structure: Any,
    *,
    calculator: MLIPCalculator,
    cache: _EnergyCache,
    stage: str,
) -> tuple[Mapping[str, Any], bool]:
    source_hash = _structure_hash(structure)
    cached = cache.get(
        source_hash,
        calculator.evaluator,
        stage,
        calculator.contract_hash,
    )
    if cached is not None:
        return cached, True
    started = time.monotonic()
    try:
        result = (
            calculator.relax(structure)
            if stage == "relaxed"
            else calculator.single_point(structure)
        )
        payload = {
            "schema": "wqcodiff_energy_cache_v1",
            "structure_hash": source_hash,
            "evaluator": calculator.evaluator,
            "stage": stage,
            "contract_hash": calculator.contract_hash,
            "status": "succeeded",
            **result,
            "walltime_s": time.monotonic() - started,
        }
    except Exception as exc:
        payload = {
            "schema": "wqcodiff_energy_cache_v1",
            "structure_hash": source_hash,
            "evaluator": calculator.evaluator,
            "stage": stage,
            "contract_hash": calculator.contract_hash,
            "status": "failed",
            "reason": f"{type(exc).__name__}:{exc}",
            "walltime_s": time.monotonic() - started,
        }
    return cache.append(payload), False


def _ehull(phase_diagram: Any, structure: Any, energy_total_ev: float) -> float:
    from pymatgen.entries.computed_entries import ComputedEntry

    value = float(
        phase_diagram.get_e_above_hull(
            ComputedEntry(structure.composition, float(energy_total_ev))
        )
    )
    if not np.isfinite(value):
        raise ValueError("missing_hull:nonfinite_e_above_hull")
    return max(0.0, value)


def evaluate(config: EvaluationConfig, *, protocol_path: str | Path) -> dict[str, Any]:
    protocol = _load_evaluation_protocol(protocol_path)
    if config.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("MLIP evaluation requires CUDA inside Slurm")
    generation = _subset(_read_generation(config.input_jsonl), config.subset_size)
    subset_ids = sorted(
        str(record.get("pair_id") or record["attempt_id"]) for record in generation
    )
    subset_hash = hashlib.sha256("\n".join(subset_ids).encode("utf-8")).hexdigest()
    if config.stage == "common-refiner":
        invalid = [
            record["attempt_id"]
            for record in generation
            if record.get("status") == "succeeded"
            and record.get("stage") != "common_refiner"
        ]
        if invalid:
            raise ValueError(
                "common-refiner evaluation input lacks frozen refiner provenance; "
                f"first={invalid[0]}"
            )
    lock = EvaluatorLock.load(config.asset_lock)
    calculator = MLIPCalculator(
        evaluator=config.evaluator,
        asset_lock=lock,
        model_root=config.model_root,
        device=config.device,
    )
    hull_payload, phase_diagram = load_frozen_hull(config.hull_path)
    if hull_payload["evaluator"] != config.evaluator:
        raise ValueError("hull evaluator does not match requested evaluator")
    if hull_payload["contract_hash"] != calculator.contract_hash:
        raise ValueError("hull/calculator contract hash mismatch")
    train_structures, train_proto_keys, novelty_summary = load_novelty_reference(
        config.novelty_reference
    )
    if novelty_summary["matcher_contract_sha256"] != matcher_contract_hash():
        raise ValueError("novelty-reference matcher contract mismatch")
    output = ArtifactLedger(
        config.output_jsonl,
        key_fields=("attempt_id", "evaluator", "stage", "hull_sha256"),
    )
    if output.records():
        raise ValueError("evaluation output is immutable and already contains records")
    cache = _EnergyCache(config.energy_cache_jsonl)
    attempts = AttemptLedger(config.attempt_ledger)
    deriver = SeedDeriver(protocol.name, config.experiment_id)
    ledger_stage = f"sun/{config.evaluator}/{config.stage}"
    existing_keys = {record.key for record in attempts.records()}
    for record in generation:
        key = (str(record["attempt_id"]), ledger_stage)
        if key in existing_keys:
            raise ValueError(f"evaluation would retry immutable attempt stage {key}")
        seed = deriver.derive(
            training_seed=int(record["training_seed"]),
            sampling_seed=int(record["sampling_seed"]),
            attempt_id=str(record["attempt_id"]),
            stage=ledger_stage,
        )
        attempts.append(
            AttemptRecord(
                attempt_id=str(record["attempt_id"]),
                method=str(record["method"]),
                training_seed=int(record["training_seed"]),
                sampling_seed=int(record["sampling_seed"]),
                stage=ledger_stage,
                status=AttemptStatus.SUBMITTED,
                seed=seed,
            )
        )

    from pymatgen.core import Structure

    interim: list[dict[str, Any]] = []
    successful_indices: list[int] = []
    successful_attempt_ids: list[str] = []
    successful_structures: list[Any] = []
    successful_states: list[Mapping[str, Any] | None] = []
    started_all = time.monotonic()
    for index, generated in enumerate(generation):
        base = {
            "schema": "wqcodiff_mlip_sun_attempt_v1",
            "attempt_id": str(generated["attempt_id"]),
            "experiment_id": generated.get("experiment_id"),
            "pairing_id": generated.get("pairing_id"),
            "method": str(generated["method"]),
            "training_seed": int(generated["training_seed"]),
            "sampling_seed": int(generated["sampling_seed"]),
            "ordinal": generated.get("ordinal"),
            "pair_id": generated.get("pair_id"),
            "paired_seed": generated.get("paired_seed"),
            "atom_count": generated.get("atom_count"),
            "orbit_count": generated.get("orbit_count"),
            "revision_control": generated.get("revision_control"),
            "revision_threshold": generated.get("revision_threshold"),
            "disc_once_tau": generated.get("disc_once_tau"),
            "temperature": generated.get("temperature"),
            "revision_initial_field_count": generated.get(
                "revision_initial_field_count"
            ),
            "revision_churn": generated.get("revision_churn"),
            "backbone_calls": generated.get("backbone_calls"),
            "generation_calls": generated.get("calls", {}),
            "generation_walltime_s": generated.get("walltime_s"),
            "generation_flops_lower_bound": generated.get(
                "generation_flops_lower_bound"
            ),
            "generation_flops_estimator": generated.get(
                "generation_flops_estimator"
            ),
            "common_refiner_calls": generated.get("common_refiner_calls"),
            "common_refiner_flops_lower_bound": generated.get(
                "common_refiner_flops_lower_bound"
            ),
            "material_family": _material_family_from_state(generated.get("state")),
            "intended_space_group": generated.get("intended_space_group"),
            **_trace_summary(generated),
            "evaluator": config.evaluator,
            "stage": config.stage,
            "contract_hash": calculator.contract_hash,
            "hull_sha256": hull_payload["hull_sha256"],
            "novelty_reference_sha256": novelty_summary["reference_sha256"],
            "matcher_contract_sha256": matcher_contract_hash(),
            "subset_hash": subset_hash,
        }
        if generated.get("status") != "succeeded":
            interim.append(
                {
                    **base,
                    "status": "failed",
                    "reason": f"upstream_generation:{generated.get('reason', generated.get('status'))}",
                    "cache_hit": False,
                }
            )
            continue
        try:
            structure = Structure.from_dict(generated["structure"])
            energy_record, cache_hit = _evaluate_energy(
                structure,
                calculator=calculator,
                cache=cache,
                stage=config.stage,
            )
            if energy_record["status"] != "succeeded":
                raise RuntimeError(str(energy_record["reason"]))
            evaluated = (
                Structure.from_dict(energy_record["structure"])
                if "structure" in energy_record
                else structure
            )
            ehull = _ehull(
                phase_diagram,
                evaluated,
                float(energy_record["energy_total_ev"]),
            )
            symmetry = _symmetry_diagnostics(
                evaluated,
                None
                if generated.get("method") == "B-ATOM-JOINT"
                else generated.get("state"),
            )
            interim.append(
                {
                    **base,
                    "status": "succeeded",
                    "cache_hit": cache_hit,
                    "input_structure_hash": _structure_hash(structure),
                    "evaluated_structure_hash": _structure_hash(evaluated),
                    "structure": evaluated.as_dict(),
                    "energy_total_ev": float(energy_record["energy_total_ev"]),
                    "energy_per_atom_ev": float(energy_record["energy_per_atom_ev"]),
                    "e_above_hull_ev_per_atom": ehull,
                    "stable_at_0p0": ehull <= 0.0,
                    "stable_at_0p1": ehull <= 0.1,
                    "max_force_ev_per_angstrom": energy_record.get(
                        "max_force_ev_per_angstrom"
                    ),
                    "relaxation_steps": energy_record.get("relaxation_steps"),
                    "mlip_walltime_s": float(energy_record.get("walltime_s", 0.0)),
                    "material_family": _material_family(evaluated),
                    "density_g_cm3": float(evaluated.density),
                    "element_count": len(evaluated.composition.elements),
                    **symmetry,
                }
            )
            successful_indices.append(index)
            successful_attempt_ids.append(str(generated["attempt_id"]))
            successful_structures.append(evaluated)
            successful_states.append(generated.get("state"))
        except Exception as exc:
            interim.append(
                {
                    **base,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "cache_hit": False,
                }
            )

    if successful_structures:
        relational = compute_relational_metrics(
            successful_attempt_ids,
            successful_structures,
            successful_states,
            train_structures=train_structures,
            train_protostructure_keys=train_proto_keys,
        )
        for index, metrics in zip(successful_indices, relational):
            row = interim[index]
            row["duplicate_cluster"] = dict(metrics.duplicate_cluster)
            row["unique"] = dict(metrics.unique)
            row["full_novel"] = dict(metrics.full_novel)
            row["anonymous_prototype_novel"] = metrics.anonymous_prototype_novel
            row["protostructure_novel"] = metrics.protostructure_novel
            row["substitution_aware_novel"] = metrics.substitution_aware_novel
            row["matcher_diagnostics"] = dict(metrics.matcher_diagnostics)
            row["novel_unique_standard"] = bool(
                metrics.unique["standard"] and metrics.full_novel["standard"]
            )
            row["mlip_sun_at_0p0"] = bool(
                row["stable_at_0p0"] and row["novel_unique_standard"]
            )
            row["mlip_sun_at_0p1"] = bool(
                row["stable_at_0p1"] and row["novel_unique_standard"]
            )
            row["substitution_aware_mlip_sun_at_0p1"] = bool(
                row["stable_at_0p1"]
                and metrics.unique["standard"]
                and metrics.substitution_aware_novel
            )
            row["matcher_sensitivity_sun_at_0p1"] = {
                sensitivity: bool(
                    row["stable_at_0p1"]
                    and metrics.unique[sensitivity]
                    and metrics.full_novel[sensitivity]
                )
                for sensitivity in ("strict", "standard", "lenient")
            }

    succeeded = 0
    sun0 = 0
    sun1 = 0
    novel_unique = 0
    failure_reasons: dict[str, int] = {}
    matcher_diagnostic_totals: dict[str, int] = {}
    matcher_diagnostic_attempts = 0
    for generated, row in zip(generation, interim):
        if row["status"] != "succeeded":
            row.setdefault("stable_at_0p0", False)
            row.setdefault("stable_at_0p1", False)
            row.setdefault("novel_unique_standard", False)
            row.setdefault("mlip_sun_at_0p0", False)
            row.setdefault("mlip_sun_at_0p1", False)
            row.setdefault("substitution_aware_mlip_sun_at_0p1", False)
            row.setdefault("matcher_diagnostics", {})
            row.setdefault(
                "matcher_sensitivity_sun_at_0p1",
                {"strict": False, "standard": False, "lenient": False},
            )
            reason = str(row["reason"])
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        diagnostics = {
            str(key): int(value)
            for key, value in row.get("matcher_diagnostics", {}).items()
        }
        if any(value > 0 for value in diagnostics.values()):
            matcher_diagnostic_attempts += 1
        for key, value in diagnostics.items():
            matcher_diagnostic_totals[key] = (
                matcher_diagnostic_totals.get(key, 0) + value
            )
        digest = output.append(row)
        seed = deriver.derive(
            training_seed=int(generated["training_seed"]),
            sampling_seed=int(generated["sampling_seed"]),
            attempt_id=str(generated["attempt_id"]),
            stage=ledger_stage,
        )
        status = (
            AttemptStatus.SUCCEEDED
            if row["status"] == "succeeded"
            else _failure_status(str(row["reason"]))
        )
        attempts.append(
            AttemptRecord(
                attempt_id=str(generated["attempt_id"]),
                method=str(generated["method"]),
                training_seed=int(generated["training_seed"]),
                sampling_seed=int(generated["sampling_seed"]),
                stage=ledger_stage,
                status=status,
                reason="" if status.success else str(row["reason"]),
                artifact_hash=digest,
                seed=seed,
                calls={
                    "mlip_relax": int(
                        config.stage == "relaxed"
                        and row["status"] == "succeeded"
                        and not row.get("cache_hit", False)
                    ),
                    "mlip_single_point": int(
                        config.stage != "relaxed"
                        and row["status"] == "succeeded"
                        and not row.get("cache_hit", False)
                    ),
                },
                walltime_s=float(row.get("mlip_walltime_s", 0.0)),
                metadata={
                    "ordinal": row.get("ordinal"),
                    "pair_id": row.get("pair_id"),
                    "subset_hash": subset_hash,
                },
            )
        )
        succeeded += int(row["status"] == "succeeded")
        sun0 += int(row["mlip_sun_at_0p0"])
        sun1 += int(row["mlip_sun_at_0p1"])
        novel_unique += int(row["novel_unique_standard"])

    denominator = len(generation)
    summary = {
        "schema": "wqcodiff_mlip_sun_summary_v1",
        "protocol_name": protocol.name,
        "protocol_sha256": protocol.sha256,
        "method": sorted({str(record["method"]) for record in generation}),
        "denominator": "all_submitted_attempts",
        "attempts": denominator,
        "succeeded": succeeded,
        "stage": config.stage,
        "evaluator": config.evaluator,
        "reference": hull_payload["hull_sha256"],
        "matcher": matcher_contract_hash(),
        "tolerance": {"mlip_sun_at_0p0": 0.0, "mlip_sun_at_0p1": 0.1},
        "subset_hash": subset_hash,
        "selection_rule": (
            "all_attempts"
            if config.subset_size is None
            else f"lowest_sha256_attempt_id_{config.subset_size}"
        ),
        "mlip_sun_at_0p0": sun0 / denominator,
        "mlip_sun_at_0p1": sun1 / denominator,
        "novel_unique_standard": novel_unique / denominator,
        "failure_rate": (denominator - succeeded) / denominator,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "matcher_diagnostics": {
            "attempts_with_timeout_or_error": matcher_diagnostic_attempts,
            "attempt_rate": matcher_diagnostic_attempts / denominator,
            "event_counts": dict(sorted(matcher_diagnostic_totals.items())),
            "policy": "conservative_duplicate_or_non_novel_never_improves_sun",
        },
        "elapsed_s": time.monotonic() - started_all,
        "output_jsonl": str(Path(config.output_jsonl).resolve()),
    }
    write_json_exclusive(
        Path(config.output_jsonl).with_suffix(".summary.json"), summary
    )
    return summary
