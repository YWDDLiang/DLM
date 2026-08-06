"""Command-line entry point for the registered WQ co-diffusion workflow."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Sequence

from .bridge import ChartCatalog, ChartSpec, TargetStratumBridge
from .contracts import AttemptLedger, SeedDeriver
from .dataset import audit_wq_dataset, build_hash_fixed_subset, preprocess_mp20_csv
from .formal import audit_pyxtal_chart_catalog, run_synthetic_transition_audit
from .kernel import TopologyEventKernel
from .protocol import load_protocol
from .state import OrbitState, StratifiedState


DEFAULT_PROTOCOL = Path("configs/experiments/wyckoff_codiffusion/protocol_v3.yaml")
DEFAULT_REGISTRY = Path(
    "configs/experiments/wyckoff_codiffusion/experiment_registry_v1.yaml"
)
MODEL_VARIANTS = (
    "B-ATOM-JOINT",
    "B-WQ-AR",
    "B-WQ-D3PM",
    "B-WQ-DLM-MONO",
    "B-WQ-JOINT-NOREV",
    "B-WQ-DISC-ONCE",
    "M-WQ-STRAT-CONF",
    "M-WQ-STRAT-GEO",
)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


class _SyntheticCatalog(ChartCatalog):
    def __init__(self) -> None:
        self.specs = {
            (1, 0): ChartSpec(1, 0, "a", 1, 3),
            (1, 1): ChartSpec(1, 1, "b", 2, 1),
            (1, 2): ChartSpec(1, 2, "c", 4, 0),
        }

    def get(self, space_group: int, wyckoff_type: int) -> ChartSpec:
        return self.specs[(space_group, wyckoff_type)]

    def types(self, space_group: int) -> tuple[int, ...]:
        return tuple(sorted(value for group, value in self.specs if group == space_group))


def _synthetic_state() -> StratifiedState:
    return StratifiedState(
        space_group=1,
        lattice_system="triclinic",
        lattice_chart=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        orbits=(
            OrbitState("o0", 0, 6, 1, 3, (0.1, 0.2, 0.3)),
            OrbitState("o1", 1, 8, 2, 1, (0.4,)),
        ),
    )


def command_protocol_check(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    _print(
        {
            "ok": True,
            "path": str(protocol.path),
            "name": protocol.name,
            "schema_version": protocol.schema_version,
            "sha256": protocol.sha256,
        }
    )
    return 0


def command_experiment_plan(args: argparse.Namespace) -> int:
    from .registry import load_experiment_registry, materialize_day7_plan

    registry = load_experiment_registry(args.registry)
    result = materialize_day7_plan(
        registry,
        run_id=args.run_id,
        source_bundle_sha256=args.source_bundle_sha256,
        output=args.output,
    )
    _print(
        {
            "ok": True,
            "registry_sha256": registry.sha256,
            "protocol_sha256": registry.protocol.sha256,
            "source_bundle_sha256": args.source_bundle_sha256,
            "phase_summary": result["phase_summary"],
            "jobs": len(result["jobs"]),
            "output": str(args.output),
        }
    )
    return 0


def command_week2_training_plan(args: argparse.Namespace) -> int:
    from .registry import load_experiment_registry, materialize_week2_training_plan

    registry = load_experiment_registry(args.registry)
    result = materialize_week2_training_plan(
        registry,
        run_id=args.run_id,
        discrete_engine=args.discrete_engine,
        source_bundle_sha256=args.source_bundle_sha256,
        dataset_paths=(
            None if not args.dataset else tuple(str(path) for path in args.dataset)
        ),
        output=args.output,
    )
    _print(
        {
            "ok": True,
            "registry_sha256": registry.sha256,
            "protocol_sha256": registry.protocol.sha256,
            "discrete_engine": result["discrete_engine_from_day7"],
            "summary": result["summary"],
            "output": str(args.output),
        }
    )
    return 0


def command_week2_sampling_plan(args: argparse.Namespace) -> int:
    from .registry import load_experiment_registry, materialize_week2_sampling_plan

    registry = load_experiment_registry(args.registry)
    result = materialize_week2_sampling_plan(
        registry,
        run_id=args.run_id,
        training_plan=args.training_plan,
        revision_lock=args.revision_lock,
        project_root=args.project_root,
        output=args.output,
    )
    _print(
        {
            "ok": True,
            "registry_sha256": registry.sha256,
            "protocol_sha256": registry.protocol.sha256,
            "configuration_count": result["configuration_count"],
            "phase_summary": result["phase_summary"],
            "output": str(args.output),
        }
    )
    return 0


def command_week2_champion_lock(args: argparse.Namespace) -> int:
    from .screening import (
        freeze_week2_champion,
        parse_named_path,
    )

    result = freeze_week2_champion(
        sampling_plan_path=args.sampling_plan,
        evaluation_paths=parse_named_path(args.evaluation),
        output=args.output,
    )
    _print(
        {
            "ok": True,
            "selected_disc_once_configuration": result[
                "selected_disc_once_configuration"
            ],
            "selected_champion": result["selected_champion"][
                "configuration_id"
            ],
            "output": str(args.output),
        }
    )
    return 0


def command_preprocess(args: argparse.Namespace) -> int:
    load_protocol(args.protocol)
    summary = preprocess_mp20_csv(
        csv_path=args.csv,
        split=args.split,
        output_path=args.output,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        limit=args.limit,
    )
    payload = {
        "schema": "wqcodiff_preprocess_summary_v1",
        "ok": summary.rows_written == summary.rows_seen,
        **summary.to_dict(),
    }
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    payload["summary_path"] = str(summary_path)
    _print(payload)
    # Individual codec failures are registered observations, not job failures;
    # the combined P1 audit enforces the 95% coverage gate.
    return 0 if payload["ok"] else 4


def command_dataset_audit(args: argparse.Namespace) -> int:
    load_protocol(args.protocol)
    mapping: dict[str, list[str]] = {}
    for item in args.split:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise SystemExit("--split must be NAME=PATH and may be repeated")
        mapping.setdefault(name, []).append(path)
    result = audit_wq_dataset(
        mapping,
        expected_total=args.expected_total,
        allow_nonpaper_counts=args.allow_nonpaper_counts,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    _print(result)
    return 0 if result["ok"] else 5


def command_dataset_subset(args: argparse.Namespace) -> int:
    load_protocol(args.protocol)
    result = build_hash_fixed_subset(
        [str(path) for path in args.dataset],
        output_path=args.output,
        fraction=args.fraction,
        count=args.count,
        salt=args.salt,
    )
    _print(result)
    return 0


def command_attempt_audit(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    deriver = SeedDeriver(protocol.name, args.experiment_id)
    expected = None
    if args.expected_ids:
        expected = [line.strip() for line in Path(args.expected_ids).read_text().splitlines() if line.strip()]
    result = AttemptLedger(args.ledger).audit(
        seed_deriver=deriver,
        terminal_stage=args.terminal_stage,
        expected_attempt_ids=expected,
    )
    payload = dataclasses.asdict(result)
    payload["ok"] = result.ok
    _print(payload)
    return 0 if result.ok else 6


def command_formal_audit(args: argparse.Namespace) -> int:
    load_protocol(args.protocol)
    catalog = _SyntheticCatalog()
    bridge = TargetStratumBridge(catalog)
    kernel = TopologyEventKernel(catalog=catalog, bridge=bridge, species=(6, 8, 14))
    result = run_synthetic_transition_audit(
        kernel,
        _synthetic_state(),
        transitions=args.transitions,
        seed=args.seed,
    )
    payload = dataclasses.asdict(result)
    payload["passed"] = result.passed
    payload["schema"] = "wqcodiff_formal_audit_v1"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    _print(payload)
    return 0 if result.passed else 7


def command_chart_audit(args: argparse.Namespace) -> int:
    load_protocol(args.protocol)
    result = audit_pyxtal_chart_catalog()
    payload = dataclasses.asdict(result)
    payload["passed"] = result.passed
    payload["schema"] = "wqcodiff_pyxtal_chart_audit_v1"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    _print(payload)
    return 0 if result.passed else 7


def command_model_smoke(args: argparse.Namespace) -> int:
    load_protocol(args.protocol)
    import torch

    from .model import WQCoDenoiser, WQTensorBatch, WQVariant

    device = torch.device(args.device)
    model = WQCoDenoiser().to(device)
    batch = WQTensorBatch(
        atom_species=torch.tensor([6, 6, 8], device=device),
        frac_coords=torch.tensor(
            [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.0, 0.0, 0.0]],
            device=device,
        ),
        lattices=torch.eye(3, device=device)[None].repeat(2, 1, 1) * 5.0,
        atom_batch=torch.tensor([0, 0, 1], device=device),
        atom_to_orbit=torch.tensor([0, 0, 1], device=device),
        orbit_species=torch.tensor([6, 8], device=device),
        orbit_wyckoff=torch.tensor([1, 2], device=device),
        orbit_batch=torch.tensor([0, 1], device=device),
        space_group=torch.tensor([1, 2], device=device),
        time=torch.tensor([0.7, 0.4], device=device),
        geometry_evidence=torch.zeros((2, 6), device=device),
    )
    with torch.no_grad():
        output = model(batch, variant=WQVariant(args.variant))
        prior = model.forward_prior(
            torch.ones(2, device=device),
            torch.zeros(2, dtype=torch.long, device=device),
        )
    _print(
        {
            "ok": True,
            "device": str(device),
            "parameter_count": model.parameter_count(),
            "space_group_logits": list(output.space_group_logits.shape),
            "orbit_logits": list(output.species_logits.shape),
            "coordinate_score": list(output.atom_coordinate_score.shape),
            "prior_space_group_logits": list(prior.space_group_logits.shape),
            "prior_first_orbit_logits": list(prior.first_wyckoff_logits.shape),
        }
    )
    return 0


def command_train(args: argparse.Namespace) -> int:
    from .model import WQVariant
    from .training import TrainingConfig, train

    result = train(
        TrainingConfig(
            dataset_paths=tuple(str(path) for path in args.dataset),
            output_dir=str(args.output),
            variant=WQVariant(args.variant),
            training_seed=args.training_seed,
            source_bundle_sha256=args.source_bundle_sha256,
            updates=args.updates,
            microbatch_size=args.microbatch_size,
            checkpoint_interval=args.checkpoint_interval,
            log_interval=args.log_interval,
            device=args.device,
            allow_nonpaper_updates=args.allow_nonpaper_updates,
            resume_checkpoint=str(args.resume) if args.resume else None,
            shared_checkpoint=(
                str(args.shared_checkpoint) if args.shared_checkpoint else None
            ),
            stop_after_shared=args.stop_after_shared,
            stop_after_update=args.stop_after_update,
        ),
        protocol_path=args.protocol,
    )
    _print(result)
    return 0


def command_sample(args: argparse.Namespace) -> int:
    from .model import WQVariant
    from .sampling import SamplingConfig, sample

    result = sample(
        SamplingConfig(
            checkpoint=str(args.checkpoint),
            output_jsonl=str(args.output),
            attempt_ledger=str(args.ledger),
            experiment_id=args.experiment_id,
            pairing_id=args.pairing_id,
            runtime_source_bundle_sha256=args.runtime_source_bundle_sha256,
            variant=WQVariant(args.variant),
            training_seed=args.training_seed,
            sampling_seed=args.sampling_seed,
            attempts=args.attempts,
            start_ordinal=args.start_ordinal,
            backbone_calls=args.backbone_calls,
            revision_control=args.revision_control,
            revision_threshold=args.revision_threshold,
            revision_lock=str(args.revision_lock) if args.revision_lock else None,
            temperature=args.temperature,
            disc_once_tau=args.disc_once_tau,
            inference_batch_size=args.inference_batch_size,
            device=args.device,
        ),
        protocol_path=args.protocol,
    )
    _print(result)
    return 0 if result["ok"] else 8


def _recovery_config_kwargs(
    args: argparse.Namespace,
    *,
    variant: Any,
) -> dict[str, Any]:
    """Translate the recovery CLI namespace without dropping provenance fields."""

    return {
        "checkpoint": str(args.checkpoint),
        "dataset_paths": tuple(str(path) for path in args.dataset),
        "output_jsonl": str(args.output),
        "attempt_ledger": str(args.ledger),
        "experiment_id": args.experiment_id,
        "pairing_id": args.pairing_id,
        "runtime_source_bundle_sha256": args.runtime_source_bundle_sha256,
        "variant": variant,
        "training_seed": args.training_seed,
        "corruption_seed": args.corruption_seed,
        "structures": args.structures,
        "corruption_level": args.corruption_level,
        "operator": args.operator,
        "geometry_condition": args.geometry_condition,
        "schedule": args.schedule,
        "control": args.control,
        "calls": args.calls,
        "revision_threshold": args.revision_threshold,
        "temperature": args.temperature,
        "inference_batch_size": args.inference_batch_size,
        "runtime_workers": args.runtime_workers,
        "device": args.device,
    }


def command_recovery(args: argparse.Namespace) -> int:
    from .model import WQVariant
    from .recovery import RecoveryConfig, run_recovery_cell

    result = run_recovery_cell(
        RecoveryConfig(
            **_recovery_config_kwargs(args, variant=WQVariant(args.variant))
        ),
        protocol_path=args.protocol,
    )
    _print(result)
    return 0 if result["ok"] else 9


def command_recovery_aggregate(args: argparse.Namespace) -> int:
    from .recovery_aggregate import aggregate_recovery

    result = aggregate_recovery(
        [str(path) for path in args.input],
        output_path=args.output,
        primary_geometry=args.primary_geometry,
        primary_schedule=args.primary_schedule,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    _print(
        {
            "dlm_promoted": result["dlm_promoted"],
            "required_claim_action": result["required_claim_action"],
            "gates": result["gates"],
            "output": str(args.output),
        }
    )
    # A negative scientific gate is a registered downgrade decision, not an
    # infrastructure/job failure.
    return 0


def command_generation_pool(args: argparse.Namespace) -> int:
    from .pooling import (
        GenerationPoolConfig,
        parse_seed_count,
        pool_generation_artifacts,
    )

    result = pool_generation_artifacts(
        GenerationPoolConfig(
            inputs=tuple(str(path) for path in args.input),
            output_jsonl=str(args.output),
            manifest_json=str(args.manifest),
            expected_method=args.expected_method,
            expected_total=args.expected_total,
            expected_training_seed_counts=tuple(
                parse_seed_count(value) for value in args.expected_training_seed_count
            ),
        )
    )
    _print(result)
    return 0


def command_revision_calibrate(args: argparse.Namespace) -> int:
    from .revision import calibrate_revision_threshold_from_recovery

    protocol = load_protocol(args.protocol)
    result = calibrate_revision_threshold_from_recovery(
        [str(path) for path in args.input],
        output=args.output,
        protocol_name=protocol.name,
        protocol_sha256=protocol.sha256,
    )
    _print(result)
    return 0


def command_novelty_reference(args: argparse.Namespace) -> int:
    from .novelty_reference import build_novelty_reference

    result = build_novelty_reference(
        train_csv=args.train_csv,
        output_jsonl=args.output,
        limit=args.limit,
    )
    _print(result)
    return 0 if result["gate_passed"] else 11


def command_reference_evaluate(args: argparse.Namespace) -> int:
    from .reference_eval import ReferenceEvaluationConfig, evaluate_references

    result = evaluate_references(
        ReferenceEvaluationConfig(
            csv_splits=tuple(args.csv),
            output_jsonl=str(args.output),
            evaluator=args.evaluator,
            asset_lock=str(args.asset_lock),
            model_root=str(args.model_root),
            stage=args.stage,
            device=args.device,
            queue_path=str(args.queue) if args.queue else None,
            limit=args.limit,
        )
    )
    _print(result)
    return 0 if result["complete"] else 12


def command_hull_step(args: argparse.Namespace) -> int:
    from .hull import build_hull_closure_step

    result = build_hull_closure_step(
        [str(path) for path in args.energy],
        output_path=args.output,
        round_index=args.round_index,
        expected_reference_count=args.expected_reference_count,
        allow_nonpaper_reference_count=args.allow_nonpaper_reference_count,
    )
    _print(
        {
            "evaluator": result["evaluator"],
            "round_index": result["round_index"],
            "pending_count": result["pending_count"],
            "closed": result["closed"],
            "gate_passed": result["gate_passed"],
            "hull_sha256": result["hull_sha256"],
        }
    )
    return 0 if result["gate_passed"] or result["pending_count"] else 13


def command_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import EvaluationConfig, evaluate

    result = evaluate(
        EvaluationConfig(
            input_jsonl=str(args.input),
            output_jsonl=str(args.output),
            energy_cache_jsonl=str(args.energy_cache),
            attempt_ledger=str(args.ledger),
            experiment_id=args.experiment_id,
            evaluator=args.evaluator,
            stage=args.stage,
            asset_lock=str(args.asset_lock),
            model_root=str(args.model_root),
            hull_path=str(args.hull),
            novelty_reference=str(args.novelty_reference),
            device=args.device,
            subset_size=args.subset_size,
        ),
        protocol_path=args.protocol,
    )
    _print(result)
    return 0


def command_aggregate(args: argparse.Namespace) -> int:
    from .aggregate import FinalAggregateConfig, aggregate_final

    result = aggregate_final(
        FinalAggregateConfig(
            inputs=tuple(str(path) for path in args.input),
            output=str(args.output),
            champion=args.champion,
            final_method=args.final_method,
            train_data=tuple(str(path) for path in args.train_data),
            usage_inputs=tuple(str(path) for path in args.usage),
            headline_stage=args.headline_stage,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
            allow_nonpaper_counts=args.allow_nonpaper_counts,
        )
    )
    _print(
        {
            "oral_eligible": result["oral_eligible"],
            "decision": result["decision"],
            "gates": result["gates"],
            "output": str(args.output),
        }
    )
    return 0


def command_refiner_lock(args: argparse.Namespace) -> int:
    from .refine import freeze_common_refiner

    result = freeze_common_refiner(
        checkpoint=args.checkpoint,
        output=args.output,
        protocol_path=args.protocol,
        frozen_day=args.frozen_day,
    )
    _print(result)
    return 0


def command_refine(args: argparse.Namespace) -> int:
    from .refine import RefineConfig, refine

    result = refine(
        RefineConfig(
            input_jsonl=str(args.input),
            output_jsonl=str(args.output),
            attempt_ledger=str(args.ledger),
            experiment_id=args.experiment_id,
            refiner_lock=str(args.refiner_lock),
            device=args.device,
        ),
        protocol_path=args.protocol,
    )
    _print(result)
    return 0 if result["ok"] else 14


def command_audit(args: argparse.Namespace) -> int:
    from .audit import WorkflowAuditConfig, audit_workflow

    ledgers: list[tuple[str, str]] = []
    for value in args.ledger:
        experiment_id, separator, path = value.partition("=")
        if not separator or not experiment_id or not path:
            raise SystemExit("--ledger must be EXPERIMENT_ID=PATH")
        ledgers.append((experiment_id, path))
    result = audit_workflow(
        WorkflowAuditConfig(
            ledgers=tuple(ledgers),
            artifacts=tuple(str(path) for path in args.artifact),
            output=str(args.output),
            formal_reports=tuple(str(path) for path in args.formal_report),
            dataset_reports=tuple(str(path) for path in args.dataset_report),
            final_aggregate=str(args.final_aggregate) if args.final_aggregate else None,
            source_manifest=str(args.source_manifest) if args.source_manifest else None,
            asset_lock=str(args.asset_lock) if args.asset_lock else None,
            model_root=str(args.model_root) if args.model_root else None,
            revision_lock=str(args.revision_lock) if args.revision_lock else None,
        ),
        protocol_path=args.protocol,
    )
    _print(
        {
            "integrity_passed": result["integrity_passed"],
            "main_claim_eligible": result["main_claim_eligible"],
            "output": str(args.output),
        }
    )
    return 0 if result["integrity_passed"] else 15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m crystal_dlm.wqcodiff")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("protocol-check")
    check.set_defaults(function=command_protocol_check)

    experiment_plan = subparsers.add_parser("experiment-plan")
    experiment_plan.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    experiment_plan.add_argument("--run-id", required=True)
    experiment_plan.add_argument("--source-bundle-sha256", required=True)
    experiment_plan.add_argument("--output", type=Path, required=True)
    experiment_plan.set_defaults(function=command_experiment_plan)

    week2_plan = subparsers.add_parser("week2-training-plan")
    week2_plan.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    week2_plan.add_argument("--run-id", required=True)
    week2_plan.add_argument(
        "--discrete-engine",
        choices=("B-WQ-AR", "B-WQ-D3PM", "B-WQ-DLM-MONO"),
        required=True,
    )
    week2_plan.add_argument("--source-bundle-sha256", required=True)
    week2_plan.add_argument("--dataset", type=Path, action="append")
    week2_plan.add_argument("--output", type=Path, required=True)
    week2_plan.set_defaults(function=command_week2_training_plan)

    week2_sampling = subparsers.add_parser("week2-sampling-plan")
    week2_sampling.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    week2_sampling.add_argument("--run-id", required=True)
    week2_sampling.add_argument("--training-plan", type=Path, required=True)
    week2_sampling.add_argument("--revision-lock", type=Path, required=True)
    week2_sampling.add_argument("--project-root", type=Path)
    week2_sampling.add_argument("--output", type=Path, required=True)
    week2_sampling.set_defaults(function=command_week2_sampling_plan)

    week2_champion = subparsers.add_parser("week2-champion-lock")
    week2_champion.add_argument("--sampling-plan", type=Path, required=True)
    week2_champion.add_argument(
        "--evaluation",
        action="append",
        required=True,
        help="CONFIG=PATH; provide each of the eight frozen raw MatterSim artifacts",
    )
    week2_champion.add_argument("--output", type=Path, required=True)
    week2_champion.set_defaults(function=command_week2_champion_lock)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--csv", type=Path, required=True)
    preprocess.add_argument("--split", required=True)
    preprocess.add_argument("--output", type=Path, required=True)
    preprocess.add_argument("--shard-index", type=int, default=0)
    preprocess.add_argument("--shard-count", type=int, default=1)
    preprocess.add_argument("--limit", type=int)
    preprocess.add_argument("--summary", type=Path)
    preprocess.set_defaults(function=command_preprocess)

    dataset_audit = subparsers.add_parser("dataset-audit")
    dataset_audit.add_argument("--split", action="append", required=True)
    dataset_audit.add_argument("--output", type=Path)
    dataset_audit.add_argument("--expected-total", type=int, default=45_229)
    dataset_audit.add_argument("--allow-nonpaper-counts", action="store_true")
    dataset_audit.set_defaults(function=command_dataset_audit)

    dataset_subset = subparsers.add_parser("dataset-subset")
    dataset_subset.add_argument("--dataset", type=Path, action="append", required=True)
    dataset_subset.add_argument("--output", type=Path, required=True)
    subset_size = dataset_subset.add_mutually_exclusive_group(required=True)
    subset_size.add_argument("--fraction", type=float)
    subset_size.add_argument("--count", type=int)
    dataset_subset.add_argument("--salt", default="wqcodiff-hash-fixed-v1")
    dataset_subset.set_defaults(function=command_dataset_subset)

    attempt_audit = subparsers.add_parser("attempt-audit")
    attempt_audit.add_argument("--ledger", type=Path, required=True)
    attempt_audit.add_argument("--experiment-id", required=True)
    attempt_audit.add_argument("--terminal-stage", default="sun")
    attempt_audit.add_argument("--expected-ids", type=Path)
    attempt_audit.set_defaults(function=command_attempt_audit)

    formal = subparsers.add_parser("formal-audit")
    formal.add_argument("--transitions", type=int, default=1_000_000)
    formal.add_argument("--seed", type=int, default=0)
    formal.add_argument("--output", type=Path)
    formal.set_defaults(function=command_formal_audit)

    chart_audit = subparsers.add_parser("chart-audit")
    chart_audit.add_argument("--output", type=Path)
    chart_audit.set_defaults(function=command_chart_audit)

    smoke = subparsers.add_parser("model-smoke")
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--variant", choices=MODEL_VARIANTS, default="M-WQ-STRAT-GEO")
    smoke.set_defaults(function=command_model_smoke)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", type=Path, action="append", required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--variant", choices=MODEL_VARIANTS, required=True)
    train_parser.add_argument("--training-seed", type=int, choices=(11, 23, 47), required=True)
    train_parser.add_argument("--source-bundle-sha256", required=True)
    train_parser.add_argument("--updates", type=int, default=100_000)
    train_parser.add_argument("--microbatch-size", type=int, choices=(64, 128), default=128)
    train_parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    train_parser.add_argument("--log-interval", type=int, default=100)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--allow-nonpaper-updates", action="store_true")
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument(
        "--shared-checkpoint",
        type=Path,
        help="Fork method-specific training from the immutable 60% shared boundary",
    )
    train_parser.add_argument(
        "--stop-after-shared",
        action="store_true",
        help="Produce the reusable shared-stage checkpoint and stop at 60%",
    )
    train_parser.add_argument(
        "--stop-after-update",
        type=int,
        help="Stop on the registered full-run schedule (e.g. 85000 for the 25k screen)",
    )
    train_parser.set_defaults(function=command_train)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.add_argument("--output", type=Path, required=True)
    sample_parser.add_argument("--ledger", type=Path, required=True)
    sample_parser.add_argument("--experiment-id", required=True)
    sample_parser.add_argument(
        "--pairing-id",
        help="Shared namespace across matched methods/controls; defaults to experiment-id",
    )
    sample_parser.add_argument("--variant", choices=MODEL_VARIANTS, required=True)
    sample_parser.add_argument("--training-seed", type=int, choices=(11, 23, 47), required=True)
    sample_parser.add_argument("--sampling-seed", type=int, required=True)
    sample_parser.add_argument("--attempts", type=int, required=True)
    sample_parser.add_argument("--start-ordinal", type=int, default=0)
    sample_parser.add_argument("--backbone-calls", type=int, choices=(16, 32, 64, 128), default=64)
    sample_parser.add_argument(
        "--revision-control",
        choices=(
            "auto",
            "none",
            "confidence",
            "geometry",
            "random-count",
            "shuffled-geometry",
            "extra-call",
        ),
        default="auto",
    )
    sample_parser.add_argument(
        "--revision-threshold", type=float, choices=(0.5, 0.6, 0.7, 0.8, 0.9), default=0.7
    )
    sample_parser.add_argument(
        "--revision-lock",
        type=Path,
        help="Immutable Day-7 threshold lock; required for paper-eligible revision sampling",
    )
    sample_parser.add_argument("--temperature", type=float, default=1.0)
    sample_parser.add_argument(
        "--disc-once-tau", type=float, choices=(0.25, 0.5, 0.75, 1.0), default=0.5
    )
    sample_parser.add_argument(
        "--inference-batch-size", type=int, choices=(16, 32, 64, 128), default=64
    )
    sample_parser.add_argument("--device", default="cuda")
    sample_parser.set_defaults(function=command_sample)

    generation_pool = subparsers.add_parser("generation-pool")
    generation_pool.add_argument("--input", type=Path, action="append", required=True)
    generation_pool.add_argument("--output", type=Path, required=True)
    generation_pool.add_argument("--manifest", type=Path, required=True)
    generation_pool.add_argument("--expected-method")
    generation_pool.add_argument("--expected-total", type=int)
    generation_pool.add_argument(
        "--expected-training-seed-count",
        action="append",
        default=[],
        metavar="SEED=COUNT",
    )
    generation_pool.set_defaults(function=command_generation_pool)

    recovery_parser = subparsers.add_parser("recovery")
    recovery_parser.add_argument("--checkpoint", type=Path, required=True)
    recovery_parser.add_argument("--dataset", type=Path, action="append", required=True)
    recovery_parser.add_argument("--output", type=Path, required=True)
    recovery_parser.add_argument("--ledger", type=Path, required=True)
    recovery_parser.add_argument("--experiment-id", required=True)
    recovery_parser.add_argument(
        "--pairing-id",
        help="Shared namespace across matched recovery methods/controls",
    )
    recovery_parser.add_argument("--runtime-source-bundle-sha256", required=True)
    recovery_parser.add_argument("--variant", choices=MODEL_VARIANTS, required=True)
    recovery_parser.add_argument("--training-seed", type=int, choices=(11, 23, 47), required=True)
    recovery_parser.add_argument("--corruption-seed", type=int, required=True)
    recovery_parser.add_argument("--structures", type=int, default=4096)
    recovery_parser.add_argument(
        "--corruption-level", type=float, choices=(0.3, 0.5, 0.7, 0.9), required=True
    )
    recovery_parser.add_argument(
        "--operator",
        choices=(
            "none",
            "deletion",
            "false-insertion",
            "wrong-wyckoff",
            "wrong-species",
            "joint",
        ),
        required=True,
    )
    recovery_parser.add_argument(
        "--control",
        choices=("none", "random-count", "shuffled-geometry", "extra-call"),
        default="none",
    )
    recovery_parser.add_argument(
        "--geometry-condition", choices=("clean", "noisy", "shuffled", "absent"), required=True
    )
    recovery_parser.add_argument(
        "--schedule",
        choices=(
            "fixed",
            "discrete-first",
            "continuous-first",
            "confidence-adaptive",
            "geometry-adaptive",
        ),
        required=True,
    )
    recovery_parser.add_argument("--calls", type=int, choices=(16, 32, 64, 128), default=16)
    recovery_parser.add_argument(
        "--revision-threshold", type=float, choices=(0.5, 0.6, 0.7, 0.8, 0.9), default=0.7
    )
    recovery_parser.add_argument("--temperature", type=float, default=1.0)
    recovery_parser.add_argument(
        "--inference-batch-size", type=int, choices=(16, 32, 64, 128), default=64
    )
    recovery_parser.add_argument(
        "--runtime-workers", type=int, choices=(1, 2, 4, 8, 12), default=12
    )
    recovery_parser.add_argument("--device", default="cuda")
    recovery_parser.set_defaults(function=command_recovery)

    recovery_aggregate = subparsers.add_parser("recovery-aggregate")
    recovery_aggregate.add_argument("--input", type=Path, action="append", required=True)
    recovery_aggregate.add_argument("--output", type=Path, required=True)
    recovery_aggregate.add_argument(
        "--primary-geometry", choices=("clean", "noisy", "shuffled", "absent"), default="noisy"
    )
    recovery_aggregate.add_argument(
        "--primary-schedule",
        choices=(
            "fixed",
            "discrete-first",
            "continuous-first",
            "confidence-adaptive",
            "geometry-adaptive",
        ),
        default="fixed",
    )
    recovery_aggregate.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    recovery_aggregate.add_argument("--bootstrap-seed", type=int, default=20260710)
    recovery_aggregate.set_defaults(function=command_recovery_aggregate)

    revision_calibrate = subparsers.add_parser("revision-calibrate")
    revision_calibrate.add_argument("--input", type=Path, action="append", required=True)
    revision_calibrate.add_argument("--output", type=Path, required=True)
    revision_calibrate.set_defaults(function=command_revision_calibrate)

    novelty_reference = subparsers.add_parser("novelty-reference")
    novelty_reference.add_argument("--train-csv", type=Path, required=True)
    novelty_reference.add_argument("--output", type=Path, required=True)
    novelty_reference.add_argument("--limit", type=int)
    novelty_reference.set_defaults(function=command_novelty_reference)

    reference_evaluate = subparsers.add_parser("reference-evaluate")
    reference_evaluate.add_argument("--csv", action="append", required=True, help="SPLIT=PATH")
    reference_evaluate.add_argument("--output", type=Path, required=True)
    reference_evaluate.add_argument("--evaluator", choices=("chgnet", "mattersim", "mace"), required=True)
    reference_evaluate.add_argument("--asset-lock", type=Path, required=True)
    reference_evaluate.add_argument("--model-root", type=Path, required=True)
    reference_evaluate.add_argument("--stage", choices=("raw", "relaxed"), required=True)
    reference_evaluate.add_argument("--queue", type=Path)
    reference_evaluate.add_argument("--device", default="cuda")
    reference_evaluate.add_argument("--limit", type=int)
    reference_evaluate.set_defaults(function=command_reference_evaluate)

    hull_step = subparsers.add_parser("hull-step")
    hull_step.add_argument("--energy", type=Path, action="append", required=True)
    hull_step.add_argument("--output", type=Path, required=True)
    hull_step.add_argument("--round-index", type=int, choices=(0, 1, 2, 3), required=True)
    hull_step.add_argument("--expected-reference-count", type=int, default=45_229)
    hull_step.add_argument("--allow-nonpaper-reference-count", action="store_true")
    hull_step.set_defaults(function=command_hull_step)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--energy-cache", type=Path, required=True)
    evaluate_parser.add_argument("--ledger", type=Path, required=True)
    evaluate_parser.add_argument("--experiment-id", required=True)
    evaluate_parser.add_argument("--evaluator", choices=("chgnet", "mattersim", "mace"), required=True)
    evaluate_parser.add_argument("--stage", choices=("raw", "common-refiner", "relaxed"), required=True)
    evaluate_parser.add_argument("--asset-lock", type=Path, required=True)
    evaluate_parser.add_argument("--model-root", type=Path, required=True)
    evaluate_parser.add_argument("--hull", type=Path, required=True)
    evaluate_parser.add_argument("--novelty-reference", type=Path, required=True)
    evaluate_parser.add_argument("--subset-size", type=int)
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.set_defaults(function=command_evaluate)

    refiner_lock_parser = subparsers.add_parser("refiner-lock")
    refiner_lock_parser.add_argument("--checkpoint", type=Path, required=True)
    refiner_lock_parser.add_argument("--output", type=Path, required=True)
    refiner_lock_parser.add_argument("--frozen-day", type=int, default=14)
    refiner_lock_parser.set_defaults(function=command_refiner_lock)

    refine_parser = subparsers.add_parser("refine")
    refine_parser.add_argument("--input", type=Path, required=True)
    refine_parser.add_argument("--output", type=Path, required=True)
    refine_parser.add_argument("--ledger", type=Path, required=True)
    refine_parser.add_argument("--experiment-id", required=True)
    refine_parser.add_argument("--refiner-lock", type=Path, required=True)
    refine_parser.add_argument("--device", default="cuda")
    refine_parser.set_defaults(function=command_refine)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser.add_argument("--champion", required=True)
    aggregate_parser.add_argument("--final-method", required=True)
    aggregate_parser.add_argument("--train-data", type=Path, action="append", default=[])
    aggregate_parser.add_argument(
        "--usage",
        type=Path,
        action="append",
        default=[],
        help="Final sampling *.job_usage.json files with method/stage labels",
    )
    aggregate_parser.add_argument(
        "--headline-stage",
        choices=("raw", "common-refiner", "relaxed"),
        default="relaxed",
    )
    aggregate_parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    aggregate_parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    aggregate_parser.add_argument("--allow-nonpaper-counts", action="store_true")
    aggregate_parser.set_defaults(function=command_aggregate)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument(
        "--ledger",
        action="append",
        default=[],
        help="EXPERIMENT_ID=PATH; repeat for each immutable attempt ledger",
    )
    audit_parser.add_argument("--artifact", type=Path, action="append", default=[])
    audit_parser.add_argument("--formal-report", type=Path, action="append", default=[])
    audit_parser.add_argument("--dataset-report", type=Path, action="append", default=[])
    audit_parser.add_argument("--final-aggregate", type=Path)
    audit_parser.add_argument("--source-manifest", type=Path)
    audit_parser.add_argument("--asset-lock", type=Path)
    audit_parser.add_argument("--model-root", type=Path)
    audit_parser.add_argument("--revision-lock", type=Path)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.set_defaults(function=command_audit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.function(args))
