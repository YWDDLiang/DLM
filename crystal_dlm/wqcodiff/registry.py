"""Validate and materialize the frozen four-week experiment registry."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import RegisteredProtocol, load_protocol


REGISTRY_NAME = "wqcodiff_iclr_four_week_v1"
REGISTRY_SCHEMA = 1
DAY7_METHODS = {
    "B-WQ-AR",
    "B-WQ-D3PM",
    "B-WQ-DLM-MONO",
    "M-WQ-STRAT-CONF",
    "M-WQ-STRAT-GEO",
}
DAY7_DISCRETE_ENGINES = ("B-WQ-AR", "B-WQ-D3PM", "B-WQ-DLM-MONO")
WEEK2_TRAIN_DATASETS = tuple(
    f"data/wqcodiff/p1_v3/train.part-{index:03d}-of-008.jsonl"
    for index in range(8)
)
WEEK2_SAMPLING_PHASES = ("preflight", "development")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"experiment registry violation: {message}")


@dataclasses.dataclass(frozen=True, slots=True)
class ExperimentRegistry:
    path: Path
    sha256: str
    data: Mapping[str, Any]
    protocol: RegisteredProtocol


def _project_root(registry_path: Path) -> Path:
    for parent in registry_path.parents:
        if (parent / "crystal_dlm/wqcodiff").is_dir() and (parent / "configs").is_dir():
            return parent
    raise ValueError("cannot locate project root from experiment registry")


def validate_experiment_registry(data: Mapping[str, Any], protocol: RegisteredProtocol) -> None:
    header = data["registry"]
    _expect(header["name"] == REGISTRY_NAME, "wrong registry name")
    _expect(int(header["schema_version"]) == REGISTRY_SCHEMA, "wrong registry schema")
    execution = data["execution_contract"]
    _expect(execution["one_model_lane_per_gpu"] is True, "multi-model GPU lane enabled")
    _expect(int(execution["maximum_concurrent_lanes"]) == 4, "concurrency cap changed")
    _expect(execution["retry_attempts"] is False, "attempt retry enabled")

    week1 = data["week1"]
    _expect(int(week1["cumulative_gpu_hours_max"]) == 180, "Week-1 budget changed")
    _expect(set(week1["training"]["variants"]) == DAY7_METHODS, "Day-7 methods changed")
    _expect(
        (int(week1["training"]["target_updates"]), int(week1["training"]["shared_boundary_update"]))
        == (10_000, 6_000),
        "Day-7 training boundary changed",
    )
    calibration = week1["threshold_calibration"]
    _expect(
        tuple(float(value) for value in calibration["thresholds"])
        == (0.5, 0.6, 0.7, 0.8, 0.9),
        "threshold grid changed",
    )
    _expect(any(cell["operator"] == "none" for cell in calibration["cells"]), "clean calibration cell removed")
    primary = week1["day7_primary_falsification"]
    primary_jobs = (
        len(primary["methods"])
        * len(primary["corruption_levels"])
        * len(primary["operators"])
        * len(primary["corruption_seeds"])
    )
    _expect(primary_jobs == int(primary["expected_jobs"]) == 180, "primary job count changed")
    _expect(
        primary_jobs * int(primary["structures"]) == int(primary["expected_attempts"]),
        "primary attempt count changed",
    )
    _expect(
        set(primary["methods"]) == {"B-WQ-AR", "B-WQ-D3PM", "B-WQ-DLM-MONO"},
        "primary DLM falsification engines changed",
    )
    interventions = week1["day7_interventions"]
    intervention_jobs = (
        len(interventions["cells"])
        * len(interventions["corruption_levels"])
        * len(interventions["corruption_seeds"])
    )
    _expect(intervention_jobs == int(interventions["expected_jobs"]) == 72, "intervention job count changed")
    _expect(
        intervention_jobs * int(interventions["structures"])
        == int(interventions["expected_attempts"]),
        "intervention attempt count changed",
    )
    all_geometries = {primary["geometry"]} | {
        cell["geometry"] for cell in interventions["cells"]
    }
    all_schedules = {primary["schedule"]} | {
        cell["schedule"] for cell in interventions["cells"]
    }
    _expect(all_geometries == {"clean", "noisy", "shuffled", "absent"}, "geometry interventions incomplete")
    _expect(
        all_schedules
        == {"fixed", "discrete-first", "continuous-first", "confidence-adaptive", "geometry-adaptive"},
        "schedule interventions incomplete",
    )

    week2 = data["week2"]
    _expect(int(week2["cumulative_gpu_hours_max"]) == 750, "Week-2 budget changed")
    _expect(
        int(week2["screening_stop_update"]) - int(week2["shared_boundary_update"])
        == int(week2["method_specific_screen_updates"])
        == 25_000,
        "screening is no longer 25k method-specific updates on the 100k schedule",
    )
    _expect(int(week2["full_schedule_updates"]) == 100_000, "full schedule changed")
    _expect(len(week2["routes"]) == 5, "screening route count changed")

    week3 = data["week3"]
    _expect(int(week3["cumulative_gpu_hours_max"]) == 1250, "Week-3 budget changed")
    _expect(int(week3["freeze_day"]) == 21, "final freeze day changed")
    week4 = data["week4"]
    _expect(int(week4["total_gpu_hours_hard_max"]) == 2050, "hard GPU-hour cap changed")
    _expect(int(week4["reserved_gpu_hours_min"]) >= 800, "Week-4 reserve reduced")
    _expect(tuple(week4["train_seed_allocation"]) == (3334, 3333, 3333), "final allocation changed")
    _expect(
        tuple(week4["train_seed_allocation_order"]) == (11, 23, 47),
        "final allocation seed order changed",
    )
    _expect(int(week4["multi_mlip_common_attempts_per_method"]) == 6000, "multi-MLIP subset changed")
    _expect(int(week4["bootstrap_repetitions"]) == 10_000, "bootstrap count changed")

    protocol_data = protocol.data
    _expect(
        int(protocol_data["compute_funnel"]["usable_a800_gpu_hours_hard_ceiling"])
        == int(week4["total_gpu_hours_hard_max"]),
        "registry/protocol GPU-hour caps disagree",
    )
    _expect(
        int(protocol_data["training"]["shared_pretraining_updates"])
        == int(week2["shared_boundary_update"]),
        "registry/protocol shared boundaries disagree",
    )


def load_experiment_registry(path: str | Path) -> ExperimentRegistry:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load the experiment registry") from exc
    location = Path(path).resolve()
    raw = location.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, Mapping):
        raise ValueError("experiment registry root must be a mapping")
    root = _project_root(location)
    protocol = load_protocol(root / str(data["registry"]["protocol"]))
    validate_experiment_registry(data, protocol)
    return ExperimentRegistry(
        path=location,
        sha256=hashlib.sha256(raw).hexdigest(),
        data=data,
        protocol=protocol,
    )


def _slug(value: Any) -> str:
    return str(value).lower().replace(".", "p").replace("-", "_")


def _file_identity(path: str | Path) -> dict[str, Any]:
    location = Path(path).resolve()
    digest = hashlib.sha256()
    with location.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _recovery_job(
    *,
    phase: str,
    run_id: str,
    method: str,
    checkpoint: str,
    level: float,
    operator: str,
    geometry: str,
    schedule: str,
    control: str,
    seed: int,
    structures: int,
    calls: int,
    threshold: float,
    pairing_id: str,
    runtime_source_bundle_sha256: str,
) -> dict[str, Any]:
    cell = "-".join(
        _slug(value)
        for value in (method, level, operator, geometry, schedule, control, seed, threshold)
    )
    experiment_id = f"{run_id}-{phase}-{cell}"
    root = f"runs/{run_id}/outputs/{phase}/{cell}"
    argv = [
        "python",
        "-m",
        "crystal_dlm.wqcodiff",
        "recovery",
        "--checkpoint",
        checkpoint,
        "--dataset",
        "${DAY7_VAL_WQ}",
        "--output",
        f"{root}.jsonl",
        "--ledger",
        f"{root}.attempts.jsonl",
        "--experiment-id",
        experiment_id,
        "--pairing-id",
        pairing_id,
        "--runtime-source-bundle-sha256",
        runtime_source_bundle_sha256,
        "--variant",
        method,
        "--training-seed",
        "11",
        "--corruption-seed",
        str(seed),
        "--structures",
        str(structures),
        "--corruption-level",
        str(level),
        "--operator",
        operator,
        "--geometry-condition",
        geometry,
        "--schedule",
        schedule,
        "--control",
        control,
        "--calls",
        str(calls),
        "--revision-threshold",
        str(threshold),
        "--device",
        "cuda",
    ]
    return {
        "phase": phase,
        "cell_id": cell,
        "experiment_id": experiment_id,
        "pairing_id": pairing_id,
        "method": method,
        "attempts": structures,
        "backbone_calls_per_attempt": calls * (2 if control == "extra-call" else 1),
        "argv": argv,
    }


def materialize_day7_plan(
    registry: ExperimentRegistry,
    *,
    run_id: str,
    source_bundle_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    """Expand every preregistered Day-7 cell without submitting any job."""

    if not run_id or any(value.isspace() for value in run_id):
        raise ValueError("run_id must be a nonempty whitespace-free identifier")
    if not re.fullmatch(r"[0-9a-f]{64}", source_bundle_sha256):
        raise ValueError("Day-7 plan requires a lowercase source-bundle SHA256")
    week1 = registry.data["week1"]
    checkpoint = {
        method: "${CHECKPOINT_" + method.replace("-", "_") + "}"
        for method in DAY7_METHODS
    }
    jobs: list[dict[str, Any]] = []
    calibration = week1["threshold_calibration"]
    for threshold, cell, seed in itertools.product(
        calibration["thresholds"],
        calibration["cells"],
        calibration["corruption_seeds"],
    ):
        pairing = "-".join(
            _slug(value)
            for value in (
                run_id,
                "threshold",
                cell["level"],
                cell["operator"],
                cell["geometry"],
                cell["schedule"],
                seed,
            )
        )
        jobs.append(
            _recovery_job(
                phase="threshold-calibration",
                run_id=run_id,
                method=calibration["method"],
                checkpoint=checkpoint[calibration["method"]],
                level=float(cell["level"]),
                operator=cell["operator"],
                geometry=cell["geometry"],
                schedule=cell["schedule"],
                control=cell["control"],
                seed=int(seed),
                structures=int(calibration["structures"]),
                calls=int(calibration["calls"]),
                threshold=float(threshold),
                pairing_id=pairing,
                runtime_source_bundle_sha256=source_bundle_sha256,
            )
        )
    primary = week1["day7_primary_falsification"]
    # The frozen threshold value is injected after calibration. This explicit
    # placeholder prevents silently falling back to a CLI default.
    frozen_threshold = "${REVISION_THRESHOLD}"
    for method, level, operator, seed in itertools.product(
        primary["methods"],
        primary["corruption_levels"],
        primary["operators"],
        primary["corruption_seeds"],
    ):
        pairing = "-".join(
            _slug(value) for value in (run_id, "primary", level, operator, seed)
        )
        job = _recovery_job(
            phase="day7-primary",
            run_id=run_id,
            method=method,
            checkpoint=checkpoint[method],
            level=float(level),
            operator=operator,
            geometry=primary["geometry"],
            schedule=primary["schedule"],
            control=primary["control"],
            seed=int(seed),
            structures=int(primary["structures"]),
            calls=int(primary["calls"]),
            threshold=0.7,
            pairing_id=pairing,
            runtime_source_bundle_sha256=source_bundle_sha256,
        )
        job["argv"][job["argv"].index("--revision-threshold") + 1] = frozen_threshold
        jobs.append(job)
    interventions = week1["day7_interventions"]
    for cell, level, seed in itertools.product(
        interventions["cells"],
        interventions["corruption_levels"],
        interventions["corruption_seeds"],
    ):
        pairing = "-".join(
            _slug(value)
            for value in (
                run_id,
                "intervention",
                level,
                interventions["operator"],
                cell["geometry"],
                cell["schedule"],
                cell["control"],
                seed,
            )
        )
        job = _recovery_job(
            phase="day7-intervention",
            run_id=run_id,
            method=cell["method"],
            checkpoint=checkpoint[cell["method"]],
            level=float(level),
            operator=interventions["operator"],
            geometry=cell["geometry"],
            schedule=cell["schedule"],
            control=cell["control"],
            seed=int(seed),
            structures=int(interventions["structures"]),
            calls=int(interventions["calls"]),
            threshold=0.7,
            pairing_id=pairing,
            runtime_source_bundle_sha256=source_bundle_sha256,
        )
        job["argv"][job["argv"].index("--revision-threshold") + 1] = frozen_threshold
        jobs.append(job)
    by_phase: dict[str, dict[str, int]] = {}
    for phase in sorted({job["phase"] for job in jobs}):
        selected = [job for job in jobs if job["phase"] == phase]
        by_phase[phase] = {
            "jobs": len(selected),
            "attempts": sum(job["attempts"] for job in selected),
            "backbone_calls": sum(
                job["attempts"] * job["backbone_calls_per_attempt"] for job in selected
            ),
        }
    result = {
        "schema": "wqcodiff_materialized_job_plan_v1",
        "run_id": run_id,
        "registry_path": str(registry.path),
        "registry_sha256": registry.sha256,
        "protocol_path": str(registry.protocol.path),
        "protocol_sha256": registry.protocol.sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "maximum_concurrent_lanes": 4,
        "requires_frozen_revision_threshold_for_noncalibration_jobs": True,
        "phase_summary": by_phase,
        "jobs": jobs,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def _week2_training_argv(
    *,
    protocol_path: str,
    datasets: Sequence[str],
    output_dir: str,
    variant: str,
    source_bundle_sha256: str,
    shared_checkpoint: str | None,
) -> list[str]:
    argv = [
        "python",
        "-m",
        "crystal_dlm.wqcodiff",
        "--protocol",
        protocol_path,
        "train",
    ]
    for dataset in datasets:
        argv.extend(("--dataset", dataset))
    argv.extend(
        (
            "--output",
            output_dir,
            "--variant",
            variant,
            "--training-seed",
            "11",
            "--source-bundle-sha256",
            source_bundle_sha256,
            "--updates",
            "100000",
            "--microbatch-size",
            "128",
            "--checkpoint-interval",
            "5000",
            "--log-interval",
            "100",
            "--device",
            "cuda",
        )
    )
    if shared_checkpoint is None:
        argv.append("--stop-after-shared")
    else:
        argv.extend(
            (
                "--shared-checkpoint",
                shared_checkpoint,
                "--stop-after-update",
                "85000",
            )
        )
    return argv


def materialize_week2_training_plan(
    registry: ExperimentRegistry,
    *,
    run_id: str,
    discrete_engine: str,
    source_bundle_sha256: str,
    output: str | Path,
    dataset_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Freeze the two shared stages and five matched Week-2 screens.

    The discrete-engine choice is the sole Day-7-dependent input.  Screening
    continuation is deliberately bound to optimizer checkpoints at update
    60,000; the validation-only 85k EMA can never be supplied as resume state.
    """

    if not run_id or any(value.isspace() for value in run_id):
        raise ValueError("run_id must be a nonempty whitespace-free identifier")
    if discrete_engine not in DAY7_DISCRETE_ENGINES:
        raise ValueError("Week-2 discrete engine must be the frozen Day-7 AR/D3PM/DLM choice")
    if not re.fullmatch(r"[0-9a-f]{64}", source_bundle_sha256):
        raise ValueError("Week-2 plan requires a lowercase source-bundle SHA256")
    datasets = tuple(dataset_paths or WEEK2_TRAIN_DATASETS)
    if len(datasets) != 8 or len(set(datasets)) != 8 or any(not value for value in datasets):
        raise ValueError("Week-2 training requires exactly eight unique full-data shards")

    protocol_path = str(registry.data["registry"]["protocol"])
    training_root = f"runs/{run_id}/outputs/training"
    shared_specs = (
        ("shared-wyckoff", "B-WQ-JOINT-NOREV", "wyckoff_quotient"),
        ("shared-atom", "B-ATOM-JOINT", "dynamic_atom_set_p1"),
    )
    jobs: list[dict[str, Any]] = []
    shared_checkpoints: dict[str, str] = {}
    for job_id, variant, representation in shared_specs:
        output_dir = f"{training_root}/{job_id}"
        checkpoint = f"{output_dir}/checkpoint_0060000.pt"
        shared_checkpoints[job_id] = checkpoint
        jobs.append(
            {
                "job_id": job_id,
                "phase": "shared-60000",
                "depends_on": [],
                "variant": variant,
                "representation": representation,
                "output_dir": output_dir,
                "target_update": 60000,
                "continuation_checkpoint": checkpoint,
                "validation_ema": None,
                "argv": _week2_training_argv(
                    protocol_path=protocol_path,
                    datasets=datasets,
                    output_dir=output_dir,
                    variant=variant,
                    source_bundle_sha256=source_bundle_sha256,
                    shared_checkpoint=None,
                ),
            }
        )

    route_specs = (
        ("best-discrete-engine", discrete_engine, "shared-wyckoff", "wyckoff_quotient"),
        ("joint-no-revision", "B-WQ-JOINT-NOREV", "shared-wyckoff", "wyckoff_quotient"),
        ("disc-once", "B-WQ-DISC-ONCE", "shared-wyckoff", "wyckoff_quotient"),
        ("atom-joint", "B-ATOM-JOINT", "shared-atom", "dynamic_atom_set_p1"),
        ("stratified-geometry", "M-WQ-STRAT-GEO", "shared-wyckoff", "wyckoff_quotient"),
    )
    for route, variant, dependency, representation in route_specs:
        job_id = f"screen-{route}"
        output_dir = f"{training_root}/{job_id}"
        jobs.append(
            {
                "job_id": job_id,
                "phase": "screen-60000-to-85000",
                "depends_on": [dependency],
                "route": route,
                "variant": variant,
                "representation": representation,
                "output_dir": output_dir,
                "target_update": 85000,
                "continuation_checkpoint": f"{output_dir}/checkpoint_0085000.pt",
                "validation_ema": f"{output_dir}/model_ema_final.pt",
                "argv": _week2_training_argv(
                    protocol_path=protocol_path,
                    datasets=datasets,
                    output_dir=output_dir,
                    variant=variant,
                    source_bundle_sha256=source_bundle_sha256,
                    shared_checkpoint=shared_checkpoints[dependency],
                ),
            }
        )

    result = {
        "schema": "wqcodiff_week2_training_plan_v1",
        "run_id": run_id,
        "registry_path": str(registry.path),
        "registry_sha256": registry.sha256,
        "protocol_path": str(registry.protocol.path),
        "protocol_sha256": registry.protocol.sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "training_seed": 11,
        "full_schedule_updates": 100000,
        "shared_boundary_update": 60000,
        "screening_stop_update": 85000,
        "discrete_engine_from_day7": discrete_engine,
        "datasets": list(datasets),
        "maximum_concurrent_lanes": 4,
        "optimizer_checkpoint_required_for_continuation": True,
        "summary": {"jobs": 7, "shared": 2, "screening": 5},
        "jobs": jobs,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def _week2_sampling_argv(
    *,
    protocol_path: str,
    checkpoint: str,
    output: str,
    ledger: str,
    experiment_id: str,
    pairing_id: str,
    variant: str,
    sampling_seed: int,
    attempts: int,
    calls: int,
    revision_threshold: float,
    revision_lock: str,
    disc_once_tau: float,
) -> list[str]:
    return [
        "python",
        "-m",
        "crystal_dlm.wqcodiff",
        "--protocol",
        protocol_path,
        "sample",
        "--checkpoint",
        checkpoint,
        "--output",
        output,
        "--ledger",
        ledger,
        "--experiment-id",
        experiment_id,
        "--pairing-id",
        pairing_id,
        "--variant",
        variant,
        "--training-seed",
        "11",
        "--sampling-seed",
        str(sampling_seed),
        "--attempts",
        str(attempts),
        "--start-ordinal",
        "0",
        "--backbone-calls",
        str(calls),
        "--revision-control",
        "auto",
        "--revision-threshold",
        str(revision_threshold),
        "--revision-lock",
        revision_lock,
        "--disc-once-tau",
        str(disc_once_tau),
        "--inference-batch-size",
        "64",
        "--device",
        "cuda",
    ]


def materialize_week2_sampling_plan(
    registry: ExperimentRegistry,
    *,
    run_id: str,
    training_plan: str | Path,
    revision_lock: str | Path,
    output: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze matched Week-2 preflight and development generation cells."""

    from .revision import load_revision_threshold_lock

    if not run_id or any(value.isspace() for value in run_id):
        raise ValueError("run_id must be a nonempty whitespace-free identifier")
    root = (
        _project_root(registry.path)
        if project_root is None
        else Path(project_root).resolve()
    )
    training_plan_path = Path(training_plan).resolve()
    training = json.loads(training_plan_path.read_text(encoding="utf-8"))
    if training.get("schema") != "wqcodiff_week2_training_plan_v1":
        raise ValueError("unsupported Week-2 training plan schema")
    if training.get("run_id") != run_id:
        raise ValueError("Week-2 sampling/training run IDs differ")
    if (
        training.get("registry_sha256") != registry.sha256
        or training.get("protocol_sha256") != registry.protocol.sha256
    ):
        raise ValueError("Week-2 training plan is not bound to this registry/protocol")
    if int(training.get("training_seed", -1)) != 11:
        raise ValueError("Week-2 sampling requires training seed 11 checkpoints")

    lock_path = Path(revision_lock).resolve()
    lock = load_revision_threshold_lock(
        lock_path,
        protocol_name=registry.protocol.name,
        protocol_sha256=registry.protocol.sha256,
    )
    threshold = float(lock["selected_threshold"])
    screen_jobs = {
        str(job.get("route")): job
        for job in training["jobs"]
        if job.get("phase") == "screen-60000-to-85000"
    }
    expected_routes = {
        "best-discrete-engine",
        "joint-no-revision",
        "disc-once",
        "atom-joint",
        "stratified-geometry",
    }
    if set(screen_jobs) != expected_routes:
        raise ValueError("Week-2 sampling requires exactly five completed screen routes")

    checkpoint_identities: dict[str, dict[str, Any]] = {}
    for route, job in screen_jobs.items():
        relative = job.get("validation_ema")
        if not relative:
            raise ValueError(f"screen route lacks a validation EMA: {route}")
        checkpoint = (root / str(relative)).resolve()
        try:
            checkpoint.relative_to(root / "runs")
        except ValueError as exc:
            raise ValueError("Week-2 checkpoint escapes the project runs directory") from exc
        checkpoint_identities[route] = _file_identity(checkpoint)

    configuration_specs: list[tuple[str, str, float | None]] = [
        ("best-discrete-engine", "best-discrete-engine", None),
        ("joint-no-revision", "joint-no-revision", None),
        ("disc-once-tau-0p25", "disc-once", 0.25),
        ("disc-once-tau-0p5", "disc-once", 0.5),
        ("disc-once-tau-0p75", "disc-once", 0.75),
        ("disc-once-tau-1p0", "disc-once", 1.0),
        ("atom-joint", "atom-joint", None),
        ("stratified-geometry", "stratified-geometry", None),
    ]
    development = registry.data["week2"]["development_sampling"]
    calls = int(development["backbone_calls"])
    dev_attempts = int(development["attempts_per_sampling_seed"])
    dev_seeds = tuple(int(value) for value in development["sampling_seeds"])
    preflight_attempts = int(development["preflight_attempts"])
    dev_pairing = str(development["pairing_id"])
    protocol_path = str(registry.data["registry"]["protocol"])
    jobs: list[dict[str, Any]] = []
    for configuration_id, route, tau in configuration_specs:
        training_job = screen_jobs[route]
        variant = str(training_job["variant"])
        checkpoint = checkpoint_identities[route]
        for phase, seeds, attempts, pairing_id in (
            (
                "preflight",
                (dev_seeds[0],),
                preflight_attempts,
                f"{dev_pairing}-preflight",
            ),
            ("development", dev_seeds, dev_attempts, dev_pairing),
        ):
            for sampling_seed in seeds:
                cell_id = f"{configuration_id}-seed-{sampling_seed}"
                experiment_id = f"{run_id}-{phase}-{cell_id}"
                output_root = f"runs/{run_id}/outputs/sampling/{phase}/{cell_id}"
                disc_once_tau = 0.5 if tau is None else tau
                jobs.append(
                    {
                        "phase": phase,
                        "cell_id": cell_id,
                        "configuration_id": configuration_id,
                        "route": route,
                        "variant": variant,
                        "disc_once_tau": tau,
                        "training_seed": 11,
                        "sampling_seed": sampling_seed,
                        "attempts": attempts,
                        "backbone_calls_per_attempt": calls,
                        "pairing_id": pairing_id,
                        "checkpoint": checkpoint,
                        "output": f"{output_root}.jsonl",
                        "ledger": f"{output_root}.attempts.jsonl",
                        "argv": _week2_sampling_argv(
                            protocol_path=protocol_path,
                            checkpoint=str(checkpoint["path"]),
                            output=f"{output_root}.jsonl",
                            ledger=f"{output_root}.attempts.jsonl",
                            experiment_id=experiment_id,
                            pairing_id=pairing_id,
                            variant=variant,
                            sampling_seed=sampling_seed,
                            attempts=attempts,
                            calls=calls,
                            revision_threshold=threshold,
                            revision_lock=str(lock_path),
                            disc_once_tau=disc_once_tau,
                        ),
                    }
                )

    phase_summary: dict[str, dict[str, int]] = {}
    for phase in WEEK2_SAMPLING_PHASES:
        selected = [job for job in jobs if job["phase"] == phase]
        phase_summary[phase] = {
            "jobs": len(selected),
            "attempts": sum(int(job["attempts"]) for job in selected),
            "backbone_calls": sum(
                int(job["attempts"]) * int(job["backbone_calls_per_attempt"])
                for job in selected
            ),
        }
    if phase_summary != {
        "preflight": {"jobs": 8, "attempts": 2048, "backbone_calls": 131072},
        "development": {"jobs": 24, "attempts": 24000, "backbone_calls": 1536000},
    }:
        raise ValueError("Week-2 sampling expansion differs from the frozen matrix")

    result = {
        "schema": "wqcodiff_week2_sampling_plan_v1",
        "run_id": run_id,
        "registry_sha256": registry.sha256,
        "protocol_sha256": registry.protocol.sha256,
        "source_bundle_sha256": training["source_bundle_sha256"],
        "project_root": str(root),
        "training_plan": _file_identity(training_plan_path),
        "revision_threshold_lock": _file_identity(lock_path),
        "selected_revision_threshold": threshold,
        "maximum_concurrent_lanes": 4,
        "configuration_count": len(configuration_specs),
        "matched_development_pairing_id": dev_pairing,
        "phase_summary": phase_summary,
        "checkpoints": checkpoint_identities,
        "jobs": jobs,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result
