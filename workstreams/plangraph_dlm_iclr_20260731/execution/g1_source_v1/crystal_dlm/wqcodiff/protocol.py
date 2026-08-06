"""Load and reject deviations from the registered protocol-v3 contract."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Mapping


ACTIVE_PROTOCOL_NAME = "sun_iclr_stratified_wyckoff_v3"
ACTIVE_SCHEMA_VERSION = 3


@dataclasses.dataclass(frozen=True, slots=True)
class RegisteredProtocol:
    path: Path
    sha256: str
    data: Mapping[str, Any]

    @property
    def name(self) -> str:
        return str(self.data["protocol"]["name"])

    @property
    def schema_version(self) -> int:
        return int(self.data["protocol"]["schema_version"])


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"protocol violation: {message}")


def validate_protocol(data: Mapping[str, Any]) -> None:
    protocol = data.get("protocol", {})
    _expect(protocol.get("name") == ACTIVE_PROTOCOL_NAME, "wrong protocol name")
    _expect(int(protocol.get("schema_version", -1)) == ACTIVE_SCHEMA_VERSION, "wrong schema")
    _expect("superseded" not in str(protocol.get("status", "")), "superseded protocol")

    representation = data["wyckoff_representation"]
    _expect(
        set(representation["allowed_topology_events"])
        == {"orbit_birth", "orbit_death", "wyckoff_type_change", "species_change"},
        "registered orbit event support changed",
    )
    _expect(representation["committed_space_group_change"] is False, "SG rollback enabled")
    _expect(
        representation["space_group_generation"] == "masked_once_then_fixed",
        "space-group commitment semantics changed",
    )
    _expect(
        int(representation["max_topology_events_per_reverse_step"]) == 1
        and representation["topology_event_budget_scope"]
        == "explicit_topology_event_kernel"
        and representation["field_updates_may_be_parallel"] is True
        and representation["parallel_field_update_semantics"]
        == "categorical_chain_or_mask_commit_not_an_additional_event_kernel_draw",
        "reverse-step topology-event semantics changed",
    )
    _expect(
        tuple(float(value) for value in representation["symmetry_tolerance_grid_angstrom"])
        == (0.001, 0.01, 0.1),
        "symmetry tolerance grid changed",
    )
    _expect(
        float(representation["primary_symmetry_tolerance_angstrom"]) == 0.01,
        "primary symmetry tolerance changed",
    )
    _expect(
        representation["batching_representation"] == "native_ragged_orbit_set",
        "semantic batching is no longer ragged",
    )
    empty_start = representation["empty_set_start"]
    _expect(empty_start["learned_start_token"] is True, "empty-set learned prior removed")
    _expect(
        set(empty_start["heads"])
        == {
            "masked_space_group",
            "first_wyckoff_type",
            "first_species",
            "first_free_coordinate_chart",
            "lattice_chart",
        },
        "empty-set unconditional prior heads changed",
    )

    attempts = data["attempt_contract"]
    _expect(
        "method" in attempts["attempt_id"]["fields"],
        "attempt identity is no longer method-specific",
    )
    _expect(
        "method" not in attempts["pair_id"]["fields"],
        "matched-noise pair identity became method-dependent",
    )
    _expect(attempts["retry_failed_attempt"] == "forbidden", "attempt retries enabled")

    shared = data["models"]["shared"]
    _expect(shared["continuous_backend"] == "native_torch_cspnet_wyckoff_tangent", "backend changed")
    _expect(
        (int(shared["hidden_dim"]), int(shared["csp_layers"]), int(shared["orbit_set_layers"]))
        == (256, 6, 4),
        "registered model size changed",
    )
    _expect(shared["torch_geometric_dependency"] is False, "PyG dependency reintroduced")
    continuous = data["models"]["continuous"]
    _expect(
        continuous["orbit_noise"] == "wrapped_gaussian"
        and continuous["wrapped_score_target"] == "periodic_log_density_gradient"
        and int(continuous["wrapped_score_integer_image_radius"]) == 8,
        "wrapped-Gaussian score contract changed",
    )
    _expect(
        continuous["wrapped_score_loss_weight"]
        == "sigma_squared_denoising_score_matching",
        "wrapped-Gaussian loss weighting changed",
    )
    _expect(
        continuous["periodic_coordinate_likelihood"]
        == "wrapped_normal_logsumexp"
        and int(continuous["periodic_coordinate_integer_image_radius"]) == 8,
        "periodic-coordinate likelihood changed",
    )
    _expect(
        float(continuous["periodic_coordinate_scale_min"]) == 0.02
        and float(continuous["periodic_coordinate_scale_max"]) == 0.5,
        "periodic-coordinate scale bounds changed",
    )
    _expect(
        set(continuous["periodic_coordinate_heads"])
        == {"first_orbit", "birth", "target_stratum_bridge"},
        "periodic-coordinate head coverage changed",
    )

    refiner = data["common_refiner"]
    _expect(refiner["topology_updates"] is False, "common refiner changes topology")
    _expect(
        (int(refiner["calls"]), float(refiner["start_time"])) == (16, 0.1),
        "common-refiner contract changed",
    )

    revisable = set(data["discrete_revision"]["revisable_fields"])
    _expect(
        revisable == {"orbit_existence", "wyckoff_type", "species"},
        "revisable field set changed",
    )
    _expect("early_space_group" not in revisable, "committed SG became revisable")
    revision = data["discrete_revision"]
    _expect(
        revision["training_evidence_corruption_labels_allowed"] is False,
        "corruption labels became available to geometry evidence",
    )
    _expect(
        revision["continuous_score_norm_source"]
        == "current_pre_evidence_coordinate_head_detached_orbit_rms_log1p200",
        "continuous-score evidence source changed",
    )
    threshold_selection = revision["threshold_selection"]
    _expect(
        threshold_selection["immutable_lock_required_for_paper_eligible_revision_sampling"]
        is True,
        "paper revision sampling no longer requires the Day-7 threshold lock",
    )
    _expect(
        float(threshold_selection["clean_false_remask_max"]) == 0.05
        and threshold_selection["tie_break"] == "higher_threshold",
        "revision threshold selection rule changed",
    )
    day7 = data["day7_falsification"]
    _expect(
        len(day7["methods"]) == len(set(day7["methods"])),
        "Day-7 method registry contains duplicates",
    )
    _expect(
        set(day7["corruption_operators"])
        == {"deletion", "false-insertion", "wrong-wyckoff", "wrong-species", "joint"},
        "Day-7 corruption operators changed",
    )
    _expect(
        day7["attempt_denominator_failure_penalties"]
        == {
            "exact_recovery": 0,
            "topology_edit_distance": 20,
            "tangent_coordinate_error": 1.0,
            "net_correction": 0,
        },
        "Day-7 failure penalties changed",
    )

    training = data["training"]
    _expect(int(training["full_optimizer_updates"]) == 100000, "full update count changed")
    _expect(
        int(training["shared_pretraining_updates"])
        + int(training["method_specific_updates"])
        == int(training["full_optimizer_updates"]),
        "training stages do not sum to full updates",
    )
    screening = data["screening"]
    _expect(
        int(screening["shared_boundary_update"]) == 60_000
        and int(screening["method_specific_screen_updates"]) == 25_000
        and int(screening["screening_stop_update_on_full_100k_schedule"]) == 85_000,
        "screening no longer uses 60k shared + 25k method-specific on the full schedule",
    )

    mlip = data["mlip"]
    _expect(str(mlip["guide"]["package_version"]) == "0.4.2", "CHGNet version changed")
    _expect(str(mlip["primary_heldout"]["package_version"]) == "1.1.2", "MatterSim version changed")
    _expect(str(mlip["secondary_heldout"]["package_version"]) == "0.3.13", "MACE version changed")
    sun_executor = mlip["sun_executor"]
    _expect(
        sun_executor["lineage"] == "r5c_full1000_sun"
        and sun_executor["script"] == "scripts/run_mattergen_sun_eval.py"
        and sun_executor["script_sha256"]
        == "510bcf297247dfab7a77ff7aa564072806f49b0c212fe670d3221d1788ef305b"
        and sun_executor["implementation"]
        == "exact_frozen_r5c_mattergen_mattersim_executor"
        and sun_executor["primary_checkpoint_override"]
        == "MatterSim-v1.0.0-5M.pth"
        and sun_executor["primary_reference"]
        == "evaluator_specific_frozen_mp20_lmdb"
        and sun_executor["frozen_arguments"]
        == {
            "device": "cuda",
            "relax_max_steps": 500,
            "relax_fmax": 0.05,
            "max_natoms_per_batch": 512,
            "structure_matcher": "disordered",
        }
        and sun_executor["denominator"]
        == "all_submitted_structures_including_unsupported_and_nonconverged"
        and sun_executor["output_join"]
        == "preserve_input_order_and_join_back_to_attempt_id",
        "R5-C S.U.N. executor contract changed",
    )
    runtime = mlip["runtime_isolation"]
    _expect(
        runtime["asset_lock_schema"] == "wqcodiff_mlip_asset_lock_v4"
        and runtime["asset_lock"] == "mlip_asset_lock_v4.json"
        and runtime["one_evaluator_per_process"] is True
        and runtime["one_evaluator_per_slurm_lane"] is True,
        "multi-MLIP process-isolation contract changed",
    )
    core_runtime = runtime["core_environment"]
    _expect(
        core_runtime["active_evaluator_stack"] == "wqcodiff-evaluator-stack-v4"
        and core_runtime["wheelhouse"] == "wheelhouse_v4"
        and core_runtime["wheelhouse_lock"] == "wheelhouse_lock_v4.json"
        and core_runtime["source_sdists"] == "source_sdists_v4"
        and core_runtime["dependency_waiver"]
        == "chgnet_torch_metadata_waiver_v4.json"
        and tuple(core_runtime["evaluators"]) == ("chgnet", "mace")
        and str(core_runtime["torch"]) == "2.4.0+cu121"
        and str(core_runtime["e3nn"]) == "0.4.4"
        and core_runtime["torch_geometric_required_by_wq_model"] is False
        and tuple(core_runtime["no_deps_metadata_validated_packages"])
        == ("chgnet", "mace-torch")
        and core_runtime["source_built_pure_python_wheels"]
        == {"nvidia-ml-py3": "7.352.0", "python-hostlist": "2.3.0"}
        and core_runtime["source_build_isolation"] is False
        and core_runtime["source_build_reproducibility"]
        == {
            "source_date_epoch": 315532800,
            "python_hash_seed": 0,
            "pip_cache": "disabled",
        }
        and core_runtime["failed_predecessor_locks"]
        == [
            {
                "filename": "wheelhouse_lock.json",
                "status": "failed_source_wheel_rebuild_evidence_only",
                "preservation": "immutable",
                "eligible_for_runtime_binding": False,
            },
            {
                "filename": "wheelhouse_lock_v2.json",
                "status": "failed_mattersim_import_evidence_only",
                "preservation": "immutable",
                "eligible_for_runtime_binding": False,
            },
        ]
        and core_runtime["global_pip_check_policy"]
        == "hash_locked_preexisting_lines_must_be_unchanged"
        and tuple(core_runtime["allowed_project_pip_check_mismatches"])
        == ("chgnet_0.4.2_requires_torch_ge_2.4.1_but_retained_2.4.0",),
        "core evaluator runtime changed",
    )
    mattersim_runtime = runtime["mattersim_environment"]
    _expect(
        mattersim_runtime["evaluator"] == "mattersim"
        and mattersim_runtime["implementation"]
        == "immutable_pythonpath_inference_target"
        and mattersim_runtime["relative_path"]
        == "runtimes/mattersim-1.1.2-py310-v4"
        and mattersim_runtime["dependency_waiver"]
        == "mattersim_inference_runtime_waiver_v4.json"
        and mattersim_runtime["runtime_lock"] == "mattersim_runtime_lock_v4.json"
        and mattersim_runtime["tree_manifest"] == "mattersim_runtime_tree_v4.json"
        and str(mattersim_runtime["e3nn_min"]) == "0.5.0"
        and mattersim_runtime["compatibility_pins"]
        == {
            "ase": {
                "version": "3.27.0",
                "wheel_sha256": "058c48ea504fe7fbbe7c932f778415243ef2df45b1ab869866f24efcc17f0538",
                "reason": "last_tested_release_exporting_stress_helper_from_ase_constraints",
            },
            "setuptools": {
                "version": "81.0.0",
                "wheel_sha256": "fdd925d5c5d9f62e4b74b30d6dd7828ce236fd6ed998a08d81de62ce5a6310d6",
                "reason": "last_release_family_containing_pkg_resources_for_mattersim_1p1p2",
            },
        }
        and mattersim_runtime["torch_geometric_scope"] == "evaluator_only"
        and mattersim_runtime["excluded_noninference_dependencies_locked"] is True
        and mattersim_runtime["full_tree_sha256_required"] is True,
        "MatterSim inference-runtime isolation changed",
    )

    evaluation = data["evaluation"]
    _expect(
        evaluation["common_subset_selection"]
        == "lowest_sha256_method_independent_pair_id",
        "multi-MLIP subset is no longer matched across methods",
    )
    _expect(
        int(evaluation["final_primary_attempts_per_method"]) == 10000
        and int(evaluation["final_multi_mlip_common_attempts_per_method"]) == 6000,
        "final evaluation counts changed",
    )
    _expect(
        evaluation["compute_accounting"][
            "actual_slurm_gpuh_and_peak_memory_required"
        ]
        is True
        and evaluation["compute_accounting"]["parameter_count_flops_proxy_label"]
        == "lower_bound_not_actual_flops",
        "compute accounting contract changed",
    )

    statistics = data["statistics"]
    _expect(int(statistics["bootstrap_repetitions"]) == 10000, "bootstrap count changed")
    _expect(
        tuple(statistics["hierarchy"])
        == ("training_seed", "sampling_seed", "duplicate_cluster"),
        "hierarchical bootstrap levels changed",
    )

    compute = data["compute_funnel"]
    _expect(int(compute["gpu_count"]) == 4, "GPU cap changed")
    _expect(int(compute["calendar_days_hard_max"]) == 28, "calendar cap changed")
    _expect(int(compute["usable_a800_gpu_hours_hard_ceiling"]) == 2050, "GPU-hour cap changed")
    _expect(int(compute["week4_frozen_champion_and_final_gpu_hours_reserved_min"]) >= 800, "Week-4 reserve reduced")

    execution = data["execution"]
    _expect(
        execution["threads"]
        == {
            "OPENBLAS_NUM_THREADS": 1,
            "OMP_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
        },
        "numeric thread lock changed",
    )


def load_protocol(path: str | Path) -> RegisteredProtocol:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - installed locally/server
        raise RuntimeError("PyYAML is required to load the protocol") from exc
    location = Path(path).resolve()
    raw = location.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, Mapping):
        raise ValueError("protocol root must be a mapping")
    validate_protocol(data)
    return RegisteredProtocol(location, hashlib.sha256(raw).hexdigest(), data)
