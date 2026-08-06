"""Strict loader for immutable CrysLLMGen/Wyckoff protocol v4 and registry v2."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_NAME = "crysllmgen_wyckoff_georev_v4"
PROTOCOL_SCHEMA = 4
REGISTRY_NAME = "crysllmgen_wq_iclr_four_week_v2"
REGISTRY_SCHEMA = 2
PRIMARY_METHODS = {
    "C-ATOM-OFFICIAL",
    "C-ATOM-MATCHED",
    "C-WQ-HANDOFF",
    "C-WQ-CONFEDIT",
    "C-WQ-GEOREV",
}


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"CrysLLMGen protocol violation: {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "crystal_dlm/wqcodiff").is_dir() and (parent / "configs").is_dir():
            return parent
    raise ValueError("cannot locate project root from protocol path")


@dataclasses.dataclass(frozen=True, slots=True)
class RegisteredCrysLLMGenProtocol:
    path: Path
    sha256: str
    data: Mapping[str, Any]

    @property
    def name(self) -> str:
        return str(self.data["protocol"]["name"])


@dataclasses.dataclass(frozen=True, slots=True)
class RegisteredCrysLLMGenRegistry:
    path: Path
    sha256: str
    data: Mapping[str, Any]
    protocol: RegisteredCrysLLMGenProtocol


def validate_protocol_v4(data: Mapping[str, Any]) -> None:
    header = data["protocol"]
    _expect(header["name"] == PROTOCOL_NAME, "wrong protocol name")
    _expect(int(header["schema_version"]) == PROTOCOL_SCHEMA, "wrong protocol schema")
    _expect("superseded" not in str(header["status"]), "superseded protocol")
    scope = data["scope"]
    _expect(scope["domain"] == "unconditional_crystal_generation_only", "scope widened")
    _expect(scope["initial_global_mask"] == "forbidden", "initial MASK was restored")
    _expect(
        scope["semantic_padding_or_null_canvas"] == "forbidden",
        "semantic padding was restored",
    )
    _expect(scope["committed_space_group_rollback"] == "forbidden", "SG rollback enabled")
    _expect(
        set(scope["active_topology_events"])
        == {"orbit_birth", "orbit_death", "wyckoff_type_change", "species_change"},
        "topology event support changed",
    )
    _expect(scope["dft"] == "forbidden", "new DFT enabled")

    lineage = data["upstream_lineage"]
    _expect(
        lineage["commit"] == "94bb287751cd20a882c7c1df7ca736633d78e5e1",
        "upstream commit changed",
    )
    _expect(lineage["license"] == "MIT", "upstream license changed")
    _expect(int(lineage["parity_proposals"]) == 256, "parity denominator changed")
    _expect(float(lineage["parity_numeric_tolerance"]) <= 1.0e-6, "parity tolerance loosened")

    assets = data["assets"]
    _expect(assets["llama"]["local_files_only"] is True, "network model resolution enabled")
    _expect(assets["llama"]["dtype"] == "bfloat16", "Llama precision changed")
    _expect(
        assets["official_atom_lora"]["use"] == "C-ATOM-OFFICIAL_only"
        and int(assets["official_atom_lora"]["rank"]) == 8,
        "official atom adapter escaped its registered scope",
    )
    _expect(
        assets["cspdiffusion"]["sha256"]
        == "573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e",
        "CSPDiffusion checkpoint identity changed",
    )
    _expect(
        int(assets["cspdiffusion"]["scheduler_timesteps"]) == 1000,
        "CSPDiffusion scheduler length changed",
    )
    _expect(
        int(assets["cspdiffusion"]["official_reverse_start_timestep"]) == 800,
        "official CrysLLMGen reverse start changed",
    )
    _expect(
        assets["cspdiffusion"]["upstream_run_type"] == "train",
        "official CrysLLMGen parent run type changed",
    )
    _expect(
        assets["missing_or_compute_inaccessible_action"] == "stop_and_ask_user",
        "missing model substitution was enabled",
    )

    grammar = data["grammar"]
    _expect(grammar["finite_state_constrained_decoding"] is True, "grammar masking removed")
    _expect(grammar["de_novo_initial_mask"] is False, "grammar starts from MASK")
    _expect(grammar["tokenizer_vocabulary_resize"] is False, "tokenizer resize added")
    _expect(int(grammar["continuous_bins"]) == 256, "continuous byte code changed")
    _expect(
        set(grammar["edit_commands"]) == {"NOOP", "BIRTH", "DEATH", "TYPE", "SPECIES"},
        "direct edit grammar changed",
    )

    attempts = data["attempt_contract"]
    _expect(attempts["retry_failed_attempt"] == "forbidden", "attempt retry enabled")
    _expect(attempts["replacement_sampling"] == "forbidden", "replacement sampling enabled")
    _expect("method" in attempts["attempt_id_fields"], "attempt IDs lost method identity")
    _expect("method" not in attempts["pair_id_fields"], "pair IDs became method-dependent")

    methods = data["methods"]["primary"]
    _expect(set(methods) == PRIMARY_METHODS, "primary method set changed")
    _expect(
        int(methods["C-ATOM-OFFICIAL"]["csp_forwards"]) == 1600,
        "official CrysLLMGen call contract changed",
    )
    for name in PRIMARY_METHODS - {"C-ATOM-OFFICIAL"}:
        _expect(int(methods[name]["csp_forwards"]) == 64, f"matched calls changed for {name}")
    _expect(data["methods"]["controls_train_separate_models"] is False, "control model added")
    _expect(
        data["methods"]["matched_count_control_reference"]
        == "frozen_C-WQ-GEOREV_pair_id_revision_steps",
        "matched-count control reference changed",
    )

    llama = data["llama_training"]
    lora = llama["lora"]
    _expect(
        (int(lora["rank"]), int(lora["alpha"]), float(lora["dropout"]))
        == (16, 32, 0.05),
        "matched LoRA capacity changed",
    )
    _expect(
        set(lora["target_modules"])
        == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
        "matched LoRA target modules changed",
    )
    _expect(int(llama["max_sequence_length"]) == 512, "Llama sequence length changed")
    _expect(llama["answer_only_supervision"] is True, "answer supervision changed")
    _expect(
        int(llama["shared_wq_edit_stage"]["optimizer_updates"]) == 20_000,
        "edit curriculum update count changed",
    )
    _expect(
        (
            float(llama["shared_wq_edit_stage"]["initial_proposal_fraction"]),
            float(llama["shared_wq_edit_stage"]["direct_edit_fraction"]),
            float(llama["shared_wq_edit_stage"]["direct_edit_clean_negative_fraction"]),
        )
        == (0.5, 0.5, 0.2),
        "mixed edit curriculum fractions changed",
    )

    refiner = data["wq_refiner_training"]
    _expect(int(refiner["total_optimizer_updates"]) == 100_000, "WQ refiner update count changed")
    _expect(
        sum(int(stage["updates"]) for stage in refiner["stages"]) == 100_000,
        "WQ refiner stages do not sum to 100k",
    )
    closed = data["closed_loop"]
    _expect(closed["space_group_fixed_after_proposal"] is True, "closed-loop SG changes enabled")
    _expect(closed["target_stratum_bridge_retry"] == "forbidden", "bridge retry enabled")
    _expect(int(closed["topology_events_per_reverse_step_max"]) == 1, "event cap changed")
    _expect(closed["mlip_evidence_allowed"] is False, "MLIP leaked into revision evidence")
    _expect(
        int(closed["threshold_calibration_attempts"]) == 1024
        and closed["threshold_calibration_uses_actual_llama_edit"] is True
        and closed["threshold_lock_required_for_all_editing_methods"] is True,
        "revision threshold calibration contract changed",
    )
    _expect(
        closed["matched_controls"]
        == {
            "random_count": "same_pair_same_initial_noise_same_georev_revision_steps",
            "shuffled_geometry": "same_pair_same_initial_noise_same_georev_revision_steps",
            "extra_call_ignored": "same_pair_same_initial_noise_one_ignored_csp_call_per_georev_revision",
        },
        "matched-count control contract changed",
    )

    sampling = data["sampling"]
    _expect(
        tuple(float(value) for value in sampling["handoff_tau_grid"])
        == (0.25, 0.5, 0.75, 1.0),
        "handoff tau grid changed",
    )
    _expect(tuple(sampling["final_training_seeds"]) == (11, 23, 47), "training seeds changed")
    _expect(tuple(sampling["final_attempt_allocation"]) == (3334, 3333, 3333), "attempt allocation changed")
    _expect(int(sampling["final_attempts_per_method"]) == 10_000, "final attempt count changed")
    _expect(int(sampling["common_multi_mlip_attempts_per_method"]) == 6000, "multi-MLIP subset changed")
    respacing = sampling["matched_atom_respacing"]
    _expect(respacing["algorithm"] == "parent_pc_skip_v1", "matched atom respacing changed")
    _expect(respacing["direct_crysllmgen_injection"] is True, "matched atom injection changed")
    _expect(int(respacing["csp_calls_per_timestep"]) == 2, "matched atom calls per step changed")
    _expect(
        tuple(int(value) for value in respacing["timesteps"])
        == (800, 774, 748, 723, 697, 671, 645, 620, 594, 568, 542, 516, 491, 465, 439, 413, 388, 362, 336, 310, 285, 259, 233, 207, 181, 156, 130, 104, 78, 53, 27, 1),
        "matched atom timestep grid changed",
    )

    gates = data["gates"]
    _expect(gates["A"]["before_training"] is True, "training no longer waits for Gate A")
    gate_b = gates["B"]
    _expect(int(gate_b["smoke_attempts_per_configuration"]) == 256, "Gate-B smoke denominator changed")
    _expect(float(gate_b["wq_roundtrip_rate_min"]) == 0.99, "Gate-B WQ roundtrip gate changed")
    _expect(
        gate_b["handoff_tau_selection_rule"]
        == "max_r5c_sun0p1_then_novel_unique_then_success_then_lower_flops_then_higher_tau",
        "Gate-B handoff selection changed",
    )
    _expect(
        gates["C"]["sampling_seed_attempts"] == {101: 1000, 202: 1000, 303: 1000},
        "Gate-C sampling-seed denominators changed",
    )
    _expect(float(gates["C"]["mattersim_mlip_sun_at_0p1_gain_min_pp"]) == 2.0, "Gate C changed")
    _expect(
        (float(gates["oral"]["mattersim_mlip_sun_at_0p1_gain_min_pp"]),
         float(gates["oral"]["gain_95ci_lower_bound_min_pp"]))
        == (5.0, 2.0),
        "oral SUN gate changed",
    )

    evaluation = data["evaluation"]
    _expect((int(evaluation["families"]), int(evaluation["configuration_ids"])) == (10, 13), "evaluation inventory changed")
    sun = evaluation["sun_executor"]
    _expect(
        sun["lineage"] == "r5c_full1000_sun"
        and sun["script"] == "scripts/run_mattergen_sun_eval.py"
        and sun["script_sha256"]
        == "510bcf297247dfab7a77ff7aa564072806f49b0c212fe670d3221d1788ef305b",
        "R5-C SUN executor changed",
    )
    _expect("every_attempt" in sun["denominator"], "SUN denominator became survivor-only")

    execution = data["execution"]
    _expect(int(execution["maximum_a800_gpus"]) == 4, "GPU cap changed")
    _expect(execution["cuda_only_through_slurm"] is True, "CUDA outside Slurm enabled")
    _expect(
        set(int(value) for value in execution["threads"].values()) == {1},
        "numeric thread lock changed",
    )
    _expect(set(int(value) for value in execution["offline"].values()) == {1}, "offline lock changed")
    compute = data["compute_funnel"]
    _expect(int(compute["a800_gpu_hours_hard_ceiling"]) == 2050, "GPU-hour ceiling changed")
    _expect(int(compute["final_reserve_gpu_hours_min"]) >= 800, "final reserve reduced")
    _expect(int(compute["freeze_day"]) == 21, "freeze day changed")


def _validate_bound_hashes(data: Mapping[str, Any], root: Path) -> None:
    bindings = [
        (data["upstream_lineage"]["snapshot_manifest"], data["upstream_lineage"]["snapshot_manifest_sha256"]),
        (data["upstream_lineage"]["disabled_extension_contract"], data["upstream_lineage"]["disabled_extension_contract_sha256"]),
        (data["assets"]["llama"]["preflight"], data["assets"]["llama"]["preflight_sha256"]),
        (data["assets"]["official_atom_lora"]["preflight"], data["assets"]["official_atom_lora"]["preflight_sha256"]),
        (data["assets"]["cspdiffusion"]["preflight"], data["assets"]["cspdiffusion"]["preflight_sha256"]),
        (data["grammar"]["contract"], data["grammar"]["contract_sha256"]),
        (data["evaluation"]["sun_executor"]["script"], data["evaluation"]["sun_executor"]["script_sha256"]),
    ]
    for relative, expected in bindings:
        path = root / str(relative)
        _expect(path.is_file(), f"bound artifact missing: {relative}")
        _expect(_sha256(path) == str(expected), f"bound artifact hash changed: {relative}")


def load_protocol_v4(path: str | Path) -> RegisteredCrysLLMGenProtocol:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load protocol v4") from exc
    location = Path(path).resolve()
    raw = location.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, Mapping):
        raise ValueError("protocol v4 root must be a mapping")
    validate_protocol_v4(data)
    _validate_bound_hashes(data, _project_root(location))
    return RegisteredCrysLLMGenProtocol(
        path=location,
        sha256=hashlib.sha256(raw).hexdigest(),
        data=data,
    )


def validate_registry_v2(
    data: Mapping[str, Any], protocol: RegisteredCrysLLMGenProtocol
) -> None:
    header = data["registry"]
    _expect(header["name"] == REGISTRY_NAME, "wrong registry name")
    _expect(int(header["schema_version"]) == REGISTRY_SCHEMA, "wrong registry schema")
    execution = data["execution_contract"]
    _expect(int(execution["maximum_concurrent_gpu_lanes"]) == 4, "registry GPU cap changed")
    _expect(execution["retry_or_replacement"] is False, "registry retry enabled")
    _expect(int(execution["blas_threads"]) == 1, "registry BLAS lock changed")
    gate_a = data["gate_a"]
    _expect(gate_a["training_blocked_until_all_pass"] is True, "Gate A training block removed")
    _expect(len(gate_a["gpu_jobs"]) == 4 and len(gate_a["cpu_jobs"]) == 1, "Gate A job inventory changed")
    inventory = data["training_inventory"]
    counts = inventory["counts"]
    _expect(
        (int(counts["main_families"]), int(counts["main_runs"]), int(counts["trained_ablation_runs"]), int(counts["maximum_training_runs"]))
        == (3, 9, 1, 10),
        "training model count changed",
    )
    _expect(int(counts["inference_control_training_runs"]) == 0, "control training run added")
    _expect(tuple(inventory["seeds"]) == (11, 23, 47), "registry training seeds changed")
    lanes = data["seed11_lanes"]
    _expect(len(lanes) == 4 and {int(item["lane"]) for item in lanes} == {0, 1, 2, 3}, "seed-11 lanes changed")
    _expect(set(data["screening"]["primary_methods"]) == PRIMARY_METHODS, "screening methods changed")
    evaluation = data["evaluation_inventory"]
    _expect(
        (len(evaluation["families"]), int(evaluation["total_configuration_ids"])) == (10, 13),
        "registry evaluation inventory changed",
    )
    final = data["final"]
    _expect(int(final["attempts_per_method"]) == 10_000, "registry final attempts changed")
    _expect(tuple(final["allocation"]) == (3334, 3333, 3333), "registry final allocation changed")
    compute = data["compute"]
    _expect(int(compute["global_gpu_hours_hard_ceiling"]) == 2050, "registry GPU-hour cap changed")
    _expect(int(compute["final_reserve_gpu_hours_min"]) >= 800, "registry reserve reduced")
    _expect(
        int(protocol.data["compute_funnel"]["a800_gpu_hours_hard_ceiling"])
        == int(compute["global_gpu_hours_hard_ceiling"]),
        "protocol/registry GPU-hour caps disagree",
    )


def load_registry_v2(path: str | Path) -> RegisteredCrysLLMGenRegistry:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load registry v2") from exc
    location = Path(path).resolve()
    raw = location.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, Mapping):
        raise ValueError("registry v2 root must be a mapping")
    root = _project_root(location)
    protocol = load_protocol_v4(root / str(data["registry"]["protocol"]))
    validate_registry_v2(data, protocol)
    return RegisteredCrysLLMGenRegistry(
        path=location,
        sha256=hashlib.sha256(raw).hexdigest(),
        data=data,
        protocol=protocol,
    )
