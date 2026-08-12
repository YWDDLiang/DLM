#!/usr/bin/env python3
"""Small fail-closed helpers for the retrained H1-A2 recovery bundle."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


DENOMINATOR = 256
PLANNER_RAW_ATTEMPTS = 1200
ARMS = ("R03",)
REPEATS = (0, 1, 2, 3)
PAIRED_SEED_NAMESPACE = "frozen_20260731_h1a2c_p0_p1_sun256_attempt_ledger"
REFERENCE_LEDGER_SHA256 = "24295854aac87f3eb9ad7cc293f2bf2d2eb1d8c292b7f05aeaad8348b6665c8f"
HEX_SHA = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return dict(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(dict(value))
    return rows


def _exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_exclusive(path: Path, value: Any) -> None:
    _exclusive_text(
        path,
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _exclusive_text(path, "".join(canonical_json(dict(row)) + "\n" for row in rows))


def require_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    location = Path(path).resolve()
    expected = require_hex_sha256(expected_sha256, label)
    if not location.is_file():
        raise FileNotFoundError(location)
    observed = sha256_file(location)
    if observed != expected:
        raise ValueError(f"{label} changed: expected={expected} observed={observed}")
    return location


def require_source_manifest(source_dir: Path, expected_sha256: str) -> None:
    source = source_dir.resolve()
    manifest = require_file(source / "SOURCE_SHA256.txt", expected_sha256, "source manifest")
    listed: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise ValueError(f"malformed manifest line {line_number}")
        expected, relative = pieces
        require_hex_sha256(expected, f"manifest line {line_number}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe manifest path at line {line_number}")
        listed.add(relative_path.as_posix())
        if sha256_file(source / relative_path) != expected:
            raise ValueError(f"source file changed: {relative}")
    observed = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_SHA256.txt"
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if listed != observed:
        raise ValueError(f"source file set changed: missing={sorted(listed-observed)}, extra={sorted(observed-listed)}")


def require_hex_sha256(value: Any, label: str) -> str:
    text = str(value).strip().lower()
    if HEX_SHA.fullmatch(text) is None:
        raise ValueError(f"{label} is not a frozen SHA-256")
    return text


def ordered_rows(rows: Iterable[Mapping[str, Any]], *, ordinal_field: str) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row[ordinal_field]))
    if len(ordered) != DENOMINATOR or [int(row.get(ordinal_field, -1)) for row in ordered] != list(range(DENOMINATOR)):
        raise ValueError(f"{ordinal_field} coverage changed")
    return ordered


def validate_arm(value: str) -> str:
    arm = str(value)
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    return arm


def validate_repeat(value: int | str) -> int:
    repeat = int(value)
    if repeat not in REPEATS:
        raise ValueError(f"repeat must be one of {REPEATS}")
    return repeat


def validate_frozen_cohort_row(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("frozen 1,000-row cohort validation is forbidden in the 256-row wrapper")


@functools.lru_cache(maxsize=1)
def _reference_seed_rows() -> tuple[dict[str, Any], ...]:
    raw = os.environ.get("H1_REFERENCE_ATTEMPT_LEDGER", "")
    if not raw:
        raise ValueError("H1_REFERENCE_ATTEMPT_LEDGER is required")
    rows = read_jsonl(require_file(raw, REFERENCE_LEDGER_SHA256, "frozen seed ledger"))
    if (
        len(rows) != DENOMINATOR
        or [int(row.get("ordinal", -1)) for row in rows] != list(range(DENOMINATOR))
        or any(not isinstance(row.get("body_noise_seed"), int) or not isinstance(row.get("refiner_noise_seed"), int) for row in rows)
    ):
        raise ValueError("frozen seed ledger contract changed")
    return tuple(rows)


def paired_seed(repeat: int, ordinal: int, channel: str) -> int:
    validate_repeat(repeat)
    if ordinal not in range(DENOMINATOR):
        raise ValueError("ordinal outside frozen denominator")
    field = {"body": "body_noise_seed", "refiner": "refiner_noise_seed"}.get(channel)
    if field is None:
        raise ValueError("unknown seed channel")
    return int(_reference_seed_rows()[ordinal][field])


def attempt_id(arm: str, repeat: int, ordinal: int, stage: str) -> str:
    validate_arm(arm)
    validate_repeat(repeat)
    if stage not in {"pre_model494", "post_model494"}:
        raise ValueError("invalid stage")
    panel = os.environ.get("H1_RECOVERY_PANEL", "fresh_cohort")
    if panel not in {"fresh_cohort", "topology_match"}:
        raise ValueError("invalid H1_RECOVERY_PANEL")
    prefix = (
        "h1a2-retrained-topology-r03"
        if panel == "topology_match"
        else "h1a2-retrained-r03"
    )
    return f"{prefix}-c{repeat}-{stage}-{ordinal:04d}"


def validate_config(config: Mapping[str, Any]) -> None:
    continuation = config.get("continuation") or {}
    body_continuation = config.get("body_continuation") or {}
    preparation_history = config.get("preparation_history") or {}
    scheduler = config.get("scheduler_repair") or {}
    body = config.get("body") or {}
    refiner = config.get("refiner") or {}
    planner = config.get("planner") or {}
    official_sun = config.get("official_sun") or {}
    seed_ledger = config.get("reference_seed_ledger") or {}
    historical_planner = config.get("historical_planner_reference") or {}
    topology = config.get("topology_match") or {}
    topology_planner = topology.get("planner") or {}
    h1a2_control = config.get("historical_h1a2_dlm_control") or {}
    inference = config.get("inference") or {}
    authorization = config.get("authorization") or {}
    downstream = config.get("downstream_cohorts") or []
    r03 = (body.get("models") or {}).get("R03") or {}
    if (
        config.get("schema")
        != "h1a2_retrained_world2_r03_refine_import_contract_repair_config_v3"
        or config.get("run_id")
        != "20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8"
        or config.get("run_root")
        != "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8"
        or continuation
        != {
            "upstream_run_root": "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5",
            "upstream_source_manifest_sha256": "29ca85d57d311f455a4f91d83e6d1c4c896d62b47835958dd37719fdd4d9f162",
            "upstream_slurm_job_id": "31900",
            "upstream_slurm_state": "FAILED",
            "upstream_slurm_exit_code": "1:0",
            "required_planner_terminal_sha256": "5a9676dcc10a4c4938d29aa39bbe509a4635f08f2215d49f756ea165729ca966",
            "required_planner_distribution_sha256": "649eddbc90148f74fb11cab865e59a21282750f134d2345522880754f61a9953",
            "required_planner_topology_sha256": "dce0b7cdcb2ad24b3aa7f418e9ede3e3a32689c61e1a7b3cbe34470ac4120b3a",
            "required_cohort_sha256": {
                "retrained_seed52021_world2_b4": "b5a897c029947e1cf88d3abbd2b48efed4cfc1694e4dc73ab9a447ce9f3d178c",
                "retrained_seed62023_world2_b4": "3913bae3e07d6a06902dc15e8ff8f484cdf32754b204d5fdaeb86b176c663dd9",
                "retrained_seed72031_world2_b4": "daf5b2fa9ec0e00929a91521fac5978e0d8d55008fcf6a2da93ca01224d44b5b",
                "retrained_seed82037_world2_b4": "63d764ecb63dd522ffd0fe1844298461fd9525e2169d38c57f2a79e8e8a9c93f",
                "retrained_seed17_world2_b4_topology_match": "3a3107489866b551069870cac12a40f02b7c6a8f313845c6df91bc450e529f39",
            },
            "failure_stage": "v7_fresh_r03_after_body_before_model494_load",
            "failure_signature": "ModuleNotFoundError: No module named 'scripts.refine_dlm_with_crysllmgen'",
            "planner_sampling_rerun": False,
            "planner_outputs_byte_reused": True,
            "generation_outputs_reused": True,
            "slurm_job_ordinal_since_v5": 3,
            "remaining_official_slurm_jobs": 0,
        }
        or body_continuation
        != {
            "upstream_run_root": "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v7",
            "upstream_slurm_job_id": "31965",
            "upstream_slurm_state": "FAILED",
            "upstream_slurm_exit_code": "1:0",
            "reuse_scope": "four_completed_fresh_r03_body256_outputs_only",
            "body_generation_rerun": False,
            "refinement_started": False,
            "required_body_succeeded": [246, 242, 250, 245],
            "import_root_cause": "R03D_PYTHONPATH_not_switched_to_R03E_before_refine1000",
            "corrected_refine_pythonpath": "SOURCE:R03E_RUNTIME:R03E:R03D:PROJECT",
        }
        or preparation_history
        != {
            "aborted_run_root": "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v6",
            "aborted_before_submission": True,
            "abort_reason": "user_requested_no_large_artifact_rehash",
            "large_artifact_rehash": False,
            "body_adapter_identity_basis": "previously_registered_sha256_plus_current_path_and_byte_size",
            "refiner_identity_basis": "previously_registered_sha256_plus_current_path_and_byte_size",
        }
        or scheduler.get("variant")
        != "refine_import_continuation_single_job_max4_a800_32cpu_v8"
        or int(scheduler.get("slurm_job_count", -1)) != 1
        or int(scheduler.get("requested_cpus", -1)) != 32
        or int(scheduler.get("maximum_concurrent_cpu_threads", -1)) != 32
        or int(scheduler.get("requested_a800_gpus", -1)) != 4
        or int(scheduler.get("maximum_visible_a800_gpus", -1)) != 4
        or scheduler.get("planner_waves") != []
        or scheduler.get("pre_refine_evaluated") is not False
        or int(config.get("denominator", -1)) != DENOMINATOR
        or len(downstream) != 4
        or downstream != [str(spec["cohort_id"]) for spec in planner.get("cohorts", [])]
        or int(planner.get("world_size", -1)) != 2
        or int(planner.get("batch_size_per_rank", -1)) != 4
        or int(planner.get("num_samples_per_cohort", -1))
        != PLANNER_RAW_ATTEMPTS
        or int(planner.get("frozen_attempts_per_cohort", -1)) != DENOMINATOR
        or planner.get("base_model")
        != "/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B/"
        or planner.get("sampler")
        != "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/scripts/sample_llama_h1_formula_plans.py"
        or planner.get("sampler_sha256")
        != "d38743f2f647d798800724b09537fbe492706805c00d7ee34c5ca8d74e39adc8"
        or float(planner.get("temperature", -1)) != 0.9
        or float(planner.get("top_p", -1)) != 0.95
        or int(planner.get("top_k", -1)) != 50
        or int(planner.get("max_new_tokens", -1)) != 96
        or planner.get("prompt_style") != "h1_rich_plan_v1"
        or planner.get("include_sample_id") is not False
        or planner.get("seed_mode") != "legacy_rank"
        or planner.get("do_sample") is not True
        or planner.get("stop_after_plan_marker") is not True
        or planner.get("truncate_after_plan_marker") is not True
        or planner.get("formula_constraint_mode") != "off"
        or planner.get("rng") != "stateful_torch_seed_plus_rank"
        or planner.get("merge_order") != "rank_concatenated_file_order"
        or planner.get("selection")
        != "first_256_raw_records_in_merged_file_order_with_failures_preserved"
        or planner.get("historical_sampling_contract")
        != "generate_1200_world2_rank_concatenated_then_freeze_first256"
        or [int(spec.get("seed", -1)) for spec in planner.get("cohorts", [])]
        != [52021, 62023, 72031, 82037]
        or topology.get("panel_id") != "historical_best_topology_match"
        or topology_planner
        != {
            "cohort_id": "retrained_seed17_world2_b4_topology_match",
            "seed": 17,
        }
        or int(topology.get("body_process_realizations", -1)) != 1
        or int(topology.get("refiner_process_realizations", -1)) != 4
        or int(topology.get("refiner_array_concurrency", -1)) != 2
        or topology.get("reuse_identical_body_and_proposal_graphs") is not True
        or topology.get("reuse_identical_refiner_seed_vector") is not True
        or topology.get("historical_target_strict_counts") != [28, 32, 30, 30]
        or h1a2_control
        != {
            "panel_id": "retrained_seed17_world2_historical_h1a2_b0_d1",
            "planner_cohort_id": "retrained_seed17_world2_b4_topology_match",
            "planner_seed": 17,
            "body_arm": "B0",
            "body_checkpoint": "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final",
            "body_adapter_sha256": "5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d",
            "generation_policy": "d1_exact_plan_schedule",
            "body_world_size": 2,
            "body_effective_batch_size": 1,
            "refiner_world_size": 2,
            "refiner_effective_batch_size": 1,
            "refiner_steps": 800,
            "evaluate_stage": "post_model494_only",
            "controlled_seed_source": "frozen_20260731_attempt_ledger",
        }
        or set((body.get("models") or {}).keys()) != set(ARMS)
        or body.get("base_model")
        != "/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct"
        or body.get("checkpoint")
        != "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
        or body.get("adapter_file") != "adapter_model.safetensors"
        or int(body.get("adapter_expected_bytes", -1)) != 6391016776
        or body.get("adapter_sha256")
        != "5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d"
        or body.get("large_artifact_rehash") is not False
        or body.get("tokenizer_vocab_sha256")
        != "3acc073da85047265769f2dccd93543fa9d7cbfa95021aef54ef282b13ce2f37"
        or body.get("tokenizer_json_sha256")
        != "3a21588abca8e56155cc7b6cabb81df51992ccd2e89704aec770912f24e75509"
        or body.get("tokenizer_config_sha256")
        != "8e89acaa54a8fb8fc7d228165ac483f61b7fef7c4c9761214092511190f75de2"
        or int(body.get("tokenizer_size", -1)) != 128830
        or r03.get("checkpoint")
        != "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
        or r03.get("adapter_sha256")
        != "5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d"
        or body.get("prompt_contract") != "historical_r5c_plan_state_json_exact_length"
        or body.get("schedule") != "d2_safe_axis"
        or float(body.get("temperature", -1)) != 0.7
        or float(body.get("cfg_scale", -1)) != 0.0
        or int(body.get("max_batch_size", -1)) != 8
        or int(refiner.get("timesteps", -1)) != 1000
        or int(refiner.get("diffusion_steps", -1)) != 800
        or int(refiner.get("effective_batch_size", -1)) != 1
        or refiner.get("checkpoint")
        != "/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt"
        or refiner.get("checkpoint_sha256")
        != "573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e"
        or int(refiner.get("checkpoint_expected_bytes", -1)) != 147645242
        or refiner.get("large_artifact_rehash") is not False
        or seed_ledger.get("sha256") != REFERENCE_LEDGER_SHA256
        or seed_ledger.get("reuse_across_cohorts") is not True
        or historical_planner.get("cohort_id")
        != "h1a2_original_seed17_world2_b4"
        or historical_planner.get("cohort256")
        != "/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1_planner_h1a2_scientific_replay_r03_sun_recovery_v5/planner/h1a2_original_seed17_world2_b4/frozen/cohort256.jsonl"
        or historical_planner.get("cohort256_sha256")
        != "26df9e9c3f995de155d67477adac72c0dcba5e0d19dbc84dfa3795eca8d20e54"
        or historical_planner.get("role")
        != "deep_distribution_reference_only"
        or official_sun.get("query_method")
        != "mp_api.client.MPRester.get_entries_in_chemsys"
        or official_sun.get("compatible_only") is not True
        or official_sun.get("thermo_type") != "GGA_GGA+U"
        or official_sun.get("unresolved_policy")
        != "explicit_hull_unknown_skip_unknown_headline"
        or inference.get("evaluated_stage") != "post_model494_only"
        or inference.get("pre_refine_role") != "intermediate_only_not_scored"
        or inference.get("exact_mcnemar_per_cohort_pre_post") is not False
        or int(inference.get("hierarchical_paired_bootstrap_draws", -1)) != 50000
        or int(inference.get("bootstrap_seed", -1)) != 20260812
        or inference.get("pooled_1024_role") != "descriptive_only"
        or authorization.get("planner_sampling") is not False
        or any(
            authorization.get(key) is not True
            for key in (
                "body_generation",
                "diffusion_refinement",
                "direct_evaluation",
                "official_sun_incremental_query",
            )
        )
        or any(
            authorization.get(key) is not False
            for key in (
                "retry",
                "replacement",
                "repair",
                "filter",
                "rerank",
                "training",
                "rl",
            )
        )
        or any(config.get(key) is not False for key in ("retry", "replacement", "repair", "filter", "rerank", "training", "rl"))
    ):
        raise ValueError("retrained recovery config contract changed")
