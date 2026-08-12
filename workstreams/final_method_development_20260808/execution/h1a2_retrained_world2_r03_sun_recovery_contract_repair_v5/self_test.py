#!/usr/bin/env python3
"""Offline contract tests for the retrained-world2 downstream bundle."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import audit_refiner_contract
import protocol


def main() -> None:
    config = json.loads(Path(__file__).with_name("CONFIG.json").read_text())
    protocol.validate_config(config)
    upstream = config["training_upstream"]
    for key in ("adapter_sha256", "adapter_config_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(upstream[key])) is None:
            raise ValueError(f"{key} is not frozen")
    assert protocol.DENOMINATOR == 256
    assert protocol.PLANNER_RAW_ATTEMPTS == 1200
    assert protocol.PAIRED_SEED_NAMESPACE == (
        "frozen_20260731_h1a2c_p0_p1_sun256_attempt_ledger"
    )
    assert config["planner"]["world_size"] == 2
    assert config["planner"]["batch_size_per_rank"] == 4
    assert config["planner"]["num_samples_per_cohort"] == 1200
    assert config["planner"]["frozen_attempts_per_cohort"] == 256
    assert config["planner"]["historical_sampling_contract"] == (
        "generate_1200_world2_rank_concatenated_then_freeze_first256"
    )
    scheduler = config["scheduler_repair"]
    assert scheduler == {
        "variant": "single_slurm_job_max4_a800_32cpu_generation_v5",
        "slurm_job_count": 1,
        "requested_cpus": 32,
        "maximum_concurrent_cpu_threads": 32,
        "requested_a800_gpus": 4,
        "maximum_visible_a800_gpus": 4,
        "requested_memory_gb": 500,
        "planner_sampler_sha256": "d38743f2f647d798800724b09537fbe492706805c00d7ee34c5ca8d74e39adc8",
        "legacy_rank_rng_interface": "implicit_seed_plus_rank_no_seed_mode_cli",
        "formula_constraint_interface": "absent_in_historical_sampler_equivalent_to_off",
        "planner_waves": [[0, 1], [2, 3], [4]],
        "generation_waves": [
            ["fresh_r03_0", "fresh_r03_1", "fresh_r03_2", "fresh_r03_3"],
            ["h1a2_b0_d1", "topology_body"],
            [
                "topology_refine_0",
                "topology_refine_1",
                "topology_refine_2",
                "topology_refine_3",
            ],
        ],
        "pre_refine_evaluated": False,
    }
    planner_launcher = Path(__file__).with_name("planner_world2.sbatch").read_text(
        encoding="utf-8"
    )
    assert "--seed-mode" not in planner_launcher
    assert "--formula-constraint-mode" not in planner_launcher
    assert "--seed \"$SEED\"" in planner_launcher
    combined = Path(__file__).with_name("combined_generation_all.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4" in combined
    assert "#SBATCH --cpus-per-task=32" in combined
    assert "#SBATCH --cpus-per-task=64" not in combined
    assert 'test "${SLURM_CPUS_PER_TASK:-0}" -eq 32' in combined
    assert "#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:8" not in combined
    assert "len(names) != 4" in combined
    assert "run_planner_wave 0 1" in combined
    assert "run_planner_wave 2 3" in combined
    assert "run_planner_wave 4" in combined
    assert 'SLURM_CPUS_PER_TASK=16' in combined
    assert combined.count('SLURM_CPUS_PER_TASK=8') == 3
    assert 'CUDA_VISIBLE_DEVICES=0,1 SLURM_CPUS_PER_TASK=16' in combined
    assert 'CUDA_VISIBLE_DEVICES=2 SLURM_CPUS_PER_TASK=8' in combined
    assert "CUDA_VISIBLE_DEVICES=0,1" in combined
    assert "CUDA_VISIBLE_DEVICES=2" in combined
    assert "CUDA_VISIBLE_DEVICES=4" not in combined
    assert "CUDA_VISIBLE_DEVICES=5" not in combined
    assert "CUDA_VISIBLE_DEVICES=6" not in combined
    assert "CUDA_VISIBLE_DEVICES=7" not in combined
    assert "assemble_generation.py" in combined
    assert "pre_model494" not in combined
    submit = Path(__file__).with_name("prepare_and_submit_once.sh").read_text(
        encoding="utf-8"
    )
    assert submit.count("sbatch --parsable") == 1
    assert "31897" in submit
    historical_rank_concat = list(range(0, 1200, 2)) + list(range(1, 1200, 2))
    assert historical_rank_concat[: protocol.DENOMINATOR] == list(range(0, 512, 2))
    refiner = config["refiner"]
    assert (
        refiner["timesteps"],
        refiner["diffusion_steps"],
        refiner["num_evals"],
        refiner["effective_batch_size"],
    ) == (1000, 800, 1, 1)
    provenance = refiner["historical_best_provenance"]
    for key in (
        "historical_wrapper_sha256",
        "current_wrapper_sha256",
        "runtime_loader_sha256",
        "diffusion_kernel_sha256",
        "refiner_seed_vector_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", provenance[key])
    assert provenance["historical_refined_counts"] == [248, 248, 248, 248]
    assert provenance["historical_clean_strict_sun_counts"] == [28, 32, 30, 30]
    assert audit_refiner_contract.diffusion_kernel_contract(
        Path(provenance["diffusion_kernel"])
    ) == {
        "time_start_is_diff_steps": True,
        "loop": "tqdm(range(diff_steps, 0, -1))",
        "exact_reverse_updates_when_diff_steps_800": 800,
    }
    assert config["topology_match"] == {
        "panel_id": "historical_best_topology_match",
        "planner": {
            "cohort_id": "retrained_seed17_world2_b4_topology_match",
            "seed": 17,
        },
        "body_process_realizations": 1,
        "refiner_process_realizations": 4,
        "refiner_array_concurrency": 2,
        "reuse_identical_body_and_proposal_graphs": True,
        "reuse_identical_refiner_seed_vector": True,
        "historical_target_strict_counts": [28, 32, 30, 30],
    }
    assert config["historical_h1a2_dlm_control"] == {
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
    assert config["inference"]["evaluated_stage"] == "post_model494_only"
    assert config["inference"]["pre_refine_role"] == "intermediate_only_not_scored"
    assert config["inference"]["exact_mcnemar_per_cohort_pre_post"] is False
    for script_name in (
        "body_refine.sbatch",
        "topology_refine.sbatch",
        "h1a2_dlm_control.sbatch",
    ):
        script = Path(__file__).with_name(script_name).read_text(encoding="utf-8")
        assert "finalize_pre" not in script
        assert "pre_model494_direct" not in script
    assert [spec["seed"] for spec in config["planner"]["cohorts"]] == [
        52021,
        62023,
        72031,
        82037,
    ]
    historical = config["historical_planner_reference"]
    historical_cohort = protocol.require_file(
        historical["cohort256"],
        historical["cohort256_sha256"],
        "historical H1-A2 world2 planner cohort",
    )
    assert len(protocol.read_jsonl(historical_cohort)) == protocol.DENOMINATOR
    assert protocol.attempt_id("R03", 3, 255, "post_model494").endswith(
        "post_model494-0255"
    )
    previous_panel = os.environ.get("H1_RECOVERY_PANEL")
    os.environ["H1_RECOVERY_PANEL"] = "topology_match"
    try:
        assert protocol.attempt_id("R03", 3, 255, "post_model494").startswith(
            "h1a2-retrained-topology-r03"
        )
    finally:
        if previous_panel is None:
            os.environ.pop("H1_RECOVERY_PANEL", None)
        else:
            os.environ["H1_RECOVERY_PANEL"] = previous_panel
    try:
        protocol.validate_frozen_cohort_row({}, repeat=0, ordinal=0)
    except RuntimeError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("compatibility sentinel must fail closed")
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "value.json"
        protocol.write_json_exclusive(target, {"ok": True})
        assert protocol.read_json(target) == {"ok": True}
    print("self_test: PASS")


if __name__ == "__main__":
    main()
