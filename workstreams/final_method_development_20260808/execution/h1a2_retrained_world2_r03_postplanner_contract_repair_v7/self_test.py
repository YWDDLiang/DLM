#!/usr/bin/env python3
"""Offline regression tests for the immutable post-planner V7 repair."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import audit_body_contract
import audit_refiner_contract
import protocol


def main() -> None:
    source = Path(__file__).resolve().parent
    config = json.loads((source / "CONFIG.json").read_text(encoding="utf-8"))
    protocol.validate_config(config)
    assert protocol.DENOMINATOR == 256
    assert protocol.PLANNER_RAW_ATTEMPTS == 1200

    continuation = config["continuation"]
    assert continuation["upstream_slurm_job_id"] == "31900"
    assert continuation["planner_sampling_rerun"] is False
    assert continuation["planner_outputs_byte_reused"] is True
    assert continuation["generation_outputs_reused"] is False
    assert continuation["slurm_job_ordinal_since_v5"] == 2
    assert continuation["remaining_official_slurm_jobs"] == 1
    preparation_history = config["preparation_history"]
    assert preparation_history["aborted_before_submission"] is True
    assert preparation_history["abort_reason"] == (
        "user_requested_no_large_artifact_rehash"
    )
    assert preparation_history["large_artifact_rehash"] is False
    for key in (
        "upstream_source_manifest_sha256",
        "required_planner_terminal_sha256",
        "required_planner_distribution_sha256",
        "required_planner_topology_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", continuation[key])
    assert len(continuation["required_cohort_sha256"]) == 5
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value)
        for value in continuation["required_cohort_sha256"].values()
    )

    body = config["body"]
    assert set(audit_body_contract.REQUIRED_BODY_KEYS).issubset(body)
    assert body["adapter_file"] == "adapter_model.safetensors"
    assert body["adapter_expected_bytes"] == 6391016776
    assert body["adapter_sha256"] == (
        "5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d"
    )
    assert body["tokenizer_size"] == 128830
    assert body["large_artifact_rehash"] is False

    scheduler = config["scheduler_repair"]
    assert scheduler["variant"] == "postplanner_single_slurm_job_max4_a800_32cpu_v7"
    assert scheduler["slurm_job_count"] == 1
    assert scheduler["planner_waves"] == []
    assert scheduler["requested_a800_gpus"] == 4
    assert scheduler["requested_cpus"] == 32
    assert scheduler["pre_refine_evaluated"] is False
    assert config["authorization"]["planner_sampling"] is False

    combined = (source / "combined_generation_all.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4" in combined
    assert "#SBATCH --cpus-per-task=32" in combined
    assert 'test "${SLURM_CPUS_PER_TASK:-0}" -eq 32' in combined
    assert "run_planner_wave" not in combined
    assert "planner_world2.sbatch" not in combined
    assert "STAGE upstream_planner_evidence_audit" in combined
    assert "run_fresh_r03" in combined
    assert "run_h1a2_and_topology_body" in combined
    assert "run_topology_refiners" in combined
    assert "assemble_generation.py" in combined
    assert "pre_model494" not in combined
    assert combined.count("SLURM_CPUS_PER_TASK=8") == 3
    assert "CUDA_VISIBLE_DEVICES=0,1 SLURM_CPUS_PER_TASK=16" in combined
    assert "CUDA_VISIBLE_DEVICES=2 SLURM_CPUS_PER_TASK=8" in combined
    for forbidden in ("CUDA_VISIBLE_DEVICES=4", "CUDA_VISIBLE_DEVICES=5"):
        assert forbidden not in combined

    submit = (source / "prepare_and_submit_once.sh").read_text(encoding="utf-8")
    assert submit.count("sbatch --parsable") == 1
    assert "31900" in submit
    assert "audit_body_contract.py" in submit
    assert "planner_world2.sbatch" not in submit
    assert "preparation_ABORTED_BY_USER_NO_LARGE_REHASH" in submit

    body_audit = (source / "audit_body_contract.py").read_text(encoding="utf-8")
    refiner_audit = (source / "audit_refiner_contract.py").read_text(
        encoding="utf-8"
    )
    assert "sha256_file(adapter)" not in body_audit
    assert '"checkpoint": require_hash' not in refiner_audit
    assert '"checkpoint_rehashed": False' in refiner_audit

    assert not (source / "planner_world2.sbatch").exists()
    assert not (source / "planner_assembly.sbatch").exists()
    assert not (source / "generation_assembly.sbatch").exists()

    refiner = config["refiner"]
    assert refiner["checkpoint_expected_bytes"] == 147645242
    assert refiner["large_artifact_rehash"] is False
    assert (
        refiner["timesteps"],
        refiner["diffusion_steps"],
        refiner["num_evals"],
        refiner["effective_batch_size"],
    ) == (1000, 800, 1, 1)
    provenance = refiner["historical_best_provenance"]
    assert provenance["historical_refined_counts"] == [248, 248, 248, 248]
    assert provenance["historical_clean_strict_sun_counts"] == [28, 32, 30, 30]
    assert audit_refiner_contract.diffusion_kernel_contract(
        Path(provenance["diffusion_kernel"])
    )["exact_reverse_updates_when_diff_steps_800"] == 800

    for script_name in (
        "body_refine.sbatch",
        "topology_body.sbatch",
        "topology_refine.sbatch",
        "h1a2_dlm_control.sbatch",
    ):
        script = (source / script_name).read_text(encoding="utf-8")
        upstream_run_id = Path(continuation["upstream_run_root"]).name
        assert upstream_run_id in script or script_name == "topology_refine.sbatch"
        assert config["run_root"] in script
        assert "finalize_pre" not in script
        assert "pre_model494_direct" not in script
        assert 'sha256sum "$BODY_CHECKPOINT/adapter_model.safetensors"' not in script

    historical_rank_concat = list(range(0, 1200, 2)) + list(range(1, 1200, 2))
    assert historical_rank_concat[: protocol.DENOMINATOR] == list(range(0, 512, 2))
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

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "value.json"
        protocol.write_json_exclusive(target, {"ok": True})
        assert protocol.read_json(target) == {"ok": True}
    print("self_test: PASS")


if __name__ == "__main__":
    main()
