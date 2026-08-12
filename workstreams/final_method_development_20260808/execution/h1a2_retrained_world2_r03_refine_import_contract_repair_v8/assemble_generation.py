#!/usr/bin/env python3
"""Audit post-refine recovery panels and freeze official-S query inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from protocol import (
    DENOMINATOR,
    canonical_sha256,
    read_json,
    read_jsonl,
    require_source_manifest,
    sha256_file,
    validate_config,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def normalized_chemsys(structure_dict: dict[str, Any]) -> str:
    structure = Structure.from_dict(structure_dict)
    return "-".join(sorted(element.symbol for element in structure.composition.elements))


def validate_post_cell(
    *,
    cell: Path,
    panel: str,
    cell_id: str,
    cohort_id: str,
    cohort_index: int | None,
    process_repeat: int | None,
    expected_body: str,
    expected_schedule: str,
    wanted: set[str],
) -> tuple[dict[str, Any], list[tuple[int, int]], dict[str, Any]]:
    generation_dir = cell / "post_model494" / "generation"
    direct_dir = cell / "post_model494" / "direct"
    generation_path = generation_dir / "generation.jsonl"
    rows = read_jsonl(generation_path)
    generation_report = read_json(generation_dir / "generation_report.json")
    direct_report = read_json(direct_dir / "report.json")
    direct_attempts = read_jsonl(direct_dir / "attempt_metrics.jsonl")
    if (
        len(rows) != DENOMINATOR
        or [int(row.get("ordinal", -1)) for row in rows]
        != list(range(DENOMINATOR))
        or len({str(row.get("attempt_id")) for row in rows}) != DENOMINATOR
        or int(generation_report.get("attempts", -1)) != DENOMINATOR
        or generation_report.get("stage") != "post_model494"
        or generation_report.get("retry_replacement_repair_filter_rerank") is not False
        or int(direct_report.get("attempts", -1)) != DENOMINATOR
        or len(direct_attempts) != DENOMINATOR
    ):
        raise ValueError(f"post-refine all-attempt contract changed: {cell_id}")
    successes = 0
    for row in rows:
        if row.get("status") != "succeeded":
            continue
        structure = row.get("structure")
        if not isinstance(structure, dict):
            raise ValueError(f"successful row lacks structure: {cell_id}")
        wanted.add(normalized_chemsys(structure))
        successes += 1
    seed_vector = [
        (int(row["body_noise_seed"]), int(row["refiner_noise_seed"]))
        for row in rows
    ]
    stage_report = {
        "generation_report": generation_report,
        "direct_report": direct_report,
        "generation_sha256": sha256_file(generation_path),
        "direct_attempts_sha256": sha256_file(direct_dir / "attempt_metrics.jsonl"),
        "generation_succeeded": successes,
        "generation_failed": DENOMINATOR - successes,
    }
    evaluation_cell = {
        "panel": panel,
        "cell_id": cell_id,
        "cohort_id": cohort_id,
        "cohort_index": cohort_index,
        "process_repeat": process_repeat,
        "stage": "post_model494",
        "planner": "P0",
        "body": expected_body,
        "schedule": expected_schedule,
        "generation_jsonl": str(generation_path),
        "generation_sha256": sha256_file(generation_path),
        "direct_report": str(direct_dir / "report.json"),
        "direct_report_sha256": sha256_file(direct_dir / "report.json"),
        "attempts": DENOMINATOR,
    }
    return stage_report, seed_vector, evaluation_cell


def validate_current_refinement(
    *, cell: Path, expected_repeat: int, expected_arm: str = "R03"
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    rows = read_jsonl(cell / "refinement" / "refinement_attempts.jsonl")
    report = read_json(cell / "refinement" / "refinement_metrics.json")
    if (
        len(rows) != DENOMINATOR
        or [int(row.get("ordinal", -1)) for row in rows]
        != list(range(DENOMINATOR))
        or {str(row.get("arm")) for row in rows} != {expected_arm}
        or {int(row.get("repeat", -1)) for row in rows} != {expected_repeat}
        or int(report.get("all_attempt_denominator", -1)) != DENOMINATOR
        or int(report.get("diff_steps", -1)) != 800
        or int(report.get("num_evals", -1)) != 1
        or int(report.get("effective_batch_size", -1)) != 1
    ):
        raise ValueError(f"refinement contract changed: {cell}")
    vector = [
        (int(row["body_noise_seed"]), int(row["refiner_noise_seed"]))
        for row in rows
    ]
    return report, vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    source = args.source_dir.resolve()
    require_source_manifest(source, args.source_manifest_sha256)
    config = read_json(source / "CONFIG.json")
    validate_config(config)
    continuation = config["continuation"]
    planner_root = Path(continuation["upstream_run_root"]).resolve()
    planner_terminal_path = planner_root / "planner_terminal_report.json"
    planner_terminal = read_json(planner_terminal_path)
    if (
        sha256_file(planner_terminal_path)
        != continuation["required_planner_terminal_sha256"]
        or not (planner_root / "status" / "planner_assembly_SUCCESS").is_file()
        or not (run_root / "status" / "upstream_planner_evidence_SUCCESS").is_file()
        or planner_terminal.get("status") != "complete"
        or planner_terminal.get("ok") is not True
        or int(planner_terminal.get("cohort_count", -1)) != 5
        or int(planner_terminal.get("fresh_cohort_count", -1)) != 4
        or int(planner_terminal.get("topology_match_cohort_count", -1)) != 1
        or int(planner_terminal.get("world_size", -1)) != 2
    ):
        raise ValueError("planner terminal contract changed")
    deep_distribution = planner_terminal.get("deep_distribution_audit") or {}
    deep_distribution_path = planner_root / "planner_distribution_deep_audit.json"
    topology_planner = planner_terminal.get("topology_match_audit") or {}
    topology_planner_path = planner_root / "planner_topology_match_audit.json"
    if (
        sha256_file(deep_distribution_path)
        != continuation["required_planner_distribution_sha256"]
        or sha256_file(topology_planner_path)
        != continuation["required_planner_topology_sha256"]
        or deep_distribution.get("path") != str(deep_distribution_path)
        or deep_distribution.get("sha256") != sha256_file(deep_distribution_path)
        or int(deep_distribution.get("fresh_cohort_count", -1)) != 4
        or deep_distribution.get("reference_cohort_id")
        != "h1a2_original_seed17_world2_b4"
        or topology_planner.get("path") != str(topology_planner_path)
        or topology_planner.get("sha256") != sha256_file(topology_planner_path)
    ):
        raise ValueError("planner distribution/topology evidence changed")

    refiner_contract_path = run_root / "status" / "refiner_contract_report.json"
    refiner_contract = read_json(refiner_contract_path)
    if (
        not (run_root / "status" / "refiner_contract_SUCCESS").is_file()
        or refiner_contract.get("status") != "pass"
        or refiner_contract.get("scientific_wrapper_contract")
        != {
            "model_timesteps": 1000,
            "reverse_updates": 800,
            "effective_batch_size": 1,
            "shuffle": False,
            "seed_calls": ["manual_seed", "manual_seed_all", "seed"],
        }
        or refiner_contract.get("diffusion_kernel_contract", {}).get(
            "exact_reverse_updates_when_diff_steps_800"
        )
        != 800
        or refiner_contract.get("checkpoint_rehashed") is not False
        or int(
            (refiner_contract.get("checkpoint_identity") or {}).get("bytes", -1)
        )
        != 147645242
    ):
        raise ValueError("historical-best refiner contract audit changed")

    wanted: set[str] = set()
    evaluation_cells: list[dict[str, Any]] = []
    all_attempt_ids: set[str] = set()
    all_seed_vectors: list[list[tuple[int, int]]] = []
    fresh_reports: list[dict[str, Any]] = []

    for index, cohort_id in enumerate(config["downstream_cohorts"]):
        cell = run_root / "cells" / cohort_id
        marker = run_root / "status" / f"cell_{index}_generation_SUCCESS"
        exit_code = run_root / "status" / f"cell_{index}_exit_code.txt"
        if not marker.is_file() or not exit_code.is_file() or exit_code.read_text().strip() != "0":
            raise RuntimeError(f"fresh cell is not terminal-success: {cohort_id}")
        cohort_dir = planner_root / "planner" / cohort_id / "frozen"
        cohort_path = cohort_dir / "cohort256.jsonl"
        cohort_report = read_json(cohort_dir / "cohort_report.json")
        cohort_sha256 = sha256_file(cohort_path)
        if (
            cohort_report.get("cohort_id") != cohort_id
            or cohort_sha256
            != continuation["required_cohort_sha256"][cohort_id]
            or cohort_report.get("cohort256_sha256") != cohort_sha256
            or int(cohort_report.get("attempts", -1)) != DENOMINATOR
            or int(cohort_report.get("world_size", -1)) != 2
            or int(cohort_report.get("batch_size_per_rank", -1)) != 4
        ):
            raise ValueError(f"fresh planner cohort report changed: {cohort_id}")
        refinement_report, refinement_vector = validate_current_refinement(
            cell=cell, expected_repeat=index
        )
        stage_report, stage_vector, evaluation_cell = validate_post_cell(
            cell=cell,
            panel="fresh_cohort",
            cell_id=f"fresh_{index}",
            cohort_id=cohort_id,
            cohort_index=index,
            process_repeat=index,
            expected_body="R03",
            expected_schedule="D2_SAFE_AXIS",
            wanted=wanted,
        )
        if stage_vector != refinement_vector:
            raise ValueError(f"fresh post-refine seed vector changed: {cohort_id}")
        rows = read_jsonl(Path(evaluation_cell["generation_jsonl"]))
        ids = {str(row["attempt_id"]) for row in rows}
        if ids & all_attempt_ids:
            raise ValueError("attempt IDs overlap across evaluation cells")
        all_attempt_ids.update(ids)
        all_seed_vectors.append(stage_vector)
        evaluation_cells.append(evaluation_cell)
        fresh_reports.append(
            {
                "cohort_id": cohort_id,
                "cohort_index": index,
                "planner": cohort_report,
                "refinement_report": refinement_report,
                "post_model494": stage_report,
            }
        )

    projection_path = run_root / "topology_match" / "process_inputs" / "projection_report.json"
    projection = read_json(projection_path)
    if (
        not (run_root / "status" / "topology_body_SUCCESS").is_file()
        or not (run_root / "topology_match" / "process_inputs" / "projection_SUCCESS").is_file()
        or projection.get("status") != "complete"
        or int(projection.get("body_process_realizations", -1)) != 1
        or int(projection.get("refiner_process_realizations", -1)) != 4
        or projection.get("identical_body_and_proposal_graphs") is not True
        or projection.get("identical_body_seed_vector") is not True
        or projection.get("identical_refiner_seed_vector") is not True
        or len(projection.get("processes") or []) != 4
    ):
        raise ValueError("topology process projection contract changed")
    topology_reports: list[dict[str, Any]] = []
    for repeat in range(4):
        marker = run_root / "status" / f"topology_refine_{repeat}_SUCCESS"
        exit_code = run_root / "status" / f"topology_refine_{repeat}_exit_code.txt"
        if not marker.is_file() or not exit_code.is_file() or exit_code.read_text().strip() != "0":
            raise RuntimeError(f"topology refiner process is not terminal-success: {repeat}")
        cell = run_root / "topology_match" / "processes" / f"repeat_{repeat}"
        refinement_report, refinement_vector = validate_current_refinement(
            cell=cell, expected_repeat=repeat
        )
        run_config = read_json(cell / "refinement" / "run_config.json")
        expected_input = run_root / "topology_match" / "process_inputs" / f"repeat_{repeat}"
        if (
            run_config.get("body_attempts") != str(expected_input / "body_attempts.jsonl")
            or run_config.get("proposal_graphs") != str(expected_input / "proposal_graphs.pt")
            or int(run_config.get("repeat", -1)) != repeat
        ):
            raise ValueError("topology refiner process input identity changed")
        stage_report, stage_vector, evaluation_cell = validate_post_cell(
            cell=cell,
            panel="topology_match",
            cell_id=f"topology_repeat_{repeat}",
            cohort_id="retrained_seed17_world2_b4_topology_match",
            cohort_index=None,
            process_repeat=repeat,
            expected_body="R03",
            expected_schedule="D2_SAFE_AXIS",
            wanted=wanted,
        )
        if stage_vector != refinement_vector:
            raise ValueError("topology post-refine seed vector changed")
        rows = read_jsonl(Path(evaluation_cell["generation_jsonl"]))
        ids = {str(row["attempt_id"]) for row in rows}
        if ids & all_attempt_ids:
            raise ValueError("attempt IDs overlap across evaluation cells")
        all_attempt_ids.update(ids)
        all_seed_vectors.append(stage_vector)
        evaluation_cells.append(evaluation_cell)
        topology_reports.append(
            {
                "process_repeat": repeat,
                "refinement_report": refinement_report,
                "post_model494": stage_report,
            }
        )

    h1a2_root = run_root / "historical_h1a2_dlm_control"
    h1a2_exit = run_root / "status" / "h1a2_dlm_control_exit_code.txt"
    if (
        not (run_root / "status" / "h1a2_dlm_control_SUCCESS").is_file()
        or not h1a2_exit.is_file()
        or h1a2_exit.read_text().strip() != "0"
    ):
        raise RuntimeError("historical H1-A2 DLM control is not terminal-success")
    input_adapter = read_json(h1a2_root / "inputs" / "input_adapter_report.json")
    body_metrics = read_json(h1a2_root / "body" / "body_metrics.json")
    body_run = read_json(h1a2_root / "body" / "run_config.json")
    h1a2_refinement = read_json(h1a2_root / "refinement" / "refinement_metrics.json")
    h1a2_refiner_run = read_json(h1a2_root / "refinement" / "run_config.json")
    h1a2_refinement_rows = read_jsonl(
        h1a2_root / "refinement" / "refinement_attempts.jsonl"
    )
    if (
        input_adapter.get("status") != "complete"
        or input_adapter.get("body_policy")
        != "historical_H1A2_B0_d1_exact_plan_schedule"
        or int(body_metrics.get("all_attempt_denominator", -1)) != DENOMINATOR
        or body_run.get("factorial_arm") != "M00"
        or body_run.get("body_arm") != "B0"
        or body_run.get("generation_policy") != "d1"
        or int(body_run.get("world_size", -1)) != 2
        or int(body_run.get("effective_generation_batch_size", -1)) != 1
        or int(h1a2_refinement.get("all_attempt_denominator", -1)) != DENOMINATOR
        or int(h1a2_refinement.get("diff_steps", -1)) != 800
        or int(h1a2_refinement.get("effective_batch_size", -1)) != 1
        or int(h1a2_refiner_run.get("world_size", -1)) != 2
        or len(h1a2_refinement_rows) != DENOMINATOR
    ):
        raise ValueError("historical H1-A2 B0/D1 control contract changed")
    h1a2_cell_report, h1a2_vector, h1a2_evaluation = validate_post_cell(
        cell=h1a2_root,
        panel="historical_h1a2_dlm_control",
        cell_id="h1a2_b0_d1_once",
        cohort_id="retrained_seed17_world2_b4_topology_match",
        cohort_index=None,
        process_repeat=0,
        expected_body="B0",
        expected_schedule="D1_EXACT_PLAN",
        wanted=wanted,
    )
    h1a2_refinement_vector = [
        (int(row["body_sampling_seed"]), int(row["refiner_sampling_seed"]))
        for row in h1a2_refinement_rows
    ]
    if h1a2_vector != h1a2_refinement_vector:
        raise ValueError("H1-A2 post-refine seed vector changed")
    rows = read_jsonl(Path(h1a2_evaluation["generation_jsonl"]))
    ids = {str(row["attempt_id"]) for row in rows}
    if ids & all_attempt_ids:
        raise ValueError("attempt IDs overlap across evaluation cells")
    all_attempt_ids.update(ids)
    all_seed_vectors.append(h1a2_vector)
    evaluation_cells.append(h1a2_evaluation)

    if len(fresh_reports) != 4 or len(topology_reports) != 4 or len(evaluation_cells) != 9:
        raise ValueError("post-refine recovery panel shape changed")
    if any(vector != all_seed_vectors[0] for vector in all_seed_vectors[1:]):
        raise ValueError("seed vectors differ across controlled panels")

    wanted_rows = [
        {"query_index": index, "chemsys": value, "elements": value.split("-")}
        for index, value in enumerate(sorted(wanted))
    ]
    inputs = run_root / "official_sun_inputs"
    if inputs.exists():
        raise FileExistsError(inputs)
    inputs.mkdir()
    write_jsonl_exclusive(inputs / "wanted_all_reconstructed_chemsys.jsonl", wanted_rows)
    write_jsonl_exclusive(inputs / "evaluation_cells.jsonl", evaluation_cells)
    input_manifest = {
        "schema": "h1_plan_recovery_official_sun_input_manifest_v3",
        "recovery_variant": "h1a2_epoch2_exact_retrain_postrefine_panels_v1",
        "evaluated_stage": "post_model494_only",
        "pre_refine_role": "intermediate_only_not_scored",
        "evaluation_cell_count": len(evaluation_cells),
        "fresh_r03_cohort_count": len(fresh_reports),
        "topology_r03_refiner_process_count": len(topology_reports),
        "historical_h1a2_dlm_control_count": 1,
        "attempts_per_cell": DENOMINATOR,
        "wanted_chemsys_count": len(wanted_rows),
        "wanted_chemsys_sha256": canonical_sha256(wanted_rows),
        "evaluation_cells_sha256": canonical_sha256(evaluation_cells),
        "same_seed_vector_across_all_controlled_panels": True,
        "seed_vector_sha256": canonical_sha256(all_seed_vectors[0]),
        "training_adapter_sha256": config["training_upstream"]["adapter_sha256"],
        "upstream_planner_root": str(planner_root),
        "upstream_planner_terminal_sha256": sha256_file(planner_terminal_path),
        "planner_deep_distribution_audit": deep_distribution,
        "planner_topology_match_audit": topology_planner,
        "historical_best_refiner_contract": {
            "path": str(refiner_contract_path),
            "sha256": sha256_file(refiner_contract_path),
            "report": refiner_contract,
        },
        "topology_process_projection": {
            "path": str(projection_path),
            "sha256": sha256_file(projection_path),
            "report": projection,
        },
        "panels": {
            "fresh_r03": fresh_reports,
            "topology_r03": topology_reports,
            "historical_h1a2_dlm_control": {
                "input_adapter": input_adapter,
                "body_metrics": body_metrics,
                "refinement_metrics": h1a2_refinement,
                "post_model494": h1a2_cell_report,
            },
        },
        "official_stability_pending": True,
        "retry_replacement_repair_filter_rerank": False,
    }
    write_json_exclusive(inputs / "input_manifest.json", input_manifest)
    (inputs / "inputs_SUCCESS").touch(exist_ok=False)
    terminal = {
        **input_manifest,
        "generation_terminal_variant": "h1a2_retrained_postrefine_generation_terminal_v1",
        "status": "complete",
        "ok": True,
        "source_manifest_sha256": args.source_manifest_sha256,
    }
    write_json_exclusive(run_root / "generation_terminal_report.json", terminal)
    (run_root / "status" / "generation_assembly_SUCCESS").touch(exist_ok=False)
    print(json.dumps(terminal, sort_keys=True))


if __name__ == "__main__":
    main()
