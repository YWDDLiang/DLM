#!/usr/bin/env python3
"""Freeze Planner-512 ordinals 0..255 into the factorial runtime contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
RUNTIME_ROOT = HERE / "runtime"
for location in (RUNTIME_ROOT, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from transformers import AutoTokenizer  # noqa: E402

from crystal_dlm.h1a2_factorial_contract import (  # noqa: E402
    MODEL_SAMPLED_PLAN_PROVENANCE,
    assert_planner_input_identity,
    build_factorial_ordinal_record,
    build_planner_input_contract,
    persist_model_sampled_plan,
)
from crystal_dlm.ordinal_rng import sha256_text  # noqa: E402
from protocol import (  # noqa: E402
    read_json,
    read_jsonl,
    require_hex_sha,
    require_runtime_manifest,
    require_sha,
    require_source_manifest,
    sha256_file,
    write_json_exclusive,
    write_jsonl_exclusive,
)


PLANNER_ARMS = ("P0", "Pstar")
DENOMINATOR = 256
FROZEN_RUN_CONFIG = {
    "num_samples": 512,
    "max_new_tokens": 96,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "max_atoms": 20,
    "prompt_style": "h1_rich_plan_v1",
    "include_sample_id": False,
    "seed": 17029,
    "seed_mode": "stateless_ordinal_v1",
    "rank_independent_sampling": True,
    "effective_generation_batch_size": 1,
}


def _validate_run_config(config: Mapping[str, Any]) -> None:
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in FROZEN_RUN_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Planner-512 run config changed: {mismatches}")


def _source_row_sha256(source: Mapping[str, Any]) -> str:
    return sha256_text(
        json.dumps(
            source,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def _failed_planner_attempt(
    *,
    arm: str,
    sample_idx: int,
    expected_seed: int,
    source: Mapping[str, Any],
    failure_reason: str,
    failure_message: str,
) -> dict[str, Any]:
    """Preserve one raw model attempt as a Planner-stage denominator failure."""

    return {
        "schema": "h1a2_factorial_contract_v1",
        "sample_idx": sample_idx,
        "planner_arm": arm,
        "attempt_status": "failed",
        "earliest_failure_stage": "planner",
        "failure_reason": str(failure_reason),
        "failure_message": str(failure_message),
        "raw_model_sampled_plan_text": str(source.get("raw_plan_text") or ""),
        "planner_sampling_seed": expected_seed,
        "plan_provenance": MODEL_SAMPLED_PLAN_PROVENANCE,
        "model_proposed_plan": True,
        "retry_used": False,
        "replacement_used": False,
        "repair_used": False,
        "filter_used": False,
        "rerank_used": False,
        "source_planner512_raw_row_sha256": _source_row_sha256(source),
    }


def _convert_source_attempt(
    arm: str,
    *,
    sample_idx: int,
    source: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    base_seed: int,
) -> dict[str, Any]:
    ordinal = build_factorial_ordinal_record(
        int(base_seed),
        sample_idx=sample_idx,
    )
    expected_seed = int(ordinal["planner_sampling_seed"])
    if int(source.get("planner_sampling_seed", -1)) != expected_seed:
        raise ValueError(f"{arm} ordinal {sample_idx} Planner seed changed")

    if source.get("parsed") is not True:
        return _failed_planner_attempt(
            arm=arm,
            sample_idx=sample_idx,
            expected_seed=expected_seed,
            source=source,
            failure_reason=str(source.get("reason") or "planner_parse_failed"),
            failure_message=str(source.get("message") or ""),
        )

    raw_text = str(source.get("raw_plan_text") or "")
    canonical_text = str(source.get("plan_text") or "")
    if not canonical_text:
        raise ValueError(
            f"{arm} ordinal {sample_idx} is parsed but lacks frozen plan_text"
        )

    # The historical H1-A2 parser decision is authoritative. Raw formatting
    # differences are advisory only: preserve the exact model output for
    # analysis, while the already-frozen canonical plan_text feeds the body
    # compiler. No new value is inferred, filled, retried, or substituted.
    record = persist_model_sampled_plan(
        canonical_text,
        planner_arm=arm,
        sample_idx=sample_idx,
        planner_sampling_seed=expected_seed,
        planner_input_contract=input_contract,
        max_atoms=20,
    )
    raw_contract_warning: str | None = None
    try:
        persist_model_sampled_plan(
            raw_text,
            planner_arm=arm,
            sample_idx=sample_idx,
            planner_sampling_seed=expected_seed,
            planner_input_contract=input_contract,
            max_atoms=20,
        )
    except ValueError as exc:
        raw_contract_warning = str(exc)

    record.update(
        {
            "attempt_status": "complete",
            "earliest_failure_stage": None,
            "failure_reason": None,
            "failure_message": None,
            "raw_model_sampled_plan_text": raw_text,
            "frozen_canonical_plan_text": canonical_text,
            "canonical_plan_text_source": "frozen_planner512_plan_text",
            "raw_plan_format_gate": "advisory_nonblocking",
            "raw_plan_contract_conforming": raw_contract_warning is None,
            "raw_plan_contract_warning": raw_contract_warning,
            "canonicalization_used": raw_text != canonical_text,
            "source_planner512_raw_row_sha256": _source_row_sha256(source),
        }
    )
    return record


def _convert_arm(
    arm: str,
    *,
    arm_config: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    base_seed: int,
) -> list[dict[str, Any]]:
    raw_path = require_sha(
        arm_config["raw_generations"],
        arm_config["raw_generations_sha256"],
        f"{arm} raw Planner generations",
    )
    require_sha(
        arm_config["run_config"],
        arm_config["run_config_sha256"],
        f"{arm} Planner run config",
    )
    require_sha(
        arm_config["plan_report"],
        arm_config["plan_report_sha256"],
        f"{arm} Planner report",
    )
    run_config = read_json(arm_config["run_config"])
    _validate_run_config(run_config)
    report = read_json(arm_config["plan_report"])
    if (
        report.get("status") != "complete"
        or report.get("arm") not in {arm, "P-star" if arm == "Pstar" else arm}
        or int(report.get("step", -1)) != int(arm_config["step"])
        or int(report.get("denominator", -1)) != 512
        or report.get("checkpoint_identity_sha256")
        != arm_config["checkpoint_identity_sha256"]
        or report.get("model_proposed_plan") is not True
        or report.get("sample_id_used") is not False
        or report.get("retry_replacement_repair_filter_or_rerank_used") is not False
    ):
        raise ValueError(f"{arm} Planner report is not the frozen endpoint")

    raw_rows = read_jsonl(raw_path)
    if (
        len(raw_rows) != 512
        or [int(row.get("sample_idx", -1)) for row in raw_rows]
        != list(range(512))
    ):
        raise ValueError(f"{arm} Planner raw ledger is not ordinals 0..511")

    attempts: list[dict[str, Any]] = []
    for sample_idx, source in enumerate(raw_rows[:DENOMINATOR]):
        attempts.append(
            _convert_source_attempt(
                arm,
                sample_idx=sample_idx,
                source=source,
                input_contract=input_contract,
                base_seed=base_seed,
            )
        )
    return attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    args = parser.parse_args()

    execution_sha = require_hex_sha(
        args.execution_manifest_sha256,
        "execution source manifest",
    )
    source = args.source_dir.resolve()
    project = args.project_root.resolve()
    require_source_manifest(source, execution_sha)
    require_runtime_manifest(project, source)
    config = read_json(args.config.resolve())
    if (
        config.get("status") != "user_authorized_diagnostic_execution"
        or config["panel"].get("raw_plan_format_gate")
        != "advisory_nonblocking"
        or config["panel"].get(
            "historical_h1a2_parsed_flag_is_authoritative"
        )
        is not True
        or config["panel"].get("frozen_canonical_plan_text_feeds_body")
        is not True
        or config["panel"].get("raw_model_text_preserved") is not True
        or config["decision_firewall"].get("diagnostic_only") is not True
        or config["decision_firewall"].get("formal_g3") is not False
        or config["decision_firewall"].get("automatic_downstream") is not False
    ):
        raise ValueError("diagnostic authorization firewall changed")
    authorization = read_json(source / config["authorization"]["path"])
    authorization_scope = authorization.get("scope") or {}
    authorization_firewall = authorization.get("decision_firewall") or {}
    if (
        authorization.get("status") != "user_authorized"
        or authorization_scope.get("crystal_generation") is not True
        or authorization_scope.get(
            "diffusion_refinement_required_for_every_arm"
        )
        is not True
        or int(authorization_scope.get("diffusion_reverse_steps", -1)) != 800
        or authorization_scope.get("frozen_sun_evaluation") is not True
        or authorization_scope.get("raw_plan_format_gate")
        != "advisory_nonblocking"
        or authorization_scope.get(
            "historical_h1a2_parsed_flag_is_authoritative"
        )
        is not True
        or authorization_scope.get("frozen_canonical_plan_text_feeds_body")
        is not True
        or authorization_scope.get(
            "raw_model_text_and_format_warning_preserved"
        )
        is not True
        or authorization_firewall.get("formal_g3") is not False
        or authorization_firewall.get("automatic_downstream") is not False
    ):
        raise ValueError("bundled authorization scope or firewall changed")

    planner_source = config["planner_source"]
    terminal_path = require_sha(
        planner_source["terminal_report"]["path"],
        planner_source["terminal_report"]["sha256"],
        "Planner-512 terminal report",
    )
    terminal = read_json(terminal_path)
    if (
        terminal.get("status") != "complete"
        or terminal.get("decision")
        != planner_source["terminal_report"]["decision"]
        or terminal.get("Pstar_selected") is not False
        or terminal.get("automatic_downstream") is not False
    ):
        raise ValueError("post-stop Planner terminal identity changed")

    require_sha(
        config["refiner"]["checkpoint"],
        config["refiner"]["checkpoint_sha256"],
        "frozen diffusion refiner",
    )
    for label, identity in config["sun"].items():
        if not isinstance(identity, Mapping) or "path" not in identity:
            continue
        location = Path(identity["path"])
        if not location.is_absolute():
            location = project / location
        require_sha(location, identity["sha256"], f"frozen S.U.N. asset {label}")

    tokenizer = AutoTokenizer.from_pretrained(
        planner_source["model_path"],
        trust_remote_code=True,
        local_files_only=True,
    )
    contracts = {
        arm: build_planner_input_contract(
            tokenizer,
            planner_arm=arm,
            checkpoint_sha256=planner_source["arms"][arm][
                "checkpoint_identity_sha256"
            ],
        )
        for arm in PLANNER_ARMS
    }
    assert_planner_input_identity(contracts["P0"], contracts["Pstar"])

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    counts: dict[str, dict[str, int]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for arm in PLANNER_ARMS:
        attempts = _convert_arm(
            arm,
            arm_config=planner_source["arms"][arm],
            input_contract=contracts[arm],
            base_seed=int(config["panel"]["planner_base_seed"]),
        )
        arm_dir = output / arm
        arm_dir.mkdir()
        attempts_path = arm_dir / "planner_attempts.jsonl"
        contract_path = arm_dir / "planner_input_contract.json"
        write_jsonl_exclusive(attempts_path, attempts)
        write_json_exclusive(contract_path, contracts[arm])
        counts[arm] = {
            "attempts": len(attempts),
            "complete": sum(
                row.get("attempt_status") == "complete" for row in attempts
            ),
            "failed": sum(row.get("attempt_status") == "failed" for row in attempts),
        }
        identities[arm] = {
            "planner_attempts": {
                "path": str(attempts_path),
                "sha256": sha256_file(attempts_path),
            },
            "planner_input_contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
        }
    manifest = {
        "schema": "h1a2_v3_poststop_sun256_plan_ledger_v1",
        "status": "complete",
        "attempts_per_arm": DENOMINATOR,
        "ordinals": "0..255",
        "planner_base_seed": int(config["panel"]["planner_base_seed"]),
        "counts": counts,
        "identities": identities,
        "planner_input_identity_equal_except_checkpoint": True,
        "model_proposed_plan": True,
        "reused_frozen_planner512": True,
        "retry_replacement_repair_filter_rerank": False,
        "formal_g3": False,
        "automatic_downstream": False,
        "execution_manifest_sha256": execution_sha,
    }
    write_json_exclusive(output / "ledger_manifest.json", manifest)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
