#!/usr/bin/env python3
"""Freeze the paired-256 panel and every execution asset before GPU work."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT_FALLBACK = HERE.parents[3]
for location in (PROJECT_ROOT_FALLBACK, HERE):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from crystal_dlm.r5_plan_state import build_body_prompt  # noqa: E402
from crystal_dlm.wqcodiff.contracts import SeedDeriver  # noqa: E402
from protocol import (  # noqa: E402
    ARM_ORDER,
    canonical_sha256,
    plan_body_eligible,
    read_json,
    read_jsonl,
    require_hex_sha,
    require_runtime_manifest,
    require_sha,
    require_source_manifest,
    resolve_project_path,
    sha256_file,
    write_json_exclusive,
    write_jsonl_exclusive,
)

FULL_HASH_MAX_BYTES = 64 * 1024 * 1024


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verified_identity(path: Path, expected: str, label: str) -> dict[str, Any]:
    location = path.resolve()
    if not location.is_file():
        raise FileNotFoundError(location)
    expected_sha = require_hex_sha(expected, label)
    observed_sha = sha256_file(location)
    if observed_sha != expected_sha:
        raise ValueError(
            f"{label} changed: expected={expected_sha}, "
            f"observed={observed_sha}, path={location}"
        )
    return {
        "path": str(location),
        "bytes": location.stat().st_size,
        "sha256": observed_sha,
    }


def _directory_identity(
    path: Path, *, precomputed_sha256: dict[str, str] | None = None
) -> dict[str, Any]:
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    known = precomputed_sha256 or {}
    files = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        relative = str(item.relative_to(root))
        size = item.stat().st_size
        digest = known.get(relative)
        if digest is None:
            digest = sha256_file(item) if size <= FULL_HASH_MAX_BYTES else None
        else:
            require_hex_sha(digest, f"precomputed asset SHA256 for {relative}")
        files.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": digest,
            }
        )
    if not files:
        raise ValueError(f"asset directory is empty: {root}")
    return {
        "path": str(root),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "hash_policy": (
            "sha256 for files <=64 MiB plus separately verified precomputed assets; "
            "path and byte inventory for other weight shards"
        ),
        "canonical_manifest_sha256": canonical_sha256(files),
    }


def _require_plan_rows(
    path: Path, *, expected_sha256: str, label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require_sha(path, expected_sha256, f"{label} frozen Planner output")
    rows = read_jsonl(path)
    if len(rows) != 512:
        raise ValueError(f"{label} Planner output must retain exactly 512 rows")
    indices = [int(row.get("sample_idx", -1)) for row in rows]
    if indices != list(range(512)):
        raise ValueError(f"{label} Planner sample_idx order changed")
    return rows, rows[:256]


def _validate_terminal(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    source = config["source_plan_run"]
    path = resolve_project_path(project_root, source["corrected_terminal_report"])
    require_sha(
        path,
        source["corrected_terminal_sha256"],
        "corrected JointChem terminal report",
    )
    terminal = read_json(path)
    if terminal.get("decision") != source["corrected_decision"]:
        raise ValueError("corrected JointChem terminal decision changed")
    if terminal.get("selected") is not None:
        raise ValueError("formal JointChem selection is no longer null")
    if (
        terminal.get("initial_adapter_sha256")
        != source["arms"]["P0"]["planner_adapter_sha256"]
    ):
        raise ValueError("frozen P0 Planner adapter identity changed")
    for key in (
        "automatic_crystal_evaluation_authorized",
        "automatic_downstream_authorized",
    ):
        if terminal.get(key) is not False:
            raise ValueError(f"corrected JointChem firewall changed: {key}")
    return {"terminal_report": _identity(path), "decision": terminal["decision"]}


def _validate_p1_selection(
    config: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    spec = config["source_plan_run"]["arms"]["P1"]
    path = resolve_project_path(project_root, spec["checkpoint_selection"])
    selection_report = read_json(path)
    selection = selection_report.get("selected")
    if not isinstance(selection, dict):
        raise ValueError("frozen P1 checkpoint selection is missing")
    checkpoint_dir = Path(selection["checkpoint_dir"]).resolve()
    expected_checkpoint_dir = resolve_project_path(
        project_root, spec["planner_checkpoint"]
    )
    margins = selection.get("margins") or {}
    if (
        int(selection.get("step", -1)) != int(spec["planner_step"])
        or checkpoint_dir != expected_checkpoint_dir
        or selection.get("checkpoint_manifest_sha256")
        != spec["planner_checkpoint_manifest_sha256"]
        or selection.get("eligible") is not True
        or selection.get("nll_noninferior") is not True
        or float(margins.get("chemistry", 0.0)) <= 0.0
        or float(margins.get("joint", 0.0)) <= 0.0
    ):
        raise ValueError("frozen P1 selected Planner checkpoint identity changed")
    return {
        "selection": _identity(path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_manifest_sha256": selection["checkpoint_manifest_sha256"],
        "step": int(selection["step"]),
    }


def _validate_plan_reports(
    config: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    source = config["source_plan_run"]
    identities: dict[str, Any] = {}
    for arm in ARM_ORDER:
        spec = source["arms"][arm]
        path = resolve_project_path(project_root, spec["plan_report"])
        report = read_json(path)
        expected_checkpoint_identity = (
            spec["planner_adapter_sha256"]
            if arm == "P0"
            else spec["planner_checkpoint_manifest_sha256"]
        )
        if (
            report.get("execution_manifest_sha256")
            != source["execution_manifest_sha256"]
            or report.get("initial_adapter_sha256")
            != source["arms"]["P0"]["planner_adapter_sha256"]
            or report.get("checkpoint_identity_sha256") != expected_checkpoint_identity
        ):
            raise ValueError(f"{arm} frozen Planner report identity changed")
        identities[arm] = _identity(path)
    return identities


def _freeze_assets(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    body = config["body"]
    parent = config["parent_refiner"]
    sun = config["sun"]
    body_checkpoint = resolve_project_path(project_root, body["checkpoint"])
    body_adapter = resolve_project_path(project_root, body["adapter_file"])
    if body_adapter.stat().st_size != int(body["adapter_bytes"]):
        raise ValueError("R5-C body adapter byte count changed")
    expected_body_adapter_sha = require_hex_sha(
        body["adapter_sha256"], "R5-C body adapter"
    )
    observed_body_adapter_sha = sha256_file(body_adapter)
    if observed_body_adapter_sha != expected_body_adapter_sha:
        raise ValueError(
            "R5-C body adapter changed: "
            f"expected={expected_body_adapter_sha}, "
            f"observed={observed_body_adapter_sha}, path={body_adapter}"
        )
    try:
        body_adapter_relative = str(body_adapter.relative_to(body_checkpoint))
    except ValueError as exc:
        raise ValueError(
            "R5-C adapter must be inside the frozen body checkpoint"
        ) from exc
    parent_checkpoint = resolve_project_path(project_root, parent["checkpoint"])
    parent_identity = _verified_identity(
        parent_checkpoint,
        parent["checkpoint_sha256"],
        "CrysLLMGen parent checkpoint",
    )
    assets: dict[str, Any] = {
        "body_base_model": _directory_identity(
            resolve_project_path(project_root, body["base_model"])
        ),
        "body_checkpoint": _directory_identity(
            body_checkpoint,
            precomputed_sha256={
                body_adapter_relative: observed_body_adapter_sha,
            },
        ),
        "body_adapter": {
            "path": str(body_adapter),
            "bytes": body_adapter.stat().st_size,
            "sha256": observed_body_adapter_sha,
        },
        "parent_checkpoint": parent_identity,
    }
    for name in (
        "eval_sun_py",
        "eval_sun_resumable_py",
        "train_csv",
        "training_index_cache",
        "mp_hull_cache",
        "chgnet_relax_cache",
        "chgnet_model_asset",
        "chgnet_runtime_checkpoint",
    ):
        spec = sun[name]
        location = resolve_project_path(project_root, spec["path"])
        assets[name] = _verified_identity(
            location,
            spec["sha256"],
            f"S.U.N. asset {name}",
        )
    metrics = config["direct_metrics"]
    gt_csv = resolve_project_path(project_root, metrics["gt_csv"])
    snapshot = resolve_project_path(project_root, metrics["upstream_snapshot"])
    assets["direct_metrics_gt_csv"] = _identity(gt_csv)
    assets["direct_metrics_compute_metrics"] = _identity(
        snapshot / "compute_metrics.py"
    )
    assets["direct_metrics_eval_utils"] = _identity(snapshot / "eval_utils.py")
    return assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    source_dir = args.source_dir.resolve()
    output = args.output_dir.resolve()
    execution_sha = require_hex_sha(
        args.execution_manifest_sha256, "execution source manifest"
    )
    source_manifest = require_source_manifest(source_dir, execution_sha)
    runtime_manifest = require_runtime_manifest(project_root, source_dir)
    config_path = args.config.resolve()
    config = read_json(config_path)
    if config.get("schema") != "h1a2c_p0_p1_sun256_exploratory_v1":
        raise ValueError("exploratory protocol identity changed")
    if config.get("status") != "user_authorized_exploratory_execution":
        raise ValueError("exploratory execution is not user-authorized")
    if config["authorization"].get("manual_crystal_evaluation_authorized") is not True:
        raise ValueError("manual crystal evaluation is not user-authorized")
    if (
        config["authorization"].get(
            "manual_authorization_includes_afterok_sun_evaluation"
        )
        is not True
    ):
        raise ValueError("afterok S.U.N. evaluation is not manually authorized")
    if (
        config["authorization"].get("automatic_crystal_evaluation_authorized")
        is not False
    ):
        raise ValueError("automatic crystal evaluation firewall changed")
    if config["authorization"].get("automatic_followup_submission") is not False:
        raise ValueError("automatic follow-up firewall changed")
    if config["decision_firewall"].get("automatic_downstream_authorized") is not False:
        raise ValueError("automatic downstream firewall changed")
    panel = config["panel"]
    if (
        int(panel["attempts_per_arm"]) != 256
        or int(panel["sample_idx_start"]) != 0
        or int(panel["sample_idx_end_inclusive"]) != 255
        or panel.get("failure_denominator") != "all_registered_attempts"
        or any(
            panel.get(key) is not False
            for key in ("retry", "replacement", "repair", "filter", "rerank")
        )
        or panel.get("sample_id_in_prompt") is not False
    ):
        raise ValueError("paired-256 denominator or no-intervention policy changed")

    terminal_identity = _validate_terminal(config, project_root)
    p1_selection_identity = _validate_p1_selection(config, project_root)
    plan_report_identities = _validate_plan_reports(config, project_root)
    arms_config = config["source_plan_run"]["arms"]
    all_rows: dict[str, list[dict[str, Any]]] = {}
    panel_rows: dict[str, list[dict[str, Any]]] = {}
    raw_identities: dict[str, Any] = {}
    for arm in ARM_ORDER:
        spec = arms_config[arm]
        path = resolve_project_path(project_root, spec["raw_generations"])
        all_rows[arm], panel_rows[arm] = _require_plan_rows(
            path,
            expected_sha256=spec["raw_generations_sha256"],
            label=arm,
        )
        raw_identities[arm] = _identity(path)
        expected_model = resolve_project_path(project_root, spec["planner_model"])
        observed_models = {
            resolve_project_path(project_root, str(row.get("planner_model_path", "")))
            for row in all_rows[arm]
        }
        if observed_models != {expected_model}:
            raise ValueError(f"{arm} frozen Planner base-model path changed")
        if arm == "P0":
            expected_checkpoint = resolve_project_path(
                project_root, spec["planner_checkpoint"]
            )
        else:
            expected_checkpoint = Path(
                p1_selection_identity["checkpoint_dir"]
            ).resolve()
        observed_checkpoints = {
            resolve_project_path(
                project_root, str(row.get("planner_checkpoint_path", ""))
            )
            for row in all_rows[arm]
        }
        if observed_checkpoints != {expected_checkpoint}:
            raise ValueError(f"{arm} frozen Planner checkpoint path changed")

    body_prompt_hashes: dict[str, list[str | None]] = {arm: [] for arm in ARM_ORDER}
    for ordinal in range(256):
        left = panel_rows["P0"][ordinal]
        right = panel_rows["P1"][ordinal]
        if int(left["sample_idx"]) != ordinal or int(right["sample_idx"]) != ordinal:
            raise ValueError("Planner panel order is not exactly sample_idx 0..255")
        for arm, row in (("P0", left), ("P1", right)):
            prompt = row.get("prompt")
            if isinstance(prompt, str) and prompt:
                lowered = prompt.lower()
                if any(
                    token in lowered
                    for token in ("sample_id", "sample-id", "sample idx", "sample_idx")
                ):
                    raise ValueError(
                        f"sample identity leaked into the {arm} body prompt at {ordinal}"
                    )
                body_prompt_hashes[arm].append(canonical_sha256(prompt))
            else:
                body_prompt_hashes[arm].append(None)
            eligible, _ = plan_body_eligible(row)
            if eligible:
                expected_prompt = (
                    build_body_prompt(dict(row["plan_state"])).rstrip() + "\n"
                )
                if prompt != expected_prompt:
                    raise ValueError(
                        f"{arm} source body prompt does not match plan_state at {ordinal}"
                    )

    training_seed = int(panel["pairing_training_seed"])
    sampling_seed = int(panel["planner_global_sampling_seed"])
    deriver = SeedDeriver(
        protocol_name=str(config["identity"]),
        experiment_id=str(panel["pairing_experiment"]),
    )
    ledger_rows = []
    seen_attempts: set[str] = set()
    for ordinal in range(256):
        pair_id = deriver.pair_id(
            training_seed=training_seed,
            sampling_seed=sampling_seed,
            ordinal=ordinal,
        )
        cell: dict[str, Any] = {
            "schema": "h1a2c_p0_p1_sun256_pair_v1",
            "ordinal": ordinal,
            "sample_idx": ordinal,
            "evaluation_ordinal": ordinal,
            "pair_id": pair_id,
            "training_seed": training_seed,
            "planner_sampling_seed": sampling_seed,
            "body_noise_seed": deriver.paired_derive(
                training_seed=training_seed,
                sampling_seed=sampling_seed,
                ordinal=ordinal,
                stage="r5c_exact_body_suffix_noise",
            ),
            "refiner_noise_seed": deriver.paired_derive(
                training_seed=training_seed,
                sampling_seed=sampling_seed,
                ordinal=ordinal,
                stage="crysllmgen_parent_reverse_noise_max20",
            ),
            "arms": {},
        }
        for arm in ARM_ORDER:
            source_row = panel_rows[arm][ordinal]
            method = str(arms_config[arm]["method"])
            attempt_id = deriver.attempt_id(
                training_seed=training_seed,
                sampling_seed=sampling_seed,
                ordinal=ordinal,
                method=method,
            )
            if attempt_id in seen_attempts:
                raise ValueError("attempt identity collision")
            seen_attempts.add(attempt_id)
            eligible, reason = plan_body_eligible(source_row)
            plan = source_row.get("plan_state")
            cell["arms"][arm] = {
                "attempt_id": attempt_id,
                "method": method,
                "source_record_sha256": canonical_sha256(source_row),
                "planner_parsed": bool(source_row.get("parsed")),
                "planner_valid_N": bool(source_row.get("valid_N")),
                "planner_valid_formula": bool(source_row.get("valid_formula")),
                "body_eligible": eligible,
                "ineligible_reason": reason,
                "plan_state": plan if isinstance(plan, dict) else None,
                "plan_state_sha256": (
                    canonical_sha256(plan) if isinstance(plan, dict) else None
                ),
                "source_body_prompt_sha256": body_prompt_hashes[arm][ordinal],
            }
        ledger_rows.append(cell)
    if len(seen_attempts) != 512:
        raise ValueError("paired ledger must register exactly 512 arm attempts")
    observed_eligible = {
        arm: sum(bool(row["arms"][arm]["body_eligible"]) for row in ledger_rows)
        for arm in ARM_ORDER
    }
    expected_eligible = {
        arm: int(panel["expected_body_eligible_by_arm"][arm]) for arm in ARM_ORDER
    }
    if observed_eligible != expected_eligible:
        raise ValueError(
            "frozen Planner body-eligibility counts changed: "
            f"expected={expected_eligible}, observed={observed_eligible}"
        )

    assets = _freeze_assets(config, project_root)
    output.mkdir(parents=True, exist_ok=False)
    ledger_path = output / "attempt_ledger.jsonl"
    asset_path = output / "asset_manifest.json"
    manifest_path = output / "ledger_manifest.json"
    write_jsonl_exclusive(ledger_path, ledger_rows)
    asset_manifest = {
        "schema": "h1a2c_p0_p1_sun256_asset_manifest_v1",
        "identity": config["identity"],
        "execution_manifest_sha256": execution_sha,
        "assets": assets,
    }
    write_json_exclusive(asset_path, asset_manifest)
    ledger_manifest = {
        "schema": "h1a2c_p0_p1_sun256_ledger_manifest_v1",
        "ok": True,
        "identity": config["identity"],
        "run_id": config["run_id"],
        "execution_manifest_sha256": execution_sha,
        "source_manifest": _identity(source_manifest),
        "runtime_manifest": _identity(runtime_manifest),
        "config": _identity(config_path),
        "corrected_jointchem_terminal": terminal_identity,
        "p1_checkpoint_selection": p1_selection_identity,
        "frozen_plan_reports": plan_report_identities,
        "frozen_planner_outputs": raw_identities,
        "source_body_prompt_vector_sha256": {
            arm: canonical_sha256(body_prompt_hashes[arm]) for arm in ARM_ORDER
        },
        "pairs": 256,
        "registered_arm_attempts": 512,
        "arms": list(ARM_ORDER),
        "body_eligible_counts": observed_eligible,
        "attempt_ledger": _identity(ledger_path),
        "asset_manifest": _identity(asset_path),
        "all_attempt_denominator_per_arm": 256,
        "retry_or_replacement_used": False,
        "automatic_downstream_authorized": False,
    }
    write_json_exclusive(manifest_path, ledger_manifest)
    with (output / "_SUCCESS").open("x", encoding="ascii") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(ledger_manifest, sort_keys=True))


if __name__ == "__main__":
    main()
