#!/usr/bin/env python3
"""Run one paired proposal-only WQ chemistry termination pilot.

Each ordinal produces exactly one baseline proposal and one proposal with the
charge-aware STOP mask under the same sampling seed.  The job does not call the
continuous parent, CrysLLMGen metrics, CHGNet, S.U.N., an external API, or any
training code.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.composition_validity import composition_record  # noqa: E402
from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog  # noqa: E402
from crystal_dlm.wqcodiff.contracts import SeedDeriver, write_json_exclusive  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.gate import (  # noqa: E402
    GateALock,
    sha256_file,
)
from crystal_dlm.wqcodiff.crysllmgen.inference import WQLlamaEngine  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.lora import (  # noqa: E402
    validate_trained_adapter,
)
from crystal_dlm.wqcodiff.crysllmgen.protocol import load_protocol_v4  # noqa: E402
from crystal_dlm.wqcodiff.runtime import expand_state  # noqa: E402


SCHEMA = "wq_llm_charge_stop_paired64_contract_v1"
IDENTITY = "wq_llm_charge_stop_paired64_v1"
ROW_SCHEMA = "wq_llm_charge_stop_paired_attempt_v1"
REPORT_SCHEMA = "wq_llm_charge_stop_paired64_terminal_v1"


def _require_sha(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("identity") != IDENTITY
        or contract.get("status") != "authorized_paired64_pilot"
    ):
        raise ValueError("unexpected WQ charge-stop pilot contract")
    panel = contract["panel"]
    if (
        int(panel["attempts_per_arm"]) != 64
        or int(panel["start_ordinal"]) != 1024
        or int(panel["end_ordinal_inclusive"]) != 1087
        or panel["arms"] != ["baseline", "charge_stop"]
        or panel["same_seed_per_pair"] is not True
    ):
        raise ValueError("WQ charge-stop pilot panel changed")
    resources = contract["resources"]
    if (
        resources["partition"] != "gpu"
        or int(resources["a800"]) != 1
        or int(resources["cpus"]) != 8
        or int(resources["cpus"]) > 8 * int(resources["a800"])
    ):
        raise ValueError("WQ charge-stop pilot resource contract changed")
    forbidden = contract["forbidden_actions"]
    if not forbidden or not all(bool(value) for value in forbidden.values()):
        raise ValueError("one WQ charge-stop forbidden action was enabled")
    return contract, sha256_file(path)


def _require_runtime() -> tuple[Any, str]:
    import torch

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("WQ charge-stop pilot must run through Slurm")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 8:
        raise RuntimeError("WQ charge-stop pilot requires exactly 8 CPU")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("WQ charge-stop pilot requires one visible GPU")
    device_name = torch.cuda.get_device_name(0)
    if "A800" not in device_name:
        raise RuntimeError(f"WQ charge-stop pilot requires A800, observed {device_name}")
    return torch, device_name


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(statistics.fmean(values)) if values else None


def summarize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    attempts_per_arm: int,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("baseline", "charge_stop"):
        selected = [row for row in values if row.get("arm") == arm]
        succeeded = [row for row in selected if row.get("status") == "succeeded"]
        reasons = collections.Counter(
            str(row.get("composition_reason", "generation_failed"))
            for row in selected
        )
        formulas = {
            str(row["formula"])
            for row in succeeded
            if row.get("formula")
        }
        proposals = {
            str(row["proposal_text_sha256"])
            for row in succeeded
            if row.get("proposal_text_sha256")
        }
        arms[arm] = {
            "terminal": len(selected),
            "succeeded": len(succeeded),
            "failed": len(selected) - len(succeeded),
            "composition_valid_count": sum(
                int(row.get("composition_valid") is True) for row in selected
            ),
            "composition_valid_rate": sum(
                int(row.get("composition_valid") is True) for row in selected
            )
            / attempts_per_arm,
            "composition_reason_counts": dict(sorted(reasons.items())),
            "unique_formula_count": len(formulas),
            "unique_formula_all_attempt_rate": len(formulas) / attempts_per_arm,
            "unique_proposal_count": len(proposals),
            "unique_proposal_all_attempt_rate": len(proposals) / attempts_per_arm,
            "mean_atom_count_on_success": _mean(succeeded, "atom_count"),
            "mean_orbit_count_on_success": _mean(succeeded, "orbit_count"),
            "mean_generated_tokens_on_success": _mean(
                (
                    {
                        **row,
                        "generated_tokens": row.get("usage", {}).get(
                            "generated_tokens"
                        ),
                    }
                    for row in succeeded
                ),
                "generated_tokens",
            ),
            "unique_stop_deferrals": sum(
                int(
                    row.get("usage", {})
                    .get("chemistry_constraint", {})
                    .get("unique_stop_deferrals", 0)
                )
                for row in selected
            ),
        }

    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in values:
        by_pair.setdefault(str(row["pair_id"]), {})[str(row["arm"])] = row
    complete_pairs = {
        pair_id: pair
        for pair_id, pair in by_pair.items()
        if set(pair) == {"baseline", "charge_stop"}
    }
    baseline_invalid_to_mask_valid = 0
    baseline_valid_to_mask_invalid = 0
    both_valid = 0
    both_invalid = 0
    for pair in complete_pairs.values():
        before = pair["baseline"].get("composition_valid") is True
        after = pair["charge_stop"].get("composition_valid") is True
        baseline_invalid_to_mask_valid += int(not before and after)
        baseline_valid_to_mask_invalid += int(before and not after)
        both_valid += int(before and after)
        both_invalid += int(not before and not after)

    baseline = arms["baseline"]
    masked = arms["charge_stop"]
    charge_reason = "charge_neutrality_fail"
    checks = {
        "exact_all_attempt_denominators": bool(
            baseline["terminal"] == attempts_per_arm
            and masked["terminal"] == attempts_per_arm
            and len(complete_pairs) == attempts_per_arm
        ),
        "masked_generation_success_noninferior": bool(
            masked["succeeded"]
            >= baseline["succeeded"]
            - int(gate["maximum_generation_success_loss"])
        ),
        "composition_valid_gain": bool(
            masked["composition_valid_count"]
            - baseline["composition_valid_count"]
            >= int(gate["minimum_composition_valid_gain_count"])
        ),
        "charge_failure_reduction": bool(
            baseline["composition_reason_counts"].get(charge_reason, 0)
            - masked["composition_reason_counts"].get(charge_reason, 0)
            >= int(gate["minimum_charge_failure_reduction_count"])
        ),
        "valid_to_invalid_pairs_bounded": bool(
            baseline_valid_to_mask_invalid
            <= int(gate["maximum_valid_to_invalid_pairs"])
        ),
        "formula_diversity_noncollapse": bool(
            masked["unique_formula_all_attempt_rate"]
            >= baseline["unique_formula_all_attempt_rate"]
            - float(gate["maximum_unique_formula_rate_loss"])
        ),
        "atom_count_shift_bounded": bool(
            masked["mean_atom_count_on_success"] is not None
            and baseline["mean_atom_count_on_success"] is not None
            and masked["mean_atom_count_on_success"]
            <= baseline["mean_atom_count_on_success"]
            + float(gate["maximum_mean_atom_count_increase"])
        ),
    }
    return {
        "arms": arms,
        "paired": {
            "complete_pairs": len(complete_pairs),
            "baseline_invalid_to_mask_valid": baseline_invalid_to_mask_valid,
            "baseline_valid_to_mask_invalid": baseline_valid_to_mask_invalid,
            "both_valid": both_valid,
            "both_invalid": both_invalid,
            "composition_valid_gain_count": (
                masked["composition_valid_count"]
                - baseline["composition_valid_count"]
            ),
            "composition_valid_gain_pp": 100.0
            * (
                masked["composition_valid_count"]
                - baseline["composition_valid_count"]
            )
            / attempts_per_arm,
        },
        "promotion_checks": checks,
        "promotion_pass": bool(checks and all(checks.values())),
    }


def _attempt_row(
    *,
    engine: WQLlamaEngine,
    catalog: PyXtalChartCatalog,
    arm: str,
    ordinal: int,
    pair_id: str,
    attempt_id: str,
    proposal_seed: int,
    execution_patch_sha256: str,
    contract_sha256: str,
) -> dict[str, Any]:
    started = time.monotonic()
    row: dict[str, Any] = {
        "schema": ROW_SCHEMA,
        "identity": IDENTITY,
        "ordinal": int(ordinal),
        "pair_id": pair_id,
        "arm": arm,
        "method": (
            "WQ-BASELINE-GRAMMAR"
            if arm == "baseline"
            else "WQ-CHARGE-AWARE-STOP"
        ),
        "attempt_id": attempt_id,
        "proposal_seed": int(proposal_seed),
        "status": "failed",
        "reason": "",
        "proposal_text": "",
        "proposal_text_sha256": None,
        "proposal_state": None,
        "topology_hash": None,
        "formula": None,
        "composition_valid": False,
        "composition_reason": "generation_failed",
        "atom_count": None,
        "orbit_count": None,
        "usage": {},
        "retry_or_replacement_used": False,
        "best_of_or_rerank_used": False,
        "training_performed": False,
        "parent_or_mlip_called": False,
        "execution_patch_sha256": execution_patch_sha256,
        "contract_sha256": contract_sha256,
    }
    try:
        if arm == "baseline":
            state, text, usage = engine.propose(
                catalog=catalog,
                seed=proposal_seed,
                attempt_id=attempt_id,
            )
        else:
            state, text, usage = engine.propose_charge_aware_stop(
                catalog=catalog,
                seed=proposal_seed,
                attempt_id=attempt_id,
            )
        expanded = expand_state(state, catalog, redetect_space_group=False)
        composition = composition_record(expanded.atomic_numbers.tolist())
        row.update(
            {
                "status": "succeeded",
                "proposal_text": text,
                "proposal_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "proposal_state": state.to_dict(canonical_storage=True),
                "topology_hash": state.topology_hash(),
                "formula": composition["formula"],
                "composition_valid": bool(composition["comp_valid"]),
                "composition_reason": str(composition["reason"]),
                "atom_count": int(expanded.atom_count),
                "orbit_count": len(state.orbits),
                "usage": usage,
            }
        )
    except Exception as exc:  # noqa: BLE001 - one attempt stays in the denominator.
        row["reason"] = f"{type(exc).__name__}:{exc}"
        row["traceback"] = traceback.format_exc()
    row["walltime_s"] = time.monotonic() - started
    return row


def _run(args: argparse.Namespace) -> dict[str, Any]:
    torch, device_name = _require_runtime()
    contract, contract_sha256 = _load_contract(args.contract)
    execution_patch_sha256 = _require_sha(
        args.execution_patch_sha256,
        label="execution patch",
    )
    protocol = load_protocol_v4(args.protocol)
    frozen = contract["frozen_assets"]
    if protocol.sha256 != frozen["protocol_sha256"]:
        raise ValueError("WQ charge-stop protocol identity changed")
    project_root = args.protocol.resolve().parents[3]
    gate = GateALock.load(
        args.gate_a_lock,
        project_root=project_root,
        protocol_path=args.protocol,
        execution_patch_manifest_sha256=execution_patch_sha256,
    )
    if gate.sha256 != frozen["gate_a_lock_sha256"]:
        raise ValueError("WQ charge-stop Gate-A lock changed")
    adapter = validate_trained_adapter(
        adapter_root=args.llama_adapter,
        gate_a_lock_sha256=gate.sha256,
        source_bundle_sha256=gate.source_bundle_sha256,
        representation="wyckoff",
        training_stage="mixed_edit",
        training_seed=int(contract["panel"]["training_seed"]),
        execution_patch_sha256=frozen[
            "adapter_training_execution_patch_sha256"
        ],
    )
    for key in (
        "adapter_model_sha256",
        "adapter_config_sha256",
        "training_report_sha256",
    ):
        if adapter[key] != frozen[key]:
            raise ValueError(f"WQ charge-stop adapter identity changed: {key}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = args.output_dir / "attempts.jsonl"
    report_path = args.output_dir / "terminal_report.json"
    panel = contract["panel"]
    training_seed = int(panel["training_seed"])
    sampling_seed = int(panel["sampling_seed"])
    pairing = SeedDeriver(protocol.name, str(panel["pairing_id"]))
    attempts = SeedDeriver(protocol.name, IDENTITY)
    catalog = PyXtalChartCatalog()
    engine = WQLlamaEngine.load(
        base_root=args.llama_root,
        adapter_root=args.llama_adapter,
    )

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with rows_path.open("x", encoding="utf-8") as handle:
        for ordinal in range(
            int(panel["start_ordinal"]),
            int(panel["end_ordinal_inclusive"]) + 1,
        ):
            pair_id = pairing.pair_id(
                training_seed=training_seed,
                sampling_seed=sampling_seed,
                ordinal=ordinal,
            )
            proposal_seed = pairing.paired_derive(
                training_seed=training_seed,
                sampling_seed=sampling_seed,
                ordinal=ordinal,
                stage="wq_proposal_generation",
            )
            for arm in panel["arms"]:
                attempt_id = attempts.attempt_id(
                    training_seed=training_seed,
                    sampling_seed=sampling_seed,
                    ordinal=ordinal,
                    method=str(arm),
                )
                row = _attempt_row(
                    engine=engine,
                    catalog=catalog,
                    arm=str(arm),
                    ordinal=ordinal,
                    pair_id=pair_id,
                    attempt_id=attempt_id,
                    proposal_seed=proposal_seed,
                    execution_patch_sha256=execution_patch_sha256,
                    contract_sha256=contract_sha256,
                )
                rows.append(row)
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())

    summary = summarize_rows(
        rows,
        attempts_per_arm=int(panel["attempts_per_arm"]),
        gate=contract["promotion_gate"],
    )
    report = {
        "schema": REPORT_SCHEMA,
        "identity": IDENTITY,
        "ok": bool(summary["promotion_pass"]),
        "acceptance": "PASS" if summary["promotion_pass"] else "FAIL",
        "decision": (
            "expand_to_three_by_256_proposal_only"
            if summary["promotion_pass"]
            else "stop_charge_stop_mask_and_prepare_formula_plan_sft"
        ),
        "contract_sha256": contract_sha256,
        "execution_patch_sha256": execution_patch_sha256,
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": gate.sha256,
        "model_identity": {
            "llama": engine.identity,
            "adapter_training": adapter,
        },
        "panel": dict(panel),
        **summary,
        "attempts_path": str(rows_path),
        "attempts_sha256": sha256_file(rows_path),
        "all_attempts_in_denominator": True,
        "retry_or_replacement_used": False,
        "best_of_or_rerank_used": False,
        "training_performed": False,
        "parent_called": False,
        "mlip_used": False,
        "external_api_used": False,
        "automatic_downstream_authorized": False,
        "gpu": {
            "name": device_name,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        },
        "walltime_s": time.monotonic() - started,
    }
    write_json_exclusive(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-a-lock", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--llama-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    args = parser.parse_args()
    for name in (
        "contract",
        "protocol",
        "gate_a_lock",
        "llama_root",
        "llama_adapter",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    try:
        report = _run(args)
    except Exception as exc:
        if args.output_dir.is_dir():
            terminal = args.output_dir / "terminal_report.json"
            if not terminal.exists():
                write_json_exclusive(
                    terminal,
                    {
                        "schema": REPORT_SCHEMA,
                        "identity": IDENTITY,
                        "ok": False,
                        "acceptance": "FAIL",
                        "decision": "stop_no_retry",
                        "reason": f"{type(exc).__name__}:{exc}",
                        "traceback": traceback.format_exc(),
                        "retry_or_replacement_used": False,
                        "training_performed": False,
                        "automatic_downstream_authorized": False,
                    },
                )
        raise
    print(json.dumps(report, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
