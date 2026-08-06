#!/usr/bin/env python3
"""Evaluate formula-plan SFT against the frozen paired-64 WQ baseline."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
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
from crystal_dlm.wqcodiff.contracts import (  # noqa: E402
    SeedDeriver,
    write_json_exclusive,
)
from crystal_dlm.wqcodiff.crysllmgen.gate import sha256_file  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.inference import WQLlamaEngine  # noqa: E402
from crystal_dlm.wqcodiff.runtime import expand_state  # noqa: E402


IDENTITY = "wq_formula_plan_sft_pilot_v1"
CONTRACT_SCHEMA = "wq_formula_plan_sft_pilot_contract_v1"
ROW_SCHEMA = "wq_formula_plan_paired_attempt_v1"
REPORT_SCHEMA = "wq_formula_plan_paired64_terminal_v1"


def _require_sha(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(statistics.fmean(values)) if values else None


def _arm_summary(rows: list[Mapping[str, Any]], denominator: int) -> dict[str, Any]:
    succeeded = [row for row in rows if row.get("status") == "succeeded"]
    reasons = collections.Counter(
        str(row.get("composition_reason", "generation_failed")) for row in rows
    )
    formulas = {str(row["formula"]) for row in succeeded if row.get("formula")}
    return {
        "terminal": len(rows),
        "succeeded": len(succeeded),
        "failed": len(rows) - len(succeeded),
        "plan_succeeded": sum(
            int(row.get("plan_status") == "succeeded") for row in rows
        ),
        "body_succeeded": sum(
            int(row.get("body_status") == "succeeded") for row in rows
        ),
        "composition_valid_count": sum(
            int(row.get("composition_valid") is True) for row in rows
        ),
        "composition_valid_rate": sum(
            int(row.get("composition_valid") is True) for row in rows
        )
        / denominator,
        "composition_reason_counts": dict(sorted(reasons.items())),
        "unique_formula_count": len(formulas),
        "unique_formula_all_attempt_rate": len(formulas) / denominator,
        "mean_atom_count_on_success": _mean(succeeded, "atom_count"),
        "mean_orbit_count_on_success": _mean(succeeded, "orbit_count"),
        "plan_body_exact_match_count": sum(
            int(row.get("plan_body_exact_match") is True) for row in rows
        ),
    }


def summarize(
    baseline_rows: Iterable[Mapping[str, Any]],
    formula_rows: Iterable[Mapping[str, Any]],
    *,
    denominator: int,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = [dict(row) for row in baseline_rows]
    formula = [dict(row) for row in formula_rows]
    baseline_summary = _arm_summary(baseline, denominator)
    formula_summary = _arm_summary(formula, denominator)
    by_pair_baseline = {str(row["pair_id"]): row for row in baseline}
    by_pair_formula = {str(row["pair_id"]): row for row in formula}
    paired_ids = set(by_pair_baseline) & set(by_pair_formula)
    invalid_to_valid = 0
    valid_to_invalid = 0
    for pair_id in paired_ids:
        before = by_pair_baseline[pair_id].get("composition_valid") is True
        after = by_pair_formula[pair_id].get("composition_valid") is True
        invalid_to_valid += int(not before and after)
        valid_to_invalid += int(before and not after)

    gain = (
        formula_summary["composition_valid_count"]
        - baseline_summary["composition_valid_count"]
    )
    charge_reason = "charge_neutrality_fail"
    checks = {
        "exact_all_attempt_denominators": bool(
            len(baseline) == denominator
            and len(formula) == denominator
            and len(paired_ids) == denominator
        ),
        "formula_generation_success": bool(
            formula_summary["succeeded"]
            >= int(gate["minimum_formula_generation_success_count"])
        ),
        "plan_body_exact_on_every_success": bool(
            formula_summary["plan_body_exact_match_count"]
            == formula_summary["succeeded"]
        ),
        "composition_valid_absolute": bool(
            formula_summary["composition_valid_count"]
            >= int(gate["minimum_formula_composition_valid_count"])
        ),
        "composition_valid_gain": bool(
            gain >= int(gate["minimum_composition_valid_gain_count"])
        ),
        "charge_failures_nonincreasing": bool(
            formula_summary["composition_reason_counts"].get(charge_reason, 0)
            <= baseline_summary["composition_reason_counts"].get(charge_reason, 0)
        ),
        "valid_to_invalid_pairs_bounded": bool(
            valid_to_invalid <= int(gate["maximum_valid_to_invalid_pairs"])
        ),
        "formula_diversity_noncollapse": bool(
            formula_summary["unique_formula_all_attempt_rate"]
            >= float(gate["minimum_unique_formula_all_attempt_rate"])
        ),
        "atom_count_shift_bounded": bool(
            formula_summary["mean_atom_count_on_success"] is not None
            and baseline_summary["mean_atom_count_on_success"] is not None
            and abs(
                formula_summary["mean_atom_count_on_success"]
                - baseline_summary["mean_atom_count_on_success"]
            )
            <= float(gate["maximum_absolute_mean_atom_count_shift"])
        ),
    }
    return {
        "arms": {
            "frozen_baseline": baseline_summary,
            "formula_plan_sft": formula_summary,
        },
        "paired": {
            "complete_pairs": len(paired_ids),
            "baseline_invalid_to_formula_valid": invalid_to_valid,
            "baseline_valid_to_formula_invalid": valid_to_invalid,
            "composition_valid_gain_count": gain,
            "composition_valid_gain_pp": 100.0 * gain / denominator,
        },
        "promotion_checks": checks,
        "promotion_pass": bool(checks and all(checks.values())),
    }


def _body_seed(plan_seed: int, *, identity: str = IDENTITY) -> int:
    digest = hashlib.sha256(
        f"{identity}|formula_conditioned_body|{int(plan_seed)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _formula_attempt(
    *,
    engine: WQLlamaEngine,
    catalog: PyXtalChartCatalog,
    baseline: Mapping[str, Any],
    attempt_id: str,
    execution_patch_sha256: str,
    contract_sha256: str,
    identity: str = IDENTITY,
    row_schema: str = ROW_SCHEMA,
) -> dict[str, Any]:
    plan_seed = int(baseline["proposal_seed"])
    body_seed = _body_seed(plan_seed, identity=identity)
    started = time.monotonic()
    row: dict[str, Any] = {
        "schema": row_schema,
        "identity": identity,
        "ordinal": int(baseline["ordinal"]),
        "pair_id": str(baseline["pair_id"]),
        "arm": "formula_plan_sft",
        "method": "WQ-FORMULA-PLAN-SFT",
        "attempt_id": attempt_id,
        "plan_seed": plan_seed,
        "body_seed": body_seed,
        "status": "failed",
        "plan_status": "failed",
        "body_status": "not_started",
        "failure_stage": "formula_plan",
        "reason": "",
        "formula_plan_text": "",
        "formula_plan_sha256": None,
        "proposal_text": "",
        "proposal_text_sha256": None,
        "formula": None,
        "composition_valid": False,
        "composition_reason": "generation_failed",
        "plan_body_exact_match": False,
        "atom_count": None,
        "orbit_count": None,
        "usage": {},
        "retry_or_replacement_used": False,
        "best_of_or_rerank_used": False,
        "execution_patch_sha256": execution_patch_sha256,
        "contract_sha256": contract_sha256,
    }
    try:
        plan, plan_text, plan_usage = engine.generate_formula_plan(
            plan_seed=plan_seed,
        )
        row.update(
            {
                "plan_status": "succeeded",
                "body_status": "running",
                "failure_stage": "formula_body",
                "formula_plan_text": plan_text,
                "formula_plan_sha256": hashlib.sha256(
                    plan_text.encode("utf-8")
                ).hexdigest(),
                "formula_plan_composition_valid": bool(plan.composition_valid),
                "formula_plan": plan.as_dict(),
                "usage": {"plan": plan_usage},
            }
        )
        state, body_text, body_usage = engine.generate_formula_body(
            catalog=catalog,
            plan=plan,
            body_seed=body_seed,
            attempt_id=attempt_id,
        )
        expanded = expand_state(state, catalog, redetect_space_group=False)
        composition = composition_record(expanded.atomic_numbers.tolist())
        row.update(
            {
                "status": "succeeded",
                "body_status": "succeeded",
                "failure_stage": None,
                "proposal_text": body_text,
                "proposal_text_sha256": hashlib.sha256(
                    body_text.encode("utf-8")
                ).hexdigest(),
                "formula": composition["formula"],
                "composition_valid": bool(composition["comp_valid"]),
                "composition_reason": str(composition["reason"]),
                "plan_body_exact_match": bool(
                    body_usage["plan_body_exact_match"]
                ),
                "atom_count": int(expanded.atom_count),
                "orbit_count": len(state.orbits),
                "usage": {
                    "llama_invocations": 2,
                    "prompt_tokens": int(plan_usage["prompt_tokens"])
                    + int(body_usage["prompt_tokens"]),
                    "generated_tokens": int(plan_usage["generated_tokens"])
                    + int(body_usage["generated_tokens"]),
                    "walltime_s": float(plan_usage["walltime_s"])
                    + float(body_usage["walltime_s"]),
                    "plan": plan_usage,
                    "body": body_usage,
                    "formula_plan": plan.as_dict(),
                    "plan_body_exact_match": True,
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - every attempt remains in denominator.
        if row.get("failure_stage") == "formula_body":
            row["body_status"] = "failed"
        row["reason"] = f"{type(exc).__name__}:{exc}"
        row["traceback"] = traceback.format_exc()
    row["walltime_s"] = time.monotonic() - started
    return row


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if (
        not os.environ.get("SLURM_JOB_ID")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 8
        or torch.cuda.device_count() != 1
        or "A800" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("formula-plan evaluation runtime/resource contract failed")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_sha = sha256_file(args.contract)
    if (
        contract.get("schema") != args.contract_schema
        or contract.get("identity") != args.identity
        or contract.get("status") != args.contract_status
        or contract_sha != args.contract_sha256
    ):
        raise ValueError("formula-plan pilot contract changed")
    execution_patch = _require_sha(args.execution_patch_sha256, "execution patch")
    frozen = contract["frozen_assets"]
    if (
        sha256_file(args.baseline_attempts)
        != frozen["baseline_attempts_sha256"]
    ):
        raise ValueError("frozen paired-64 baseline attempts changed")
    training_report_path = args.trained_adapter.parent / "training_report.json"
    training = json.loads(training_report_path.read_text(encoding="utf-8"))
    if (
        training.get("schema") != args.training_report_schema
        or training.get("identity") != args.identity
        or training.get("ok") is not True
        or training.get("execution_patch_sha256") != execution_patch
        or training.get("contract_sha256") != contract_sha
        or int(training["optimizer"]["completed_updates"])
        < args.minimum_training_updates
        or int(training["optimizer"]["completed_updates"])
        > args.maximum_training_updates
        or float(training["optimizer"].get("completed_epochs", 0.0))
        < args.minimum_training_epochs
        or training.get("heldout_metrics_or_generations_used_for_training") is not False
        or training["adapter_final"]["adapter_model_sha256"]
        != sha256_file(args.trained_adapter / "adapter_model.safetensors")
        or training["adapter_final"]["adapter_config_sha256"]
        != sha256_file(args.trained_adapter / "adapter_config.json")
    ):
        raise ValueError("formula-plan continuation adapter did not pass its training gate")

    baseline_all = _read_jsonl(args.baseline_attempts)
    baseline = [row for row in baseline_all if row.get("arm") == "baseline"]
    baseline.sort(key=lambda row: int(row["ordinal"]))
    panel = contract["panel"]
    if (
        len(baseline) != 64
        or [int(row["ordinal"]) for row in baseline]
        != list(
            range(
                int(panel["start_ordinal"]),
                int(panel["end_ordinal_inclusive"]) + 1,
            )
        )
        or any(
            row.get("status") != "succeeded"
            or row.get("retry_or_replacement_used") is not False
            for row in baseline
        )
    ):
        raise ValueError("frozen baseline paired panel changed")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    attempts_path = args.output_dir / "attempts.jsonl"
    report_path = args.output_dir / "terminal_report.json"

    catalog = PyXtalChartCatalog()
    engine = WQLlamaEngine.load(
        base_root=args.llama_root,
        adapter_root=args.trained_adapter,
    )
    attempt_ids = SeedDeriver(str(panel["protocol_name"]), args.identity)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with attempts_path.open("x", encoding="utf-8") as handle:
        for baseline_row in baseline:
            ordinal = int(baseline_row["ordinal"])
            attempt_id = attempt_ids.attempt_id(
                training_seed=int(panel["training_seed"]),
                sampling_seed=int(panel["sampling_seed"]),
                ordinal=ordinal,
                method="formula_plan_sft",
            )
            row = _formula_attempt(
                engine=engine,
                catalog=catalog,
                baseline=baseline_row,
                attempt_id=attempt_id,
                execution_patch_sha256=execution_patch,
                contract_sha256=contract_sha,
                identity=args.identity,
                row_schema=args.row_schema,
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

    summary = summarize(
        baseline,
        rows,
        denominator=64,
        gate=contract["promotion_gate"],
    )
    report = {
        "schema": args.report_schema,
        "identity": args.identity,
        "ok": bool(summary["promotion_pass"]),
        "acceptance": "PASS" if summary["promotion_pass"] else "FAIL",
        "decision": (
            "promote_formula_plan_to_three_by_256_direct_metrics"
            if summary["promotion_pass"]
            else "stop_and_analyze_formula_plan_pilot"
        ),
        "contract_sha256": contract_sha,
        "execution_patch_sha256": execution_patch,
        "baseline": {
            "attempts_path": str(args.baseline_attempts),
            "attempts_sha256": sha256_file(args.baseline_attempts),
            "regenerated": False,
        },
        "training": {
            "report_path": str(training_report_path),
            "report_sha256": sha256_file(training_report_path),
            "adapter": training["adapter_final"],
        },
        **summary,
        "formula_attempts_path": str(attempts_path),
        "formula_attempts_sha256": sha256_file(attempts_path),
        "all_attempts_in_denominator": True,
        "retry_or_replacement_used": False,
        "best_of_or_rerank_used": False,
        "training_updates": int(training["optimizer"]["completed_updates"]),
        "training_epochs": float(training["optimizer"]["completed_epochs"]),
        "parent_called": False,
        "mlip_used": False,
        "sun_used": False,
        "external_api_used": False,
        "automatic_downstream_authorized": False,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        },
        "walltime_s": time.monotonic() - started,
    }
    if not math.isfinite(float(report["walltime_s"])):
        raise RuntimeError("non-finite formula-plan evaluation walltime")
    write_json_exclusive(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--baseline-attempts", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--trained-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--identity", default=IDENTITY)
    parser.add_argument("--contract-schema", default=CONTRACT_SCHEMA)
    parser.add_argument("--contract-status", default="authorized_short_sft_pilot")
    parser.add_argument(
        "--training-report-schema",
        default="wq_formula_plan_sft_training_report_v1",
    )
    parser.add_argument("--row-schema", default=ROW_SCHEMA)
    parser.add_argument("--report-schema", default=REPORT_SCHEMA)
    parser.add_argument("--minimum-training-updates", type=int, default=200)
    parser.add_argument("--maximum-training-updates", type=int, default=200)
    parser.add_argument("--minimum-training-epochs", type=float, default=0.0)
    args = parser.parse_args()
    for name in (
        "contract",
        "baseline_attempts",
        "llama_root",
        "trained_adapter",
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
                        "schema": args.report_schema,
                        "identity": args.identity,
                        "ok": False,
                        "acceptance": "FAIL",
                        "decision": "stop_no_retry",
                        "reason": f"{type(exc).__name__}:{exc}",
                        "traceback": traceback.format_exc(),
                        "retry_or_replacement_used": False,
                        "automatic_downstream_authorized": False,
                    },
                )
        raise
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
