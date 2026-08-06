#!/usr/bin/env python3
"""Generate the immutable WQ source panel for WTB-256.

Exactly one constrained WQ proposal is attempted for each frozen ordinal.
Failures remain terminal rows in the 256-attempt denominator.  This stage does
not load the CrysLLMGen diffusion parent, evaluate metrics, call an MLIP, query
an API, train, retry, replace, rerank, or select outcomes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystal_dlm.wqcodiff.charts import PyXtalChartCatalog  # noqa: E402
from crystal_dlm.wqcodiff.contracts import (  # noqa: E402
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
    write_json_exclusive,
)
from crystal_dlm.wqcodiff.crysllmgen.gate import (  # noqa: E402
    GateALock,
    sha256_file,
)
from crystal_dlm.wqcodiff.crysllmgen.inference import WQLlamaEngine  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.lora import (  # noqa: E402
    validate_trained_adapter,
)
from crystal_dlm.wqcodiff.crysllmgen.protocol import load_protocol_v4  # noqa: E402
from crystal_dlm.wqcodiff.crysllmgen.wtb_confirmatory import (  # noqa: E402
    ATTEMPTS,
    END_ORDINAL_INCLUSIVE,
    IDENTITY,
    PAIRING_ID,
    PROTOCOL_NAME,
    SAMPLING_SEED,
    SOURCE_EXPERIMENT_ID,
    SOURCE_METHOD,
    START_ORDINAL,
    TRAINING_SEED,
    build_confirmatory_cells,
    composition_signature,
    source_signature,
)
from crystal_dlm.wqcodiff.runtime import expand_state  # noqa: E402


CONTRACT_SCHEMA = "wq_wyckoff_chart_retraction_confirmatory256_contract_v1"
SOURCE_SCHEMA = "wq_wyckoff_chart_retraction_source_attempt_v1"
REPORT_SCHEMA = "wq_wyckoff_chart_retraction_source_report_v1"
STAGE = "wq_source_generation"


def _require_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be one lowercase SHA256")
    return value


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("identity") != IDENTITY:
        raise ValueError("unexpected WTB-256 contract identity")
    if payload.get("status") != "local_built_remote_execution_not_authorized":
        raise ValueError("WTB-256 contract status changed")
    panel = payload["panel"]
    if (
        int(panel["training_seed"]) != TRAINING_SEED
        or int(panel["sampling_seed"]) != SAMPLING_SEED
        or int(panel["start_ordinal"]) != START_ORDINAL
        or int(panel["end_ordinal_inclusive"]) != END_ORDINAL_INCLUSIVE
        or int(panel["attempts"]) != ATTEMPTS
        or panel["source_method"] != SOURCE_METHOD
        or panel["pairing_id"] != PAIRING_ID
    ):
        raise ValueError("WTB-256 proposal panel differs from the frozen contract")
    resources = payload["resources"]
    if int(resources["a800"]) != 1 or int(resources["cpus"]) > 8:
        raise ValueError("WTB-256 resource contract exceeds 8 CPU per A800")
    if not all(bool(value) for value in payload["forbidden_actions"].values()):
        raise ValueError("a forbidden WTB-256 action was enabled")
    return payload, sha256_file(path)


def _require_runtime() -> tuple[Any, str]:
    import torch

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("WTB-256 source generation must run through Slurm")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if not 1 <= cpus <= 8:
        raise RuntimeError("WTB-256 source generation exceeds 8 CPU per A800")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("WTB-256 source generation requires one visible GPU")
    device_name = torch.cuda.get_device_name(0)
    if "A800" not in device_name:
        raise RuntimeError(f"WTB-256 requires A800, observed {device_name}")
    return torch, device_name


def _run(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    torch, device_name = _require_runtime()
    contract, contract_sha256 = _load_contract(args.contract)
    execution_patch = _require_sha256(
        args.execution_patch_sha256,
        name="execution patch",
    )
    protocol = load_protocol_v4(args.protocol)
    if protocol.name != PROTOCOL_NAME:
        raise ValueError("WTB-256 protocol name changed")
    if protocol.sha256 != contract["frozen_assets"]["protocol_sha256"]:
        raise ValueError("WTB-256 protocol SHA256 changed")
    project_root = args.protocol.resolve().parents[3]
    gate = GateALock.load(
        args.gate_a_lock,
        project_root=project_root,
        protocol_path=args.protocol,
        execution_patch_manifest_sha256=execution_patch,
    )
    if gate.sha256 != contract["frozen_assets"]["gate_a_lock_sha256"]:
        raise ValueError("WTB-256 Gate A lock changed")
    adapter = validate_trained_adapter(
        adapter_root=args.llama_adapter,
        gate_a_lock_sha256=gate.sha256,
        source_bundle_sha256=gate.source_bundle_sha256,
        representation="wyckoff",
        training_stage="mixed_edit",
        training_seed=TRAINING_SEED,
        execution_patch_sha256=contract["frozen_assets"][
            "adapter_training_execution_patch_sha256"
        ],
    )
    if (
        adapter["adapter_model_sha256"]
        != contract["frozen_assets"]["adapter_model_sha256"]
        or adapter["adapter_config_sha256"]
        != contract["frozen_assets"]["adapter_config_sha256"]
        or adapter["training_report_sha256"]
        != contract["frozen_assets"]["adapter_training_report_sha256"]
    ):
        raise ValueError("WTB-256 adapter bytes changed")

    output.mkdir(parents=True, exist_ok=False)
    source_path = output / "source_attempts.jsonl"
    ledger_path = output / "attempt_ledger.jsonl"
    artifacts = ArtifactLedger(source_path)
    attempts = AttemptLedger(ledger_path)
    source_deriver = SeedDeriver(PROTOCOL_NAME, SOURCE_EXPERIMENT_ID)
    cells = build_confirmatory_cells()
    expected_attempt_ids = [cell.source_attempt_id for cell in cells]
    for cell in cells:
        attempts.append(
            AttemptRecord(
                attempt_id=cell.source_attempt_id,
                method=SOURCE_METHOD,
                training_seed=TRAINING_SEED,
                sampling_seed=SAMPLING_SEED,
                stage=STAGE,
                status=AttemptStatus.SUBMITTED,
                seed=source_deriver.derive(
                    training_seed=TRAINING_SEED,
                    sampling_seed=SAMPLING_SEED,
                    attempt_id=cell.source_attempt_id,
                    stage=STAGE,
                ),
                metadata={
                    "ordinal": cell.ordinal,
                    "pair_id": cell.pair_id,
                    "proposal_seed": cell.proposal_seed,
                },
            )
        )

    catalog = PyXtalChartCatalog()
    llama = WQLlamaEngine.load(
        base_root=args.llama_root,
        adapter_root=args.llama_adapter,
    )
    started_all = time.monotonic()
    succeeded = 0
    failed = 0
    for cell in cells:
        started = time.monotonic()
        status = AttemptStatus.FAILED
        reason = ""
        calls = {"llama": 0, "llama_tokens": 0}
        row: dict[str, Any] = {
            "schema": SOURCE_SCHEMA,
            "identity": IDENTITY,
            "attempt_id": cell.source_attempt_id,
            "method": SOURCE_METHOD,
            "status": "failed",
            "reason": "",
            **cell.to_dict(),
            "training_seed": TRAINING_SEED,
            "sampling_seed": SAMPLING_SEED,
            "proposal_text": "",
            "proposal_state": None,
            "proposal_topology_hash": None,
            "composition_signature": None,
            "source_signature": None,
            "structure": None,
            "retry_or_replacement_used": False,
            "best_of_or_rerank_used": False,
            "execution_patch_sha256": execution_patch,
            "contract_sha256": contract_sha256,
        }
        try:
            state, text, usage = llama.propose(
                catalog=catalog,
                seed=cell.proposal_seed,
                attempt_id=cell.source_attempt_id,
            )
            calls = {
                "llama": int(usage["llama_invocations"]),
                "llama_tokens": int(usage["generated_tokens"]),
            }
            expanded = expand_state(
                state,
                catalog,
                redetect_space_group=False,
            )
            proposal_state = state.to_dict()
            topology_hash = state.topology_hash()
            ordered_composition = composition_signature(expanded.atomic_numbers)
            row.update(
                {
                    "status": "succeeded",
                    "proposal_text": text,
                    "proposal_text_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "proposal_usage": usage,
                    "proposal_state": proposal_state,
                    "proposal_topology_hash": topology_hash,
                    "composition_signature": ordered_composition,
                    "source_signature": source_signature(
                        proposal_state=proposal_state,
                        topology_hash=topology_hash,
                        atomic_numbers=expanded.atomic_numbers,
                    ),
                    "structure": expanded.pymatgen_structure().as_dict(),
                    "atom_count": int(expanded.atom_count),
                    "atomic_numbers": [
                        int(value) for value in expanded.atomic_numbers
                    ],
                }
            )
            status = AttemptStatus.SUCCEEDED
            succeeded += 1
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            row.update(
                {
                    "reason": reason,
                    "traceback": traceback.format_exc(),
                }
            )
            failed += 1
        elapsed = time.monotonic() - started
        row["calls"] = calls
        row["walltime_s"] = elapsed
        digest = artifacts.append(row)
        attempts.append(
            AttemptRecord(
                attempt_id=cell.source_attempt_id,
                method=SOURCE_METHOD,
                training_seed=TRAINING_SEED,
                sampling_seed=SAMPLING_SEED,
                stage=STAGE,
                status=status,
                reason=reason,
                artifact_hash=digest,
                seed=source_deriver.derive(
                    training_seed=TRAINING_SEED,
                    sampling_seed=SAMPLING_SEED,
                    attempt_id=cell.source_attempt_id,
                    stage=STAGE,
                ),
                calls=calls,
                walltime_s=elapsed,
                metadata={
                    "ordinal": cell.ordinal,
                    "pair_id": cell.pair_id,
                    "proposal_seed": cell.proposal_seed,
                },
            )
        )

    audit = attempts.audit(
        seed_deriver=source_deriver,
        terminal_stage=STAGE,
        expected_attempt_ids=expected_attempt_ids,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "ok": bool(audit.ok and succeeded + failed == ATTEMPTS),
        "acceptance": "PASS" if audit.ok and succeeded + failed == ATTEMPTS else "FAIL",
        "identity": IDENTITY,
        "contract_sha256": contract_sha256,
        "execution_patch_sha256": execution_patch,
        "submitted": ATTEMPTS,
        "terminal": succeeded + failed,
        "succeeded": succeeded,
        "failed": failed,
        "start_ordinal": START_ORDINAL,
        "end_ordinal_inclusive": END_ORDINAL_INCLUSIVE,
        "method": SOURCE_METHOD,
        "pairing_id": PAIRING_ID,
        "retry_or_replacement_used": False,
        "best_of_or_rerank_used": False,
        "training_performed": False,
        "mlip_used": False,
        "external_api_used": False,
        "source_attempts": str(source_path),
        "source_attempts_sha256": sha256_file(source_path),
        "attempt_ledger": str(ledger_path),
        "attempt_ledger_sha256": sha256_file(ledger_path),
        "attempt_audit": dataclasses.asdict(audit),
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": gate.sha256,
        "model_identity": {
            "llama": llama.identity,
            "adapter_training": adapter,
        },
        "gpu": {
            "name": device_name,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        },
        "walltime_s": time.monotonic() - started_all,
    }
    write_json_exclusive(output / "source_report.json", report)
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
        report = _run(args, args.output_dir)
    except Exception as exc:
        if args.output_dir.is_dir():
            terminal = args.output_dir / "source_report.json"
            if not terminal.exists():
                write_json_exclusive(
                    terminal,
                    {
                        "schema": REPORT_SCHEMA,
                        "ok": False,
                        "acceptance": "FAIL",
                        "identity": IDENTITY,
                        "reason": f"{type(exc).__name__}:{exc}",
                        "traceback": traceback.format_exc(),
                        "retry_or_replacement_used": False,
                        "training_performed": False,
                        "mlip_used": False,
                        "external_api_used": False,
                    },
                )
        raise
    print(json.dumps(report, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
