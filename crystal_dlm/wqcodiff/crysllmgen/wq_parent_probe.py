"""Same-proposal probe through the released parent CrysLLMGen diffusion.

This diagnostic intentionally stops before metrics or checkpoint selection.
It answers one causal question: can the released atom-level CSP diffusion
produce a finite crystal from the exact WQ-Llama proposal seeds that collapsed
under the quotient-space reverse integrator?
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from ..charts import PyXtalChartCatalog
from ..contracts import (
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
    write_json_exclusive,
)
from ..runtime import expand_state
from .atom_sampling import (
    _output_structure,
    _sample_respaced_32,
    expanded_state_to_parent_batch,
    load_registered_csp,
)
from .gate import GateALock, sha256_file
from .inference import WQLlamaEngine
from .lora import validate_trained_adapter
from .protocol import load_protocol_v4


METHOD = "DIAG-WQ-PROPOSAL-PARENT-CSP32"


@dataclasses.dataclass(frozen=True, slots=True)
class WQParentCSPProbeConfig:
    protocol_path: str
    gate_a_lock: str
    csp_checkpoint: str
    llama_root: str
    llama_adapter: str
    output_jsonl: str
    attempt_ledger: str
    report_path: str
    experiment_id: str
    pairing_id: str
    training_seed: int
    sampling_seed: int
    attempts: int
    adapter_training_execution_patch_sha256: str
    diagnostic_execution_patch_sha256: str
    start_ordinal: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.training_seed not in {11, 23, 47}:
            raise ValueError("training seed is outside 11/23/47")
        if self.attempts <= 0 or self.start_ordinal < 0:
            raise ValueError("invalid probe denominator or start ordinal")
        if not self.experiment_id or not self.pairing_id:
            raise ValueError("probe experiment and pairing identities are required")
        for label, value in (
            (
                "adapter training execution patch",
                self.adapter_training_execution_patch_sha256,
            ),
            ("diagnostic execution patch", self.diagnostic_execution_patch_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{label} must be one lowercase SHA256")


class WQParentProbeFailure(RuntimeError):
    def __init__(
        self,
        cause: Exception,
        *,
        calls: Mapping[str, int],
        proposal_text: str = "",
        proposal_usage: Mapping[str, Any] | None = None,
        proposal_state: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{type(cause).__name__}:{cause}")
        self.calls = dict(calls)
        self.proposal_text = proposal_text
        self.proposal_usage = dict(proposal_usage or {})
        self.proposal_state = dict(proposal_state or {})


def _derived_subseed(seed: int, label: str, step: int) -> int:
    """Mirror the WQ closed-loop proposal seed derivation exactly."""

    raw = f"{seed}:{label}:{step}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _probe_one(
    *,
    llama: WQLlamaEngine,
    csp: Any,
    catalog: PyXtalChartCatalog,
    attempt_id: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    calls = {"llama": 0, "llama_tokens": 0, "csp": 0}
    proposal_text = ""
    proposal_usage: dict[str, Any] = {}
    proposal_state: dict[str, Any] = {}
    try:
        state, proposal_text, proposal_usage = llama.propose(
            catalog=catalog,
            seed=_derived_subseed(seed, "proposal", -1),
            attempt_id=attempt_id,
        )
        calls["llama"] = 1
        calls["llama_tokens"] = int(proposal_usage["generated_tokens"])
        proposal_state = dict(state.to_dict())
        expanded = expand_state(state, catalog)
        batch = expanded_state_to_parent_batch(expanded, device)
        csp_seed = _derived_subseed(seed, "parent_csp_refinement", -1)
        random.seed(csp_seed)
        torch.manual_seed(csp_seed)
        torch.cuda.manual_seed_all(csp_seed)
        with torch.inference_mode():
            output, calls["csp"] = _sample_respaced_32(csp, batch)
        final_structure = _output_structure(output)
        initial_structure = expanded.pymatgen_structure()
    except Exception as exc:
        raise WQParentProbeFailure(
            exc,
            calls=calls,
            proposal_text=proposal_text,
            proposal_usage=proposal_usage,
            proposal_state=proposal_state,
        ) from exc
    return {
        "schema": "wq_parent_csp_probe_attempt_v1",
        "producer_schema": "same_wq_proposal_parent_csp32_v1",
        "attempt_id": attempt_id,
        "method": METHOD,
        "status": AttemptStatus.SUCCEEDED.value,
        "reason": "",
        "proposal_text": proposal_text,
        "proposal_text_sha256": hashlib.sha256(
            proposal_text.encode("utf-8")
        ).hexdigest(),
        "proposal_usage": proposal_usage,
        "proposal_state": proposal_state,
        "proposal_topology_hash": state.topology_hash(),
        "initial_structure": initial_structure.as_dict(),
        "initial_volume": float(initial_structure.volume),
        "final_structure": final_structure.as_dict(),
        "final_volume": float(final_structure.volume),
        "atom_count": len(final_structure),
        "reverse_steps": 32,
        "calls": calls,
    }


def probe(config: WQParentCSPProbeConfig) -> dict[str, Any]:
    protocol = load_protocol_v4(config.protocol_path)
    project_root = Path(config.protocol_path).resolve().parents[3]
    gate = GateALock.load(
        config.gate_a_lock,
        project_root=project_root,
        protocol_path=config.protocol_path,
        execution_patch_manifest_sha256=(
            config.diagnostic_execution_patch_sha256
        ),
    )
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the parent-CSP probe requires CUDA")
    expected_csp = str(protocol.data["assets"]["cspdiffusion"]["sha256"])
    if sha256_file(config.csp_checkpoint) != expected_csp:
        raise ValueError("probe CSP checkpoint differs from protocol")
    adapter_training = validate_trained_adapter(
        adapter_root=config.llama_adapter,
        gate_a_lock_sha256=gate.sha256,
        source_bundle_sha256=gate.source_bundle_sha256,
        representation="wyckoff",
        training_stage="mixed_edit",
        training_seed=config.training_seed,
        execution_patch_sha256=config.adapter_training_execution_patch_sha256,
    )
    llama = WQLlamaEngine.load(
        base_root=config.llama_root,
        adapter_root=config.llama_adapter,
    )
    csp, csp_identity = load_registered_csp(
        snapshot_root=project_root / "crystal_dlm/wqcodiff/crysllmgen/upstream",
        checkpoint=config.csp_checkpoint,
        device=device,
    )
    catalog = PyXtalChartCatalog()

    output_path = Path(config.output_jsonl).resolve()
    ledger_path = Path(config.attempt_ledger).resolve()
    report_path = Path(config.report_path).resolve()
    for path in (output_path, ledger_path, report_path):
        if path.exists():
            raise FileExistsError(f"parent-CSP probe evidence is immutable: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = ArtifactLedger(output_path)
    attempts = AttemptLedger(ledger_path)
    derivation = SeedDeriver(protocol.name, config.experiment_id)
    pairing = SeedDeriver(protocol.name, config.pairing_id)
    expected_ids: list[str] = []
    for ordinal in range(config.start_ordinal, config.start_ordinal + config.attempts):
        attempt_id = derivation.attempt_id(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            ordinal=ordinal,
            method=METHOD,
        )
        expected_ids.append(attempt_id)
        ledger_seed = derivation.derive(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            attempt_id=attempt_id,
            stage="parent_csp_probe",
        )
        attempts.append(
            AttemptRecord(
                attempt_id=attempt_id,
                method=METHOD,
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                stage="parent_csp_probe",
                status=AttemptStatus.SUBMITTED,
                seed=ledger_seed,
                metadata={
                    "ordinal": ordinal,
                    "pair_id": pairing.pair_id(
                        training_seed=config.training_seed,
                        sampling_seed=config.sampling_seed,
                        ordinal=ordinal,
                    ),
                },
            )
        )

    succeeded = 0
    failed = 0
    started_all = time.monotonic()
    for ordinal, attempt_id in zip(
        range(config.start_ordinal, config.start_ordinal + config.attempts),
        expected_ids,
    ):
        paired_seed = pairing.paired_derive(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            ordinal=ordinal,
            stage="generation_sampling",
        )
        ledger_seed = derivation.derive(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            attempt_id=attempt_id,
            stage="parent_csp_probe",
        )
        started = time.monotonic()
        calls = {"llama": 0, "llama_tokens": 0, "csp": 0}
        try:
            row = _probe_one(
                llama=llama,
                csp=csp,
                catalog=catalog,
                attempt_id=attempt_id,
                seed=paired_seed,
                device=device,
            )
            calls = dict(row["calls"])
            status = AttemptStatus.SUCCEEDED
            reason = ""
            succeeded += 1
        except WQParentProbeFailure as exc:
            calls = dict(exc.calls)
            status = AttemptStatus.FAILED
            reason = str(exc)
            row = {
                "schema": "wq_parent_csp_probe_attempt_v1",
                "producer_schema": "same_wq_proposal_parent_csp32_v1",
                "attempt_id": attempt_id,
                "method": METHOD,
                "status": status.value,
                "reason": reason,
                "proposal_text": exc.proposal_text,
                "proposal_text_sha256": hashlib.sha256(
                    exc.proposal_text.encode("utf-8")
                ).hexdigest(),
                "proposal_usage": exc.proposal_usage,
                "proposal_state": exc.proposal_state,
                "calls": calls,
            }
            failed += 1
        elapsed = time.monotonic() - started
        pair_id = pairing.pair_id(
            training_seed=config.training_seed,
            sampling_seed=config.sampling_seed,
            ordinal=ordinal,
        )
        row.update(
            {
                "ordinal": ordinal,
                "pair_id": pair_id,
                "paired_seed": paired_seed,
                "ledger_seed": ledger_seed,
                "experiment_id": config.experiment_id,
                "pairing_id": config.pairing_id,
                "training_seed": config.training_seed,
                "sampling_seed": config.sampling_seed,
                "stage": "parent_csp_probe",
                "walltime_s": elapsed,
                "retry_or_replacement_used": False,
                "adapter_training_execution_patch_sha256": (
                    config.adapter_training_execution_patch_sha256
                ),
                "diagnostic_execution_patch_sha256": (
                    config.diagnostic_execution_patch_sha256
                ),
                "model_identity": {
                    "llama": llama.identity,
                    "adapter_training": adapter_training,
                    "csp": csp_identity,
                },
            }
        )
        digest = artifacts.append(row)
        attempts.append(
            AttemptRecord(
                attempt_id=attempt_id,
                method=METHOD,
                training_seed=config.training_seed,
                sampling_seed=config.sampling_seed,
                stage="parent_csp_probe",
                status=status,
                reason=reason,
                artifact_hash=digest,
                seed=ledger_seed,
                calls=calls,
                walltime_s=elapsed,
                metadata={
                    "ordinal": ordinal,
                    "pairing_id": config.pairing_id,
                    "pair_id": pair_id,
                    "paired_seed": paired_seed,
                },
            )
        )

    audit = attempts.audit(
        seed_deriver=derivation,
        terminal_stage="parent_csp_probe",
        expected_attempt_ids=expected_ids,
    )
    report = {
        "schema": "wq_parent_csp_probe_report_v1",
        "ok": audit.ok and succeeded + failed == config.attempts,
        "scientific_acceptance": (
            "parent_path_nonzero_success" if succeeded > 0 else "parent_path_all_failed"
        ),
        "submitted": config.attempts,
        "terminal": succeeded + failed,
        "succeeded": succeeded,
        "failed": failed,
        "method": METHOD,
        "reverse_steps": 32,
        "pairing_id": config.pairing_id,
        "retry_or_replacement_used": False,
        "adapter_training_execution_patch_sha256": (
            config.adapter_training_execution_patch_sha256
        ),
        "diagnostic_execution_patch_sha256": (
            config.diagnostic_execution_patch_sha256
        ),
        "audit": dataclasses.asdict(audit),
        "output_jsonl": str(output_path),
        "output_sha256": sha256_file(output_path),
        "attempt_ledger": str(ledger_path),
        "attempt_ledger_sha256": sha256_file(ledger_path),
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": gate.sha256,
        "model_identity": {
            "llama": llama.identity,
            "adapter_training": adapter_training,
            "csp": csp_identity,
        },
        "walltime_s": time.monotonic() - started_all,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
    }
    write_json_exclusive(report_path, report)
    if not report["ok"]:
        raise RuntimeError("parent-CSP probe attempt audit failed")
    return report
