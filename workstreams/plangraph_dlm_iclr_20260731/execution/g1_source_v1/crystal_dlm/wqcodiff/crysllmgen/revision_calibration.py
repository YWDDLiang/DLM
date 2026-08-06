"""Validation-only revision threshold calibration for CrysLLMGen/WQ."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..charts import PyXtalChartCatalog
from ..contracts import ArtifactLedger, write_json_exclusive
from ..revision import FieldRef, REVISION_THRESHOLDS, select_revision_threshold
from ..runtime import compute_geometry_evidence, expand_state, tensorize_state
from ..state import GeometryEvidence
from .gate import GateALock, sha256_file
from .inference import WQLlamaEngine
from .lora import validate_trained_adapter
from .protocol import load_protocol_v4
from .sft_data import build_direct_edit_example
from .wq_sampling import _field_scores, _load_refiner
from .wq_text import TopologyEdit, parse_topology_edit, parse_wq_proposal


CALIBRATION_ATTEMPTS = 1024
CALIBRATION_SEED = 2026072004


def _read_selected(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for raw in paths:
        with Path(raw).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("selected"):
                    continue
                material_id = str(row.get("material_id", ""))
                if not material_id or material_id in records:
                    raise ValueError("calibration data has missing/duplicate material IDs")
                records[material_id] = row
    if len(records) < CALIBRATION_ATTEMPTS:
        raise ValueError("revision calibration has fewer than 1024 selected records")
    return sorted(
        records.values(),
        key=lambda row: (
            hashlib.sha256(
                f"crysllmgen_revision_calibration_v1\0{row['material_id']}".encode()
            ).hexdigest(),
            str(row["material_id"]),
        ),
    )[:CALIBRATION_ATTEMPTS]


def _seed(label: str, ordinal: int) -> int:
    digest = hashlib.sha256(
        f"{CALIBRATION_SEED}:{label}:{ordinal}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _condition_evidence(
    state: Any,
    *,
    catalog: Any,
    condition: str,
    seed: int,
) -> list[GeometryEvidence]:
    expanded = expand_state(state, catalog, redetect_space_group=True)
    values = list(compute_geometry_evidence(state, expanded))
    rng = random.Random(seed)
    if condition == "absent":
        return [GeometryEvidence(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in values]
    if condition == "noisy":
        return [
            GeometryEvidence(
                *(min(1.0, max(0.0, item + rng.gauss(0.0, 0.10))) for item in value.as_tuple())
            )
            for value in values
        ]
    if condition == "shuffled":
        rng.shuffle(values)
        return values
    if condition != "clean":
        raise ValueError("unknown revision calibration geometry condition")
    return values


def _target_score(
    target: TopologyEdit,
    state: Any,
    scores: Mapping[FieldRef, float],
) -> float:
    if not scores:
        raise ValueError("revision head produced no field scores")
    if target.kind in {"noop", "birth"}:
        return max(float(value) for value in scores.values())
    if target.orbit_index is None:
        raise ValueError("calibration target lacks an orbit pointer")
    orbit = state.orbits[target.orbit_index]
    field = {
        "death": "existence",
        "type_change": "wyckoff_type",
        "species_change": "species",
    }[target.kind]
    return float(scores[FieldRef(orbit.orbit_id, field)])


@dataclasses.dataclass(frozen=True, slots=True)
class RevisionCalibrationLock:
    path: Path
    sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        gate_a_lock_sha256: str,
        refiner_checkpoint_sha256: str,
        llama_adapter_sha256: str,
    ) -> "RevisionCalibrationLock":
        location = Path(path).resolve()
        payload = json.loads(location.read_text(encoding="utf-8"))
        if payload.get("schema") != "crysllmgen_revision_calibration_lock_v1" or not payload.get("ok"):
            raise ValueError("revision calibration did not pass")
        if int(payload.get("attempts", -1)) != CALIBRATION_ATTEMPTS:
            raise ValueError("revision calibration denominator changed")
        if payload.get("gate_a_lock_sha256") != gate_a_lock_sha256:
            raise ValueError("revision calibration/Gate A mismatch")
        if payload.get("refiner_checkpoint_sha256") != refiner_checkpoint_sha256:
            raise ValueError("revision calibration/refiner mismatch")
        if payload.get("llama_adapter_sha256") != llama_adapter_sha256:
            raise ValueError("revision calibration/Llama mismatch")
        threshold = float(payload.get("selected_threshold", -1.0))
        if threshold not in REVISION_THRESHOLDS:
            raise ValueError("revision calibration threshold is not registered")
        if float(payload.get("clean_false_remask_rate", 1.0)) > 0.05:
            raise ValueError("revision calibration violates the clean false-remask gate")
        attempts_path = Path(str(payload.get("attempts_jsonl", "")))
        if not attempts_path.is_file() or sha256_file(attempts_path) != payload.get(
            "attempts_sha256"
        ):
            raise ValueError("revision calibration attempt evidence changed")
        return cls(location, sha256_file(location), payload)

    @property
    def threshold(self) -> float:
        return float(self.payload["selected_threshold"])


def calibrate(
    *,
    protocol_path: str | Path,
    gate_a_lock_path: str | Path,
    refiner_checkpoint: str | Path,
    llama_root: str | Path,
    llama_adapter: str | Path,
    validation_paths: Sequence[str | Path],
    output_dir: str | Path,
    training_seed: int,
    device: str = "cuda",
) -> dict[str, Any]:
    protocol = load_protocol_v4(protocol_path)
    project_root = Path(protocol_path).resolve().parents[3]
    gate = GateALock.load(
        gate_a_lock_path,
        project_root=project_root,
        protocol_path=protocol_path,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("revision calibration requires a Slurm CUDA allocation")
    refiner_path = Path(refiner_checkpoint).resolve()
    model, checkpoint = _load_refiner(
        refiner_path,
        project_root=project_root,
        device=target_device,
    )
    if (
        checkpoint.get("source_bundle_sha256") != gate.source_bundle_sha256
        or int(checkpoint.get("training_seed", -1)) != int(training_seed)
    ):
        raise ValueError("revision calibration refiner identity mismatch")
    adapter_identity = validate_trained_adapter(
        adapter_root=llama_adapter,
        gate_a_lock_sha256=gate.sha256,
        source_bundle_sha256=gate.source_bundle_sha256,
        representation="wyckoff",
        training_stage="mixed_edit",
        training_seed=training_seed,
    )
    llama = WQLlamaEngine.load(base_root=llama_root, adapter_root=llama_adapter)
    catalog = PyXtalChartCatalog(hall_style="spglib")
    records = _read_selected(validation_paths)
    attempts_path = output / "attempts.jsonl"
    ledger = ArtifactLedger(attempts_path)
    clean_scores: list[float] = []
    wrong_scores: list[float] = []
    predictions: list[tuple[bool, float, bool, bool]] = []
    failures = 0
    started = time.monotonic()
    for ordinal, record in enumerate(records):
        example = build_direct_edit_example(
            record,
            ordinal=ordinal,
            training_seed=CALIBRATION_SEED,
            catalog=catalog,
        )
        clean = str(example["actual_operator"]) == "clean"
        score = 1.0 if clean else 0.0
        predicted = TopologyEdit("noop")
        target = TopologyEdit("noop")
        status = "succeeded"
        reason = ""
        try:
            user = str(example["user_prompt"])
            prefix = "P="
            if not user.startswith(prefix) or ";G=" not in user:
                raise ValueError("malformed direct-edit validation prompt")
            proposal, _ = user[len(prefix) :].rsplit(";G=", 1)
            state = parse_wq_proposal(
                proposal,
                catalog,
                attempt_id=f"cal-{ordinal}",
                timestep=0.5,
            )
            target = parse_topology_edit(str(example["answer"]), state, catalog)
            evidence = _condition_evidence(
                state,
                catalog=catalog,
                condition=str(example["geometry_condition"]),
                seed=_seed("evidence", ordinal),
            )
            expanded = expand_state(state, catalog)
            batch = tensorize_state(state, expanded, evidence, time=0.5).to(target_device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model_output = model(batch, use_geometry_evidence=True)
            scores = _field_scores(state, model_output, control="geometry")
            score = _target_score(target, state, scores)
            if score >= min(REVISION_THRESHOLDS):
                predicted, predicted_text, usage = llama.edit(
                    state,
                    evidence,
                    catalog=catalog,
                    seed=_seed("llama", ordinal),
                )
            else:
                predicted_text = "NOT_CALLED_BELOW_MIN_THRESHOLD"
                usage = {"llama_invocations": 0, "generated_tokens": 0}
        except Exception as exc:
            status = "failed"
            reason = f"{type(exc).__name__}:{exc}"
            failures += 1
            predicted_text = ""
            usage = {"llama_invocations": 0, "generated_tokens": 0}
            score = 1.0 if clean else 0.0
        correct = predicted == target
        wronged_clean = clean and predicted.kind != "noop"
        if clean:
            clean_scores.append(score)
        else:
            wrong_scores.append(score)
        predictions.append((clean, score, correct, wronged_clean))
        ledger.append(
            {
                "schema": "crysllmgen_revision_calibration_attempt_v1",
                "attempt_id": f"cal-{ordinal:04d}",
                "ordinal": ordinal,
                "material_id": record["material_id"],
                "status": status,
                "reason": reason,
                "clean": clean,
                "actual_operator": example["actual_operator"],
                "geometry_condition": example["geometry_condition"],
                "score": score,
                "target": example["answer"],
                "predicted": predicted_text,
                "correct": correct,
                "wronged_clean": wronged_clean,
                "llama_usage": usage,
                "retry_or_replacement_used": False,
            }
        )
    wrong_to_right: dict[float, int] = {}
    right_to_wrong: dict[float, int] = {}
    threshold_rows = []
    for threshold in REVISION_THRESHOLDS:
        wrong_to_right[threshold] = sum(
            (not clean) and score >= threshold and correct
            for clean, score, correct, _ in predictions
        )
        right_to_wrong[threshold] = sum(
            clean and score >= threshold and wronged
            for clean, score, _, wronged in predictions
        )
        threshold_rows.append(
            {
                "threshold": threshold,
                "clean_false_remask_rate": sum(value >= threshold for value in clean_scores)
                / len(clean_scores),
                "wrong_selected": sum(value >= threshold for value in wrong_scores),
                "wrong_to_right": wrong_to_right[threshold],
                "right_to_wrong": right_to_wrong[threshold],
                "net_correction": wrong_to_right[threshold] - right_to_wrong[threshold],
            }
        )
    selected = select_revision_threshold(
        clean_scores=clean_scores,
        wrong_scores=wrong_scores,
        right_to_wrong_if_selected=right_to_wrong,
        wrong_to_right_if_selected=wrong_to_right,
    )
    result = {
        "schema": "crysllmgen_revision_calibration_lock_v1",
        "ok": selected.clean_false_remask_rate <= 0.05,
        "attempts": len(records),
        "terminal_attempts": len(predictions),
        "clean_attempts": len(clean_scores),
        "corrupted_attempts": len(wrong_scores),
        "failed_attempts": failures,
        "selected_threshold": selected.threshold,
        "clean_false_remask_rate": selected.clean_false_remask_rate,
        "selected_net_correction": selected.net_correction,
        "thresholds": threshold_rows,
        "selection_rule": "clean_false_remask_le_0p05_then_max_net_correction_then_higher_threshold",
        "training_seed": training_seed,
        "protocol_sha256": protocol.sha256,
        "gate_a_lock_sha256": gate.sha256,
        "source_bundle_sha256": gate.source_bundle_sha256,
        "refiner_checkpoint": str(refiner_path),
        "refiner_checkpoint_sha256": sha256_file(refiner_path),
        "llama_adapter": str(Path(llama_adapter).resolve()),
        "llama_adapter_sha256": adapter_identity["adapter_model_sha256"],
        "validation_paths": [str(Path(value).resolve()) for value in validation_paths],
        "validation_sha256": [sha256_file(value) for value in validation_paths],
        "attempts_jsonl": str(attempts_path),
        "attempts_sha256": sha256_file(attempts_path),
        "walltime_s": time.monotonic() - started,
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(output / "revision_calibration_lock.json", result)
    if not result["ok"]:
        raise RuntimeError("no registered revision threshold passed calibration")
    return result
