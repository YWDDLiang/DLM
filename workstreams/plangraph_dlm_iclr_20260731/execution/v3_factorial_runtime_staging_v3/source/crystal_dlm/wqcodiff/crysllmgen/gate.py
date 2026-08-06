"""Immutable Gate-A aggregation and source-identity verification."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .protocol import load_registry_v2, load_protocol_v4


SHA256 = re.compile(r"^[0-9a-f]{64}$")
PATCH_ALLOWED_TOP_LEVEL = {
    "configs",
    "crystal_dlm",
    "diagnostics",
    "docs",
    "scripts",
    "tests",
}
PATCH_ALLOWED_AUTHORIZATIONS = {
    "user_explicit_chgnet_waiver_2026-07-17",
    "user_environment_scope_explicit_2026-07-17",
    "user_r5c_a100_sun_and_three_epoch_checkpoint_selection_2026-07-21",
    "user_r5c_a100_sun_three_epoch_two_gpu_ddp_2026-07-21",
    "user_thread_env_superseding_diagnostics_2026-07-22",
    "user_ddp_profile_matrix_2026-07-22",
    "user_flash_profile_formal_progression_2026-07-23",
    "user_refiner_supersession26955_v2_2026-07-23",
    "user_epoch_eval_selection_supersession27104_v1_2026-07-23",
    "user_epoch_eval_selection_supersession27104_v2_bash42_2026-07-24",
    "user_wq_parent_csp_probe_and_bridge_v2_2026-07-24",
    "user_wq_parent_csp_probe_sup27407_v1_2026-07-24",
    "user_wq_parent_csp_sun256_v1_2026-07-24",
    "user_iclr_mlip_free_wq_mechanism64_2026-07-25",
    "user_wq_existing22_chgnet_sun_v1_2026-07-25",
    "user_wq_existing22_mp_completion_v1_2026-07-25",
    "user_wq_schedule_correct_bridge_parity_v1_2026-07-26",
    "user_wq_schedule_correct_bridge_parity_sup28054_v1_2026-07-26",
    "user_wq_wyckoff_tangent_bridge_preflight_v1_remote_install_2026-07-26",
    "user_wq_wyckoff_tangent_bridge_preflight_v1_audit_amendment_and_submit_2026-07-26",
    "user_wq_wyckoff_chart_retraction_preflight_sup28185_v2_2026-07-26",
    "user_wq_wyckoff_chart_retraction_confirmatory256_v1_local_preparation_2026-07-26",
    "user_wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1_2026-07-27",
    "user_wq_wyckoff_identity_mechanics_sup28195_v1_2026-07-27",
    "user_wq_llm_charge_stop_paired64_v1_2026-07-27",
    "user_wq_formula_plan_sft_pilot_v1_2026-07-27",
    "user_wq_formula_plan_sft_one_epoch_v2_2026-07-27",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: str | Path) -> dict[str, Any]:
    location = Path(path).resolve()
    payload = json.loads(location.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Gate A artifact is not a mapping: {location}")
    return payload


def audit_source_sync_record(
    source_sync_record: str | Path,
    *,
    project_root: str | Path,
    authorized_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    record_path = Path(source_sync_record).resolve()
    root = Path(project_root).resolve()
    payload = _load(record_path)
    errors: list[str] = []
    if payload.get("schema") != "wqcodiff_source_manifest_v1":
        errors.append("source_manifest_schema")
    bundle_sha = str(payload.get("bundle_sha256", ""))
    if SHA256.fullmatch(bundle_sha) is None:
        errors.append("source_bundle_sha256")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        errors.append("source_file_manifest")
        files = []
    changed: list[str] = []
    missing: list[str] = []
    overrides = dict(authorized_overrides or {})
    for entry in files:
        if not isinstance(entry, Mapping) or not entry.get("path"):
            errors.append("malformed_source_entry")
            continue
        relative = str(entry["path"])
        path = (root / relative).resolve()
        if root != path and root not in path.parents:
            errors.append(f"source_path_escape:{relative}")
            continue
        if not path.is_file():
            missing.append(relative)
            continue
        expected = overrides.get(relative, entry)
        if path.stat().st_size != int(expected["bytes"]) or sha256_file(path) != str(
            expected["sha256"]
        ):
            changed.append(relative)
    if missing:
        errors.append("source_files_missing")
    if changed:
        errors.append("source_files_changed")
    return {
        "schema": "crysllmgen_source_sync_audit_v1",
        "ok": not errors,
        "errors": errors,
        "source_sync_record": str(record_path),
        "source_sync_record_sha256": sha256_file(record_path),
        "source_bundle_sha256": bundle_sha,
        "registered_files": len(files),
        "missing": missing,
        "changed": changed,
        "authorized_override_paths": sorted(overrides),
    }


def audit_authorized_patch_record(
    *,
    project_root: str | Path,
    manifest_sha256: str,
    base_source_bundle_sha256: str,
) -> dict[str, Any]:
    """Verify one exact installed patch record and every overlaid source byte."""

    if SHA256.fullmatch(manifest_sha256) is None:
        raise ValueError("execution patch is not one lowercase SHA256")
    root = Path(project_root).resolve()
    record_path = (
        root
        / ".artifacts"
        / "source_sync"
        / f"authorized_patch_{manifest_sha256}.json"
    )
    if not record_path.is_file():
        raise ValueError("authorized execution-patch installation record is missing")
    payload = _load(record_path)
    errors: list[str] = []
    if payload.get("schema") != "wqcodiff_authorized_patch_v1":
        errors.append("patch_schema")
    if payload.get("manifest_sha256") != manifest_sha256:
        errors.append("patch_manifest_sha256")
    if payload.get("base_source_bundle_sha256") != base_source_bundle_sha256:
        errors.append("patch_base_source")
    if payload.get("authorization") not in PATCH_ALLOWED_AUTHORIZATIONS:
        errors.append("patch_authorization")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        errors.append("patch_files")
        entries = []
    overrides: dict[str, dict[str, Any]] = {}
    changed: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            errors.append("patch_entry")
            continue
        relative = str(entry.get("path", ""))
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.parts[0] not in PATCH_ALLOWED_TOP_LEVEL
            or relative in overrides
            or SHA256.fullmatch(str(entry.get("sha256", ""))) is None
        ):
            errors.append(f"patch_path:{relative}")
            continue
        location = (root / relative).resolve()
        if root != location and root not in location.parents:
            errors.append(f"patch_path_escape:{relative}")
            continue
        expected = {
            "bytes": int(entry.get("bytes", -1)),
            "sha256": str(entry.get("sha256", "")),
        }
        overrides[relative] = expected
        if (
            not location.is_file()
            or location.stat().st_size != expected["bytes"]
            or sha256_file(location) != expected["sha256"]
        ):
            changed.append(relative)
    if changed:
        errors.append("patch_files_changed")
    return {
        "schema": "crysllmgen_authorized_execution_patch_audit_v1",
        "ok": not errors,
        "errors": errors,
        "record": str(record_path),
        "record_sha256": sha256_file(record_path),
        "manifest_sha256": manifest_sha256,
        "base_source_bundle_sha256": base_source_bundle_sha256,
        "authorization": payload.get("authorization"),
        "registered_files": len(entries),
        "changed": changed,
        "overrides": overrides,
    }


def _artifact(
    label: str,
    path: str | Path,
    *,
    schema: str,
    predicate: bool,
) -> tuple[dict[str, Any], list[str]]:
    location = Path(path).resolve()
    payload = _load(location)
    errors: list[str] = []
    if payload.get("schema") != schema:
        errors.append(f"{label}:schema")
    if not predicate:
        errors.append(f"{label}:gate")
    return (
        {
            "path": str(location),
            "sha256": sha256_file(location),
            "schema": payload.get("schema"),
        },
        errors,
    )


def build_gate_a_lock(
    *,
    project_root: str | Path,
    source_sync_record: str | Path,
    protocol_path: str | Path,
    registry_path: str | Path,
    parity_audit_path: str | Path,
    llama_report_path: str | Path,
    grammar_report_path: str | Path,
    atom_smoke_report_path: str | Path,
    wq_smoke_report_path: str | Path,
    constrained_report_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    protocol = load_protocol_v4(protocol_path)
    registry = load_registry_v2(registry_path)
    if registry.protocol.sha256 != protocol.sha256:
        raise ValueError("Gate A protocol and registry bindings disagree")
    source = audit_source_sync_record(source_sync_record, project_root=root)
    errors = list(source["errors"])
    artifacts: dict[str, Any] = {}

    parity = _load(parity_audit_path)
    artifacts["disabled_extension_parity"], found = _artifact(
        "disabled_extension_parity",
        parity_audit_path,
        schema="crysllmgen_disabled_extension_parity_audit_v1",
        predicate=bool(parity.get("ok")) and not parity.get("errors"),
    )
    errors.extend(found)
    llama = _load(llama_report_path)
    artifacts["llama_offline_forward"], found = _artifact(
        "llama_offline_forward",
        llama_report_path,
        schema="crysllmgen_llama_gate_a_report_v1",
        predicate=(
            bool(llama.get("ok"))
            and bool(llama.get("offline"))
            and int(llama.get("blas_threads", -1)) == 1
            and bool(llama.get("model", {}).get("adapter_changes_logits"))
        ),
    )
    errors.extend(found)
    grammar = _load(grammar_report_path)
    artifacts["grammar"], found = _artifact(
        "grammar",
        grammar_report_path,
        schema="crysllmgen_wq_grammar_gate_report_v1",
        predicate=(
            bool(grammar.get("ok"))
            and int(grammar.get("transitions", {}).get("transitions", -1)) == 1_000_000
            and int(grammar.get("transitions", {}).get("illegal_generated", -1)) == 0
            and int(grammar.get("catalog", {}).get("roundtrip_passed", -1)) == 230
        ),
    )
    errors.extend(found)

    smoke_payloads: dict[str, dict[str, Any]] = {}
    for label, path, representation in (
        ("atom_lora_smoke", atom_smoke_report_path, "atom"),
        ("wq_lora_smoke", wq_smoke_report_path, "wyckoff"),
    ):
        payload = _load(path)
        smoke_payloads[label] = payload
        artifacts[label], found = _artifact(
            label,
            path,
            schema="crysllmgen_lora_training_report_v1",
            predicate=(
                payload.get("run_role") == "smoke"
                and payload.get("representation") == representation
                and int(payload.get("optimizer", {}).get("completed_global_step", -1)) == 100
                and int(payload.get("runtime", {}).get("threads", -1)) == 1
                and bool(payload.get("runtime", {}).get("offline"))
                and bool(payload.get("model", {}).get("adapter_sha256"))
            ),
        )
        errors.extend(found)

    constrained = _load(constrained_report_path)
    wq_adapter_sha = str(
        smoke_payloads["wq_lora_smoke"].get("model", {}).get("adapter_sha256", "")
    )
    artifacts["constrained_256"], found = _artifact(
        "constrained_256",
        constrained_report_path,
        schema="crysllmgen_constrained_gate_report_v1",
        predicate=(
            bool(constrained.get("ok"))
            and int(constrained.get("submitted_attempts", -1)) == 256
            and int(constrained.get("terminal_attempts", -1)) == 256
            and int(constrained.get("parsed_attempts", -1)) == 256
            and int(constrained.get("topology_legal_attempts", -1)) == 256
            and not bool(constrained.get("retry_or_replacement_used", True))
            and constrained.get("model", {}).get("adapter_model_sha256")
            == wq_adapter_sha
        ),
    )
    errors.extend(found)
    return {
        "schema": "crysllmgen_gate_a_lock_v1",
        "ok": not errors,
        "errors": errors,
        "project_root": str(root),
        "source": source,
        "protocol": {
            "path": str(protocol.path),
            "sha256": protocol.sha256,
        },
        "registry": {
            "path": str(registry.path),
            "sha256": registry.sha256,
        },
        "artifacts": artifacts,
        "training_unblocked": not errors,
        "retry_or_replacement_used": False,
    }


@dataclasses.dataclass(frozen=True, slots=True)
class GateALock:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    execution_patch: Mapping[str, Any] | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        project_root: str | Path,
        protocol_path: str | Path,
        execution_patch_manifest_sha256: str | None = None,
    ) -> "GateALock":
        location = Path(path).resolve()
        payload = _load(location)
        if payload.get("schema") != "crysllmgen_gate_a_lock_v1":
            raise ValueError("invalid Gate A lock schema")
        if not bool(payload.get("ok")) or not bool(payload.get("training_unblocked")):
            raise ValueError("Gate A did not unblock training")
        if payload.get("errors"):
            raise ValueError("Gate A lock contains errors")
        protocol = load_protocol_v4(protocol_path)
        if payload.get("protocol", {}).get("sha256") != protocol.sha256:
            raise ValueError("Gate A lock/protocol hash mismatch")
        patch_sha256 = execution_patch_manifest_sha256 or os.environ.get(
            "WQ_EXECUTION_PATCH_SHA256"
        )
        patch = None
        overrides = None
        if patch_sha256 is not None:
            patch = audit_authorized_patch_record(
                project_root=project_root,
                manifest_sha256=patch_sha256,
                base_source_bundle_sha256=str(
                    payload["source"]["source_bundle_sha256"]
                ),
            )
            if not patch["ok"]:
                raise ValueError("authorized execution patch changed after install")
            overrides = patch["overrides"]
        source = audit_source_sync_record(
            payload["source"]["source_sync_record"],
            project_root=project_root,
            authorized_overrides=overrides,
        )
        if not source["ok"] or source["source_bundle_sha256"] != payload["source"][
            "source_bundle_sha256"
        ]:
            raise ValueError("Gate A source tree changed after lock")
        for label, artifact in payload.get("artifacts", {}).items():
            artifact_path = Path(str(artifact["path"]))
            if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
                raise ValueError(f"Gate A artifact changed after lock: {label}")
        return cls(location, sha256_file(location), payload, patch)

    @property
    def source_bundle_sha256(self) -> str:
        return str(self.payload["source"]["source_bundle_sha256"])
