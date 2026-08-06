#!/usr/bin/env python3
"""Build the one authorized WTB-32 audit-and-submit amendment archive."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "runs/remote_audit/"
    "20260726_wq_wyckoff_tangent_bridge_preflight_v1"
)
BUILD = AUDIT / "build_submit_amendment_v1"
PARENT_ROOT = (
    AUDIT
    / "build_transfer_v1/"
    "archive_root_b8e987569f97"
)
PARENT_MANIFEST = PARENT_ROOT / "patch_manifest.json"
PARENT_MANIFEST_SHA256 = (
    "b8e987569f9794821921aea149b511f98d192673e39054848955a4cf2e39e134"
)
AUTHORIZATION = (
    "user_wq_wyckoff_tangent_bridge_preflight_v1_"
    "audit_amendment_and_submit_2026-07-26"
)
AUTHORIZATION_RECORD = (
    "runs/remote_audit/"
    "20260726_wq_wyckoff_tangent_bridge_preflight_v1/"
    "submission_authorization_record.json"
)
AUTHORIZATION_SOURCE = (
    "diagnostics/authorization_records/"
    "wq_wyckoff_tangent_bridge_preflight_v1_"
    "audit_amendment_and_submit.json"
)
AUTHORIZATION_SHA256 = (
    "14b7c095f03308fe18895d0ab0d1c6f03c4f24848392138c24b15b9f10939f42"
)
U_AUDIT = (
    ROOT
    / "runs/remote_audit/"
    "20260726_wq_schedule_correct_bridge_parity_sup28054_v1/"
    "terminal_audit.json"
)
U_AUDIT_SHA256 = (
    "4b77a10d632c33b53b2208d49db19f542081a7fd91d52f038ebe7e5280e2cf41"
)
U_AUDIT_BYTES = 9596
U_AUDIT_PAYLOAD = "amendment_payload/job28081_terminal_audit.json"
U_AUDIT_REMOTE = (
    "runs/remote_audit/"
    "20260726_wq_schedule_correct_bridge_parity_sup28054_v1/"
    "terminal_audit.json"
)
BASE_SOURCE_BUNDLE_SHA256 = (
    "6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908"
)
CONTRACT = (
    "configs/experiments/wyckoff_codiffusion/"
    "wq_wyckoff_tangent_bridge_preflight_v1.json"
)
CONTRACT_SHA256 = (
    "125a19b6eca74fca8c7637830263406568c48ff51a0878f44202af0c3fa136f2"
)
TANGENT = "crystal_dlm/wqcodiff/crysllmgen/tangent_bridge.py"
TANGENT_SHA256 = (
    "175c98dcf36ced9b4f0fee6a30845ba78c9a275114fdca3c55392bf9dafa97e8"
)
RUNNER = "scripts/a800/run_wq_wyckoff_tangent_bridge_preflight_v1.py"
RUNNER_SHA256 = (
    "edddded5bbf3aca16b47bfd3e060a374f60e22ff3473e852e88a8c72b7cec3c3"
)
SLURM = "scripts/a800/wq_wyckoff_tangent_bridge_preflight_v1/preflight.sbatch"
SLURM_SHA256 = (
    "8a46c3465e8eecf54b0e0abd85323a1926e92bf767ad2a71ae37050f52e7fc65"
)
SUBMIT = "scripts/a800/wq_wyckoff_tangent_bridge_preflight_v1/submit_once.sh"
SUBMIT_SHA256 = (
    "0d343a29f7ea31b7ee52d2dbc3b1ae6d3dc2a53a2cf0accbea379f81403b9710"
)
BUILDER = "diagnostics/build_wq_wyckoff_tangent_submit_amendment_archive.py"
NEW_PATHS = {
    AUTHORIZATION_SOURCE,
    BUILDER,
}
UPDATED_PARENT_PATHS = {
    "crystal_dlm/wqcodiff/crysllmgen/gate.py",
    "scripts/a800/install_authorized_patch.py",
    SLURM,
    SUBMIT,
    "tests/test_crysllmgen_gate_lock.py",
    "tests/test_wq_wyckoff_tangent_submission.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} identity changed: {observed}")


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_for(relative: str) -> Path:
    if relative == AUTHORIZATION_SOURCE:
        source = ROOT / AUTHORIZATION_RECORD
    elif relative in NEW_PATHS or relative in UPDATED_PARENT_PATHS:
        source = ROOT / relative
    else:
        source = PARENT_ROOT / relative
    if source.is_file() and not source.is_symlink():
        return source
    raise FileNotFoundError(f"cumulative source path is missing or unsafe: {relative}")


def add_tar_entry(
    archive: tarfile.TarFile,
    source: Path,
    arcname: str,
    *,
    is_directory: bool,
) -> None:
    info = archive.gettarinfo(str(source), arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = (
        0o755
        if is_directory
        else (0o755 if os.access(source, os.X_OK) else 0o644)
    )
    if is_directory:
        archive.addfile(info)
    else:
        with source.open("rb") as handle:
            archive.addfile(info, handle)


def build_tar(archive_path: Path, archive_root: Path) -> None:
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                root_name = archive_root.name
                add_tar_entry(
                    archive,
                    archive_root,
                    root_name,
                    is_directory=True,
                )
                directories = sorted(
                    (path for path in archive_root.rglob("*") if path.is_dir()),
                    key=lambda path: path.relative_to(archive_root).as_posix(),
                )
                files = sorted(
                    (path for path in archive_root.rglob("*") if path.is_file()),
                    key=lambda path: path.relative_to(archive_root).as_posix(),
                )
                for directory in directories:
                    relative = directory.relative_to(archive_root).as_posix()
                    add_tar_entry(
                        archive,
                        directory,
                        f"{root_name}/{relative}",
                        is_directory=True,
                    )
                for file_path in files:
                    relative = file_path.relative_to(archive_root).as_posix()
                    add_tar_entry(
                        archive,
                        file_path,
                        f"{root_name}/{relative}",
                        is_directory=False,
                    )


def main() -> None:
    require_hash(
        PARENT_MANIFEST,
        PARENT_MANIFEST_SHA256,
        "parent cumulative manifest",
    )
    require_hash(
        ROOT / AUTHORIZATION_RECORD,
        AUTHORIZATION_SHA256,
        "submission authorization record",
    )
    require_hash(U_AUDIT, U_AUDIT_SHA256, "job28081 terminal audit")
    if U_AUDIT.stat().st_size != U_AUDIT_BYTES:
        raise ValueError("job28081 terminal-audit byte count changed")
    require_hash(ROOT / CONTRACT, CONTRACT_SHA256, "scientific contract")
    require_hash(ROOT / TANGENT, TANGENT_SHA256, "tangent bridge source")
    require_hash(ROOT / RUNNER, RUNNER_SHA256, "WTB-32 runner")
    require_hash(ROOT / SLURM, SLURM_SHA256, "WTB-32 Slurm script")
    require_hash(ROOT / SUBMIT, SUBMIT_SHA256, "WTB-32 submit-once wrapper")

    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if parent.get("schema") != "wqcodiff_authorized_patch_v1":
        raise ValueError("unexpected parent manifest schema")
    parent_paths = {str(entry["path"]) for entry in parent["files"]}
    relative_paths = sorted(parent_paths | NEW_PATHS)
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("cumulative patch path set is duplicated")
    for relative in relative_paths:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.parts[0] == "runs":
            raise ValueError(f"unsafe source-patch path: {relative}")

    BUILD.mkdir(parents=True, exist_ok=False)
    temporary_root = BUILD / "archive_root_building"
    temporary_root.mkdir()
    files: list[dict[str, Any]] = []
    for relative in relative_paths:
        source = source_for(relative)
        destination = temporary_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files.append(
            {
                "path": relative,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    manifest = {
        "schema": "wqcodiff_authorized_patch_v1",
        "authorization": AUTHORIZATION,
        "authorization_record": AUTHORIZATION_RECORD,
        "authorization_record_source": AUTHORIZATION_SOURCE,
        "authorization_record_sha256": AUTHORIZATION_SHA256,
        "base_source_bundle_sha256": BASE_SOURCE_BUNDLE_SHA256,
        "cumulative_preservation": True,
        "parent_authorized_patch_manifest_sha256": PARENT_MANIFEST_SHA256,
        "scope": (
            "Preserve the frozen WTB-32 scientific contract, tangent bridge, "
            "runner, source panel, parent checkpoint, and immutable U outputs. "
            "Amend only the exact submission authorization binding and Gate "
            "registration, carry the exact missing job28081 terminal audit as "
            "a non-source sidecar, and authorize one evaluation-only WTB-32 "
            "submission after all fail-closed gates pass. No training, new "
            "generation, U rerun, MLIP/API evaluation, retry, replacement, "
            "cancellation, or scientific-contract change is authorized."
        ),
        "scientific_contract_sha256": CONTRACT_SHA256,
        "sidecar_payloads": [
            {
                "archive_path": U_AUDIT_PAYLOAD,
                "exclusive_remote_path": U_AUDIT_REMOTE,
                "bytes": U_AUDIT_BYTES,
                "sha256": U_AUDIT_SHA256,
            }
        ],
        "files": files,
    }
    manifest_path = temporary_root / "patch_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    manifest_sha = sha256(manifest_path)
    archive_root = BUILD / f"archive_root_{manifest_sha[:12]}"
    temporary_root.rename(archive_root)

    sidecar = archive_root / U_AUDIT_PAYLOAD
    sidecar.parent.mkdir(parents=True)
    shutil.copy2(U_AUDIT, sidecar)
    require_hash(sidecar, U_AUDIT_SHA256, "archived job28081 terminal audit")

    archive_name = (
        "wq_wyckoff_tangent_bridge_submit_amendment_v1_"
        f"{manifest_sha[:12]}.tar.gz"
    )
    archive_path = BUILD / archive_name
    build_tar(archive_path, archive_root)
    archive_sha = sha256(archive_path)
    transfer = {
        "schema": "wq_wyckoff_tangent_bridge_submit_amendment_transfer_v1",
        "archive": archive_name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_regular_files": len(files) + 2,
        "archive_root": archive_root.name,
        "archive_sha256": archive_sha,
        "authorized_patch_files": len(files),
        "authorization_record_sha256": AUTHORIZATION_SHA256,
        "patch_manifest_sha256": manifest_sha,
        "parent_execution_patch_sha256": PARENT_MANIFEST_SHA256,
        "scientific_contract_sha256": CONTRACT_SHA256,
        "job28081_terminal_audit": {
            "archive_path": U_AUDIT_PAYLOAD,
            "exclusive_remote_path": U_AUDIT_REMOTE,
            "bytes": U_AUDIT_BYTES,
            "sha256": U_AUDIT_SHA256,
        },
        "starteam_staging_directory": (
            "/zhdd/home/ywliang/a800/staging/"
            "wq_wyckoff_tangent_bridge_submit_amendment_v1"
        ),
        "a800_staging_directory": (
            "/public/home/jiaosz/ywliang/ai4s/.staging/"
            "wq_wyckoff_tangent_bridge_submit_amendment_v1"
        ),
        "starteam_to_a800_port": 7001,
        "starteam_to_a800_private_key_flag": False,
        "transfer_authorization": {
            "local_to_starteam5090_once": True,
            "starteam5090_to_a800_once": True,
            "reusable": False,
            "scatter_file_transfer": False,
        },
        "authorized_remote_actions": {
            "atomic_install": True,
            "exclusive_terminal_audit_materialization": True,
            "installed_byte_audit": True,
            "exact_import_and_unit_test_gate": True,
            "slurm_claim": True,
            "slurm_submission": True,
            "evaluation_only": True,
            "training": False,
            "retry_or_replacement": False,
        },
    }
    transfer_path = BUILD / "transfer_manifest.json"
    write_json_exclusive(transfer_path, transfer)
    print(
        json.dumps(
            {
                "ok": True,
                "archive": str(archive_path),
                "archive_sha256": archive_sha,
                "archive_bytes": archive_path.stat().st_size,
                "archive_regular_files": len(files) + 2,
                "authorized_patch_files": len(files),
                "patch_manifest_sha256": manifest_sha,
                "transfer_manifest": str(transfer_path),
                "transfer_manifest_sha256": sha256(transfer_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
