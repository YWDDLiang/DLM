#!/usr/bin/env python3
"""Build the one authorized cumulative bridge-parity job28054 supersession archive."""

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
    "20260726_wq_schedule_correct_bridge_parity_sup28054_v1"
)
BUILD = AUDIT / "build_execution_v2"
PARENT_ROOT = (
    ROOT
    / "runs/remote_audit/20260726_wq_schedule_correct_bridge_parity_v1/"
    "build_execution_v3/archive_root_9c51c4087abb"
)
PARENT_MANIFEST = PARENT_ROOT / "patch_manifest.json"
PARENT_MANIFEST_SHA256 = (
    "9c51c4087abb0361407021fb1a5dd2251939c2196ffcf2b619bdd7c4bbe503cb"
)
AUTHORIZATION = (
    "user_wq_schedule_correct_bridge_parity_sup28054_v1_2026-07-26"
)
AUTHORIZATION_RECORD = (
    "runs/remote_audit/"
    "20260726_wq_schedule_correct_bridge_parity_sup28054_v1/"
    "authorization_record.json"
)
AUTHORIZATION_SOURCE = (
    "diagnostics/authorization_records/"
    "wq_schedule_correct_bridge_parity_sup28054_v1_remote_execution.json"
)
AUTHORIZATION_SHA256 = (
    "d29f1d52945f3cbb4b6ccfd393dc1dc06f28f4c960d2d787eabb3691b962a914"
)
FAILURE_AUDIT_LOCAL = (
    "runs/remote_audit/20260726_wq_schedule_correct_bridge_parity_v1/"
    "job28054_terminal_failure_audit.json"
)
FAILURE_AUDIT_SOURCE = (
    "diagnostics/failure_audits/"
    "wq_schedule_correct_bridge_parity_job28054.json"
)
FAILURE_AUDIT_SHA256 = (
    "99b7b57b80c6d097adaa22bf3fa39d761573f4e76f6c74b89b55947081c95dca"
)
BASE_SOURCE_BUNDLE_SHA256 = (
    "6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908"
)
CONTRACT_SHA256 = (
    "18472b1f40147fb7f70c304647ed164c11469f577898007e72dfc103fd31fb26"
)
EXECUTION_PLAN_SHA256 = (
    "55277e0e754b4319b9e5a8a4dd23a3443849e127d6a02a889da466ace8bec0a8"
)
NEW_PATHS = {
    "configs/experiments/wyckoff_codiffusion/"
    "wq_schedule_correct_bridge_parity_sup28054_v1.json",
    "configs/experiments/wyckoff_codiffusion/"
    "wq_schedule_correct_bridge_parity_sup28054_execution_v1.json",
    AUTHORIZATION_SOURCE,
    FAILURE_AUDIT_SOURCE,
    "diagnostics/build_wq_schedule_correct_bridge_parity_sup28054_archive.py",
    "scripts/a800/wq_schedule_correct_bridge_parity_sup28054_v1/"
    "preflight.sbatch",
    "scripts/a800/wq_schedule_correct_bridge_parity_sup28054_v1/"
    "submit_once.sh",
    "tests/test_wq_schedule_correct_bridge_parity_sup28054_submission.py",
}
UPDATED_PARENT_PATHS = {
    "crystal_dlm/wqcodiff/crysllmgen/gate.py",
    "scripts/a800/install_authorized_patch.py",
    "tests/test_crysllmgen_gate_lock.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_for(relative: str) -> Path:
    if relative == AUTHORIZATION_SOURCE:
        source = ROOT / AUTHORIZATION_RECORD
    elif relative == FAILURE_AUDIT_SOURCE:
        source = ROOT / FAILURE_AUDIT_LOCAL
    elif relative in NEW_PATHS or relative in UPDATED_PARENT_PATHS:
        source = ROOT / relative
    else:
        source = PARENT_ROOT / relative
    if source.is_file():
        return source
    raise FileNotFoundError(f"cumulative source path is missing: {relative}")


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
    if sha256(PARENT_MANIFEST) != PARENT_MANIFEST_SHA256:
        raise ValueError("parent cumulative manifest identity changed")
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if parent.get("schema") != "wqcodiff_authorized_patch_v1":
        raise ValueError("unexpected parent manifest schema")
    parent_paths = {str(entry["path"]) for entry in parent["files"]}
    relative_paths = sorted(parent_paths | NEW_PATHS)
    if not relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise ValueError("cumulative path set is empty or duplicated")
    for relative in relative_paths:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.parts[0] == "runs":
            raise ValueError(f"unsafe archive path: {relative}")

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

    authorization_source = temporary_root / AUTHORIZATION_SOURCE
    if sha256(authorization_source) != AUTHORIZATION_SHA256:
        raise ValueError("archive authorization record identity changed")
    failure_audit_source = temporary_root / FAILURE_AUDIT_SOURCE
    if sha256(failure_audit_source) != FAILURE_AUDIT_SHA256:
        raise ValueError("job28054 failure-audit identity changed")
    contract = (
        temporary_root
        / "configs/experiments/wyckoff_codiffusion/"
        "wq_schedule_correct_bridge_parity_sup28054_v1.json"
    )
    if sha256(contract) != CONTRACT_SHA256:
        raise ValueError("scientific contract identity changed")
    execution_plan = (
        temporary_root
        / "configs/experiments/wyckoff_codiffusion/"
        "wq_schedule_correct_bridge_parity_sup28054_execution_v1.json"
    )
    if sha256(execution_plan) != EXECUTION_PLAN_SHA256:
        raise ValueError("execution-plan identity changed")

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
            "Preserve every previously authorized source byte and all job28054 "
            "evidence; add one new MLIP-free evaluation-only supersession whose "
            "only scientific change is binding the immutable 256-row source to "
            "its exact observed wqcodiff_generation_attempt_v1 schema. Reuse "
            "the released parent, selected eight proposals, 4x8 matrix, noise "
            "seeds, gates, and one-A800/eight-CPU/64-GiB/60-minute envelope. "
            "No generation, retry, variant replacement, API query, checkpoint "
            "selection, short training, or long training."
        ),
        "files": files,
    }
    manifest_path = temporary_root / "patch_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    manifest_sha = sha256(manifest_path)
    archive_root = BUILD / f"archive_root_{manifest_sha[:12]}"
    temporary_root.rename(archive_root)
    archive_name = (
        "wq_schedule_correct_bridge_parity_sup28054_v1_"
        f"{manifest_sha[:12]}.tar.gz"
    )
    archive_path = BUILD / archive_name
    build_tar(archive_path, archive_root)
    archive_sha = sha256(archive_path)
    transfer = {
        "schema": "wq_schedule_correct_bridge_parity_sup28054_transfer_v1",
        "archive": archive_name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_regular_files": len(files) + 1,
        "archive_root": archive_root.name,
        "archive_sha256": archive_sha,
        "authorized_patch_files": len(files),
        "authorization_record_sha256": AUTHORIZATION_SHA256,
        "job28054_failure_audit_sha256": FAILURE_AUDIT_SHA256,
        "patch_manifest_sha256": manifest_sha,
        "parent_execution_patch_sha256": PARENT_MANIFEST_SHA256,
        "scientific_contract_sha256": CONTRACT_SHA256,
        "starteam_staging_directory": (
            "/zhdd/home/ywliang/a800/staging/"
            "wq_schedule_correct_bridge_parity_sup28054_v1"
        ),
        "a800_staging_directory": (
            "/public/home/jiaosz/ywliang/ai4s/.staging/"
            "wq_schedule_correct_bridge_parity_sup28054_v1"
        ),
        "starteam_to_a800_port": 7001,
        "starteam_to_a800_private_key_flag": False,
        "transfer_authorization": {
            "local_to_starteam5090_once": True,
            "starteam5090_to_a800_once": True,
            "reusable": False,
        },
        "execution": {
            "evaluation_only": True,
            "single_non_array_slurm_job": True,
            "a800": 1,
            "cpus": 8,
            "memory_gib": 64,
            "time_limit_minutes": 60,
            "training": False,
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
                "archive_regular_files": len(files) + 1,
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
