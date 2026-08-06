#!/usr/bin/env python3
"""Build the deterministic WTB-256 job28194 audit-sidecar supersession."""

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
    / "runs"
    / "remote_audit"
    / "20260727_wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1"
)
BUILD = AUDIT / "build_local_preparation_v1"
PARENT_ROOT = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260726_wq_wyckoff_chart_retraction_confirmatory256_v1"
    / "build_local_preparation_v2_final"
    / "archive_root_487cb3ead5c0"
)
PARENT_MANIFEST = PARENT_ROOT / "patch_manifest.json"
PARENT_MANIFEST_SHA256 = (
    "487cb3ead5c07fda70234481278189cced822c177762af13ee293fca4fa2369d"
)
AUTHORIZATION = (
    "user_wq_wyckoff_chart_retraction_confirmatory256_"
    "sup28194_v1_2026-07-27"
)
AUTHORIZATION_SOURCE = (
    "diagnostics/authorization_records/"
    "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1.json"
)
BASE_SOURCE_BUNDLE_SHA256 = (
    "6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908"
)
CONTRACT = (
    "configs/experiments/wyckoff_codiffusion/"
    "wq_wyckoff_chart_retraction_confirmatory256_v1.json"
)
CONTRACT_SHA256 = (
    "293c026d2f371b592a81e8e4d3982b4cb65ae3b0d90b82bf72a639caae24b77a"
)
JOB28194_AUDIT = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260727_wq_wyckoff_chart_retraction_confirmatory256_"
    "job28194_terminal_failure_v1.json"
)
JOB28194_AUDIT_SHA256 = (
    "9d98688876d13e0b84607dba25647c18b37d133a385303a9d4cd8103dd8bad6e"
)
EXECUTION_IDENTITY = (
    "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1"
)
SCIENTIFIC_IDENTITY = "wq_wyckoff_chart_retraction_confirmatory256_v1"
BUILDER = (
    "diagnostics/"
    "build_wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1_archive.py"
)
CURRENT_PATHS = {
    AUTHORIZATION_SOURCE,
    BUILDER,
    "crystal_dlm/wqcodiff/crysllmgen/gate.py",
    (
        "docs/experiment_program/"
        "20260727_wtb256_job28194_audit_sidecar_supersession_plan.md"
    ),
    "scripts/a800/install_authorized_patch.py",
    (
        "scripts/a800/"
        "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1/"
        "pipeline.sbatch"
    ),
    (
        "scripts/a800/"
        "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1/"
        "submit_once.sh"
    ),
    "tests/test_crysllmgen_gate_lock.py",
    "tests/test_crysllmgen_wtb_confirmatory256_sup28194.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_for(relative: str) -> Path:
    source = ROOT / relative if relative in CURRENT_PATHS else PARENT_ROOT / relative
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
        with gzip.GzipFile(
            fileobj=raw,
            mode="wb",
            mtime=0,
            filename="",
        ) as zipped:
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


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} identity changed: {observed}")


def main() -> None:
    require_hash(
        PARENT_MANIFEST,
        PARENT_MANIFEST_SHA256,
        "job28194 installed parent cumulative manifest",
    )
    require_hash(ROOT / CONTRACT, CONTRACT_SHA256, "WTB-256 scientific contract")
    require_hash(
        JOB28194_AUDIT,
        JOB28194_AUDIT_SHA256,
        "job28194 immutable terminal audit",
    )
    authorization = json.loads(
        (ROOT / AUTHORIZATION_SOURCE).read_text(encoding="utf-8")
    )
    if (
        authorization.get("user_quote") != "同意"
        or authorization.get("execution_identity") != EXECUTION_IDENTITY
        or authorization.get("scientific_identity") != SCIENTIFIC_IDENTITY
        or authorization.get("supersedes_failed_job_id") != 28194
        or authorization["scientific_scope_unchanged"].get("training") is not False
        or authorization["scientific_scope_unchanged"].get(
            "retry_or_replacement_allowed"
        )
        is not False
    ):
        raise ValueError("sup28194 exact user authorization boundary changed")
    contract = json.loads((ROOT / CONTRACT).read_text(encoding="utf-8"))
    if (
        contract.get("identity") != SCIENTIFIC_IDENTITY
        or contract.get("status")
        != "local_built_remote_execution_not_authorized"
        or contract["panel"].get("start_ordinal") != 512
        or contract["panel"].get("end_ordinal_inclusive") != 767
        or contract["panel"].get("attempts") != 256
        or list(contract["matrix"].get("arms", {})) != ["R", "U", "T"]
    ):
        raise ValueError("WTB-256 scientific contract changed")
    for name, entry in contract["implementation"].items():
        require_hash(
            ROOT / entry["path"],
            entry["sha256"],
            f"WTB-256 scientific implementation {name}",
        )

    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if (
        parent.get("schema") != "wqcodiff_authorized_patch_v1"
        or parent.get("base_source_bundle_sha256")
        != BASE_SOURCE_BUNDLE_SHA256
        or len(parent.get("files", [])) != 150
    ):
        raise ValueError("unexpected job28194 parent manifest")
    parent_paths = {str(entry["path"]) for entry in parent["files"]}
    relative_paths = sorted(parent_paths | CURRENT_PATHS)
    if not relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise ValueError("cumulative path set is empty or duplicated")
    for relative in relative_paths:
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or pure.parts[0] == "runs"
        ):
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

    authorization_sha256 = sha256(ROOT / AUTHORIZATION_SOURCE)
    manifest = {
        "schema": "wqcodiff_authorized_patch_v1",
        "authorization": AUTHORIZATION,
        "authorization_record": AUTHORIZATION_SOURCE,
        "authorization_record_sha256": authorization_sha256,
        "base_source_bundle_sha256": BASE_SOURCE_BUNDLE_SHA256,
        "cumulative_preservation": True,
        "execution_identity": EXECUTION_IDENTITY,
        "scientific_identity": SCIENTIFIC_IDENTITY,
        "supersedes_failed_job_id": 28194,
        "superseded_execution_patch_sha256": PARENT_MANIFEST_SHA256,
        "job28194_terminal_audit_sha256": JOB28194_AUDIT_SHA256,
        "parent_authorized_patch_manifest_sha256": PARENT_MANIFEST_SHA256,
        "scope": (
            "Preserve all 150 bytes-locked files installed for job28194 and "
            "the unchanged WTB-256 scientific contract. Correct only the "
            "installer/runtime Gate-A authorization registry mismatch, add "
            "registry-parity and exact sup28194 regression coverage, and add "
            "a new execution wrapper that performs GateALock.load before "
            "claim creation. Authorize one exact transfer/install/submission "
            "sequence for the evaluation-only scientific panel. Training, "
            "retry, replacement, reranking, best-of selection, altered "
            "denominators, and modification of job28194 evidence remain "
            "forbidden."
        ),
        "scientific_contract_sha256": CONTRACT_SHA256,
        "files": files,
    }
    manifest_path = temporary_root / "patch_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    manifest_sha256 = sha256(manifest_path)
    archive_root = BUILD / f"archive_root_{manifest_sha256[:12]}"
    temporary_root.rename(archive_root)
    archive_name = (
        "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1_"
        f"{manifest_sha256[:12]}.tar.gz"
    )
    archive_path = BUILD / archive_name
    build_tar(archive_path, archive_root)
    transfer = {
        "schema": (
            "wq_wyckoff_chart_retraction_confirmatory256_"
            "sup28194_archive_identity_v1"
        ),
        "status": "AUTHORIZED_ONCE_NOT_YET_TRANSFERRED",
        "execution_identity": EXECUTION_IDENTITY,
        "scientific_identity": SCIENTIFIC_IDENTITY,
        "supersedes_failed_job_id": 28194,
        "archive": archive_name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_regular_files": len(files) + 1,
        "archive_root": archive_root.name,
        "archive_sha256": sha256(archive_path),
        "authorized_patch_files": len(files),
        "authorization_record_sha256": authorization_sha256,
        "patch_manifest_sha256": manifest_sha256,
        "parent_execution_patch_sha256": PARENT_MANIFEST_SHA256,
        "scientific_contract_sha256": CONTRACT_SHA256,
        "job28194_terminal_audit_sha256": JOB28194_AUDIT_SHA256,
        "starteam_staging_directory": (
            "/zhdd/home/ywliang/a800/staging/"
            "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1"
        ),
        "a800_staging_directory": (
            "/public/home/jiaosz/ywliang/ai4s/.staging/"
            "wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1"
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
            "installed_byte_audit": True,
            "exact_import_and_unit_test_gate": True,
            "preclaim_gate_a_lock": True,
            "slurm_claim": True,
            "slurm_submission": True,
            "training": False,
        },
    }
    write_json_exclusive(BUILD / "transfer_manifest.json", transfer)
    print(json.dumps(transfer, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
