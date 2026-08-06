#!/usr/bin/env python3
"""Build the deterministic WTB-256 local-preparation archive.

The archive is cumulative over the exact patch installed for successful
development mechanics job28187.  Building it does not authorize either hop of
remote transfer, remote installation, Slurm claim creation, or submission.
Those actions require a later record bound to the exact resulting archive and
patch-manifest identities.
"""

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
    / "20260726_wq_wyckoff_chart_retraction_confirmatory256_v1"
)
BUILD = AUDIT / "build_local_preparation_v2_final"
PARENT_ROOT = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260726_wq_wyckoff_chart_retraction_preflight_sup28185_v2"
    / "build_transfer_preservation_v2"
    / "archive_root_227dd62635dc"
)
PARENT_MANIFEST = PARENT_ROOT / "patch_manifest.json"
PARENT_MANIFEST_SHA256 = (
    "227dd62635dc20ac79580810defeea2b5f47e399fb70913535c5809f5e876642"
)
AUTHORIZATION = (
    "user_wq_wyckoff_chart_retraction_confirmatory256_v1_"
    "local_preparation_2026-07-26"
)
AUTHORIZATION_SOURCE = (
    "diagnostics/authorization_records/"
    "wq_wyckoff_chart_retraction_confirmatory256_v1_local_preparation.json"
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
SUPERSEDED_LOCAL_DRAFT = {
    "status": "SUPERSEDED_LOCAL_DRAFT_NOT_TRANSFERRED",
    "build_directory": "build_local_preparation_v1",
    "archive": (
        "wq_wyckoff_chart_retraction_confirmatory256_v1_"
        "a0eb8f52bb39.tar.gz"
    ),
    "archive_sha256": (
        "9708bddb1d1164230515b45ffc9929a45d9b5e23b17c859d503a6fc56389e100"
    ),
    "patch_manifest_sha256": (
        "a0eb8f52bb39d7aadd3d2a06da794b38ffbf8262d6dc6a07b4cdfe47f8111c19"
    ),
    "scientific_contract_sha256": (
        "d8902634968b33a19698a6a1c733c96cfe2798b7fafaf8228de9c786b867bdbd"
    ),
    "remote_transfer_attempted": False,
    "remote_install_attempted": False,
    "slurm_claim_created": False,
    "slurm_submission_attempted": False,
    "scientific_attempts": 0,
}
IDENTITY = "wq_wyckoff_chart_retraction_confirmatory256_v1"
BUILDER = (
    "diagnostics/"
    "build_wq_wyckoff_chart_retraction_confirmatory256_v1_archive.py"
)
CURRENT_PATHS = {
    AUTHORIZATION_SOURCE,
    BUILDER,
    CONTRACT,
    "crystal_dlm/wqcodiff/crysllmgen/wtb_confirmatory.py",
    "docs/experiment_program/20260726_wtb256_training_and_paper_execution_plan.md",
    "docs/experiment_program/20260726_wtb256_training_and_paper_execution_tasks.md",
    "scripts/a800/install_authorized_patch.py",
    "scripts/a800/run_wq_wyckoff_chart_retraction_sources256_v1.py",
    "scripts/a800/run_wq_wyckoff_chart_retraction_arms256_v1.py",
    (
        "scripts/a800/"
        "summarize_wq_wyckoff_chart_retraction_confirmatory256_v1.py"
    ),
    (
        "scripts/a800/"
        "wq_wyckoff_chart_retraction_confirmatory256_v1/pipeline.sbatch"
    ),
    (
        "scripts/a800/"
        "wq_wyckoff_chart_retraction_confirmatory256_v1/submit_once.sh"
    ),
    "tests/test_crysllmgen_wtb_confirmatory256.py",
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
        "job28187 installed parent cumulative manifest",
    )
    require_hash(ROOT / CONTRACT, CONTRACT_SHA256, "WTB-256 scientific contract")
    contract = json.loads((ROOT / CONTRACT).read_text(encoding="utf-8"))
    authorization = contract.get("authorization") or {}
    if (
        contract.get("identity") != IDENTITY
        or contract.get("status")
        != "local_built_remote_execution_not_authorized"
        or any(
            not isinstance(authorization.get(key), bool)
            for key in (
                "remote_transfer_authorized",
                "remote_install_authorized",
                "slurm_submission_authorized",
            )
        )
        or any(
            bool(authorization[key])
            for key in (
                "remote_transfer_authorized",
                "remote_install_authorized",
                "slurm_submission_authorized",
            )
        )
    ):
        raise ValueError("WTB-256 local-only authorization boundary changed")
    for name, entry in contract["implementation"].items():
        require_hash(
            ROOT / entry["path"],
            entry["sha256"],
            f"WTB-256 implementation {name}",
        )

    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if (
        parent.get("schema") != "wqcodiff_authorized_patch_v1"
        or parent.get("base_source_bundle_sha256")
        != BASE_SOURCE_BUNDLE_SHA256
    ):
        raise ValueError("unexpected job28187 parent manifest")
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
        "local_draft_supersession": SUPERSEDED_LOCAL_DRAFT,
        "parent_authorized_patch_manifest_sha256": PARENT_MANIFEST_SHA256,
        "scope": (
            "Preserve every source byte authorized and installed for successful "
            "development mechanics job28187; add the fresh WTB-256 contract, "
            "source and paired R/U/T runners, direct and exact S.U.N. evaluation "
            "pipeline, paired statistics, immutable promotion lock, tests, and "
            "one-job resource-safe wrapper. This record authorizes local "
            "preparation and deterministic archive construction only. Remote "
            "transfer, installation, Slurm claim/submission, training, retry, "
            "replacement, rerank, best-of, MP API in Slurm, and automatic "
            "follow-up remain unauthorized."
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
        "wq_wyckoff_chart_retraction_confirmatory256_v1_"
        f"{manifest_sha256[:12]}.tar.gz"
    )
    archive_path = BUILD / archive_name
    build_tar(archive_path, archive_root)
    transfer = {
        "schema": (
            "wq_wyckoff_chart_retraction_confirmatory256_"
            "local_archive_identity_v1"
        ),
        "status": "LOCAL_ONLY_REMOTE_TRANSFER_NOT_AUTHORIZED",
        "archive": archive_name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_regular_files": len(files) + 1,
        "archive_root": archive_root.name,
        "archive_sha256": sha256(archive_path),
        "authorized_patch_files": len(files),
        "authorization_record_sha256": authorization_sha256,
        "local_draft_supersession": SUPERSEDED_LOCAL_DRAFT,
        "patch_manifest_sha256": manifest_sha256,
        "parent_execution_patch_sha256": PARENT_MANIFEST_SHA256,
        "scientific_contract_sha256": CONTRACT_SHA256,
        "starteam_staging_directory": (
            "/zhdd/home/ywliang/a800/staging/"
            "wq_wyckoff_chart_retraction_confirmatory256_v1"
        ),
        "a800_staging_directory": (
            "/public/home/jiaosz/ywliang/ai4s/.staging/"
            "wq_wyckoff_chart_retraction_confirmatory256_v1"
        ),
        "starteam_to_a800_port": 7001,
        "starteam_to_a800_private_key_flag": False,
        "transfer_authorization": {
            "local_to_starteam5090_once": False,
            "starteam5090_to_a800_once": False,
            "reusable": False,
            "scatter_file_transfer": False,
        },
        "authorized_remote_actions": {
            "atomic_install": False,
            "installed_byte_audit": False,
            "exact_import_and_unit_test_gate": False,
            "slurm_claim": False,
            "slurm_submission": False,
            "training": False,
        },
    }
    write_json_exclusive(BUILD / "transfer_manifest.json", transfer)
    print(json.dumps(transfer, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
