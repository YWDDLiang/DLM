#!/usr/bin/env python3
"""Build the deterministic permutation-safe WTB mechanics patch archive."""

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
    / "20260727_wq_wyckoff_identity_mechanics_sup28195_v1"
)
BUILD = AUDIT / "build_local_preparation_v1"
PARENT_ROOT = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260727_wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1"
    / "build_local_preparation_v1"
    / "archive_root_1d15a08a07d6"
)
PARENT_MANIFEST = PARENT_ROOT / "patch_manifest.json"
PARENT_MANIFEST_SHA256 = (
    "1d15a08a07d6312fff636b0ec25078580aafd37c1e5e931783d801dbcf7c2f10"
)
AUTHORIZATION = "user_wq_wyckoff_identity_mechanics_sup28195_v1_2026-07-27"
AUTHORIZATION_SOURCE = (
    "diagnostics/authorization_records/"
    "wq_wyckoff_identity_mechanics_sup28195_v1.json"
)
BASE_SOURCE_BUNDLE_SHA256 = (
    "6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908"
)
CONTRACT = (
    "configs/experiments/wyckoff_codiffusion/"
    "wq_wyckoff_identity_mechanics_sup28195_v1.json"
)
CONTRACT_SHA256 = (
    "6ca0d3f292aff8fcaedd97566fc5b2367bc17c24f6a075b5a47490613cb5663d"
)
EXECUTION_IDENTITY = "wq_wyckoff_identity_mechanics_sup28195_v1"
BUILDER = "diagnostics/build_wq_wyckoff_identity_mechanics_sup28195_v1_archive.py"
CURRENT_PATHS = {
    AUTHORIZATION_SOURCE,
    BUILDER,
    CONTRACT,
    "crystal_dlm/wqcodiff/crysllmgen/gate.py",
    "crystal_dlm/wqcodiff/crysllmgen/wtb_identity_v2.py",
    "docs/experiment_program/20260726_wtb256_training_and_paper_execution_plan.md",
    (
        "docs/experiment_program/"
        "20260727_wtb256_permutation_safe_identity_and_next_execution_plan.md"
    ),
    "scripts/a800/install_authorized_patch.py",
    "scripts/a800/run_wq_wyckoff_identity_mechanics_sup28195_v1.py",
    (
        "scripts/a800/wq_wyckoff_identity_mechanics_sup28195_v1/"
        "mechanics.sbatch"
    ),
    (
        "scripts/a800/wq_wyckoff_identity_mechanics_sup28195_v1/"
        "submit_once.sh"
    ),
    "tests/test_crysllmgen_gate_lock.py",
    "tests/test_crysllmgen_wtb_identity_mechanics_sup28195.py",
    "tests/test_crysllmgen_wtb_identity_v2.py",
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
        "job28195 parent cumulative manifest",
    )
    require_hash(ROOT / CONTRACT, CONTRACT_SHA256, "identity-v2 contract")
    authorization = json.loads(
        (ROOT / AUTHORIZATION_SOURCE).read_text(encoding="utf-8")
    )
    if (
        authorization.get("execution_identity") != EXECUTION_IDENTITY
        or authorization.get("supersedes_failed_job_id") != 28195
        or authorization.get("user_quote")
        != (
            "与原顺序签名比较这个太严苛了，我觉得可以去掉，或者不作为拦截，"
            "然后你看看接下来该做什么，然后就去做"
        )
        or authorization["scientific_boundary"].get("job28195_reinterpreted")
        is not False
        or authorization["scientific_boundary"].get("confirmatory_evidence")
        is not False
        or authorization["scientific_boundary"].get("training") is not False
    ):
        raise ValueError("identity-v2 exact user authorization boundary changed")
    contract = json.loads((ROOT / CONTRACT).read_text(encoding="utf-8"))
    if (
        contract.get("identity") != EXECUTION_IDENTITY
        or contract.get("status") != "authorized_development_execution"
        or contract["lineage"].get("job28195_reinterpreted") is not False
        or contract["lineage"].get("confirmatory_evidence") is not False
        or contract["matrix"].get("source_identity_audit_attempts") != 256
        or contract["matrix"].get("mechanics_attempts_per_arm") != 32
        or list(contract["matrix"].get("arms", {})) != ["R", "U", "T"]
        or contract["identity_contract"].get("composition_identity")
        != "element_count_multiset"
        or contract["identity_contract"].get(
            "legacy_ordered_atom_mismatch_blocking"
        )
        is not False
    ):
        raise ValueError("identity-v2 development contract changed")
    for name, entry in contract["implementation"].items():
        require_hash(
            ROOT / entry["path"],
            entry["sha256"],
            f"identity-v2 implementation {name}",
        )

    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if (
        parent.get("schema") != "wqcodiff_authorized_patch_v1"
        or parent.get("base_source_bundle_sha256")
        != BASE_SOURCE_BUNDLE_SHA256
        or len(parent.get("files", [])) != 156
    ):
        raise ValueError("unexpected job28195 parent manifest")
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
        "supersedes_failed_job_id": 28195,
        "superseded_execution_patch_sha256": PARENT_MANIFEST_SHA256,
        "job28195_terminal_audit_sha256": (
            "124bb6e02d612687cd25a21b57b57e64773eff5836c788b8f5998754f1da76c9"
        ),
        "parent_authorized_patch_manifest_sha256": PARENT_MANIFEST_SHA256,
        "scope": (
            "Preserve all bytes installed for job28195 and all job28195 "
            "evidence. Replace only representation-sensitive ordered-atom "
            "composition identity with an exact element-count multiset while "
            "keeping canonical proposal, atom count, and exact species-Wyckoff "
            "topology fail-closed. Run one development-only 256-source identity "
            "audit plus 32-cell R/U/T mechanics regression. New proposals, "
            "CrysLLMGen metrics, CHGNet, S.U.N., training, retry, replacement, "
            "scientific reinterpretation, and automatic confirmatory submission "
            "remain forbidden."
        ),
        "development_contract_sha256": CONTRACT_SHA256,
        "files": files,
    }
    manifest_path = temporary_root / "patch_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    manifest_sha256 = sha256(manifest_path)
    archive_root = BUILD / f"archive_root_{manifest_sha256[:12]}"
    temporary_root.rename(archive_root)
    archive_name = (
        "wq_wyckoff_identity_mechanics_sup28195_v1_"
        f"{manifest_sha256[:12]}.tar.gz"
    )
    archive_path = BUILD / archive_name
    build_tar(archive_path, archive_root)
    transfer = {
        "schema": "wq_wyckoff_identity_mechanics_archive_identity_v1",
        "status": "AUTHORIZED_ONCE_NOT_YET_TRANSFERRED",
        "execution_identity": EXECUTION_IDENTITY,
        "supersedes_failed_job_id": 28195,
        "archive": archive_name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_regular_files": len(files) + 1,
        "archive_root": archive_root.name,
        "archive_sha256": sha256(archive_path),
        "authorized_patch_files": len(files),
        "authorization_record_sha256": authorization_sha256,
        "patch_manifest_sha256": manifest_sha256,
        "parent_execution_patch_sha256": PARENT_MANIFEST_SHA256,
        "development_contract_sha256": CONTRACT_SHA256,
        "job28195_terminal_audit_sha256": (
            "124bb6e02d612687cd25a21b57b57e64773eff5836c788b8f5998754f1da76c9"
        ),
        "starteam_staging_directory": (
            "/zhdd/home/ywliang/a800/staging/"
            "wq_wyckoff_identity_mechanics_sup28195_v1"
        ),
        "a800_staging_directory": (
            "/public/home/jiaosz/ywliang/ai4s/.staging/"
            "wq_wyckoff_identity_mechanics_sup28195_v1"
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
            "automatic_confirmatory_submission": False,
        },
    }
    write_json_exclusive(BUILD / "transfer_manifest.json", transfer)
    print(json.dumps(transfer, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
