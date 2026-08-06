#!/usr/bin/env python3
"""Build the single authorized cumulative WTB-32 v2 execution archive."""

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
    / "20260726_wq_wyckoff_chart_retraction_preflight_sup28185_v2"
)
BUILD = AUDIT / "build_transfer_v1"
PARENT_ROOT = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260726_wq_wyckoff_tangent_bridge_preflight_v1"
    / "build_transfer_v1"
    / "archive_root_b8e987569f97"
)
PARENT_MANIFEST = PARENT_ROOT / "patch_manifest.json"
PARENT_MANIFEST_SHA256 = (
    "b8e987569f9794821921aea149b511f98d192673e39054848955a4cf2e39e134"
)
AUTHORIZATION = (
    "user_wq_wyckoff_chart_retraction_preflight_sup28185_v2_2026-07-26"
)
AUTHORIZATION_SOURCE = (
    "diagnostics/authorization_records/"
    "wq_wyckoff_chart_retraction_preflight_sup28185_v2.json"
)
AUTHORIZATION_SHA256 = (
    "ba1acd31cda69a4388dca420796dd7bee7e7419239ce3b5405668e42bc6e8ce1"
)
BASE_SOURCE_BUNDLE_SHA256 = (
    "6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908"
)
CONTRACT = (
    "configs/experiments/wyckoff_codiffusion/"
    "wq_wyckoff_chart_retraction_preflight_sup28185_v2.json"
)
CONTRACT_SHA256 = (
    "518e44cc1a94334f8232ee54f4199a3c01436c0768defb9e61e2628a27324a6a"
)
RUNNER = (
    "scripts/a800/"
    "run_wq_wyckoff_chart_retraction_preflight_sup28185_v2.py"
)
RUNNER_SHA256 = (
    "63925aa0b877914b35240e55459026cf10e660665a14304a427c20253ef57a35"
)
LEGACY_RUNNER = "scripts/a800/run_wq_wyckoff_tangent_bridge_preflight_v1.py"
LEGACY_RUNNER_SHA256 = (
    "edddded5bbf3aca16b47bfd3e060a374f60e22ff3473e852e88a8c72b7cec3c3"
)
TANGENT = "crystal_dlm/wqcodiff/crysllmgen/tangent_bridge.py"
TANGENT_SHA256 = (
    "127e3c707b1bf79f2fc44d97bccecd8a6de3cb39b8e989a2806c5c7b377bfbaf"
)
RUNTIME = "crystal_dlm/wqcodiff/runtime.py"
RUNTIME_SHA256 = (
    "8b5ba104ee1be25ff7f8a14b703193b33920bfd71abd52b1ba1e0d082e909ea4"
)
SBATCH = (
    "scripts/a800/wq_wyckoff_chart_retraction_preflight_sup28185_v2/"
    "preflight.sbatch"
)
SBATCH_SHA256 = (
    "73e6dcefaccf8e7c07ca9adb84f58a52b80da8387e518a1f7c18f4a3f39daeee"
)
SUBMIT = (
    "scripts/a800/wq_wyckoff_chart_retraction_preflight_sup28185_v2/"
    "submit_once.sh"
)
SUBMIT_SHA256 = (
    "517e854c8cdc1b0bb6a9477becfac8ec45ac43e460321d9c8c13bf93f8491a90"
)
CURRENT_PATHS = {
    AUTHORIZATION_SOURCE,
    CONTRACT,
    RUNNER,
    TANGENT,
    RUNTIME,
    SBATCH,
    SUBMIT,
    "crystal_dlm/wqcodiff/crysllmgen/gate.py",
    "diagnostics/build_wq_wyckoff_chart_retraction_preflight_sup28185_v2_archive.py",
    "docs/experiment_program/"
    "20260726_wtb32_job28185_lattice_retraction_root_cause_and_v2_plan.md",
    "docs/experiment_program/20260726_wyckoff_tangent_bridge_execution_tasks.md",
    "docs/experiment_program/20260726_wyckoff_tangent_bridge_iclr_plan.md",
    "scripts/a800/install_authorized_patch.py",
    "tests/test_crysllmgen_chart_retraction_preflight_v2.py",
    "tests/test_crysllmgen_gate_lock.py",
    "tests/test_crysllmgen_tangent_bridge.py",
    "tests/test_crysllmgen_tangent_preflight_runner.py",
    "tests/test_wq_wyckoff_chart_retraction_submission_v2.py",
    "tests/test_wq_wyckoff_tangent_submission.py",
    "tests/test_wqcodiff_runtime.py",
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


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} identity changed: {observed}")


def main() -> None:
    require_hash(
        PARENT_MANIFEST,
        PARENT_MANIFEST_SHA256,
        "parent cumulative manifest",
    )
    require_hash(
        ROOT / AUTHORIZATION_SOURCE,
        AUTHORIZATION_SHA256,
        "authorization record",
    )
    require_hash(ROOT / CONTRACT, CONTRACT_SHA256, "v2 contract")
    require_hash(ROOT / RUNNER, RUNNER_SHA256, "v2 runner")
    require_hash(ROOT / LEGACY_RUNNER, LEGACY_RUNNER_SHA256, "legacy runner")
    require_hash(ROOT / TANGENT, TANGENT_SHA256, "tangent bridge source")
    require_hash(ROOT / RUNTIME, RUNTIME_SHA256, "runtime source")
    require_hash(ROOT / SBATCH, SBATCH_SHA256, "v2 Slurm script")
    require_hash(ROOT / SUBMIT, SUBMIT_SHA256, "v2 submit-once wrapper")

    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if parent.get("schema") != "wqcodiff_authorized_patch_v1":
        raise ValueError("unexpected parent manifest schema")
    parent_paths = {str(entry["path"]) for entry in parent["files"]}
    relative_paths = sorted(parent_paths | CURRENT_PATHS)
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

    manifest = {
        "schema": "wqcodiff_authorized_patch_v1",
        "authorization": AUTHORIZATION,
        "authorization_record": AUTHORIZATION_SOURCE,
        "authorization_record_sha256": AUTHORIZATION_SHA256,
        "base_source_bundle_sha256": BASE_SOURCE_BUNDLE_SHA256,
        "cumulative_preservation": True,
        "parent_authorized_patch_manifest_sha256": PARENT_MANIFEST_SHA256,
        "scope": (
            "Preserve all previously authorized source bytes and immutable "
            "job28081/job28185 evidence bindings; add the reviewed global "
            "chart retraction, exact primitive transform propagation, v2 "
            "development-only F/T32 mechanics runner, tests, and fail-closed "
            "single-job wrapper. One transfer, atomic install, remote gates, "
            "and one 1xA800/8CPU mechanics job are authorized. Confirmatory "
            "evaluation, training, generation, MLIP/API, retry, replacement, "
            "and U rerun remain forbidden."
        ),
        "scientific_contract_sha256": CONTRACT_SHA256,
        "files": files,
    }
    manifest_path = temporary_root / "patch_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    manifest_sha = sha256(manifest_path)
    archive_root = BUILD / f"archive_root_{manifest_sha[:12]}"
    temporary_root.rename(archive_root)
    archive_name = (
        "wq_wyckoff_chart_retraction_preflight_sup28185_v2_"
        f"{manifest_sha[:12]}.tar.gz"
    )
    archive_path = BUILD / archive_name
    build_tar(archive_path, archive_root)
    transfer = {
        "schema": "wq_wyckoff_chart_retraction_preflight_transfer_v2",
        "archive": archive_name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_regular_files": len(files) + 1,
        "archive_root": archive_root.name,
        "archive_sha256": sha256(archive_path),
        "authorized_patch_files": len(files),
        "authorization_record_sha256": AUTHORIZATION_SHA256,
        "patch_manifest_sha256": manifest_sha,
        "parent_execution_patch_sha256": PARENT_MANIFEST_SHA256,
        "scientific_contract_sha256": CONTRACT_SHA256,
        "starteam_staging_directory": (
            "/zhdd/home/ywliang/a800/staging/"
            "wq_wyckoff_chart_retraction_preflight_sup28185_v2"
        ),
        "a800_staging_directory": (
            "/public/home/jiaosz/ywliang/ai4s/.staging/"
            "wq_wyckoff_chart_retraction_preflight_sup28185_v2"
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
            "slurm_claim": True,
            "slurm_submission": True,
            "training": False,
            "confirmatory_evaluation": False,
        },
    }
    transfer_path = BUILD / "transfer_manifest.json"
    write_json_exclusive(transfer_path, transfer)
    print(json.dumps(transfer, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
