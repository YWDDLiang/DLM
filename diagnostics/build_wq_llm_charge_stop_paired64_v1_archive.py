#!/usr/bin/env python3
"""Build the deterministic cumulative WQ charge-stop pilot patch."""

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
AUDIT = ROOT / "runs/remote_audit/20260727_wq_llm_charge_stop_paired64_v1"
BUILD = AUDIT / "build_v1"
PARENT_ROOT = (
    ROOT
    / "runs/remote_audit/20260727_wq_wyckoff_identity_mechanics_sup28195_v1"
    / "build_local_preparation_v1/archive_root_9b79313fe7a4"
)
PARENT_MANIFEST = PARENT_ROOT / "patch_manifest.json"
PARENT_MANIFEST_SHA256 = (
    "9b79313fe7a4fc19a372e9a6c3af95785867cb8ad66537447a0bb096cc87e9c0"
)
BASE_SOURCE_BUNDLE_SHA256 = (
    "6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908"
)
AUTHORIZATION = "user_wq_llm_charge_stop_paired64_v1_2026-07-27"
IDENTITY = "wq_llm_charge_stop_paired64_v1"
CONTRACT = (
    "configs/experiments/wyckoff_codiffusion/"
    "wq_llm_charge_stop_paired64_v1.json"
)
CONTRACT_SHA256 = (
    "fdcd6b2b2a897fd4cc912e4d5f0e3053626c2cc19a7f510d5a7c0beb6d060307"
)
CURRENT_PATHS = {
    CONTRACT,
    "crystal_dlm/wqcodiff/crysllmgen/chemistry_constraint.py",
    "crystal_dlm/wqcodiff/crysllmgen/gate.py",
    "crystal_dlm/wqcodiff/crysllmgen/inference.py",
    "diagnostics/build_wq_llm_charge_stop_paired64_v1_archive.py",
    (
        "docs/experiment_program/"
        "20260727_wq_llm_mainline_and_charge_stop_execution.md"
    ),
    "scripts/a800/install_authorized_patch.py",
    "scripts/a800/run_wq_llm_charge_stop_paired64_v1.py",
    "scripts/a800/wq_llm_charge_stop_paired64_v1/pilot.sbatch",
    "scripts/a800/wq_llm_charge_stop_paired64_v1/submit_once.sh",
    "tests/test_crysllmgen_charge_stop_constraint.py",
    "tests/test_crysllmgen_gate_lock.py",
    "tests/test_wq_llm_charge_stop_pilot.py",
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
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"missing cumulative source: {relative}")
    return source


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


def build_tar(path: Path, root: Path) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                root_name = root.name
                add_tar_entry(archive, root, root_name, is_directory=True)
                for directory in sorted(
                    (value for value in root.rglob("*") if value.is_dir()),
                    key=lambda value: value.relative_to(root).as_posix(),
                ):
                    add_tar_entry(
                        archive,
                        directory,
                        f"{root_name}/{directory.relative_to(root).as_posix()}",
                        is_directory=True,
                    )
                for file_path in sorted(
                    (value for value in root.rglob("*") if value.is_file()),
                    key=lambda value: value.relative_to(root).as_posix(),
                ):
                    add_tar_entry(
                        archive,
                        file_path,
                        f"{root_name}/{file_path.relative_to(root).as_posix()}",
                        is_directory=False,
                    )


def main() -> None:
    if sha256(PARENT_MANIFEST) != PARENT_MANIFEST_SHA256:
        raise ValueError("parent cumulative patch identity changed")
    if sha256(ROOT / CONTRACT) != CONTRACT_SHA256:
        raise ValueError("WQ charge-stop contract identity changed")
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if (
        parent.get("schema") != "wqcodiff_authorized_patch_v1"
        or parent.get("base_source_bundle_sha256")
        != BASE_SOURCE_BUNDLE_SHA256
    ):
        raise ValueError("unexpected parent cumulative patch")

    parent_paths = {str(entry["path"]) for entry in parent["files"]}
    relative_paths = sorted(parent_paths | CURRENT_PATHS)
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
    temporary = BUILD / "archive_root_building"
    temporary.mkdir()
    files: list[dict[str, Any]] = []
    for relative in relative_paths:
        source = source_for(relative)
        destination = temporary / relative
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
        "authorization_source": {
            "date": "2026-07-27",
            "quote": (
                "好的，按照这个进行，然后开始实验，后续减少审计，"
                "现在我发现审计占了太多时间，反而挤占了训练的时间"
            ),
        },
        "base_source_bundle_sha256": BASE_SOURCE_BUNDLE_SHA256,
        "cumulative_preservation": True,
        "parent_authorized_patch_manifest_sha256": PARENT_MANIFEST_SHA256,
        "execution_identity": IDENTITY,
        "contract_sha256": CONTRACT_SHA256,
        "audit_policy": "minimal_four_artifacts",
        "scope": (
            "Install one charge-aware STOP constraint and run one paired "
            "baseline-vs-mask 64-cell proposal-only pilot on the existing "
            "epoch-3 WQ LLM. Parent diffusion, CrysLLMGen metrics, MLIP, SUN, "
            "MP API, training, retry, replacement, and automatic downstream "
            "remain forbidden."
        ),
        "files": files,
    }
    manifest_path = temporary / "patch_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    manifest_sha = sha256(manifest_path)
    archive_root = BUILD / f"archive_root_{manifest_sha[:12]}"
    temporary.rename(archive_root)
    archive_name = (
        f"wq_llm_charge_stop_paired64_v1_{manifest_sha[:12]}.tar.gz"
    )
    archive_path = BUILD / archive_name
    build_tar(archive_path, archive_root)
    transfer = {
        "schema": "wq_llm_charge_stop_paired64_archive_v1",
        "identity": IDENTITY,
        "archive": archive_name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "archive_regular_files": len(files) + 1,
        "patch_files": len(files),
        "patch_manifest_sha256": manifest_sha,
        "contract_sha256": CONTRACT_SHA256,
        "parent_patch_manifest_sha256": PARENT_MANIFEST_SHA256,
        "starteam_staging_directory": (
            "/zhdd/home/ywliang/a800/staging/"
            "wq_llm_charge_stop_paired64_v1"
        ),
        "a800_staging_directory": (
            "/public/home/jiaosz/ywliang/ai4s/.staging/"
            "wq_llm_charge_stop_paired64_v1"
        ),
        "starteam_to_a800_port": 7001,
        "resources": {"a800": 1, "cpus": 8, "memory_gib": 64},
        "audit_policy": "minimal_four_artifacts",
    }
    write_json_exclusive(BUILD / "transfer_manifest.json", transfer)
    print(json.dumps(transfer, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
