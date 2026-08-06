#!/usr/bin/env python3
"""Build the deterministic cumulative WQ formula-plan one-epoch v2 patch."""

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
BUILD = (
    ROOT
    / "runs/remote_audit/20260727_wq_formula_plan_sft_one_epoch_v2"
    / "build_v2_test_entry_amendment"
)
PARENT_ROOT = (
    ROOT
    / "runs/remote_audit/20260727_wq_formula_plan_sft_one_epoch_v2"
    / "build_v2/archive_root_4fff0af9f281"
)
PARENT_MANIFEST = PARENT_ROOT / "patch_manifest.json"
PARENT_MANIFEST_SHA256 = (
    "4fff0af9f281fee1e87c20c2b23934229a1229e149a747fd289d5aca5c49950a"
)
BASE_SOURCE_BUNDLE_SHA256 = (
    "6eeb48310454199a37d622e9dade6ccbe0f5c280b14999a6ca5c26c7e11e5908"
)
AUTHORIZATION = "user_wq_formula_plan_sft_one_epoch_v2_2026-07-27"
IDENTITY = "wq_formula_plan_sft_one_epoch_v2"
CONTRACT = (
    "configs/experiments/wyckoff_codiffusion/"
    "wq_formula_plan_sft_one_epoch_v2.json"
)
CONTRACT_SHA256 = (
    "f50b934cb24b026815caea6ac387e4a0604ae32310ee9c46b67e0cc6e1cb0e46"
)
CURRENT_PATHS = {
    CONTRACT,
    "crystal_dlm/wqcodiff/crysllmgen/formula_plan.py",
    "crystal_dlm/wqcodiff/crysllmgen/gate.py",
    "crystal_dlm/wqcodiff/crysllmgen/inference.py",
    "diagnostics/authorization_records/wq_formula_plan_sft_one_epoch_v2.json",
    "diagnostics/build_wq_formula_plan_sft_one_epoch_v2_archive.py",
    "docs/experiment_program/20260727_wq_formula_plan_one_epoch_v2.md",
    "scripts/a800/install_authorized_patch.py",
    "scripts/a800/run_wq_formula_plan_paired64_v1.py",
    "scripts/a800/train_wq_formula_plan_lora.py",
    "scripts/a800/wq_formula_plan_sft_one_epoch_v2/submit_once.sh",
    "scripts/a800/wq_formula_plan_sft_one_epoch_v2/train_and_eval.sbatch",
    "tests/test_crysllmgen_formula_plan.py",
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
        raise ValueError("formula-plan one-epoch contract identity changed")
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
                "你先去把12 次全都死在 Wyckoff 计量不可达的硬约束状态这个解决，"
                "然后plan生成成功概率也有点低。把第一个问题解决了，"
                "训完一个epoch后看看效果"
            ),
        },
        "base_source_bundle_sha256": BASE_SOURCE_BUNDLE_SHA256,
        "cumulative_preservation": True,
        "parent_authorized_patch_manifest_sha256": PARENT_MANIFEST_SHA256,
        "execution_identity": IDENTITY,
        "contract_sha256": CONTRACT_SHA256,
        "audit_policy": "minimal_terminal_only",
        "scope": (
            "Install exact residual Wyckoff-count reachability and valid plan "
            "termination; reuse the immutable 61393-example chemistry-plan "
            "dataset; run one total epoch from the original epoch-3 adapter and "
            "one paired-64 development gate. Retry, replacement, repair, "
            "reranking, MLIP, SUN, external API, and automatic downstream remain "
            "forbidden."
        ),
        "files": files,
    }
    manifest_path = temporary / "patch_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    manifest_sha = sha256(manifest_path)
    archive_root = BUILD / f"archive_root_{manifest_sha[:12]}"
    temporary.rename(archive_root)
    archive_name = (
        f"wq_formula_plan_sft_one_epoch_v2_{manifest_sha[:12]}.tar.gz"
    )
    archive_path = BUILD / archive_name
    build_tar(archive_path, archive_root)
    transfer = {
        "schema": "wq_formula_plan_sft_one_epoch_archive_v2",
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
            "wq_formula_plan_sft_one_epoch_v2"
        ),
        "a800_staging_directory": (
            "/public/home/jiaosz/ywliang/ai4s/.staging/"
            "wq_formula_plan_sft_one_epoch_v2"
        ),
        "starteam_to_a800_port": 7001,
        "resources": {
            "a800": 1,
            "cpus": 8,
            "memory_gib": 64,
            "time_limit": "05:00:00",
        },
        "audit_policy": "minimal_terminal_only",
    }
    write_json_exclusive(BUILD / "transfer_manifest.json", transfer)
    print(json.dumps(transfer, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
