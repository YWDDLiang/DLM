#!/usr/bin/env python3
"""Build the single sidecar archive for the missing immutable job28185 audit."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260726_wq_wyckoff_chart_retraction_preflight_sup28185_v2"
)
BUILD = AUDIT_ROOT / "build_missing_failure_audit_amendment_v1"
SOURCE = (
    ROOT
    / "runs"
    / "remote_audit"
    / "20260726_wq_wyckoff_tangent_bridge_preflight_v1"
    / "terminal_audit_job28185.json"
)
SOURCE_SHA256 = (
    "2f686b881479f12b4abdc4c0ece217947c5aeb99c072f6119137e97887905f22"
)
REMOTE_PATH = (
    "runs/remote_audit/"
    "20260726_wq_wyckoff_tangent_bridge_preflight_v1/"
    "terminal_audit_job28185.json"
)


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


def add_entry(
    archive: tarfile.TarFile,
    source: Path,
    arcname: str,
    *,
    directory: bool,
) -> None:
    info = archive.gettarinfo(str(source), arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if directory else 0o644
    if directory:
        archive.addfile(info)
    else:
        with source.open("rb") as handle:
            archive.addfile(info, handle)


def main() -> None:
    if not SOURCE.is_file() or SOURCE.is_symlink():
        raise FileNotFoundError("immutable job28185 terminal audit is unavailable")
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("immutable job28185 terminal audit changed")

    BUILD.mkdir(parents=True, exist_ok=False)
    temporary = BUILD / "archive_root_building"
    payload_dir = temporary / "amendment_payload"
    payload_dir.mkdir(parents=True)
    payload = payload_dir / "terminal_audit_job28185.json"
    shutil.copy2(SOURCE, payload)

    manifest = {
        "schema": "wq_wyckoff_chart_retraction_missing_audit_amendment_v1",
        "status": "prepared_requires_explicit_user_authorization",
        "run_id": "20260720_0401-crysllmgen-wq-final-v3",
        "identity": "wq_wyckoff_chart_retraction_preflight_sup28185_v2",
        "scientific_identity_changed": False,
        "scientific_attempts": 0,
        "claim_created": False,
        "slurm_submitted": False,
        "reason": (
            "The already-transferred cumulative v2 source archive deliberately "
            "excluded runs/, but the immutable locally completed job28185 "
            "terminal audit had not previously been materialized on A800."
        ),
        "sidecar": {
            "archive_path": "amendment_payload/terminal_audit_job28185.json",
            "exclusive_remote_path": REMOTE_PATH,
            "bytes": payload.stat().st_size,
            "sha256": SOURCE_SHA256,
        },
        "authorized_effect_if_user_approves": (
            "transfer this archive once per frozen hop, extract to a unique "
            "staging directory, and exclusively materialize only the exact "
            "sidecar at the frozen remote path"
        ),
        "forbidden": {
            "overwrite_existing_remote_path": True,
            "source_patch_change": True,
            "scientific_contract_change": True,
            "retry_or_replacement": True,
            "training_or_generation": True,
        },
    }
    write_json_exclusive(temporary / "amendment_manifest.json", manifest)
    manifest_sha = sha256(temporary / "amendment_manifest.json")
    archive_root = BUILD / f"archive_root_{manifest_sha[:12]}"
    temporary.rename(archive_root)

    archive_name = (
        "wq_wtb32_v2_missing_job28185_terminal_audit_amendment_v1_"
        f"{manifest_sha[:12]}.tar.gz"
    )
    archive_path = BUILD / archive_name
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                add_entry(
                    archive,
                    archive_root,
                    archive_root.name,
                    directory=True,
                )
                add_entry(
                    archive,
                    archive_root / "amendment_manifest.json",
                    f"{archive_root.name}/amendment_manifest.json",
                    directory=False,
                )
                add_entry(
                    archive,
                    archive_root / "amendment_payload",
                    f"{archive_root.name}/amendment_payload",
                    directory=True,
                )
                add_entry(
                    archive,
                    archive_root
                    / "amendment_payload"
                    / "terminal_audit_job28185.json",
                    (
                        f"{archive_root.name}/amendment_payload/"
                        "terminal_audit_job28185.json"
                    ),
                    directory=False,
                )

    transfer = {
        "schema": "wq_wyckoff_chart_retraction_missing_audit_transfer_v1",
        "status": "prepared_requires_explicit_user_authorization",
        "archive": archive_name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "archive_root": archive_root.name,
        "amendment_manifest_sha256": manifest_sha,
        "job28185_terminal_audit_sha256": SOURCE_SHA256,
        "starteam_staging_directory": (
            "/zhdd/home/ywliang/a800/staging/"
            "wq_wtb32_v2_missing_job28185_terminal_audit_amendment_v1"
        ),
        "a800_staging_directory": (
            "/public/home/jiaosz/ywliang/ai4s/.staging/"
            "wq_wtb32_v2_missing_job28185_terminal_audit_amendment_v1"
        ),
        "starteam_to_a800_port": 7001,
        "starteam_to_a800_private_key_flag": False,
        "transfer_attempts_authorized": 0,
        "scientific_attempts": 0,
        "claim_created": False,
        "slurm_submitted": False,
    }
    transfer_path = BUILD / "transfer_manifest.json"
    write_json_exclusive(transfer_path, transfer)
    print(
        json.dumps(
            {
                "ok": True,
                "archive": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": sha256(archive_path),
                "archive_root": archive_root.name,
                "amendment_manifest_sha256": manifest_sha,
                "transfer_manifest": str(transfer_path),
                "transfer_manifest_sha256": sha256(transfer_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
