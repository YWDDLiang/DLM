#!/usr/bin/env python3
"""Freeze the portable CPython 3.12 + exact-SMACT4 offline input bundle."""

from __future__ import annotations

import argparse
from email.parser import Parser
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
from typing import Any
import zipfile


IDENTITY = "smact4_400_runtime_v1"
SCHEMA = "smact4_400_runtime_bundle_manifest_v1"
EXPECTED_PYTHON_ARCHIVE_SHA256 = (
    "506191be3ee7bd190a8834dcdc1b3bc70aab50608deccc711935aa007239cabd"
)
EXPECTED_SMACT_WHEEL_SHA256 = (
    "e3eb968da92d47a8ef9a4af42af5589a6de61cccbca9d329937e1f4e402f0551"
)
EXPECTED_RESOLVED = {
    "numpy": "2.2.6",
    "pymatgen": "2025.4.24",
    "smact": "4.0.0",
    "transformers": "4.54.0",
}
EXPECTED_PROBE = {
    "contract_sha256": (
        "ad070f3ad8d025ec9d7918dc1aa84e3f8faaf4ee0a124fbe8602208d5609ca19"
    ),
    "oxidation_elements": 93,
    "python": "3.12.13",
    "smact": "4.0.0",
    "transformers": "4.54.0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"wheel METADATA cardinality mismatch: {path}")
        metadata = Parser().parsestr(
            archive.read(metadata_names[0]).decode("utf-8", errors="strict")
        )
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise RuntimeError(f"wheel identity missing: {path}")
    return canonical_name(name), version


def regular_files(directory: Path) -> list[Path]:
    values = sorted(directory.glob("*.whl"), key=lambda path: path.name.lower())
    if not values or any(not path.is_file() or path.is_symlink() for path in values):
        raise RuntimeError("wheelhouse must contain only regular wheel inputs")
    if any(directory.iterdir()) and len(values) != sum(1 for _ in directory.iterdir()):
        raise RuntimeError("wheelhouse contains a non-wheel payload")
    return values


def copy_input(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-archive", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--resolver-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    python_archive = args.python_archive.resolve()
    wheelhouse = args.wheelhouse.resolve()
    requirements = args.requirements.resolve()
    resolver_report = args.resolver_report.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if sha256_file(python_archive) != EXPECTED_PYTHON_ARCHIVE_SHA256:
        raise RuntimeError("portable CPython archive SHA mismatch")
    if not requirements.is_file() or not resolver_report.is_file():
        raise FileNotFoundError("requirements or resolver report missing")

    wheels = regular_files(wheelhouse)
    resolved: dict[str, str] = {}
    for wheel in wheels:
        name, version = wheel_identity(wheel)
        if name in resolved:
            raise RuntimeError(f"duplicate wheel distribution: {name}")
        resolved[name] = version
    for name, version in EXPECTED_RESOLVED.items():
        if resolved.get(name) != version:
            raise RuntimeError(f"resolved {name} identity mismatch")
    smact_wheel = next(wheel for wheel in wheels if wheel_identity(wheel)[0] == "smact")
    if sha256_file(smact_wheel) != EXPECTED_SMACT_WHEEL_SHA256:
        raise RuntimeError("SMACT wheel SHA mismatch")

    resolver = json.loads(resolver_report.read_text(encoding="utf-8"))
    installs = resolver.get("install")
    if not isinstance(installs, list):
        raise RuntimeError("standard resolver report has no install list")
    resolver_identities = {
        canonical_name(str(item["metadata"]["name"])): str(
            item["metadata"]["version"]
        )
        for item in installs
    }
    if resolver_identities != resolved:
        raise RuntimeError("standard resolver closure differs from wheelhouse")

    stage = output_root / "bundle"
    wheel_stage = stage / "wheelhouse"
    wheel_stage.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    files.append(copy_input(python_archive, stage / python_archive.name))
    files.append(copy_input(requirements, stage / "REQUIREMENTS.in"))
    files.append(copy_input(resolver_report, stage / "STANDARD_RESOLVER_REPORT.json"))
    for wheel in wheels:
        files.append(copy_input(wheel, wheel_stage / wheel.name))
    files = [
        {**entry, "path": Path(entry["path"]).relative_to(stage).as_posix()}
        for entry in files
    ]
    manifest = {
        "schema": SCHEMA,
        "identity": IDENTITY,
        "status": "pass",
        "target": {
            "implementation": "CPython",
            "platform": "x86_64-unknown-linux-gnu",
            "python_version": "3.12.13",
        },
        "python_release": {
            "archive": python_archive.name,
            "archive_sha256": EXPECTED_PYTHON_ARCHIVE_SHA256,
            "release_tag": "20260807",
            "source_url": (
                "https://github.com/astral-sh/python-build-standalone/releases/"
                "download/20260807/cpython-3.12.13%2B20260807-x86_64-unknown-"
                "linux-gnu-install_only_stripped.tar.gz"
            ),
        },
        "wheel_count": len(wheels),
        "resolved_distributions": dict(sorted(resolved.items())),
        "local_offline_probe": EXPECTED_PROBE,
        "standard_resolver_report_sha256": sha256_file(resolver_report),
        "files": sorted(files, key=lambda entry: entry["path"]),
        "remote_network_install": False,
        "global_environment_mutation": False,
        "user_site_isolation_required": True,
    }
    manifest_path = stage / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    archive_path = output_root / f"{IDENTITY}_bundle.tar.gz"
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in sorted(p for p in stage.rglob("*") if p.is_file()):
                    relative = path.relative_to(stage).as_posix()
                    info = tarfile.TarInfo(relative)
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    record = {
        "schema": "smact4_400_runtime_bundle_freeze_v1",
        "identity": IDENTITY,
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "python_archive_sha256": EXPECTED_PYTHON_ARCHIVE_SHA256,
        "smact_wheel_sha256": EXPECTED_SMACT_WHEEL_SHA256,
        "wheel_count": len(wheels),
        "standard_resolver_report_sha256": sha256_file(resolver_report),
    }
    (output_root / "FREEZE_RECORD.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
