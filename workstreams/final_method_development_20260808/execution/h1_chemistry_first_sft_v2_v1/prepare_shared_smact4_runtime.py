#!/usr/bin/env python3
"""Atomically prepare the frozen portable exact-SMACT4 runtime offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import stat
import subprocess
import tarfile
from typing import Any, Mapping


IDENTITY = "smact4_400_runtime_v1"
EXPECTED_WHEEL_SHA256 = (
    "e3eb968da92d47a8ef9a4af42af5589a6de61cccbca9d329937e1f4e402f0551"
)
EXPECTED_PYTHON_ARCHIVE_SHA256 = (
    "506191be3ee7bd190a8834dcdc1b3bc70aab50608deccc711935aa007239cabd"
)
EXPECTED_CONTRACT_SHA256 = (
    "ad070f3ad8d025ec9d7918dc1aa84e3f8faaf4ee0a124fbe8602208d5609ca19"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def safe_regular_tar_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    seen: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or member.name in seen
            or not member.isfile()
        ):
            raise RuntimeError(f"unsafe bundle archive member: {member.name!r}")
        seen.add(member.name)
    return members


def safe_python_tar_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    seen: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (
            not path.parts
            or path.parts[0] != "python"
            or path.is_absolute()
            or ".." in path.parts
            or member.name in seen
            or not (member.isfile() or member.issym())
        ):
            raise RuntimeError(f"unsafe portable-Python member: {member.name!r}")
        seen.add(member.name)
        if member.issym():
            if PurePosixPath(member.linkname).is_absolute():
                raise RuntimeError(f"absolute portable-Python link: {member.name!r}")
            resolved = posixpath.normpath(
                posixpath.join(posixpath.dirname(member.name), member.linkname)
            )
            resolved_path = PurePosixPath(resolved)
            if not resolved_path.parts or resolved_path.parts[0] != "python":
                raise RuntimeError(f"escaping portable-Python link: {member.name!r}")
    return members


def validate_bundle(
    bundle_root: Path, manifest: Mapping[str, Any], freeze: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        manifest.get("schema") != "smact4_400_runtime_bundle_manifest_v1"
        or manifest.get("identity") != IDENTITY
        or manifest.get("status") != "pass"
        or freeze.get("schema") != "smact4_400_runtime_bundle_freeze_v1"
        or freeze.get("identity") != IDENTITY
    ):
        raise RuntimeError("runtime bundle identity mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("runtime bundle file inventory is empty")
    expected: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise RuntimeError("invalid runtime bundle file inventory row")
        relative = str(item.get("path", ""))
        path = PurePosixPath(relative)
        if not path.parts or path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe runtime bundle inventory path: {relative!r}")
        target = bundle_root / Path(*path.parts)
        if not target.is_file() or target.is_symlink():
            raise RuntimeError(f"runtime bundle file missing: {relative}")
        if (
            target.stat().st_size != int(item.get("bytes", -1))
            or sha256_file(target) != item.get("sha256")
        ):
            raise RuntimeError(f"runtime bundle file identity mismatch: {relative}")
        expected.add(relative)
    observed = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    if observed != expected | {"BUNDLE_MANIFEST.json"}:
        raise RuntimeError("runtime bundle archive has missing or extra files")
    if int(manifest.get("wheel_count", -1)) != int(freeze.get("wheel_count", -2)):
        raise RuntimeError("runtime bundle wheel count mismatch")
    resolved = manifest.get("resolved_distributions")
    if not isinstance(resolved, Mapping) or resolved.get("smact") != "4.0.0":
        raise RuntimeError("runtime bundle resolved SMACT identity mismatch")
    probe = manifest.get("local_offline_probe")
    if (
        not isinstance(probe, Mapping)
        or probe.get("python") != "3.12.13"
        or probe.get("smact") != "4.0.0"
        or probe.get("transformers") != "4.54.0"
        or probe.get("contract_sha256") != EXPECTED_CONTRACT_SHA256
    ):
        raise RuntimeError("runtime bundle local offline probe mismatch")
    return {
        "file_count": len(files),
        "wheel_count": int(manifest["wheel_count"]),
        "standard_resolver_report_sha256": manifest[
            "standard_resolver_report_sha256"
        ],
    }


def freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode) & ~0o222
        path.chmod(mode)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-archive", type=Path, required=True)
    parser.add_argument("--bundle-freeze-record", type=Path, required=True)
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--project-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    bundle_archive = args.bundle_archive.resolve()
    freeze_record_path = args.bundle_freeze_record.resolve()
    wrapper_source = args.wrapper_source.resolve()
    source_root = args.project_source_root.resolve()
    output_root = args.output_root.resolve()
    build_root = output_root.with_name(f".{output_root.name}.building")
    if output_root.exists() or build_root.exists():
        raise FileExistsError("refusing to overwrite runtime or build evidence")
    freeze = read_object(freeze_record_path)
    if (
        freeze.get("schema") != "smact4_400_runtime_bundle_freeze_v1"
        or freeze.get("identity") != IDENTITY
        or bundle_archive.name != freeze.get("archive")
        or bundle_archive.stat().st_size != int(freeze.get("archive_bytes", -1))
        or sha256_file(bundle_archive) != freeze.get("archive_sha256")
        or freeze.get("python_archive_sha256") != EXPECTED_PYTHON_ARCHIVE_SHA256
        or freeze.get("smact_wheel_sha256") != EXPECTED_WHEEL_SHA256
    ):
        raise RuntimeError("runtime bundle freeze record mismatch")

    build_root.mkdir(parents=True)
    input_bundle = build_root / "input_bundle"
    input_bundle.mkdir()
    with tarfile.open(bundle_archive, "r:gz") as archive:
        members = safe_regular_tar_members(archive)
        archive.extractall(input_bundle, members=members)
    manifest_path = input_bundle / "BUNDLE_MANIFEST.json"
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != freeze.get("bundle_manifest_sha256")
    ):
        raise RuntimeError("runtime bundle manifest SHA mismatch")
    bundle = validate_bundle(input_bundle, read_object(manifest_path), freeze)
    manifest = read_object(manifest_path)

    python_name = str(manifest["python_release"]["archive"])
    python_archive = input_bundle / python_name
    if (
        not python_archive.is_file()
        or sha256_file(python_archive) != EXPECTED_PYTHON_ARCHIVE_SHA256
    ):
        raise RuntimeError("portable CPython identity mismatch")
    runtime_base = build_root / "runtime_base"
    runtime_base.mkdir()
    with tarfile.open(python_archive, "r:gz") as archive:
        members = safe_python_tar_members(archive)
        archive.extractall(runtime_base, members=members)
    base_python_relative = Path("runtime_base/python/bin/python3.12")
    base_python = build_root / base_python_relative
    if not base_python.is_file() or not os.access(base_python, os.X_OK):
        raise RuntimeError("portable CPython executable is missing")
    version_probe = subprocess.run(
        [
            str(base_python),
            "-c",
            "import json,sys; print(json.dumps({'version': list(sys.version_info[:3])}))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    version_payload = json.loads(version_probe.stdout.splitlines()[-1])
    if version_payload.get("version") != [3, 12, 13]:
        raise RuntimeError("portable CPython version mismatch")

    wheelhouse = input_bundle / "wheelhouse"
    wheels = sorted(wheelhouse.glob("*.whl"), key=lambda path: path.name.lower())
    if len(wheels) != int(manifest["wheel_count"]):
        raise RuntimeError("portable runtime wheelhouse cardinality mismatch")
    site = build_root / "site"
    site.mkdir()
    clean_env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "USE_TORCH": "0",
        "USE_TF": "0",
        "USE_FLAX": "0",
    }
    subprocess.run(
        [
            str(base_python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--target",
            str(site),
            *map(str, wheels),
        ],
        check=True,
        env=clean_env,
    )
    check_env = {**clean_env, "PYTHONPATH": str(site)}
    pip_check = subprocess.run(
        [str(base_python), "-m", "pip", "check"],
        check=True,
        capture_output=True,
        text=True,
        env=check_env,
    )

    wrapper = build_root / "python"
    shutil.copyfile(wrapper_source, wrapper)
    wrapper.chmod(0o700)
    base_python_path = build_root / "base_python_path.txt"
    base_python_path.write_text(base_python_relative.as_posix() + "\n", encoding="utf-8")
    smact_wheels = [
        path
        for path in wheels
        if path.name.lower().startswith("smact-4.0.0-")
        and sha256_file(path) == EXPECTED_WHEEL_SHA256
    ]
    if len(smact_wheels) != 1:
        raise RuntimeError("exact SMACT wheel is missing from bundle")
    copied_wheel = build_root / smact_wheels[0].name
    shutil.copyfile(smact_wheels[0], copied_wheel)

    probe_env = {
        **clean_env,
        "PYTHONPATH": str(source_root),
    }
    code = r'''
import json, sys
from importlib.metadata import version
import transformers
from crystal_dlm.h1_nocharge_ion_aux import load_smact4_icsd24_oxidation_map
mapping, contract = load_smact4_icsd24_oxidation_map()
print(json.dumps({
    "python": sys.version.split()[0],
    "smact": version("SMACT"),
    "transformers": transformers.__version__,
    "contract_sha256": contract["contract_sha256"],
    "oxidation_elements": len(mapping),
}, sort_keys=True))
'''
    completed = subprocess.run(
        [str(wrapper), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=probe_env,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("shared SMACT4 wrapper returned no audit JSON")
    probe = json.loads(lines[-1])
    expected_probe = manifest["local_offline_probe"]
    if probe != expected_probe:
        raise RuntimeError(f"shared runtime probe mismatch: {probe!r}")

    site_inventory = build_root / "SITE_SHA256.txt"
    site_entries = sorted(path for path in site.rglob("*") if path.is_file())
    with site_inventory.open("w", encoding="utf-8") as handle:
        for path in site_entries:
            handle.write(f"{sha256_file(path)}  {path.relative_to(site).as_posix()}\n")
    terminal = {
        "schema": "smact4_400_runtime_manifest_v2",
        "identity": IDENTITY,
        "status": "pass",
        "bundle_archive": bundle_archive.name,
        "bundle_archive_sha256": freeze["archive_sha256"],
        "bundle_manifest_sha256": freeze["bundle_manifest_sha256"],
        "bundle_validation": bundle,
        "wheel_sha256": sha256_file(copied_wheel),
        "wrapper_sha256": sha256_file(wrapper),
        "base_python_relative": base_python_relative.as_posix(),
        "base_python_sha256": sha256_file(base_python),
        "base_python_version": version_payload["version"],
        "site_file_count": len(site_entries),
        "site_inventory_sha256": sha256_file(site_inventory),
        "pip_check": pip_check.stdout.strip(),
        "probe": probe,
        "network": False,
        "global_environment_mutation": False,
        "user_site_isolation": True,
        "atomic_publish": True,
    }
    terminal_path = build_root / "terminal_report.json"
    terminal_path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    terminal_sha = sha256_file(terminal_path)
    (build_root / "terminal_report.sha256").write_text(
        f"{terminal_sha}  terminal_report.json\n", encoding="utf-8"
    )
    (build_root / "_SUCCESS").write_text(
        json.dumps(
            {"identity": IDENTITY, "terminal_sha256": terminal_sha}, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    freeze_tree(build_root)
    build_root.rename(output_root)
    print(json.dumps(terminal, sort_keys=True))


if __name__ == "__main__":
    main()
