#!/usr/bin/env python3
"""Create an immutable deterministic source snapshot for this route."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import tarfile
from typing import Iterable


IDENTITY = "h1_nocharge_ion_aux_sft_v1"
EXECUTION_REL = Path(
    "workstreams/plangraph_dlm_iclr_20260731/execution/h1_nocharge_ion_aux_sft_v1"
)
ANALYSIS_REL = Path(
    "workstreams/plangraph_dlm_iclr_20260731/analysis/"
    "H1_NOCHARGE_ION_AUX_SFT_EXECUTION_ANNEX_V1.md"
)
TEST_DEPENDENCIES = (
    Path(
        "workstreams/plangraph_dlm_iclr_20260731/execution/"
        "h1_crplan_r0_paired32_script_package_repair_v5/evaluate_paired32.py"
    ),
)
TEXT_SUFFIXES = {
    ".py", ".sh", ".sbatch", ".json", ".md", ".txt", ".yaml", ".yml", ".toml"
}
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allowed(path: Path) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and not any(part in EXCLUDED_NAMES for part in path.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
        and (path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE"})
    )


def source_files(root: Path) -> Iterable[Path]:
    for directory in (root / "crystal_dlm", root / "scripts", root / "tests"):
        yield from (path for path in directory.rglob("*") if allowed(path))
    for path in (root / EXECUTION_REL).rglob("*"):
        if allowed(path):
            yield path
    if (root / ANALYSIS_REL).is_file():
        yield root / ANALYSIS_REL
    for relative in TEST_DEPENDENCIES:
        path = root / relative
        if not allowed(path):
            raise FileNotFoundError(f"registered isolated-test dependency is missing: {path}")
        yield path


def normalized_mode(path: Path) -> int:
    if path.suffix in {".sh", ".sbatch"} or path.name.endswith(".py"):
        return 0o755
    return 0o644


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    source_root = output_root / "source"
    source_root.mkdir(parents=True)

    selected: dict[str, Path] = {}
    for path in source_files(project_root):
        relative = path.relative_to(project_root).as_posix()
        if relative in selected:
            continue
        selected[relative] = path
    if not selected:
        raise RuntimeError("source selection is empty")
    for relative, source in sorted(selected.items()):
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(normalized_mode(source))

    entries = [
        {
            "path": relative,
            "bytes": (source_root / relative).stat().st_size,
            "sha256": sha256_file(source_root / relative),
            "mode": oct(normalized_mode(source)),
        }
        for relative, source in sorted(selected.items())
    ]
    manifest = {
        "schema": "h1_nocharge_ion_aux_source_manifest_v1",
        "identity": IDENTITY,
        "files": entries,
        "forbidden_payloads": [
            "model_weights", "adapter_weights", "MP20_CSV", "credentials", "runs"
        ],
    }
    manifest_path = source_root / "SOURCE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksummed = [entry["path"] for entry in entries] + ["SOURCE_MANIFEST.json"]
    inventory_path = source_root / "SOURCE_SHA256.txt"
    inventory_path.write_text(
        "".join(
            f"{sha256_file(source_root / relative)}  {relative}\n"
            for relative in sorted(checksummed)
        ),
        encoding="utf-8",
    )

    archive_path = output_root / f"{IDENTITY}.tar.gz"
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
                    relative = path.relative_to(source_root).as_posix()
                    info = tarfile.TarInfo(relative)
                    info.size = path.stat().st_size
                    info.mode = normalized_mode(path)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    record = {
        "schema": "h1_nocharge_ion_aux_source_freeze_v1",
        "identity": IDENTITY,
        "source_inventory_sha256": sha256_file(inventory_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "file_count": len(entries),
    }
    (output_root / "FREEZE_RECORD.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
