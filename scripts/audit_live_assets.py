#!/usr/bin/env python3
"""Create an immutable hash manifest for explicitly named live cluster assets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("asset must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path or "/" in name or "\\" in name:
        raise argparse.ArgumentTypeError("asset name/path is invalid")
    return name, Path(raw_path)


def parse_expected_file(value: str) -> tuple[str, str, str]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("expected file must be ASSET:RELATIVE=SHA256")
    key, expected = value.split("=", 1)
    asset, relative = key.split(":", 1)
    if not asset or not relative or len(expected) != 64:
        raise argparse.ArgumentTypeError("expected file is invalid")
    return asset, relative.replace("\\", "/"), expected.lower()


def tree_digest(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: str(value["relative_path"])):
        payload = (
            f"{row['relative_path']}\0{row['size_bytes']}\0{row['sha256']}\n"
        ).encode("utf-8")
        digest.update(payload)
    return digest.hexdigest()


def audit_asset(name: str, path: Path, *, workers: int) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        return {
            "name": name,
            "path": str(resolved),
            "exists": False,
            "kind": "missing",
            "files": [],
            "file_count": 0,
            "size_bytes": 0,
            "tree_sha256": None,
        }
    if resolved.is_file():
        file_row = {
            "relative_path": resolved.name,
            "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
        return {
            "name": name,
            "path": str(resolved),
            "exists": True,
            "kind": "file",
            "files": [file_row],
            "file_count": 1,
            "size_bytes": file_row["size_bytes"],
            "tree_sha256": tree_digest([file_row]),
        }
    candidates = sorted(
        candidate
        for candidate in resolved.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    )
    if not candidates:
        return {
            "name": name,
            "path": str(resolved),
            "exists": True,
            "kind": "directory",
            "files": [],
            "file_count": 0,
            "size_bytes": 0,
            "tree_sha256": tree_digest([]),
        }

    def one(candidate: Path) -> dict[str, Any]:
        return {
            "relative_path": candidate.relative_to(resolved).as_posix(),
            "size_bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        rows = list(executor.map(one, candidates))
    return {
        "name": name,
        "path": str(resolved),
        "exists": True,
        "kind": "directory",
        "files": sorted(rows, key=lambda value: value["relative_path"]),
        "file_count": len(rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "tree_sha256": tree_digest(rows),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Rich-DLM live asset manifest",
        "",
        "| asset | kind | files | GiB | tree SHA256 | exists |",
        "|---|---|---:|---:|---|---:|",
    ]
    for asset in report["assets"]:
        lines.append(
            f"| {asset['name']} | {asset['kind']} | {asset['file_count']} | "
            f"{asset['size_bytes'] / (1024 ** 3):.4f} | "
            f"{asset['tree_sha256'] or ''} | {asset['exists']} |"
        )
    lines.extend(["", "## Expected file checks", ""])
    if report["expected_file_checks"]:
        for check in report["expected_file_checks"]:
            lines.append(
                f"- `{check['asset']}:{check['relative_path']}`: "
                f"expected `{check['expected_sha256']}`, observed "
                f"`{check.get('observed_sha256')}`, pass={check['pass']}"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            f"Overall pass: **{report['gate']['pass']}**.",
            "",
            "This is an asset identity audit. It reads no generation or stability outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", action="append", type=parse_named_path, required=True)
    parser.add_argument("--expected-file", action="append", type=parse_expected_file, default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    names = [name for name, _ in args.asset]
    if len(names) != len(set(names)):
        raise ValueError("asset names must be unique")
    assets = [
        audit_asset(name, path, workers=int(args.workers))
        for name, path in args.asset
    ]
    by_name = {asset["name"]: asset for asset in assets}
    checks = []
    for asset_name, relative, expected in args.expected_file:
        asset = by_name.get(asset_name)
        observed = None
        if asset is not None:
            observed = next(
                (
                    row["sha256"]
                    for row in asset["files"]
                    if row["relative_path"] == relative
                ),
                None,
            )
        checks.append(
            {
                "asset": asset_name,
                "relative_path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "pass": observed == expected,
            }
        )
    report = {
        "schema": "h1a2_rich_dlm_live_asset_manifest_v1",
        "assets": assets,
        "expected_file_checks": checks,
        "outcomes_read": False,
        "gpu_jobs_used": 0,
        "workers": int(args.workers),
        "gate": {
            "all_assets_exist": all(asset["exists"] for asset in assets),
            "all_assets_nonempty": all(asset["file_count"] > 0 for asset in assets),
            "expected_files_match": all(check["pass"] for check in checks),
        },
    }
    report["gate"]["pass"] = all(report["gate"].values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "RICH_DLM_LIVE_ASSET_MANIFEST.json"
    md_path = args.output_dir / "RICH_DLM_LIVE_ASSET_MANIFEST.md"
    csv_path = args.output_dir / "RICH_DLM_LIVE_ASSET_FILES.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("asset", "asset_path", "relative_path", "size_bytes", "sha256"),
        )
        writer.writeheader()
        for asset in assets:
            for row in asset["files"]:
                writer.writerow(
                    {
                        "asset": asset["name"],
                        "asset_path": asset["path"],
                        **row,
                    }
                )
    outputs = {
        path.name: sha256_file(path)
        for path in (json_path, md_path, csv_path)
    }
    marker = args.output_dir / ("_SUCCESS" if report["gate"]["pass"] else "_FAILED")
    marker.write_text(json.dumps(outputs, sort_keys=True) + "\n", encoding="utf-8")
    if not report["gate"]["pass"]:
        raise SystemExit(3)
    print(json.dumps({"assets": len(assets), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
