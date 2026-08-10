#!/usr/bin/env python3
"""Create a fresh V4 run root from byte-frozen V3 scientific inputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} changed: expected={expected} observed={observed}")


def count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def identity(actual: Path, displayed: Path) -> dict[str, Any]:
    return {
        "path": str(displayed),
        "bytes": actual.stat().st_size,
        "sha256": sha256_file(actual),
    }


def generate_source_manifest(source: Path) -> Path:
    manifest = source / "SOURCE_SHA256.txt"
    if manifest.exists():
        raise FileExistsError(manifest)
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.name != "SOURCE_SHA256.txt"
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    if not files:
        raise ValueError(f"empty source package: {source}")
    with manifest.open("x", encoding="ascii", newline="\n") as handle:
        for path in files:
            relative = path.relative_to(source).as_posix()
            handle.write(f"{sha256_file(path)}  {relative}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def validate_source_tree(source: Path) -> None:
    if any(path.name == "__pycache__" for path in source.rglob("__pycache__")):
        raise ValueError("Python cache directory found in source")
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".pyc", ".pyo"}:
            raise ValueError(f"Python cache artifact found: {path}")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if path.suffix in {".sh", ".sbatch"}:
            subprocess.run(["bash", "-n", str(path)], check=True)
        if path.suffix == ".sbatch" and "gpu_long" in path.read_text(
            encoding="utf-8"
        ):
            raise ValueError(f"gpu_long is forbidden: {path}")


def validate_cache(
    root: Path,
    contract: Mapping[str, Any],
    *,
    subdir: str,
    source_field: str,
) -> None:
    cache_root = root / subdir
    manifest_path = cache_root / "completion_manifest.json"
    cache_path = cache_root / "completed_mp_hull_cache.jsonl"
    marker = cache_root / "completion_SUCCESS"
    require_sha(
        manifest_path,
        str(contract["completion_manifest_sha256"]),
        f"{subdir} completion manifest",
    )
    require_sha(
        cache_path,
        str(contract["completed_cache_sha256"]),
        f"{subdir} completed cache",
    )
    if not marker.is_file():
        raise FileNotFoundError(marker)
    manifest = read_json(manifest_path)
    completed = manifest.get("completed_mp_hull_cache") or {}
    if (
        manifest.get("status") != "complete_all_wanted_chemsys_resolved"
        or manifest.get(source_field)
        != contract[
            "origin_body_source_manifest_sha256"
            if source_field == "source_manifest_sha256"
            else "origin_native_source_manifest_sha256"
        ]
        or manifest.get("api_key_serialized") is not False
        or manifest.get("mp_query_inside_slurm") is not False
        or manifest.get("sample_retry_or_replacement_used") is not False
        or int(manifest.get("transport_retries", -1))
        != int(contract["transport_retries"])
        or completed.get("all_rows_populated") is not True
        or int(completed.get("rows", -1)) != int(contract["rows"])
        or completed.get("sha256") != contract["completed_cache_sha256"]
    ):
        raise ValueError(f"{subdir} completion contract changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--preparing-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    args = parser.parse_args()

    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("V4 input import must run before Slurm submission")
    if any(os.environ.get(name) for name in ("MP_API_KEY", "PMG_MAPI_KEY", "MAPI_KEY")):
        raise RuntimeError("credentials must be absent during V4 input import")
    if len(args.source_git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_git_commit
    ):
        raise ValueError("source Git commit must be one lowercase 40-hex identity")

    package = args.package_root.resolve()
    preparing = args.preparing_root.resolve()
    final_root = args.final_root.resolve()
    contract_path = package / "INPUT_IMPORT_CONTRACT.json"
    contract = read_json(contract_path)
    source_root = Path(str(contract["source_run_root"])).resolve()
    if (
        contract.get("schema") != "h1_plan1200_v4_input_import_contract_v1"
        or final_root != Path(str(contract["target_run_root"])).resolve()
        or preparing.exists()
        or final_root.exists()
        or any(contract.get(key) is not False for key in (
            "sample_retry", "replacement", "repair", "filter", "rerank",
            "automatic_training", "automatic_rl",
        ))
    ):
        raise ValueError("V4 import identity or no-intervention contract changed")

    planner_contract = contract["planner"]
    require_sha(
        source_root / "planner_terminal_report.json",
        planner_contract["terminal_report_sha256"],
        "planner terminal report",
    )
    if not (source_root / "status/planner_assembly_SUCCESS").is_file():
        raise FileNotFoundError(source_root / "status/planner_assembly_SUCCESS")
    for item in planner_contract["repeats"]:
        repeat = int(item["repeat"])
        cohort = source_root / "repeats" / str(repeat) / "cohort"
        candidates = (
            source_root
            / "repeats"
            / str(repeat)
            / "crysllmgen_native_candidates"
        )
        require_sha(
            cohort / "cohort1000.jsonl",
            item["cohort1000_sha256"],
            f"repeat {repeat} cohort",
        )
        require_sha(
            cohort / "cohort_manifest.json",
            item["cohort_manifest_sha256"],
            f"repeat {repeat} cohort manifest",
        )
        require_sha(
            candidates / "candidate_pool.jsonl",
            item["candidate_pool_sha256"],
            f"repeat {repeat} candidate pool",
        )
        require_sha(
            candidates / "candidate_pool_manifest.json",
            item["candidate_pool_manifest_sha256"],
            f"repeat {repeat} candidate pool manifest",
        )
        if count_jsonl(cohort / "cohort1000.jsonl") != 1000:
            raise ValueError(f"repeat {repeat} cohort denominator changed")
        if count_jsonl(candidates / "candidate_pool.jsonl") != int(
            item["parse_successes"]
        ):
            raise ValueError(f"repeat {repeat} candidate-pool count changed")
        if not (candidates / "_SUCCESS").is_file():
            raise FileNotFoundError(candidates / "_SUCCESS")

    validate_cache(
        source_root,
        contract["main_mp_cache"],
        subdir="mp_cache",
        source_field="source_manifest_sha256",
    )
    validate_cache(
        source_root,
        contract["native_mp_cache"],
        subdir="native_mp_cache",
        source_field="native_source_manifest_sha256",
    )
    validate_source_tree(package / "body")
    validate_source_tree(package / "native")

    preparing.mkdir(parents=True)
    try:
        (preparing / "status").mkdir()
        (preparing / "logs").mkdir()
        shutil.copy2(contract_path, preparing / "INPUT_IMPORT_CONTRACT.json")
        shutil.copy2(
            source_root / "planner_terminal_report.json",
            preparing / "planner_terminal_report.json",
        )
        shutil.copy2(
            source_root / "status/planner_assembly_SUCCESS",
            preparing / "status/planner_assembly_SUCCESS",
        )
        for item in planner_contract["repeats"]:
            repeat = int(item["repeat"])
            source_repeat = source_root / "repeats" / str(repeat)
            target_repeat = preparing / "repeats" / str(repeat)
            target_repeat.mkdir(parents=True)
            shutil.copytree(source_repeat / "cohort", target_repeat / "cohort")
            shutil.copytree(
                source_repeat / "crysllmgen_native_candidates",
                target_repeat / "crysllmgen_native_candidates",
            )
        for subdir in ("mp_cache", "native_mp_cache"):
            source_cache = source_root / subdir
            target_cache = preparing / subdir
            target_cache.mkdir()
            for name in (
                "completion_manifest.json",
                "completed_mp_hull_cache.jsonl",
                "completion_SUCCESS",
            ):
                shutil.copy2(source_cache / name, target_cache / name)
        shutil.copytree(package / "body", preparing / "body_source")
        shutil.copytree(package / "native", preparing / "native1000_source")
        body_manifest = generate_source_manifest(preparing / "body_source")
        native_manifest = generate_source_manifest(preparing / "native1000_source")

        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONPATH"] = str(preparing / "body_source")
        subprocess.run(
            [sys.executable, str(preparing / "body_source/self_test.py")],
            check=True,
            env=env,
        )
        env["PYTHONPATH"] = os.pathsep.join(
            [str(preparing / "native1000_source"), str(preparing / "body_source")]
        )
        subprocess.run(
            [sys.executable, str(preparing / "native1000_source/self_test.py")],
            check=True,
            env=env,
        )

        displayed = final_root
        imported_repeats = []
        for item in planner_contract["repeats"]:
            repeat = int(item["repeat"])
            actual = preparing / "repeats" / str(repeat)
            shown = displayed / "repeats" / str(repeat)
            imported_repeats.append(
                {
                    "repeat": repeat,
                    "cohort1000": identity(
                        actual / "cohort/cohort1000.jsonl",
                        shown / "cohort/cohort1000.jsonl",
                    ),
                    "candidate_pool": identity(
                        actual
                        / "crysllmgen_native_candidates/candidate_pool.jsonl",
                        shown
                        / "crysllmgen_native_candidates/candidate_pool.jsonl",
                    ),
                }
            )
        report = {
            "schema": "h1_plan1200_v4_input_import_report_v1",
            "status": "complete",
            "source_run_id": contract["source_run_id"],
            "source_run_root": str(source_root),
            "target_run_id": contract["target_run_id"],
            "target_run_root": str(final_root),
            "source_git_commit": args.source_git_commit,
            "contract_sha256": sha256_file(
                preparing / "INPUT_IMPORT_CONTRACT.json"
            ),
            "planner_terminal": identity(
                preparing / "planner_terminal_report.json",
                displayed / "planner_terminal_report.json",
            ),
            "repeats": imported_repeats,
            "main_mp_cache": identity(
                preparing / "mp_cache/completed_mp_hull_cache.jsonl",
                displayed / "mp_cache/completed_mp_hull_cache.jsonl",
            ),
            "native_mp_cache": identity(
                preparing / "native_mp_cache/completed_mp_hull_cache.jsonl",
                displayed / "native_mp_cache/completed_mp_hull_cache.jsonl",
            ),
            "body_source_manifest": identity(
                body_manifest, displayed / "body_source/SOURCE_SHA256.txt"
            ),
            "native_source_manifest": identity(
                native_manifest, displayed / "native1000_source/SOURCE_SHA256.txt"
            ),
            "scientific_inputs_byte_identical_to_V3": True,
            "cohort_schema_repair_location": "consumer_validation_only",
            "mp_query_performed": False,
            "sample_retry_replacement_repair_filter_rerank": False,
            "automatic_training": False,
            "automatic_rl": False,
        }
        write_json_exclusive(preparing / "status/v4_input_import_report.json", report)
        (preparing / "status/v4_input_import_SUCCESS").touch(exist_ok=False)
        (preparing / "status/native1000_inputs_SUCCESS").touch(exist_ok=False)
        (preparing / "status/body_source_git_commit.txt").write_text(
            args.source_git_commit + "\n", encoding="ascii"
        )
        (preparing / "status/native_source_git_commit.txt").write_text(
            args.source_git_commit + "\n", encoding="ascii"
        )
    except BaseException:
        raise

    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
