#!/usr/bin/env python3
"""Run the exact R5-C A100 evaluator and restore the full attempt denominator."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from crystal_dlm.wqcodiff.contracts import write_json_exclusive
from crystal_dlm.wqcodiff.crysllmgen.a100_sun import (
    META_LIKE_THRESHOLD,
    STRICT_THRESHOLD,
    aggregate_a100_outputs,
    prepare_a100_input,
    verify_a100_assets,
)


REQUIRED_EVALUATION_ENV = "diff_meets_diff"
REQUIRED_EVALUATION_PREFIX = Path(
    "/public/home/jiaosz/miniconda3/envs/diff_meets_diff"
)
REQUIRED_TORCH_VERSION = "2.4.0+cu121"
REQUIRED_TORCH_SCATTER_VERSION = "2.1.2+pt24cu121"
REQUIRED_RUNTIME_MODULES = (
    "crystal_dlm",
    "crystal_dlm.wqcodiff",
    "crystal_dlm.wqcodiff.contracts",
    "crystal_dlm.wqcodiff.crysllmgen",
    "crystal_dlm.wqcodiff.crysllmgen.epoch_training",
    "crystal_dlm.wqcodiff.crysllmgen.a100_sun",
)


def _preflight_runtime_imports(expected_runtime_root: Path) -> dict[str, object]:
    """Fail before expensive work if an isolated runtime dependency is absent.

    The S.U.N. adapter imports used to live inside :func:`main`, so importing
    this runner during source-bundle tests did not exercise them.  Keeping the
    imports at module scope and checking every module origin prevents a shared
    checkout from silently filling a hole in a frozen runtime bundle.
    """

    runtime_root = expected_runtime_root.resolve()
    if not runtime_root.is_dir():
        raise FileNotFoundError(runtime_root)
    origins: dict[str, str] = {}
    candidates = {"runner": Path(__file__).resolve()}
    for name in REQUIRED_RUNTIME_MODULES:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if module is None or origin is None:
            raise RuntimeError(f"required runtime module was not imported: {name}")
        candidates[name] = Path(origin).resolve()
    for name, origin in candidates.items():
        try:
            relative = origin.relative_to(runtime_root)
        except ValueError as exc:
            raise RuntimeError(
                f"runtime dependency escaped frozen root: {name} -> {origin}"
            ) from exc
        origins[name] = relative.as_posix()
    return {
        "schema": "crysllmgen_a100_sun_runtime_import_preflight_v1",
        "status": "pass",
        "runtime_root": str(runtime_root),
        "origins": origins,
    }


def _require_runtime() -> int:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("R5-C A100 S.U.N. must run through Slurm")
    values = []
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        value = int(os.environ.get(name, "0"))
        if value not in (4, 8, 16):
            raise RuntimeError(f"{name} must be one of 4, 8, or 16")
        values.append(value)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    if len(set(values)) != 1:
        raise RuntimeError("A100 S.U.N. numerical thread settings must agree")
    if values[0] > int(os.environ.get("SLURM_CPUS_PER_TASK", "0")):
        raise RuntimeError("numerical threads exceed allocated Slurm CPUs")
    if os.environ.get("CONDA_DEFAULT_ENV") != REQUIRED_EVALUATION_ENV:
        raise RuntimeError(
            "the registered R5-C A100 evaluator requires diff_meets_diff"
        )
    if Path(os.environ.get("CONDA_PREFIX", "")).resolve() != (
        REQUIRED_EVALUATION_PREFIX
    ):
        raise RuntimeError("the registered evaluation Conda prefix changed")
    if Path(sys.executable).resolve() != (
        REQUIRED_EVALUATION_PREFIX / "bin/python3.10"
    ).resolve():
        raise RuntimeError("the registered evaluation Python executable changed")
    import torch

    if torch.__version__ != REQUIRED_TORCH_VERSION:
        raise RuntimeError("the registered evaluation PyTorch version changed")
    if (
        importlib.metadata.version("torch-scatter")
        != REQUIRED_TORCH_SCATTER_VERSION
    ):
        raise RuntimeError("the registered torch-scatter build changed")
    import torch_scatter  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("CHGNet A100 evaluation requires a Slurm CUDA device")
    return values[0]


def _run(command: list[str], *, stdout_path: Path, stderr_path: Path) -> None:
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        subprocess.run(command, check=True, stdout=stdout, stderr=stderr)


def main() -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["--preflight-runtime-root"]:
        if len(argv) != 2:
            raise SystemExit(
                "--preflight-runtime-root requires exactly one runtime directory"
            )
        print(
            json.dumps(
                _preflight_runtime_imports(Path(argv[1])),
                sort_keys=True,
            )
        )
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--expected-attempts", type=int, default=256)
    parser.add_argument("--eval-sun-py", type=Path, required=True)
    parser.add_argument("--eval-sun-resumable-py", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--training-index-cache", type=Path, required=True)
    parser.add_argument("--mp-hull-cache", type=Path, required=True)
    parser.add_argument("--chgnet-relax-cache", type=Path, required=True)
    parser.add_argument("--chgnet-model-asset", type=Path, required=True)
    parser.add_argument("--chgnet-runtime-checkpoint", type=Path, required=True)
    parser.add_argument("--base-source-bundle-sha256", required=True)
    parser.add_argument("--execution-patch-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    threads = _require_runtime()
    cuda_device = __import__("torch").cuda.get_device_name(0)
    for label, value in (
        ("base source bundle", args.base_source_bundle_sha256),
        ("execution patch", args.execution_patch_sha256),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{label} must be one lowercase SHA256")

    assets = verify_a100_assets(
        eval_sun_py=args.eval_sun_py,
        eval_sun_resumable_py=args.eval_sun_resumable_py,
        train_csv=args.train_csv,
        training_index_cache=args.training_index_cache,
        mp_hull_cache=args.mp_hull_cache,
        chgnet_relax_cache=args.chgnet_relax_cache,
        chgnet_model_asset=args.chgnet_model_asset,
        chgnet_runtime_checkpoint=args.chgnet_runtime_checkpoint,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    generated_pt = output / "generated_attempt_preserving.pt"
    input_manifest = output / "input_manifest.json"
    prepare_a100_input(
        generation_jsonl=args.generation_jsonl.resolve(),
        output_pt=generated_pt,
        output_manifest=input_manifest,
        expected_attempts=args.expected_attempts,
    )

    working_mp_cache = output / "working_mp_hull_cache.jsonl"
    working_relax_cache = output / "working_chgnet_relax_cache.jsonl"
    shutil.copyfile(args.mp_hull_cache.resolve(), working_mp_cache)
    shutil.copyfile(args.chgnet_relax_cache.resolve(), working_relax_cache)
    strict_dir = output / "exact_strict"
    meta_dir = output / "exact_meta_like"
    strict_command = [
        sys.executable,
        str(args.eval_sun_resumable_py.resolve()),
        "--gen_file",
        str(generated_pt),
        "--train_csv",
        str(args.train_csv.resolve()),
        "--output_dir",
        str(strict_dir),
        "--stable_threshold",
        str(STRICT_THRESHOLD),
        "--mp_cache_path",
        str(working_mp_cache),
        "--global_relax_cache_path",
        str(working_relax_cache),
    ]
    meta_command = [
        sys.executable,
        str(args.eval_sun_resumable_py.resolve()),
        "--gen_file",
        str(generated_pt),
        "--train_csv",
        str(args.train_csv.resolve()),
        "--output_dir",
        str(meta_dir),
        "--stable_threshold",
        str(META_LIKE_THRESHOLD),
        "--mp_cache_path",
        str(working_mp_cache),
        "--global_relax_cache_path",
        str(working_relax_cache),
    ]
    contract = {
        "schema": "crysllmgen_r5c_a100_sun_run_contract_v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "environment": os.environ["CONDA_DEFAULT_ENV"],
        "environment_prefix": os.environ["CONDA_PREFIX"],
        "python_executable": sys.executable,
        "torch_version": __import__("torch").__version__,
        "torch_scatter_version": importlib.metadata.version("torch-scatter"),
        "cuda_device": cuda_device,
        "a100_wording": "frozen R5-C A100 evaluation protocol; actual device reported separately",
        "threads": threads,
        "expected_attempts": args.expected_attempts,
        "base_source_bundle_sha256": args.base_source_bundle_sha256,
        "execution_patch_sha256": args.execution_patch_sha256,
        "assets": assets,
        "strict_command": strict_command,
        "meta_like_command": meta_command,
        "offline": True,
        "coverage_adjusted_selection_role": "report_only",
        "retry_or_replacement_used": False,
    }
    write_json_exclusive(output / "run_contract.json", contract)
    started = time.monotonic()
    try:
        _run(
            strict_command,
            stdout_path=output / "strict.stdout.log",
            stderr_path=output / "strict.stderr.log",
        )
        _run(
            meta_command,
            stdout_path=output / "meta_like.stdout.log",
            stderr_path=output / "meta_like.stderr.log",
        )
        result = aggregate_a100_outputs(
            input_manifest=input_manifest,
            strict_summary=strict_dir / "RESULTS_SUMMARY.md",
            meta_summary=meta_dir / "RESULTS_SUMMARY.md",
            strict_relax_results=strict_dir / "relax_results.jsonl",
            meta_relax_results=meta_dir / "relax_results.jsonl",
            eval_sun_py=args.eval_sun_py.resolve(),
            eval_sun_resumable_py=args.eval_sun_resumable_py.resolve(),
            train_csv=args.train_csv.resolve(),
            training_index_cache=args.training_index_cache.resolve(),
            source_mp_hull_cache=args.mp_hull_cache.resolve(),
            source_chgnet_relax_cache=args.chgnet_relax_cache.resolve(),
            working_mp_hull_cache=working_mp_cache,
            chgnet_model_asset=args.chgnet_model_asset.resolve(),
            chgnet_runtime_checkpoint=args.chgnet_runtime_checkpoint.resolve(),
            output_jsonl=output / "attempt_results.jsonl",
            output_summary=output / "attempt_summary.json",
            base_source_bundle_sha256=args.base_source_bundle_sha256,
            execution_patch_sha256=args.execution_patch_sha256,
        )
    except Exception as exc:
        failure = {
            "schema": "crysllmgen_r5c_a100_sun_executor_failure_v1",
            "type": type(exc).__name__,
            "message": str(exc),
            "walltime_s": time.monotonic() - started,
            "retry_or_replacement_used": False,
        }
        write_json_exclusive(output / "executor_failure.json", failure)
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
