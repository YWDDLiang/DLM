#!/usr/bin/env python3
"""Registered environment, asset, offline, and CUDA checks for WQ co-diffusion."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


VERSIONS = {
    "pyxtal": "1.1.4",
    "chgnet": "0.4.2",
    "mattersim": "1.1.2",
    "mace-torch": "0.3.13",
}
PACKAGE_SCOPES = {
    "core": ("pyxtal", "chgnet", "mace-torch"),
    "chgnet": ("pyxtal", "chgnet"),
    "mattersim": ("pyxtal", "mattersim"),
    "mace": ("pyxtal", "mace-torch"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/public/home/jiaosz/ywliang/models/wqcodiff"),
    )
    parser.add_argument("--asset-lock", type=Path)
    parser.add_argument(
        "--runtime-scope",
        choices=("core", "chgnet", "mattersim", "mace"),
        default="core",
    )
    parser.add_argument("--expected-numpy")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--require-offline", action="store_true")
    parser.add_argument("--mlip-smoke", action="store_true")
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Run the code/runtime gate without requiring evaluator checkpoints.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    project = args.project_root.resolve()
    model_root = args.model_root.resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    check("linux", platform.system() == "Linux", platform.platform())
    check("python_3_10", sys.version_info[:2] == (3, 10), sys.version)
    check("project_root", project.is_dir(), project)
    check("model_root", model_root.is_dir(), model_root)
    if args.skip_assets and args.runtime_scope != "core":
        parser.error("--skip-assets is allowed only for the core runtime scope")

    import numpy as np
    import torch

    check("torch_2_4_0", torch.__version__.split("+")[0] == "2.4.0", torch.__version__)
    check("numpy_import", True, np.__version__)
    if args.expected_numpy:
        check("numpy_abi_lock", np.__version__ == args.expected_numpy, np.__version__)
    check("cuda_available", not args.require_cuda or torch.cuda.is_available(), torch.cuda.is_available())
    if args.require_cuda and torch.cuda.is_available():
        check("cuda_device", True, torch.cuda.get_device_name(0))
        check("bf16_supported", torch.cuda.is_bf16_supported(), torch.cuda.is_bf16_supported())
    if args.require_slurm:
        check("inside_slurm", bool(os.environ.get("SLURM_JOB_ID")), os.environ.get("SLURM_JOB_ID"))
    if args.require_offline:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            check(name, os.environ.get(name) == "1", os.environ.get(name))
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        check(name, os.environ.get(name) == "1", os.environ.get(name))
    check(
        "PYTHONDONTWRITEBYTECODE",
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        os.environ.get("PYTHONDONTWRITEBYTECODE"),
    )
    pycache_prefix = os.environ.get("PYTHONPYCACHEPREFIX", "")
    check(
        "PYTHONPYCACHEPREFIX",
        bool(pycache_prefix) and Path(pycache_prefix).is_absolute(),
        pycache_prefix,
    )

    for distribution in PACKAGE_SCOPES[args.runtime_scope]:
        expected = VERSIONS[distribution]
        try:
            installed = importlib.metadata.version(distribution)
            check(f"package_{distribution}", installed == expected, installed)
        except Exception as exc:
            check(f"package_{distribution}", False, repr(exc))
    from crystal_dlm.wqcodiff.dependency_waiver import (
        MATTERSIM_RUNTIME_LOCK_FILENAME,
        MATTERSIM_RUNTIME_RELATIVE,
        MATTERSIM_WAIVER_FILENAME,
        MLIP_ASSET_LOCK_FILENAME,
        WAIVER_FILENAME,
        load_chgnet_torch_waiver,
        load_mattersim_inference_waiver,
        load_mattersim_runtime_lock,
        validate_mace_runtime_metadata,
    )

    if args.runtime_scope in {"core", "chgnet", "mace"}:
        try:
            waiver = load_chgnet_torch_waiver(
                model_root / WAIVER_FILENAME,
                installed_torch=torch.__version__,
            )
            check("chgnet_torch_metadata_waiver", True, waiver["validation"])
        except Exception as exc:
            check("chgnet_torch_metadata_waiver", False, repr(exc))
        try:
            installed_e3nn = importlib.metadata.version("e3nn")
            check("core_e3nn_0_4_4", installed_e3nn == "0.4.4", installed_e3nn)
        except Exception as exc:
            check("core_e3nn_0_4_4", False, repr(exc))
        try:
            mace_dependencies = validate_mace_runtime_metadata()
            check("mace_direct_dependency_closure", True, len(mace_dependencies))
        except Exception as exc:
            check("mace_direct_dependency_closure", False, repr(exc))
    else:
        expected_runtime = (model_root / MATTERSIM_RUNTIME_RELATIVE).resolve()
        runtime_entries = []
        for entry in sys.path:
            if not entry:
                continue
            try:
                runtime_entries.append(Path(entry).resolve())
            except OSError:
                continue
        check(
            "mattersim_runtime_on_pythonpath",
            expected_runtime in runtime_entries,
            expected_runtime,
        )
        try:
            waiver = load_mattersim_inference_waiver(
                model_root / MATTERSIM_WAIVER_FILENAME,
                installed_torch=torch.__version__,
            )
            load_mattersim_runtime_lock(
                model_root / MATTERSIM_RUNTIME_LOCK_FILENAME,
                model_root=model_root,
                installed_torch=torch.__version__,
            )
            check("mattersim_inference_runtime_waiver", True, waiver["validation"])
        except Exception as exc:
            check("mattersim_inference_runtime_waiver", False, repr(exc))
        try:
            from packaging.version import Version

            installed_e3nn = importlib.metadata.version("e3nn")
            check(
                "mattersim_e3nn_min_0_5",
                Version(installed_e3nn) >= Version("0.5.0"),
                installed_e3nn,
            )
        except Exception as exc:
            check("mattersim_e3nn_min_0_5", False, repr(exc))
    for module in ("pymatgen", "spglib", "ase", "yaml"):
        try:
            importlib.import_module(module)
            check(f"import_{module}", True, "ok")
        except Exception as exc:
            check(f"import_{module}", False, repr(exc))

    try:
        from crystal_dlm.wqcodiff.model import WQCoDenoiser
        from crystal_dlm.wqcodiff.protocol import load_protocol

        protocol = load_protocol(
            project / "configs/experiments/wyckoff_codiffusion/protocol_v3.yaml"
        )
        model = WQCoDenoiser()
        check("protocol_v3", True, protocol.sha256)
        check("model_parameter_count", model.parameter_count() == 6_574_421, model.parameter_count())
    except Exception as exc:
        check("wqcodiff_import", False, repr(exc))

    if args.skip_assets and args.mlip_smoke:
        parser.error("--mlip-smoke cannot be combined with --skip-assets")
    if not args.skip_assets:
        lock_path = args.asset_lock or model_root / MLIP_ASSET_LOCK_FILENAME
        try:
            from crystal_dlm.wqcodiff.mlip import EvaluatorLock, MLIPCalculator

            lock = EvaluatorLock.load(lock_path)
            evaluators = (
                (args.runtime_scope,)
                if args.runtime_scope in {"chgnet", "mattersim", "mace"}
                else ("chgnet", "mattersim", "mace")
            )
            for evaluator in evaluators:
                asset = lock.verify(
                    evaluator,
                    model_root,
                    verify_installed=args.runtime_scope != "core",
                )
                check(f"asset_{evaluator}", True, asset.checkpoint_sha256)
            if args.mlip_smoke:
                from pymatgen.core import Lattice, Structure

                structure = Structure(
                    Lattice.cubic(5.43),
                    [14, 14],
                    [[0, 0, 0], [0.25, 0.25, 0.25]],
                )
                device = "cuda" if args.require_cuda else "cpu"
                if args.runtime_scope == "core":
                    raise RuntimeError("MLIP smoke requires one evaluator runtime scope")
                for evaluator in evaluators:
                    result = MLIPCalculator(
                        evaluator=evaluator,
                        asset_lock=lock,
                        model_root=model_root,
                        device=device,
                    ).single_point(structure)
                    check(
                        f"mlip_smoke_{evaluator}",
                        np.isfinite(result["energy_per_atom_ev"]),
                        result["energy_per_atom_ev"],
                    )
        except Exception as exc:
            check("mlip_asset_lock", False, repr(exc))

    payload = {
        "schema": "wqcodiff_environment_doctor_v1",
        "ok": all(item["ok"] for item in checks),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime_scope": args.runtime_scope,
        "checks": checks,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
