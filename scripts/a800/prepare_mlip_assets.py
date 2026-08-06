#!/usr/bin/env python3
"""Login-node-only download/copy and immutable lock for the three MLIP assets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crystal_dlm.wqcodiff.mlip import (  # noqa: E402
    EVALUATOR_CHECKPOINTS,
    EVALUATOR_VERSIONS,
    EvaluatorLock,
    sha256_file,
)
from crystal_dlm.wqcodiff.dependency_waiver import (  # noqa: E402
    ACTIVE_EVALUATOR_STACK,
    MATTERSIM_ASE_VERSION,
    MATTERSIM_ASE_WHEEL_SHA256,
    MATTERSIM_RUNTIME_LOCK_FILENAME,
    MATTERSIM_RUNTIME_RELATIVE,
    MATTERSIM_SETUPTOOLS_VERSION,
    MATTERSIM_SETUPTOOLS_WHEEL_SHA256,
    MATTERSIM_WAIVER_FILENAME,
    MLIP_ASSET_LOCK_FILENAME,
    SOURCE_SDISTS_DIRNAME,
    WAIVER_FILENAME,
    WHEELHOUSE_DIRNAME,
    WHEELHOUSE_LOCK_FILENAME,
    load_chgnet_torch_waiver,
    load_mattersim_inference_waiver,
    load_mattersim_runtime_lock,
)
from crystal_dlm.wqcodiff.vocabulary import MP20_ATOMIC_NUMBERS  # noqa: E402


MODEL_URLS = {
    "mattersim": (
        "https://github.com/microsoft/mattersim/raw/refs/tags/v1.1.2/"
        "pretrained_models/mattersim-v1.0.0-5M.pth"
    ),
    "mace": (
        "https://huggingface.co/mace-foundations/mace-mp-0/resolve/main/"
        "mace-mp-0b3-medium.model?download=true"
    ),
}
KNOWN_SHA256 = {
    "mace": "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012",
}
PACKAGES = {"chgnet": "chgnet", "mattersim": "mattersim", "mace": "mace-torch"}
LICENSES = {"chgnet": "MIT", "mattersim": "MIT", "mace": "MIT"}
SOURCES = {
    "chgnet": "https://github.com/CederGroupHub/chgnet/tree/v0.4.2/chgnet/pretrained",
    "mattersim": MODEL_URLS["mattersim"],
    "mace": "https://huggingface.co/mace-foundations/mace-mp-0/blob/main/mace-mp-0b3-medium.model",
}


def _download(url: str, destination: Path) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("asset downloads are forbidden inside Slurm/GPU jobs")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise RuntimeError(f"stale partial download exists: {temporary}")
    request = urllib.request.Request(url, headers={"User-Agent": "wqcodiff-asset-lock/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if temporary.stat().st_size < 1024 * 1024:
            raise RuntimeError(f"downloaded asset is implausibly small: {temporary.stat().st_size}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _installed_chgnet_checkpoint() -> Path:
    import chgnet

    root = Path(chgnet.__file__).resolve().parent / "pretrained"
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix in {".pt", ".pth"} or path.name.endswith(".pth.tar"))
        and "0.3.0" in str(path)
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one packaged CHGNet 0.3.0 checkpoint under {root}, "
            f"found {[str(path) for path in candidates]}"
        )
    return candidates[0]


def _ensure_assets(model_root: Path, *, download: bool) -> dict[str, Path]:
    result: dict[str, Path] = {}
    chgnet_source = _installed_chgnet_checkpoint()
    chgnet_target = model_root / EVALUATOR_CHECKPOINTS["chgnet"]
    if not chgnet_target.exists():
        if not download:
            raise FileNotFoundError(chgnet_target)
        temporary = chgnet_target.with_suffix(chgnet_target.suffix + ".partial")
        shutil.copy2(chgnet_source, temporary)
        os.replace(temporary, chgnet_target)
    if sha256_file(chgnet_target) != sha256_file(chgnet_source):
        raise RuntimeError("copied CHGNet checkpoint differs from pinned package asset")
    result["chgnet"] = chgnet_target

    for evaluator in ("mattersim", "mace"):
        target = model_root / EVALUATOR_CHECKPOINTS[evaluator]
        if not target.exists():
            if not download:
                raise FileNotFoundError(target)
            _download(MODEL_URLS[evaluator], target)
        expected = KNOWN_SHA256.get(evaluator)
        if expected is not None and sha256_file(target) != expected:
            raise RuntimeError(
                f"{evaluator} checkpoint hash differs from the official frozen hash"
            )
        result[evaluator] = target
    return result


def _isolated_versions(runtime: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(runtime)]):
        name = str(distribution.metadata.get("Name") or "").lower().replace("_", "-")
        if name:
            result[name] = distribution.version
    return result


def _package_versions(model_root: Path) -> dict[str, str]:
    result = {
        "chgnet": importlib.metadata.version(PACKAGES["chgnet"]),
        "mace": importlib.metadata.version(PACKAGES["mace"]),
    }
    runtime_versions = _isolated_versions(model_root / MATTERSIM_RUNTIME_RELATIVE)
    try:
        result["mattersim"] = runtime_versions["mattersim"]
    except KeyError as exc:
        raise RuntimeError("isolated MatterSim runtime has no distribution metadata") from exc
    for evaluator, installed in result.items():
        expected = EVALUATOR_VERSIONS[evaluator]
        if installed != expected:
            raise RuntimeError(
                f"{PACKAGES[evaluator]}=={installed}; expected exactly {expected}"
            )
    return result


def _checkpoint_support(paths: dict[str, Path]) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read element support from each frozen model, never from our vocabulary.

    CHGNet and MatterSim index atomic number directly into a contiguous embedding,
    while MACE serializes its exact (potentially non-contiguous) atomic-number
    buffer.  Loading the model metadata here makes the asset lock evidentiary:
    writing 89 numbers into a JSON file cannot manufacture evaluator support.
    """

    import torch

    from chgnet.model.model import CHGNet

    chgnet_model = CHGNet.from_file(str(paths["chgnet"]), use_device="cpu")
    count = int(chgnet_model.atom_embedding.embedding.num_embeddings)
    if count < 1 or count > 118:
        raise RuntimeError(f"invalid CHGNet atomic embedding size: {count}")
    support: dict[str, tuple[tuple[int, ...], str]] = {
        "chgnet": (
            tuple(range(1, count + 1)),
            "checkpoint_model.atom_embedding.embedding.num_embeddings; direct Z-1 indexing",
        )
    }
    del chgnet_model

    matter_checkpoint = torch.load(paths["mattersim"], map_location="cpu")
    try:
        max_z = int(matter_checkpoint["model_args"]["max_z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("MatterSim checkpoint does not expose model_args.max_z") from exc
    if max_z < 1 or max_z > 118:
        raise RuntimeError(f"invalid MatterSim max_z: {max_z}")
    support["mattersim"] = (
        tuple(range(1, max_z + 1)),
        (
            "checkpoint model_args.max_z inclusive; species Z is encoded with "
            "F.one_hot(Z, num_classes=max_z+1)"
        ),
    )
    del matter_checkpoint

    from mace.calculators import mace_mp

    mace_calculator = mace_mp(
        model=str(paths["mace"]),
        device="cpu",
        default_dtype="float32",
        dispersion=False,
    )
    models = tuple(getattr(mace_calculator, "models", ()))
    if len(models) != 1 or not hasattr(models[0], "atomic_numbers"):
        raise RuntimeError("MACE checkpoint does not expose one atomic_numbers buffer")
    atomic_numbers = getattr(models[0], "atomic_numbers")
    if hasattr(atomic_numbers, "detach"):
        atomic_numbers = atomic_numbers.detach().cpu().reshape(-1).tolist()
    mace_support = tuple(sorted({int(value) for value in atomic_numbers}))
    if not mace_support or not all(1 <= value <= 118 for value in mace_support):
        raise RuntimeError(f"invalid MACE atomic-number support: {mace_support}")
    support["mace"] = (
        mace_support,
        "loaded checkpoint model.atomic_numbers buffer",
    )
    return support


def _verify_wheelhouse_lock(model_root: Path) -> Path:
    lock_path = model_root / WHEELHOUSE_LOCK_FILENAME
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "wqcodiff_wheelhouse_lock_v4":
        raise RuntimeError("invalid wheelhouse lock schema")
    if payload.get("stack_id") != ACTIVE_EVALUATOR_STACK:
        raise RuntimeError("invalid active evaluator-stack ID")
    build_environment = payload.get("build_environment", {})
    if (
        build_environment.get("build_isolation") is not False
        or build_environment.get("pip_cache") is not False
        or build_environment.get("source_date_epoch") != 315532800
        or build_environment.get("python_hash_seed") != 0
        or not all(
            str(build_environment.get(name) or "")
            for name in ("python", "pip", "setuptools", "wheel")
        )
    ):
        raise RuntimeError("wheelhouse lock lacks the frozen source-build environment")
    wheelhouse = model_root / WHEELHOUSE_DIRNAME
    expected_wheels = {str(entry["filename"]) for entry in payload.get("wheels", ())}
    actual_wheels = {path.name for path in wheelhouse.glob("*.whl") if path.is_file()}
    if actual_wheels != expected_wheels:
        raise RuntimeError(
            f"wheelhouse file set differs from lock: actual={sorted(actual_wheels)}, "
            f"expected={sorted(expected_wheels)}"
        )
    for entry in payload.get("wheels", ()):
        path = wheelhouse / entry["filename"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"wheelhouse asset differs from lock: {path}")
    locked = {
        (
            str(entry["package"]).lower().replace("_", "-"),
            str(entry["version"]),
        ): str(entry["sha256"])
        for entry in payload.get("wheels", ())
    }
    expected_compatibility = {
        ("ase", MATTERSIM_ASE_VERSION): MATTERSIM_ASE_WHEEL_SHA256,
        ("setuptools", MATTERSIM_SETUPTOOLS_VERSION): (
            MATTERSIM_SETUPTOOLS_WHEEL_SHA256
        ),
    }
    for identity, digest in expected_compatibility.items():
        if locked.get(identity) != digest:
            raise RuntimeError(f"MatterSim compatibility wheel differs from lock: {identity}")
    predecessors = payload.get("failed_predecessors")
    expected_predecessors = {
        "wheelhouse_lock.json": "source_built_wheel_was_rebuilt_after_first_lock",
        "wheelhouse_lock_v2.json": (
            "mattersim_import_failed_on_pkg_resources_and_ase_3p28_api"
        ),
    }
    if not isinstance(predecessors, list) or {
        str(entry.get("filename")): str(entry.get("reason"))
        for entry in predecessors
    } != expected_predecessors:
        raise RuntimeError("active wheelhouse lock lacks both failed predecessors")
    for predecessor in predecessors:
        path = model_root / str(predecessor["filename"])
        if not path.is_file() or sha256_file(path) != predecessor.get("sha256"):
            raise RuntimeError(f"failed predecessor lock changed: {path}")
    source_root = model_root / SOURCE_SDISTS_DIRNAME
    expected_sources: set[str] = set()
    for entry in payload.get("source_builds", ()):
        expected_sources.add(str(entry["source_filename"]))
        source = source_root / entry["source_filename"]
        if not source.is_file() or sha256_file(source) != entry["source_sha256"]:
            raise RuntimeError(f"source-built wheel input differs from lock: {source}")
        wheel = wheelhouse / entry["built_wheel_filename"]
        if not wheel.is_file() or sha256_file(wheel) != entry["built_wheel_sha256"]:
            raise RuntimeError(f"source-built wheel differs from lock: {wheel}")
    actual_sources = {path.name for path in source_root.iterdir() if path.is_file()}
    if actual_sources != expected_sources:
        raise RuntimeError(
            f"source-sdist file set differs from lock: actual={sorted(actual_sources)}, "
            f"expected={sorted(expected_sources)}"
        )
    return lock_path


def _build_lock(
    paths: dict[str, Path],
    versions: dict[str, str],
    wheelhouse_lock: Path,
    support: dict[str, tuple[tuple[int, ...], str]],
    chgnet_dependency_waiver: Path,
    mattersim_dependency_waiver: Path,
    mattersim_runtime_lock: Path,
) -> dict[str, Any]:
    assets = []
    for evaluator in ("chgnet", "mattersim", "mace"):
        supported_atomic_numbers, support_basis = support[evaluator]
        missing_mp20 = sorted(set(MP20_ATOMIC_NUMBERS) - set(supported_atomic_numbers))
        if missing_mp20:
            raise RuntimeError(
                f"{evaluator} checkpoint lacks MP20 elements: {missing_mp20}"
            )
        assets.append(
            {
                "evaluator": evaluator,
                "package": PACKAGES[evaluator],
                "package_version": versions[evaluator],
                "checkpoint": paths[evaluator].name,
                "checkpoint_sha256": sha256_file(paths[evaluator]),
                "checkpoint_bytes": paths[evaluator].stat().st_size,
                "supported_atomic_numbers": list(supported_atomic_numbers),
                "support_basis": support_basis,
                "mp20_vocabulary_covered": True,
                "source_url": SOURCES[evaluator],
                "license": LICENSES[evaluator],
            }
        )
    import torch

    return {
        "schema": "wqcodiff_mlip_asset_lock_v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "wheelhouse_lock": wheelhouse_lock.name,
        "wheelhouse_lock_sha256": sha256_file(wheelhouse_lock),
        "chgnet_dependency_waiver": chgnet_dependency_waiver.name,
        "chgnet_dependency_waiver_sha256": sha256_file(chgnet_dependency_waiver),
        "mattersim_dependency_waiver": mattersim_dependency_waiver.name,
        "mattersim_dependency_waiver_sha256": sha256_file(mattersim_dependency_waiver),
        "mattersim_runtime_lock": mattersim_runtime_lock.name,
        "mattersim_runtime_lock_sha256": sha256_file(mattersim_runtime_lock),
        "assets": assets,
    }


def _smoke(lock_path: Path, model_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    probe = PROJECT_ROOT / "scripts/a800/mlip_runtime_probe.py"
    with tempfile.TemporaryDirectory(prefix="wqcodiff-mlip-smoke-") as directory:
        output_root = Path(directory)
        for evaluator in ("chgnet", "mattersim", "mace"):
            output = output_root / f"{evaluator}.json"
            environment = dict(os.environ)
            python_paths = [str(PROJECT_ROOT)]
            if evaluator == "mattersim":
                python_paths.insert(0, str(model_root / MATTERSIM_RUNTIME_RELATIVE))
            if environment.get("PYTHONPATH"):
                python_paths.append(environment["PYTHONPATH"])
            environment["PYTHONPATH"] = os.pathsep.join(python_paths)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(
                [
                    sys.executable,
                    str(probe),
                    "--evaluator",
                    evaluator,
                    "--model-root",
                    str(model_root),
                    "--asset-lock",
                    str(lock_path),
                    "--device",
                    "cpu",
                    "--output-json",
                    str(output),
                ],
                check=True,
                env=environment,
            )
            results.append(json.loads(output.read_text(encoding="utf-8")))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/public/home/jiaosz/ywliang/models/wqcodiff"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    model_root = args.model_root.resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    wheelhouse_lock = _verify_wheelhouse_lock(model_root)
    import torch

    chgnet_dependency_waiver = model_root / WAIVER_FILENAME
    load_chgnet_torch_waiver(
        chgnet_dependency_waiver,
        installed_torch=torch.__version__,
    )
    mattersim_dependency_waiver = model_root / MATTERSIM_WAIVER_FILENAME
    load_mattersim_inference_waiver(
        mattersim_dependency_waiver,
        installed_torch=torch.__version__,
    )
    mattersim_runtime_lock = model_root / MATTERSIM_RUNTIME_LOCK_FILENAME
    load_mattersim_runtime_lock(
        mattersim_runtime_lock,
        model_root=model_root,
        installed_torch=torch.__version__,
    )
    versions = _package_versions(model_root)
    paths = _ensure_assets(model_root, download=args.download)
    support = _checkpoint_support(paths)
    lock_path = model_root / MLIP_ASSET_LOCK_FILENAME
    payload = _build_lock(
        paths,
        versions,
        wheelhouse_lock,
        support,
        chgnet_dependency_waiver,
        mattersim_dependency_waiver,
        mattersim_runtime_lock,
    )
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_payload = dict(payload)
        comparable_existing.pop("created_utc", None)
        comparable_payload.pop("created_utc", None)
        if comparable_existing != comparable_payload:
            raise RuntimeError("existing immutable MLIP lock differs from discovered assets")
    else:
        with lock_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    lock = EvaluatorLock.load(lock_path)
    for evaluator in ("chgnet", "mattersim", "mace"):
        lock.verify(evaluator, model_root, verify_installed=False)
    result: dict[str, Any] = {
        "ok": True,
        "model_root": str(model_root),
        "asset_lock": str(lock_path),
        "asset_lock_sha256": sha256_file(lock_path),
        "assets": payload["assets"],
    }
    if args.smoke:
        result["cpu_smoke"] = _smoke(lock_path, model_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
