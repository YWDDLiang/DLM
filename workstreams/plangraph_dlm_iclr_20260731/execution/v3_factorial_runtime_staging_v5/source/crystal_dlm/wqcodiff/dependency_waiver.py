"""Auditable dependency exceptions for the frozen multi-MLIP runtime stack."""

from __future__ import annotations

import email
import hashlib
import importlib
import importlib.metadata as metadata
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


WAIVER_SCHEMA = "wqcodiff_dependency_waiver_v1"
WAIVER_FILENAME = "chgnet_torch_metadata_waiver_v4.json"
MATTERSIM_WAIVER_FILENAME = "mattersim_inference_runtime_waiver_v4.json"
MATTERSIM_RUNTIME_LOCK_FILENAME = "mattersim_runtime_lock_v4.json"
MATTERSIM_TREE_MANIFEST_FILENAME = "mattersim_runtime_tree_v4.json"
MATTERSIM_RUNTIME_RELATIVE = "runtimes/mattersim-1.1.2-py310-v4"
WHEELHOUSE_LOCK_FILENAME = "wheelhouse_lock_v4.json"
WHEELHOUSE_DIRNAME = "wheelhouse_v4"
SOURCE_SDISTS_DIRNAME = "source_sdists_v4"
MLIP_ASSET_LOCK_FILENAME = "mlip_asset_lock_v4.json"
ACTIVE_EVALUATOR_STACK = "wqcodiff-evaluator-stack-v4"
CHGNET_VERSION = "0.4.2"
MATTERSIM_VERSION = "1.1.2"
MACE_VERSION = "0.3.13"
RETAINED_TORCH_BASE = "2.4.0"
WAIVED_REQUIREMENT = "torch>=2.4.1"
AUTHORIZATION = "user_explicit_2026-07-17"
CHGNET_WAIVER_VALIDATION = (
    "project_closure_satisfied_and_only_registered_project_mismatch_with_"
    "unchanged_global_pip_check_baseline"
)
CHGNET_NON_TORCH_REQUIREMENTS = (
    "ase>=3.23.0",
    "cython>=3",
    "numpy>=1.26",
    "nvidia-ml-py3>=7.352.0",
    "pymatgen>=2024.9.10",
    "typing-extensions>=4.12",
)
CHGNET_RESOLVER_REQUIREMENTS = tuple(
    value for value in CHGNET_NON_TORCH_REQUIREMENTS if not value.startswith("nvidia-ml-py3")
)
NVIDIA_ML_PY3_VERSION = "7.352.0"
NVIDIA_ML_PY3_SDIST_SHA256 = "390f02919ee9d73fe63a98c73101061a6b37fa694a793abf56673320f1f51277"
PYTHON_HOSTLIST_VERSION = "2.3.0"
PYTHON_HOSTLIST_SDIST_SHA256 = "e1a0b18e525a5fca573cb9862799f11b3f2bd3ba7aec70c4ecd8b95341bb71ea"
MACE_ACTIVE_METADATA_REQUIREMENTS = (
    "torch>=1.12",
    "e3nn==0.4.4",
    "numpy",
    "opt_einsum",
    "ase",
    "torch-ema",
    "prettytable",
    "matscipy",
    "h5py",
    "torchmetrics",
    "python-hostlist",
    "configargparse",
    "GitPython",
    "pyYAML",
    "tqdm",
    "lmdb",
    "orjson",
    "matplotlib",
    "pandas",
)
MACE_RESOLVER_REQUIREMENTS = tuple(
    value for value in MACE_ACTIVE_METADATA_REQUIREMENTS if value != "python-hostlist"
)
MATTERSIM_WHEEL_SHA256 = "e249532b6e66d9307c7a72fde252f0bcf151c588b8656ce56ef1cbaf0ed90d10"
MATTERSIM_ASE_VERSION = "3.27.0"
MATTERSIM_ASE_WHEEL_SHA256 = "058c48ea504fe7fbbe7c932f778415243ef2df45b1ab869866f24efcc17f0538"
MATTERSIM_SETUPTOOLS_VERSION = "81.0.0"
MATTERSIM_SETUPTOOLS_WHEEL_SHA256 = "fdd925d5c5d9f62e4b74b30d6dd7828ce236fd6ed998a08d81de62ce5a6310d6"
MATTERSIM_COMPATIBILITY_PINS = {
    "ase": {
        "version": MATTERSIM_ASE_VERSION,
        "wheel_sha256": MATTERSIM_ASE_WHEEL_SHA256,
        "reason": "last_tested_release_exporting_stress_helper_from_ase_constraints",
    },
    "setuptools": {
        "version": MATTERSIM_SETUPTOOLS_VERSION,
        "wheel_sha256": MATTERSIM_SETUPTOOLS_WHEEL_SHA256,
        "reason": "last_release_family_containing_pkg_resources_for_mattersim_1p1p2",
    },
}
MATTERSIM_INFERENCE_REQUIREMENTS = (
    f"ase=={MATTERSIM_ASE_VERSION}",
    "deprecated",
    "e3nn>=0.5.0",
    "loguru",
    "numpy<2",
    "opt_einsum_fx",
    "prettytable",
    "pymatgen",
    "requests",
    "scikit-learn",
    "torch-ema>=0.3",
    "torch>=2.2.0",
    "torch_geometric>=2.5.3",
    "torch_runstats>=0.2.0",
    "torchmetrics>=0.10.0",
    f"setuptools=={MATTERSIM_SETUPTOOLS_VERSION}",
)
MATTERSIM_ACTIVE_METADATA_REQUIREMENTS = (
    "ase>=3.23.0",
    "azure-identity",
    "azure-storage-blob",
    "scikit-learn",
    "deprecated",
    "e3nn>=0.5.0",
    "atomate2",
    "emmet-core>=0.84",
    "loguru",
    "mp-api",
    "numpy<2",
    "opt_einsum_fx",
    "pydantic>=2.9.2",
    "pymatgen",
    "seekpath",
    "phonopy",
    "torch-ema>=0.3",
    "torch>=2.2.0",
    "torch_geometric>=2.5.3",
    "torch_runstats>=0.2.0",
    "torchaudio>=2.2.0",
    "torchmetrics>=0.10.0",
    "torchvision>=0.17.0",
    "wandb",
)
MATTERSIM_EXCLUDED_NONINFERENCE_DISTRIBUTIONS = (
    "atomate2",
    "azure-identity",
    "azure-storage-blob",
    "emmet-core",
    "mp-api",
    "phonopy",
    "pydantic",
    "seekpath",
    "torchaudio",
    "torchvision",
    "wandb",
)
MATTERSIM_INFERENCE_IMPORTS = (
    "ase",
    "deprecated",
    "e3nn",
    "loguru",
    "numpy",
    "opt_einsum_fx",
    "prettytable",
    "pymatgen",
    "requests",
    "sklearn",
    "torch",
    "torch_ema",
    "torch_geometric",
    "torch_runstats",
    "torchmetrics",
    "pkg_resources",
    "mattersim.forcefield",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_wheel_distribution_metadata(path: Path) -> email.message.Message:
    """Read the one top-level distribution metadata record from a wheel.

    Wheels such as setuptools may vendor other distributions, including nested
    ``*.dist-info/METADATA`` records.  Those records describe bundled
    dependencies and must not be mistaken for the wheel's own identity.
    """

    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise ValueError(
                f"cannot identify exactly one top-level METADATA file in {path.name}: "
                f"{names}"
            )
        return email.message_from_bytes(archive.read(names[0]))


def build_waiver_resolver_input(source: Path, destination: Path) -> tuple[str, ...]:
    """Remove only CHGNet itself and expose all of its non-Torch dependencies."""

    retained: list[str] = []
    removed: list[str] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if canonicalize_name(requirement.name) == "chgnet":
            removed.append(str(requirement))
        else:
            retained.append(line)
    if removed != [f"chgnet=={CHGNET_VERSION}"]:
        raise ValueError(f"expected exactly chgnet=={CHGNET_VERSION}, found {removed}")
    values = tuple(retained) + CHGNET_NON_TORCH_REQUIREMENTS
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(values) + "\n")
    return values


def _deduplicate_requirements(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        requirement = Requirement(value)
        key = str(requirement)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def build_isolated_runtime_resolver_inputs(
    source: Path,
    core_destination: Path,
    mattersim_destination: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Create conflict-free core/MatterSim inference resolver inputs.

    MACE 0.3.13 pins e3nn 0.4.4 while MatterSim needs e3nn >=0.5.0, so
    MatterSim is resolved into an isolated ``PYTHONPATH`` target runtime.
    """

    retained: list[str] = []
    removed: dict[str, list[str]] = {
        "chgnet": [],
        "mattersim": [],
        "mace-torch": [],
    }
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name in removed:
            removed[name].append(str(requirement))
        else:
            retained.append(line)
    expected = {
        "chgnet": [f"chgnet=={CHGNET_VERSION}"],
        "mattersim": [f"mattersim=={MATTERSIM_VERSION}"],
        "mace-torch": [f"mace-torch=={MACE_VERSION}"],
    }
    if removed != expected:
        raise ValueError(f"top-level evaluator requirements changed: {removed}")
    core_values = _deduplicate_requirements(
        (*retained, *CHGNET_RESOLVER_REQUIREMENTS, *MACE_RESOLVER_REQUIREMENTS)
    )
    mattersim_values = _deduplicate_requirements(MATTERSIM_INFERENCE_REQUIREMENTS)
    for destination, values in (
        (core_destination, core_values),
        (mattersim_destination, mattersim_values),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(values) + "\n")
    return core_values, mattersim_values


def audit_resolver_report(
    report_path: Path,
    protected_snapshot_path: Path,
    output_path: Path,
) -> tuple[str, ...]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    before = json.loads(protected_snapshot_path.read_text(encoding="utf-8"))[
        "protected_packages"
    ]
    violations: list[dict[str, str]] = []
    resolved: list[str] = []
    for item in report.get("install", []):
        package = item.get("metadata", {})
        name = canonicalize_name(str(package.get("name", "")))
        version = str(package.get("version", ""))
        if not name or not version:
            raise ValueError("resolver report contains an unnamed/unversioned distribution")
        if name in before:
            if version != before[name]:
                violations.append(
                    {"package": name, "before": before[name], "resolver": version}
                )
            continue
        resolved.append(f"{name}=={version}")
    if violations:
        raise RuntimeError(f"resolver would replace protected packages: {violations}")
    values = tuple(sorted(set(resolved)))
    with output_path.open("x", encoding="utf-8") as handle:
        if values:
            handle.write("\n".join(values) + "\n")
    return values


def _active_requirements(distribution: metadata.Distribution) -> dict[str, Requirement]:
    active: dict[str, Requirement] = {}
    for raw in distribution.requires or ():
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(requirement.name)
        if name in active:
            raise RuntimeError(f"duplicate active distribution requirement for {name}")
        active[name] = requirement
    return active


def _requirement_signatures(
    requirements: dict[str, Requirement],
) -> dict[str, tuple[str, tuple[str, ...], str | None]]:
    return {
        name: (str(requirement.specifier), tuple(sorted(requirement.extras)), requirement.url)
        for name, requirement in requirements.items()
    }


def finalize_chgnet_torch_waiver(
    *,
    pip_check_before_output: Path,
    pip_check_before_status: int,
    pip_check_output: Path,
    pip_check_status: int,
    output_paths: Sequence[Path],
    source_bundle_sha256: str,
) -> dict[str, Any]:
    """Prove that the registered Torch metadata mismatch is the only mismatch."""

    import torch

    if torch.__version__.split("+")[0] != RETAINED_TORCH_BASE:
        raise RuntimeError(f"waiver cannot change frozen Torch: {torch.__version__}")
    if not re.fullmatch(r"[0-9a-f]{64}", source_bundle_sha256):
        raise ValueError("source bundle SHA256 is invalid")
    if metadata.version("chgnet") != CHGNET_VERSION:
        raise RuntimeError("CHGNet version differs from the authorized waiver")

    active = _active_requirements(metadata.distribution("chgnet"))
    expected = {
        canonicalize_name(Requirement(value).name): Requirement(value)
        for value in (*CHGNET_NON_TORCH_REQUIREMENTS, WAIVED_REQUIREMENT)
    }
    if _requirement_signatures(active) != _requirement_signatures(expected):
        raise RuntimeError(
            "installed CHGNet active requirements differ from the reviewed metadata: "
            f"{[str(value) for value in active.values()]}"
        )
    for name, requirement in active.items():
        if name == "torch":
            continue
        installed = metadata.version(name)
        if installed not in requirement.specifier:
            raise RuntimeError(f"unsatisfied CHGNet dependency: {requirement}; found {installed}")

    raw_before = pip_check_before_output.read_bytes()
    before_lines = [
        line.strip() for line in raw_before.decode("utf-8").splitlines() if line.strip()
    ]
    raw_output = pip_check_output.read_bytes()
    lines = [line.strip() for line in raw_output.decode("utf-8").splitlines() if line.strip()]
    if pip_check_status == 0:
        raise RuntimeError("pip check omitted the authorized CHGNet/Torch mismatch")
    required_fragments = (
        f"chgnet {CHGNET_VERSION}",
        "requirement torch>=2.4.1",
        "torch 2.4.0",
    )
    chgnet_lines = [
        line
        for line in lines
        if all(
            fragment in line.lower().replace("_", "-")
            for fragment in required_fragments
        )
    ]
    if len(chgnet_lines) != 1:
        raise RuntimeError(f"pip check must contain exactly one CHGNet waiver line: {lines}")
    before_other = sorted(
        line for line in before_lines if "chgnet 0.4.2" not in line.lower()
    )
    after_other = sorted(line for line in lines if line != chgnet_lines[0])
    if before_other != after_other:
        raise RuntimeError(
            "global pip-check baseline changed during project installation: "
            f"before={before_other}, after={after_other}"
        )
    if pip_check_before_status == 0 and before_lines:
        raise RuntimeError("pip-check baseline status/output are inconsistent")
    if pip_check_before_status != 0 and not before_lines:
        raise RuntimeError("pip-check baseline failed without diagnostic lines")

    payload: dict[str, Any] = {
        "schema": WAIVER_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": AUTHORIZATION,
        "scope": "chgnet_0p4p2_torch_minimum_metadata_only",
        "package": "chgnet",
        "package_version": CHGNET_VERSION,
        "waived_requirement": WAIVED_REQUIREMENT,
        "retained_torch": torch.__version__,
        "retained_torch_base": RETAINED_TORCH_BASE,
        "non_torch_requirements": list(CHGNET_NON_TORCH_REQUIREMENTS),
        "pip_check_output": lines,
        "pip_check_output_sha256": _sha256_bytes(raw_output),
        "authorized_chgnet_pip_check_line": chgnet_lines[0],
        "preexisting_pip_check_status": pip_check_before_status,
        "preexisting_pip_check_output": before_lines,
        "preexisting_pip_check_output_sha256": _sha256_bytes(raw_before),
        "validation": CHGNET_WAIVER_VALIDATION,
        "source_bundle_sha256": source_bundle_sha256,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    for output in output_paths:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    return payload


def validate_mace_runtime_metadata() -> dict[str, str]:
    """Prove the no-deps MACE wheel has exactly the reviewed direct closure."""

    if metadata.version("mace-torch") != MACE_VERSION:
        raise RuntimeError("MACE version differs from the registered runtime")
    active = _active_requirements(metadata.distribution("mace-torch"))
    expected = {
        canonicalize_name(Requirement(value).name): Requirement(value)
        for value in MACE_ACTIVE_METADATA_REQUIREMENTS
    }
    if _requirement_signatures(active) != _requirement_signatures(expected):
        raise RuntimeError(
            "installed MACE requirements differ from reviewed 0.3.13 metadata: "
            f"{[str(value) for value in active.values()]}"
        )
    installed_versions: dict[str, str] = {}
    for name, requirement in active.items():
        installed = metadata.version(name)
        if installed not in requirement.specifier:
            raise RuntimeError(f"unsatisfied MACE dependency: {requirement}; found {installed}")
        installed_versions[name] = installed
    return installed_versions


def load_chgnet_torch_waiver(path: Path, *, installed_torch: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "schema": WAIVER_SCHEMA,
        "authorization": AUTHORIZATION,
        "scope": "chgnet_0p4p2_torch_minimum_metadata_only",
        "package": "chgnet",
        "package_version": CHGNET_VERSION,
        "waived_requirement": WAIVED_REQUIREMENT,
        "retained_torch_base": RETAINED_TORCH_BASE,
        "validation": CHGNET_WAIVER_VALIDATION,
    }
    for key, value in exact.items():
        if payload.get(key) != value:
            raise ValueError(f"invalid dependency-waiver field {key}: {payload.get(key)!r}")
    if payload.get("retained_torch") != installed_torch:
        raise ValueError("dependency waiver was validated against a different Torch build")
    if tuple(payload.get("non_torch_requirements", ())) != CHGNET_NON_TORCH_REQUIREMENTS:
        raise ValueError("dependency waiver non-Torch requirements differ from the lock")
    lines = payload.get("pip_check_output")
    if not isinstance(lines, list) or not lines:
        raise ValueError("dependency waiver must contain pip-check evidence")
    authorized = str(payload.get("authorized_chgnet_pip_check_line", ""))
    if lines.count(authorized) != 1:
        raise ValueError("dependency waiver must identify exactly one CHGNet mismatch")
    before_lines = payload.get("preexisting_pip_check_output")
    if not isinstance(before_lines, list):
        raise ValueError("dependency waiver lacks the global pip-check baseline")
    if sorted(line for line in lines if line != authorized) != sorted(
        line for line in before_lines if "chgnet 0.4.2" not in line.lower()
    ):
        raise ValueError("dependency waiver global pip-check baseline changed")
    raw_output = ("\n".join(str(line) for line in lines) + "\n").encode("utf-8")
    if payload.get("pip_check_output_sha256") != _sha256_bytes(raw_output):
        raise ValueError("dependency waiver pip-check evidence hash mismatch")
    raw_before = (
        "\n".join(str(line) for line in before_lines)
        + ("\n" if before_lines else "")
    ).encode("utf-8")
    if payload.get("preexisting_pip_check_output_sha256") != _sha256_bytes(raw_before):
        raise ValueError("dependency waiver pre-install pip-check hash mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("source_bundle_sha256", ""))):
        raise ValueError("dependency waiver source bundle hash is invalid")
    return payload


def finalize_mattersim_inference_waiver(
    *,
    output_paths: Sequence[Path],
    source_bundle_sha256: str,
    runtime_tree_sha256: str,
) -> dict[str, Any]:
    """Validate the isolated forcefield import closure against wheel metadata."""

    import torch

    if torch.__version__.split("+")[0] != RETAINED_TORCH_BASE:
        raise RuntimeError(f"MatterSim runtime changed frozen Torch: {torch.__version__}")
    for value, name in (
        (source_bundle_sha256, "source bundle"),
        (runtime_tree_sha256, "runtime tree"),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{name} SHA256 is invalid")
    if metadata.version("mattersim") != MATTERSIM_VERSION:
        raise RuntimeError("MatterSim version differs from the isolated-runtime lock")
    active = _active_requirements(metadata.distribution("mattersim"))
    expected = {
        canonicalize_name(Requirement(value).name): Requirement(value)
        for value in MATTERSIM_ACTIVE_METADATA_REQUIREMENTS
    }
    if _requirement_signatures(active) != _requirement_signatures(expected):
        raise RuntimeError(
            "installed MatterSim requirements differ from reviewed 1.1.2 metadata: "
            f"{[str(value) for value in active.values()]}"
        )
    inference = {
        canonicalize_name(Requirement(value).name): Requirement(value)
        for value in MATTERSIM_INFERENCE_REQUIREMENTS
    }
    for name, requirement in inference.items():
        installed = metadata.version(name)
        if installed not in requirement.specifier:
            raise RuntimeError(
                f"unsatisfied MatterSim inference dependency: {requirement}; found {installed}"
            )
    imports: dict[str, str] = {}
    for module in MATTERSIM_INFERENCE_IMPORTS:
        loaded = importlib.import_module(module)
        imports[module] = str(getattr(loaded, "__version__", "import_ok"))
    payload: dict[str, Any] = {
        "schema": WAIVER_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "user_environment_scope_explicit_2026-07-17",
        "scope": "mattersim_1p1p2_forcefield_inference_runtime_only",
        "package": "mattersim",
        "package_version": MATTERSIM_VERSION,
        "official_wheel_sha256": MATTERSIM_WHEEL_SHA256,
        "retained_torch": torch.__version__,
        "retained_torch_base": RETAINED_TORCH_BASE,
        "inference_requirements": list(MATTERSIM_INFERENCE_REQUIREMENTS),
        "excluded_noninference_distributions": list(
            MATTERSIM_EXCLUDED_NONINFERENCE_DISTRIBUTIONS
        ),
        "compatibility_pins": MATTERSIM_COMPATIBILITY_PINS,
        "validated_imports": imports,
        "runtime_tree_sha256": runtime_tree_sha256,
        "validation": "reviewed_forcefield_import_closure_satisfied_in_isolated_target",
        "source_bundle_sha256": source_bundle_sha256,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    for output in output_paths:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    return payload


def load_mattersim_inference_waiver(
    path: Path,
    *,
    installed_torch: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "schema": WAIVER_SCHEMA,
        "authorization": "user_environment_scope_explicit_2026-07-17",
        "scope": "mattersim_1p1p2_forcefield_inference_runtime_only",
        "package": "mattersim",
        "package_version": MATTERSIM_VERSION,
        "official_wheel_sha256": MATTERSIM_WHEEL_SHA256,
        "retained_torch_base": RETAINED_TORCH_BASE,
        "validation": "reviewed_forcefield_import_closure_satisfied_in_isolated_target",
    }
    for key, value in exact.items():
        if payload.get(key) != value:
            raise ValueError(f"invalid MatterSim waiver field {key}: {payload.get(key)!r}")
    if payload.get("retained_torch") != installed_torch:
        raise ValueError("MatterSim waiver was validated against a different Torch build")
    if tuple(payload.get("inference_requirements", ())) != MATTERSIM_INFERENCE_REQUIREMENTS:
        raise ValueError("MatterSim inference requirements differ from the lock")
    if tuple(payload.get("excluded_noninference_distributions", ())) != (
        MATTERSIM_EXCLUDED_NONINFERENCE_DISTRIBUTIONS
    ):
        raise ValueError("MatterSim excluded dependency set differs from the lock")
    if payload.get("compatibility_pins") != MATTERSIM_COMPATIBILITY_PINS:
        raise ValueError("MatterSim compatibility pins differ from the lock")
    for key in ("runtime_tree_sha256", "source_bundle_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get(key, ""))):
            raise ValueError(f"MatterSim waiver {key} is invalid")
    return payload


def build_runtime_tree_manifest(root: Path) -> dict[str, Any]:
    """Hash every regular file in an isolated target without path-dependent data."""

    location = root.resolve()
    if not location.is_dir():
        raise FileNotFoundError(location)
    entries: list[dict[str, Any]] = []
    for path in sorted(location.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"isolated runtime may not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"isolated runtime contains a non-regular path: {path}")
        relative = path.relative_to(location).as_posix()
        if relative.endswith(".pyc") or "/__pycache__/" in f"/{relative}":
            raise RuntimeError(f"isolated runtime contains unregistered bytecode: {relative}")
        content = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": _sha256_bytes(content),
            }
        )
    if not entries:
        raise RuntimeError("isolated runtime tree is empty")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "schema": "wqcodiff_isolated_runtime_tree_v1",
        "files": entries,
        "file_count": len(entries),
        "tree_sha256": _sha256_bytes(canonical),
    }


def validate_runtime_tree_manifest(root: Path, payload: Mapping[str, Any]) -> str:
    if payload.get("schema") != "wqcodiff_isolated_runtime_tree_v1":
        raise ValueError("invalid isolated-runtime tree schema")
    discovered = build_runtime_tree_manifest(root)
    if dict(payload) != discovered:
        raise RuntimeError("isolated MatterSim runtime tree differs from its manifest")
    return str(discovered["tree_sha256"])


def load_mattersim_runtime_lock(
    path: Path,
    *,
    model_root: Path,
    installed_torch: str,
) -> dict[str, Any]:
    """Validate the immutable target runtime, its manifest, and waiver binding."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "schema": "wqcodiff_mattersim_runtime_lock_v4",
        "runtime": MATTERSIM_RUNTIME_RELATIVE,
        "tree_manifest": MATTERSIM_TREE_MANIFEST_FILENAME,
        "dependency_waiver": MATTERSIM_WAIVER_FILENAME,
        "wheelhouse_lock": WHEELHOUSE_LOCK_FILENAME,
        "source_bundle_sha256": payload.get("source_bundle_sha256"),
    }
    for key, value in exact.items():
        if payload.get(key) != value:
            raise ValueError(f"invalid MatterSim runtime-lock field {key}")
    if payload.get("compatibility_pins") != MATTERSIM_COMPATIBILITY_PINS:
        raise ValueError("MatterSim runtime-lock compatibility pins changed")
    for key in (
        "source_bundle_sha256",
        "runtime_tree_sha256",
        "tree_manifest_sha256",
        "dependency_waiver_sha256",
        "wheelhouse_lock_sha256",
        "resolver_report_sha256",
        "resolver_requirements_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get(key, ""))):
            raise ValueError(f"invalid MatterSim runtime-lock digest {key}")
    if payload.get("retained_torch") != installed_torch:
        raise ValueError("MatterSim runtime lock was created against another Torch build")

    root = model_root.resolve()
    runtime = root / MATTERSIM_RUNTIME_RELATIVE
    manifest_path = root / MATTERSIM_TREE_MANIFEST_FILENAME
    waiver_path = root / MATTERSIM_WAIVER_FILENAME
    wheelhouse_path = root / WHEELHOUSE_LOCK_FILENAME
    for required in (runtime, manifest_path, waiver_path, wheelhouse_path):
        if not required.exists():
            raise FileNotFoundError(required)
    if _sha256_bytes(manifest_path.read_bytes()) != payload["tree_manifest_sha256"]:
        raise RuntimeError("MatterSim runtime tree-manifest SHA256 mismatch")
    if _sha256_bytes(waiver_path.read_bytes()) != payload["dependency_waiver_sha256"]:
        raise RuntimeError("MatterSim inference-waiver SHA256 mismatch")
    if _sha256_bytes(wheelhouse_path.read_bytes()) != payload["wheelhouse_lock_sha256"]:
        raise RuntimeError("MatterSim runtime wheelhouse-lock SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tree_sha256 = validate_runtime_tree_manifest(runtime, manifest)
    if tree_sha256 != payload["runtime_tree_sha256"]:
        raise RuntimeError("MatterSim runtime tree SHA256 differs from runtime lock")
    waiver = load_mattersim_inference_waiver(
        waiver_path,
        installed_torch=installed_torch,
    )
    if waiver["runtime_tree_sha256"] != tree_sha256:
        raise RuntimeError("MatterSim waiver binds a different runtime tree")
    if waiver["source_bundle_sha256"] != payload["source_bundle_sha256"]:
        raise RuntimeError("MatterSim waiver and runtime lock bind different source bundles")
    return payload
