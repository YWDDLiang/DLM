#!/usr/bin/env bash
set -Eeuo pipefail

# Login-node-only environment lock. Default APPLY=0 is non-mutating.
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-wq-env-lock}"
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
MODEL_ROOT="${MODEL_ROOT:-/public/home/jiaosz/ywliang/models/wqcodiff}"
ENV_NAME="${ENV_NAME:-diff_meets_diff}"
APPLY="${APPLY:-0}"
ALLOW_CHGNET_TORCH_WAIVER="${ALLOW_CHGNET_TORCH_WAIVER:-0}"
ALLOW_MATTERSIM_INFERENCE_RUNTIME="${ALLOW_MATTERSIM_INFERENCE_RUNTIME:-0}"
SOURCE_BUNDLE_SHA256="${SOURCE_BUNDLE_SHA256:-ae5386bdfcce8ffd981cfeaf2a977e7e96749e7c90c58ba2b60c7e30f95c025b}"
LOG_DIR="${PROJECT_ROOT}/runs/${RUN_ID}/logs"
LOCK_DIR="${PROJECT_ROOT}/runs/${RUN_ID}/environment"
WHEELHOUSE="${MODEL_ROOT}/wheelhouse_v4"
WHEELHOUSE_LOCK="${MODEL_ROOT}/wheelhouse_lock_v4.json"
REPORT="${LOCK_DIR}/resolver_report.json"
MATTERSIM_REPORT="${LOCK_DIR}/mattersim_resolver_report.json"
RUNTIME_CONSTRAINTS="${LOCK_DIR}/runtime_constraints.txt"
RESOLVED_MISSING="${LOCK_DIR}/resolved_missing_requirements.txt"
MATTERSIM_RESOLVED_MISSING="${LOCK_DIR}/mattersim_resolved_missing_requirements.txt"
RESOLVER_REQUIREMENTS="${LOCK_DIR}/resolver_requirements.txt"
MATTERSIM_RESOLVER_REQUIREMENTS="${LOCK_DIR}/mattersim_resolver_requirements.txt"
PIP_CHECK_OUTPUT="${LOCK_DIR}/pip_check.txt"
PIP_CHECK_BEFORE_OUTPUT="${LOCK_DIR}/pip_check_before.txt"
PIP_CHECK_BEFORE_STATUS_FILE="${LOCK_DIR}/pip_check_before.status"
WAIVER_RUN_RECORD="${LOCK_DIR}/chgnet_torch_metadata_waiver_v4.json"
WAIVER_MODEL_RECORD="${MODEL_ROOT}/chgnet_torch_metadata_waiver_v4.json"
MATTERSIM_WAIVER_RUN_RECORD="${LOCK_DIR}/mattersim_inference_runtime_waiver_v4.json"
MATTERSIM_WAIVER_MODEL_RECORD="${MODEL_ROOT}/mattersim_inference_runtime_waiver_v4.json"
SOURCE_SDISTS="${MODEL_ROOT}/source_sdists_v4"
MATTERSIM_RUNTIME_ROOT="${MODEL_ROOT}/runtimes"
MATTERSIM_RUNTIME="${MATTERSIM_RUNTIME_ROOT}/mattersim-1.1.2-py310-v4"
MATTERSIM_RUNTIME_STAGING="${MATTERSIM_RUNTIME_ROOT}/.mattersim-1.1.2-py310-v4-${RUN_ID}.staging"
MATTERSIM_TREE_MANIFEST="${LOCK_DIR}/mattersim_runtime_tree_v4.json"
MATTERSIM_TREE_MODEL_MANIFEST="${MODEL_ROOT}/mattersim_runtime_tree_v4.json"
MATTERSIM_RUNTIME_LOCK="${MODEL_ROOT}/mattersim_runtime_lock_v4.json"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: environment resolution/download is forbidden inside Slurm." >&2
  exit 2
fi
if [ "${APPLY}" != "0" ] && [ "${APPLY}" != "1" ]; then
  echo "ERROR: APPLY must be 0 or 1." >&2
  exit 2
fi
if [ "${ALLOW_CHGNET_TORCH_WAIVER}" != "0" ] && [ "${ALLOW_CHGNET_TORCH_WAIVER}" != "1" ]; then
  echo "ERROR: ALLOW_CHGNET_TORCH_WAIVER must be 0 or 1." >&2
  exit 2
fi
if [ "${ALLOW_MATTERSIM_INFERENCE_RUNTIME}" != "0" ] && [ "${ALLOW_MATTERSIM_INFERENCE_RUNTIME}" != "1" ]; then
  echo "ERROR: ALLOW_MATTERSIM_INFERENCE_RUNTIME must be 0 or 1." >&2
  exit 2
fi
if [ "${ALLOW_MATTERSIM_INFERENCE_RUNTIME}" = "1" ] && [ "${ALLOW_CHGNET_TORCH_WAIVER}" != "1" ]; then
  echo "ERROR: the registered isolated evaluator stack also requires the CHGNet waiver." >&2
  exit 2
fi
if [ -e "${LOG_DIR}" ] || [ -e "${LOCK_DIR}" ]; then
  echo "ERROR: immutable environment RUN_ID already exists: ${RUN_ID}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "${LOCK_DIR}" "${WHEELHOUSE}"

{
  echo "WQ environment lock start: $(date '+%F %T %Z')"
  echo "host=$(hostname) user=$(whoami) environment=${ENV_NAME} apply=${APPLY} chgnet_waiver=${ALLOW_CHGNET_TORCH_WAIVER} mattersim_isolated=${ALLOW_MATTERSIM_INFERENCE_RUNTIME}"

  set +u
  if [ -f /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh ]; then
    source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
  else
    source ~/.bashrc
  fi
  set -u
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "ERROR: required conda environment ${ENV_NAME} is missing." >&2
    exit 3
  fi
  conda activate "${ENV_NAME}"
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

  python - "${LOCK_DIR}/protected_before.json" "${RUNTIME_CONSTRAINTS}" <<'PY'
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path
import numpy
import torch

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"registered Python is 3.10, found {sys.version}")
if torch.__version__.split("+")[0] != "2.4.0":
    raise SystemExit(f"registered torch is 2.4.0, found {torch.__version__}")

installed = {dist.metadata["Name"].lower(): dist.version for dist in md.distributions()}
protected_names = {"torch", "numpy", "pymatgen", "spglib", "triton"}
protected_names.update(name for name in installed if name.startswith("nvidia-"))
protected = {name: installed[name] for name in sorted(protected_names) if name in installed}
snapshot = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "numpy": numpy.__version__,
    "protected_packages": protected,
}
Path(sys.argv[1]).write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
constraints = ["torch==2.4.0"] + [
    f"{name}=={version}" for name, version in protected.items() if name != "torch"
]
Path(sys.argv[2]).write_text("\n".join(constraints) + "\n")
print(json.dumps(snapshot, indent=2, sort_keys=True))
PY

  if [ "${ALLOW_MATTERSIM_INFERENCE_RUNTIME}" = "1" ]; then
    python - "${PROJECT_ROOT}" "${PROJECT_ROOT}/requirements/wqcodiff-py310.txt" \
      "${RESOLVER_REQUIREMENTS}" "${MATTERSIM_RESOLVER_REQUIREMENTS}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from crystal_dlm.wqcodiff.dependency_waiver import build_isolated_runtime_resolver_inputs

core, mattersim = build_isolated_runtime_resolver_inputs(
    Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
)
print("authorized core resolver input:\n" + "\n".join(core))
print("authorized isolated MatterSim resolver input:\n" + "\n".join(mattersim))
PY
  elif [ "${ALLOW_CHGNET_TORCH_WAIVER}" = "1" ]; then
    python - "${PROJECT_ROOT}" "${PROJECT_ROOT}/requirements/wqcodiff-py310.txt" \
      "${RESOLVER_REQUIREMENTS}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from crystal_dlm.wqcodiff.dependency_waiver import build_waiver_resolver_input

values = build_waiver_resolver_input(Path(sys.argv[2]), Path(sys.argv[3]))
print("authorized CHGNet/Torch resolver input:")
print("\n".join(values))
PY
  else
    python - "${PROJECT_ROOT}/requirements/wqcodiff-py310.txt" \
      "${RESOLVER_REQUIREMENTS}" <<'PY'
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
with destination.open("x", encoding="utf-8") as handle:
    handle.write(source.read_text(encoding="utf-8"))
PY
  fi
  if [ ! -e "${MATTERSIM_RESOLVER_REQUIREMENTS}" ]; then
    python - "${MATTERSIM_RESOLVER_REQUIREMENTS}" <<'PY'
import sys
from pathlib import Path
with Path(sys.argv[1]).open("x", encoding="utf-8"):
    pass
PY
  fi

  conda list --explicit > "${LOCK_DIR}/conda_explicit_before.txt"
  python -m pip freeze --all > "${LOCK_DIR}/pip_freeze_before.txt"
  set +e
  python -m pip check > "${PIP_CHECK_BEFORE_OUTPUT}" 2>&1
  PIP_CHECK_BEFORE_STATUS=$?
  set -e
  printf '%s\n' "${PIP_CHECK_BEFORE_STATUS}" > "${PIP_CHECK_BEFORE_STATUS_FILE}"
  sha256sum "${LOCK_DIR}/conda_explicit_before.txt" \
    "${LOCK_DIR}/pip_freeze_before.txt" \
    "${PIP_CHECK_BEFORE_OUTPUT}" \
    "${PIP_CHECK_BEFORE_STATUS_FILE}" \
    "${LOCK_DIR}/protected_before.json" \
    "${RUNTIME_CONSTRAINTS}" \
    "${RESOLVER_REQUIREMENTS}" \
    "${MATTERSIM_RESOLVER_REQUIREMENTS}" > "${LOCK_DIR}/environment_before.sha256"

  python -m pip install --dry-run --only-binary=:all: --report "${REPORT}" \
    --constraint "${RUNTIME_CONSTRAINTS}" \
    --requirement "${RESOLVER_REQUIREMENTS}"
  if [ "${ALLOW_MATTERSIM_INFERENCE_RUNTIME}" = "1" ]; then
    python -m pip install --dry-run --only-binary=:all: --report "${MATTERSIM_REPORT}" \
      --constraint "${RUNTIME_CONSTRAINTS}" \
      --requirement "${MATTERSIM_RESOLVER_REQUIREMENTS}"
  fi
  python - "${PROJECT_ROOT}" "${REPORT}" "${LOCK_DIR}/protected_before.json" \
    "${RESOLVED_MISSING}" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from crystal_dlm.wqcodiff.dependency_waiver import audit_resolver_report
values = audit_resolver_report(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
print(json.dumps({"core_resolved_missing_distributions": values}, indent=2))
PY
  if [ "${ALLOW_MATTERSIM_INFERENCE_RUNTIME}" = "1" ]; then
    python - "${PROJECT_ROOT}" "${MATTERSIM_REPORT}" \
      "${LOCK_DIR}/protected_before.json" "${MATTERSIM_RESOLVED_MISSING}" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from crystal_dlm.wqcodiff.dependency_waiver import audit_resolver_report
values = audit_resolver_report(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
print(json.dumps({"mattersim_resolved_missing_distributions": values}, indent=2))
PY
  else
    python - "${MATTERSIM_RESOLVED_MISSING}" <<'PY'
import sys
from pathlib import Path
with Path(sys.argv[1]).open("x", encoding="utf-8"):
    pass
PY
  fi

  if [ "${APPLY}" != "1" ]; then
    echo "DRY RUN ONLY. Review ${REPORT}; no package or model was changed."
    exit 0
  fi

  python -m pip download --dest "${WHEELHOUSE}" \
    --only-binary=:all: \
    --no-deps \
    --requirement "${PROJECT_ROOT}/requirements/wqcodiff-py310.txt"
  python -m pip download --dest "${WHEELHOUSE}" \
    --only-binary=:all: \
    --no-deps \
    --requirement "${RESOLVED_MISSING}"
  if [ "${ALLOW_MATTERSIM_INFERENCE_RUNTIME}" = "1" ]; then
    python -m pip download --dest "${WHEELHOUSE}" \
      --only-binary=:all: \
      --no-deps \
      --requirement "${MATTERSIM_RESOLVED_MISSING}"
  fi
  mkdir -p "${SOURCE_SDISTS}"
  python - "${SOURCE_SDISTS}" <<'PY'
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "nvidia-ml-py3-7.352.0.tar.gz": (
        "390f02919ee9d73fe63a98c73101061a6b37fa694a793abf56673320f1f51277",
        "https://files.pythonhosted.org/packages/6d/64/cce82bddb80c0b0f5c703bbdafa94bfb69a1c5ad7a79cff00b482468f0d3/nvidia-ml-py3-7.352.0.tar.gz",
    ),
    "python_hostlist-2.3.0.tar.gz": (
        "e1a0b18e525a5fca573cb9862799f11b3f2bd3ba7aec70c4ecd8b95341bb71ea",
        "https://files.pythonhosted.org/packages/90/cc/bb6395c3f2b6bb739b1d3fc0e71f94e6a1c2e256df496237cbfd13cd74a6/python_hostlist-2.3.0.tar.gz",
    ),
}
for filename, (wanted, url) in expected.items():
    path = root / filename
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        if temporary.exists():
            raise SystemExit(f"stale source-sdist partial exists: {temporary}")
        request = urllib.request.Request(url, headers={"User-Agent": "wqcodiff-source-lock/1"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != wanted:
            raise SystemExit(f"downloaded {filename} SHA256 mismatch: {digest}")
        os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != wanted:
        raise SystemExit(f"{filename} SHA256 mismatch: {digest}")
    print(f"verified source sdist {path} {digest}")
actual = {path.name for path in root.iterdir() if path.is_file()}
if actual != set(expected):
    raise SystemExit(f"source-sdist set differs from the lock: {sorted(actual)}")
PY
  if [ ! -e "${WHEELHOUSE_LOCK}" ]; then
    SOURCE_DATE_EPOCH=315532800 PYTHONHASHSEED=0 \
      python -m pip wheel --wheel-dir "${WHEELHOUSE}" --no-deps \
      --no-build-isolation --no-cache-dir \
      "${SOURCE_SDISTS}/nvidia-ml-py3-7.352.0.tar.gz"
    SOURCE_DATE_EPOCH=315532800 PYTHONHASHSEED=0 \
      python -m pip wheel --wheel-dir "${WHEELHOUSE}" --no-deps \
      --no-build-isolation --no-cache-dir \
      "${SOURCE_SDISTS}/python_hostlist-2.3.0.tar.gz"
  else
    echo "immutable wheelhouse-v4 lock exists; source-built wheels will be verified, not rebuilt"
  fi
  find "${WHEELHOUSE}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${LOCK_DIR}/wheelhouse.sha256"
  python - "${PROJECT_ROOT}" "${WHEELHOUSE}" "${WHEELHOUSE_LOCK}" \
    "${SOURCE_SDISTS}" "${MODEL_ROOT}/wheelhouse_lock.json" \
    "${MODEL_ROOT}/wheelhouse_lock_v2.json" <<'PY'
import hashlib
import importlib.metadata as distribution_metadata
import json
import platform
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(project_root))
from crystal_dlm.wqcodiff.dependency_waiver import load_wheel_distribution_metadata

wheelhouse = Path(sys.argv[2]).resolve()
lock_path = Path(sys.argv[3]).resolve()
source_root = Path(sys.argv[4]).resolve()
failed_predecessor_paths = tuple(Path(value).resolve() for value in sys.argv[5:])
wheels = sorted(wheelhouse.glob("*.whl"))
other = sorted(path.name for path in wheelhouse.iterdir() if path.is_file() and path.suffix != ".whl")
if other:
    raise SystemExit(f"offline wheelhouse contains non-wheel distributions: {other}")
if not wheels:
    raise SystemExit("offline wheelhouse is empty")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

entries = []
for path in wheels:
    metadata = load_wheel_distribution_metadata(path)
    name = str(metadata.get("Name"))
    version = str(metadata.get("Version"))
    entries.append({
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha(path),
        "package": name,
        "version": version,
        "license": str(metadata.get("License") or "see-wheel-metadata"),
        "requires_python": str(metadata.get("Requires-Python") or ""),
        "source": f"https://pypi.org/project/{name}/{version}/",
    })
required = {
    "pyxtal": "1.1.4",
    "chgnet": "0.4.2",
    "mattersim": "1.1.2",
    "mace-torch": "0.3.13",
    "ase": "3.27.0",
    "setuptools": "81.0.0",
}
discovered = {entry["package"].lower().replace("_", "-"): entry["version"] for entry in entries}
missing = {name: version for name, version in required.items() if discovered.get(name) != version}
if missing:
    raise SystemExit(f"wheelhouse lacks required exact distributions: {missing}")
mattersim = [entry for entry in entries if entry["package"].lower() == "mattersim"]
if len(mattersim) != 1 or mattersim[0]["sha256"] != "e249532b6e66d9307c7a72fde252f0bcf151c588b8656ce56ef1cbaf0ed90d10":
    raise SystemExit(f"MatterSim 1.1.2 official CPython-3.10 wheel hash mismatch: {mattersim}")
ase = [entry for entry in entries if entry["package"].lower() == "ase" and entry["version"] == "3.27.0"]
if len(ase) != 1 or ase[0]["sha256"] != "058c48ea504fe7fbbe7c932f778415243ef2df45b1ab869866f24efcc17f0538":
    raise SystemExit(f"ASE 3.27.0 official wheel hash mismatch: {ase}")
setuptools = [
    entry for entry in entries
    if entry["package"].lower() == "setuptools" and entry["version"] == "81.0.0"
]
if len(setuptools) != 1 or setuptools[0]["sha256"] != "fdd925d5c5d9f62e4b74b30d6dd7828ce236fd6ed998a08d81de62ce5a6310d6":
    raise SystemExit(f"setuptools 81.0.0 official wheel hash mismatch: {setuptools}")
sdist = source_root / "nvidia-ml-py3-7.352.0.tar.gz"
if not sdist.is_file() or sha(sdist) != "390f02919ee9d73fe63a98c73101061a6b37fa694a793abf56673320f1f51277":
    raise SystemExit("frozen nvidia-ml-py3 source sdist is missing or changed")
nvidia_wheels = [
    entry for entry in entries
    if entry["package"].lower().replace("_", "-") == "nvidia-ml-py3"
    and entry["version"] == "7.352.0"
]
if len(nvidia_wheels) != 1:
    raise SystemExit(f"expected exactly one locally built nvidia-ml-py3 wheel: {nvidia_wheels}")
hostlist_sdist = source_root / "python_hostlist-2.3.0.tar.gz"
if not hostlist_sdist.is_file() or sha(hostlist_sdist) != "e1a0b18e525a5fca573cb9862799f11b3f2bd3ba7aec70c4ecd8b95341bb71ea":
    raise SystemExit("frozen python-hostlist source sdist is missing or changed")
hostlist_wheels = [
    entry for entry in entries
    if entry["package"].lower().replace("_", "-") == "python-hostlist"
    and entry["version"] == "2.3.0"
]
if len(hostlist_wheels) != 1:
    raise SystemExit(f"expected exactly one locally built python-hostlist wheel: {hostlist_wheels}")
payload = {
    "schema": "wqcodiff_wheelhouse_lock_v4",
    "stack_id": "wqcodiff-evaluator-stack-v4",
    "wheelhouse": str(wheelhouse),
    "build_environment": {
        "python": platform.python_version(),
        "pip": distribution_metadata.version("pip"),
        "setuptools": distribution_metadata.version("setuptools"),
        "wheel": distribution_metadata.version("wheel"),
        "build_isolation": False,
        "pip_cache": False,
        "source_date_epoch": 315532800,
        "python_hash_seed": 0,
    },
    "wheels": entries,
    "source_builds": [
        {
            "package": "nvidia-ml-py3",
            "version": "7.352.0",
            "source_filename": sdist.name,
            "source_sha256": sha(sdist),
            "source_url": "https://pypi.org/project/nvidia-ml-py3/7.352.0/",
            "built_wheel_filename": nvidia_wheels[0]["filename"],
            "built_wheel_sha256": nvidia_wheels[0]["sha256"],
        },
        {
            "package": "python-hostlist",
            "version": "2.3.0",
            "source_filename": hostlist_sdist.name,
            "source_sha256": sha(hostlist_sdist),
            "source_url": "https://pypi.org/project/python-hostlist/2.3.0/",
            "built_wheel_filename": hostlist_wheels[0]["filename"],
            "built_wheel_sha256": hostlist_wheels[0]["sha256"],
        },
    ],
}
if not all(path.is_file() for path in failed_predecessor_paths):
    raise SystemExit(f"registered predecessor locks are missing: {failed_predecessor_paths}")
failed_predecessors = []
reasons = {
    "wheelhouse_lock.json": "source_built_wheel_was_rebuilt_after_first_lock",
    "wheelhouse_lock_v2.json": "mattersim_import_failed_on_pkg_resources_and_ase_3p28_api",
}
for failed_predecessor_path in failed_predecessor_paths:
    failed_payload = json.loads(failed_predecessor_path.read_text(encoding="utf-8"))
    mismatches = []
    failed_wheelhouse = failed_predecessor_path.parent / str(
        Path(failed_payload.get("wheelhouse", "wheelhouse")).name
    )
    for entry in failed_payload.get("wheels", ()):
        candidate = failed_wheelhouse / entry["filename"]
        actual = sha(candidate) if candidate.is_file() else None
        if actual != entry["sha256"]:
            mismatches.append({"filename": entry["filename"], "locked": entry["sha256"], "actual": actual})
    failed_predecessors.append({
        "filename": failed_predecessor_path.name,
        "sha256": sha(failed_predecessor_path),
        "reason": reasons[failed_predecessor_path.name],
        "wheel_mismatches": mismatches,
    })
payload["failed_predecessors"] = failed_predecessors
if lock_path.exists():
    existing = json.loads(lock_path.read_text(encoding="utf-8"))
    if existing != payload:
        raise SystemExit("existing immutable wheelhouse lock differs from downloaded wheels")
else:
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
print(json.dumps({"wheelhouse_lock": str(lock_path), "wheels": len(entries)}, indent=2))
PY
  python -m pip install --no-index --find-links "${WHEELHOUSE}" \
    --constraint "${RUNTIME_CONSTRAINTS}" \
    --requirement "${RESOLVER_REQUIREMENTS}"
  python -m pip install --force-reinstall --no-index --find-links "${WHEELHOUSE}" \
    --no-deps "nvidia-ml-py3==7.352.0"
  python -m pip install --force-reinstall --no-index --find-links "${WHEELHOUSE}" \
    --no-deps "python-hostlist==2.3.0"
  if [ "${ALLOW_CHGNET_TORCH_WAIVER}" = "1" ]; then
    python -m pip install --no-index --find-links "${WHEELHOUSE}" \
      --no-deps "chgnet==0.4.2"
  fi
  python -m pip install --no-index --find-links "${WHEELHOUSE}" \
    --no-deps "mace-torch==0.3.13"
  set +e
  python -m pip check > "${PIP_CHECK_OUTPUT}" 2>&1
  PIP_CHECK_STATUS=$?
  set -e
  cat "${PIP_CHECK_OUTPUT}"
  if [ "${ALLOW_CHGNET_TORCH_WAIVER}" = "1" ]; then
    python - "${PROJECT_ROOT}" "${PIP_CHECK_BEFORE_OUTPUT}" \
      "${PIP_CHECK_BEFORE_STATUS}" "${PIP_CHECK_OUTPUT}" "${PIP_CHECK_STATUS}" \
      "${WAIVER_RUN_RECORD}" "${WAIVER_MODEL_RECORD}" "${SOURCE_BUNDLE_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from crystal_dlm.wqcodiff.dependency_waiver import (
    finalize_chgnet_torch_waiver,
    validate_mace_runtime_metadata,
)

validate_mace_runtime_metadata()

payload = finalize_chgnet_torch_waiver(
    pip_check_before_output=Path(sys.argv[2]),
    pip_check_before_status=int(sys.argv[3]),
    pip_check_output=Path(sys.argv[4]),
    pip_check_status=int(sys.argv[5]),
    output_paths=(Path(sys.argv[6]),),
    source_bundle_sha256=sys.argv[8],
)
model_record = Path(sys.argv[7])
if model_record.exists():
    existing = json.loads(model_record.read_text(encoding="utf-8"))
    comparable_existing = dict(existing)
    comparable_payload = dict(payload)
    comparable_existing.pop("created_utc", None)
    comparable_payload.pop("created_utc", None)
    if comparable_existing != comparable_payload:
        raise RuntimeError("existing immutable CHGNet waiver differs from current evidence")
else:
    with model_record.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  elif [ "${PIP_CHECK_STATUS}" -ne 0 ]; then
    echo "ERROR: pip check failed without an authorized exception." >&2
    exit "${PIP_CHECK_STATUS}"
  fi
  python -m pip freeze --all > "${LOCK_DIR}/pip_freeze_after.txt"
  python - "${LOCK_DIR}/protected_before.json" <<'PY'
import importlib.metadata as md
import json
import sys
before = json.load(open(sys.argv[1], encoding="utf-8"))["protected_packages"]
after = {name: md.version(name) for name in before}
if before != after:
    raise SystemExit(f"protected packages changed after install: before={before}, after={after}")
print("post-install protected-package audit passed")
PY
  if [ "${ALLOW_MATTERSIM_INFERENCE_RUNTIME}" = "1" ]; then
    if [ -e "${MATTERSIM_RUNTIME}" ] || [ -e "${MATTERSIM_RUNTIME_STAGING}" ] || \
       [ -e "${MATTERSIM_RUNTIME_LOCK}" ] || [ -e "${MATTERSIM_TREE_MODEL_MANIFEST}" ] || \
       [ -e "${MATTERSIM_WAIVER_MODEL_RECORD}" ]; then
      echo "ERROR: immutable MatterSim runtime or one of its records already exists." >&2
      exit 4
    fi
    if [ ! -s "${MATTERSIM_RESOLVED_MISSING}" ]; then
      echo "ERROR: isolated MatterSim resolver produced no pinned dependencies." >&2
      exit 4
    fi
    mkdir -p "${MATTERSIM_RUNTIME_ROOT}" "${MATTERSIM_RUNTIME_STAGING}"
    PYTHONDONTWRITEBYTECODE=1 python -m pip install \
      --no-index --find-links "${WHEELHOUSE}" --no-deps --no-compile \
      --target "${MATTERSIM_RUNTIME_STAGING}" \
      --requirement "${MATTERSIM_RESOLVED_MISSING}"
    PYTHONDONTWRITEBYTECODE=1 python -m pip install \
      --no-index --find-links "${WHEELHOUSE}" --no-deps --no-compile \
      --target "${MATTERSIM_RUNTIME_STAGING}" "mattersim==1.1.2"
    python - "${PROJECT_ROOT}" "${MATTERSIM_RUNTIME_STAGING}" \
      "${MATTERSIM_TREE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from crystal_dlm.wqcodiff.dependency_waiver import build_runtime_tree_manifest

payload = build_runtime_tree_manifest(Path(sys.argv[2]))
with Path(sys.argv[3]).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps({"runtime_tree_sha256": payload["tree_sha256"], "files": payload["file_count"]}, indent=2))
PY
    RUNTIME_TREE_SHA256="$(python - "${MATTERSIM_TREE_MANIFEST}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["tree_sha256"])
PY
)"
    PYTHONPATH="${MATTERSIM_RUNTIME_STAGING}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      PYTHONDONTWRITEBYTECODE=1 python - "${PROJECT_ROOT}" \
      "${MATTERSIM_WAIVER_RUN_RECORD}" "${SOURCE_BUNDLE_SHA256}" \
      "${RUNTIME_TREE_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from crystal_dlm.wqcodiff.dependency_waiver import finalize_mattersim_inference_waiver

payload = finalize_mattersim_inference_waiver(
    output_paths=(Path(sys.argv[2]),),
    source_bundle_sha256=sys.argv[3],
    runtime_tree_sha256=sys.argv[4],
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
    mv "${MATTERSIM_RUNTIME_STAGING}" "${MATTERSIM_RUNTIME}"
    python - "${PROJECT_ROOT}" "${MODEL_ROOT}" "${MATTERSIM_RUNTIME_LOCK}" \
      "${MATTERSIM_TREE_MANIFEST}" "${MATTERSIM_TREE_MODEL_MANIFEST}" \
      "${MATTERSIM_WAIVER_RUN_RECORD}" "${MATTERSIM_WAIVER_MODEL_RECORD}" \
      "${MATTERSIM_REPORT}" "${MATTERSIM_RESOLVER_REQUIREMENTS}" \
      "${WHEELHOUSE_LOCK}" "${SOURCE_BUNDLE_SHA256}" <<'PY'
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

project, model_root, lock_path, run_tree, model_tree, run_waiver, model_waiver, report, requirements, wheelhouse_lock = map(Path, sys.argv[1:11])
source_bundle_sha256 = sys.argv[11]
sys.path.insert(0, str(project))
from crystal_dlm.wqcodiff.dependency_waiver import load_mattersim_runtime_lock

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for source, destination in ((run_tree, model_tree), (run_waiver, model_waiver)):
    with destination.open("xb") as target, source.open("rb") as handle:
        shutil.copyfileobj(handle, target)
tree_payload = json.loads(model_tree.read_text(encoding="utf-8"))
waiver_payload = json.loads(model_waiver.read_text(encoding="utf-8"))
import torch
payload = {
    "schema": "wqcodiff_mattersim_runtime_lock_v4",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "runtime": "runtimes/mattersim-1.1.2-py310-v4",
    "runtime_tree_sha256": tree_payload["tree_sha256"],
    "tree_manifest": model_tree.name,
    "tree_manifest_sha256": sha(model_tree),
    "dependency_waiver": model_waiver.name,
    "dependency_waiver_sha256": sha(model_waiver),
    "wheelhouse_lock": wheelhouse_lock.name,
    "wheelhouse_lock_sha256": sha(wheelhouse_lock),
    "resolver_report_sha256": sha(report),
    "resolver_requirements_sha256": sha(requirements),
    "source_bundle_sha256": source_bundle_sha256,
    "retained_torch": torch.__version__,
    "python": sys.version,
    "compatibility_pins": {
        "ase": {
            "version": "3.27.0",
            "wheel_sha256": "058c48ea504fe7fbbe7c932f778415243ef2df45b1ab869866f24efcc17f0538",
            "reason": "last_tested_release_exporting_stress_helper_from_ase_constraints",
        },
        "setuptools": {
            "version": "81.0.0",
            "wheel_sha256": "fdd925d5c5d9f62e4b74b30d6dd7828ce236fd6ed998a08d81de62ce5a6310d6",
            "reason": "last_release_family_containing_pkg_resources_for_mattersim_1p1p2",
        },
    },
    "validated_imports": waiver_payload["validated_imports"],
}
with lock_path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
load_mattersim_runtime_lock(
    lock_path,
    model_root=model_root,
    installed_torch=torch.__version__,
)
print(json.dumps({"mattersim_runtime_lock": str(lock_path), "runtime_tree_sha256": tree_payload["tree_sha256"]}, indent=2))
PY
  fi
  echo "WQ environment lock complete: $(date '+%F %T %Z')"
} 2>&1 | tee -a "${LOG_DIR}/env_lock.log"
