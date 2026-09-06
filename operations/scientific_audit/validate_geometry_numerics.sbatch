#!/usr/bin/env bash
#SBATCH --job-name=dlm-geometry-numerics
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=dlm-geometry-numerics-%j.out
#SBATCH --error=dlm-geometry-numerics-%j.err
set -Eeuo pipefail
RUN="${AUDIT_ROOT:?}/runs/geometry_numerics_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "${AUDIT_MODEL_SOURCE:?}" && git archive "${AUDIT_CODE_COMMIT:?}") | tar -x -C "$RUN/code"
printf '%s\n' "$AUDIT_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
export PYTHONPATH="$RUN/code/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=1
export GEOMETRY_NUMERICS_RUN="$RUN"
"${AUDIT_PYTHON:?}" -m unittest discover -s "$RUN/code/tests" -p test_mixed_geometry_diffusion.py -v >"$RUN/tests.out" 2>"$RUN/tests.err"
"${AUDIT_PYTHON:?}" - <<'PY'
from pathlib import Path
import hashlib, json, os
import torch
from crystal_dlm.mixed_geometry_diffusion import LatticeNormalizer

run = Path(os.environ["GEOMETRY_NUMERICS_RUN"])
identity = Path(os.environ["AUDIT_IDENTITY_RUN"])
torch.set_num_threads(2)
data, receipts = {}, {}
for split in ("train", "val"):
    summary = json.loads((identity / split / "SUMMARY.json").read_text())
    if summary["original_cif_identity_gate_passed"] is not True:
        raise ValueError("original continuous source identity has not passed")
    path = identity / split / "source_identity.jsonl"
    lattice, counts, source_ids = [], [], []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["identity_verified"] is not True:
                raise ValueError("unverified source in continuous input")
            geometry = row["continuous_aligned"]
            lattice.append(geometry["lattice_matrix_A"])
            counts.append(len(geometry["species"]))
            source_ids.append(row["source_row_idx"])
    data[split] = (torch.tensor(lattice, dtype=torch.float64), torch.tensor(counts))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    receipts[split] = {"path": str(path), "sha256": digest.hexdigest(),
                       "source_records": len(source_ids), "unique_source_ids": len(set(source_ids)),
                       "identity_receipts": summary["input_receipts"]}
normalizer = LatticeNormalizer.fit(*data["train"], split="train")
normalizer.save(run / "normalizer.json")
statistics = {}
for split, (lattice, counts) in data.items():
    z = normalizer.encode(lattice, counts)
    decoded = normalizer.decode(z, counts)
    gram, recovered = lattice @ lattice.transpose(-1, -2), decoded @ decoded.transpose(-1, -2)
    relative = (gram - recovered).abs().flatten(1).amax(1) / gram.abs().flatten(1).amax(1)
    statistics[split] = {
        "sources": len(counts), "atoms": int(counts.sum()), "min_atoms": int(counts.min()),
        "max_atoms": int(counts.max()), "chart_mean": z.mean(0).tolist(),
        "chart_population_std": z.std(0, correction=0).tolist(),
        "max_squared_chart_norm": float(z.square().sum(-1).max()),
        "maximum_relative_gram_roundtrip_error": float(relative.max()),
        "finite_chart": bool(torch.isfinite(z).all())}
report = {"schema": "mixed_geometry_numerical_validation_v1", "torch": torch.__version__,
          "model_forwards": 0, "GPU_calls": 0, "normalizer": normalizer.to_dict(),
          "input_receipts": receipts, "statistics": statistics}
(run / "SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"normalizer": normalizer.to_dict(), "statistics": statistics}), flush=True)
PY
touch "$RUN/_SUCCESS"
