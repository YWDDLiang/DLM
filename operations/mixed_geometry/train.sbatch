#!/usr/bin/env bash
#SBATCH --job-name=dlm-mixed-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=440G
#SBATCH --time=06:00:00
#SBATCH --output=dlm-mixed-train-%j.out
#SBATCH --error=dlm-mixed-train-%j.err
set -Eeuo pipefail
RUN="${MIXED_ROOT:?}/runs/train_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test "$(cd "${MIXED_OBJECT_SOURCE:?}" && git rev-parse "${MIXED_CODE_COMMIT:?}")" = "$MIXED_CODE_COMMIT"
(cd "$MIXED_OBJECT_SOURCE" && git archive "$MIXED_CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$MIXED_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
cp "${MIXED_TRAIN_MANIFEST:?}" "$RUN/LAUNCH_MANIFEST.json"
export PYTHONPATH="$RUN/code/src" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MIXED_TRAIN_RUN="$RUN"
"${MIXED_PYTHON:?}" - <<'PY'
import hashlib, json, os, pathlib, subprocess, sys
run = pathlib.Path(os.environ["MIXED_TRAIN_RUN"])
manifest = json.loads((run / "LAUNCH_MANIFEST.json").read_text())
if manifest["code_commit"] != (run / "CODE_COMMIT").read_text().strip():
    raise ValueError("launch and archived code commit differ")
for gate in manifest["completed_gates"]:
    path = pathlib.Path(gate["summary_path"])
    if not pathlib.Path(gate["success_marker"]).is_file():
        raise ValueError(f"required completed gate lacks its marker: {gate['name']}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != gate["summary_sha256"]:
        raise ValueError(f"required gate summary changed: {gate['name']}")
command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={manifest['resources']['gpus']}",
           str(run / "code/src/scripts/train_mixed_geometry_dlm.py"),
           "--model-path", manifest["model_path"], "--v2-checkpoint", manifest["v2_checkpoint"],
           "--output-dir", str(run / "train"), "--launch-manifest", str(run / "LAUNCH_MANIFEST.json")]
for name in ("train_data", "val_data", "train_identity", "val_identity", "normalizer"):
    command += ["--" + name.replace("_", "-"), manifest["inputs"][name]]
for name in ("seed", "data_seed", "validation_seed", "validation_sources", "expected_train_sources",
             "expected_val_sources", "epochs", "microbatch", "effective_batch", "max_length"):
    command += ["--" + name.replace("_", "-"), str(manifest["training"][name])]
(run / "TRAIN_COMMAND.json").write_text(json.dumps(command, indent=2) + "\n")
with (run / "train.out").open("w") as out, (run / "train.err").open("w") as err:
    subprocess.run(command, check=True, stdout=out, stderr=err)
PY
test -f "$RUN/train/_SUCCESS"
cp "$RUN/train/POLICY_PATH" "$RUN/POLICY_PATH"
touch "$RUN/_SUCCESS"
