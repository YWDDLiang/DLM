#!/usr/bin/env bash
#SBATCH --job-name=spad-state-path-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=06:00:00
#SBATCH --output=spad-state-path-train-%j.out
#SBATCH --error=spad-state-path-train-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
TEACHER="${SPAD_PATH_TEACHER_JSON:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
WORLD="${SPAD_TRAIN_WORLD_SIZE:-4}"
RUN="$ROOT/runs/spad_state_path_train_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
export PYTHONPATH="$RUN/code/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
POLICY=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["provenance"]["checkpoint"])' "$TEACHER")
extra=()
if test -n "${SPAD_POST_OPTIMIZER_STATE:-}"; then extra+=(--optimizer-state "$SPAD_POST_OPTIMIZER_STATE"); fi
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$WORLD" \
  "$RUN/code/src/scripts/train_programmed_path_policy.py" \
  --model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct --checkpoint-path "$POLICY" \
  --teacher-json "$TEACHER" --ce-data-jsonl "$ROOT/data/spad_basin_closure_sft_v1_20260904/train.jsonl" \
  --output-dir "$RUN/train" --engineering-steps "${SPAD_ENGINEERING_STEPS:-0}" "${extra[@]}" \
  >"$RUN/train.out" 2>"$RUN/train.err"
test -f "$RUN/train/_SUCCESS"
touch "$RUN/_SUCCESS"
