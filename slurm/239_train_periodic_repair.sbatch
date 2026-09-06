#!/usr/bin/env bash
#SBATCH --job-name=periodic-repair-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=periodic-repair-train-%j.out
#SBATCH --error=periodic-repair-train-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
EXPERIMENT="${PERIODIC_REPAIR_ROOT:?}"
ROUND="${PERIODIC_REPAIR_TRAIN_ROUND:?}"
K4_RUN="${PERIODIC_REPAIR_K4_RUN:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
test -f "$K4_RUN/_SUCCESS"
test -f "$K4_RUN/preflight/_SUCCESS"
RUN="$EXPERIMENT/runs/train_${ROUND}_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
prepared=("$K4_RUN/targets")
labels=("$K4_RUN/labels/labels.jsonl")
if [[ "$ROUND" == 0 ]]; then
  expected=4096
  INITIAL="$ROOT/runs/spad_state_path_train_39892/train/checkpoints/step-1020"
  REFERENCE="$INITIAL"
elif [[ "$ROUND" == 1 ]]; then
  K8_RUN="${PERIODIC_REPAIR_K8_RUN:?}"
  test -f "$K8_RUN/_SUCCESS"
  prepared+=("$K8_RUN/targets")
  labels+=("$K8_RUN/labels/labels.jsonl")
  expected=12288
  INITIAL="${PERIODIC_REPAIR_PREVIOUS_POLICY:?}"
  REFERENCE="${PERIODIC_REPAIR_FINAL_K8_POLICY:?}"
else
  exit 2
fi
"$PY" "$CODE/scripts/build_terminal_repair_dataset.py" --mode finalize \
  --prepared "${prepared[@]}" --labels "${labels[@]}" --expected-requests "$expected" \
  --output "$RUN/data" >"$RUN/data.out" 2>"$RUN/data.err"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$CODE/src/scripts/train_periodic_terminal_repair.py" \
  --model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct \
  --checkpoint-path "$INITIAL" --reference-checkpoint-path "$REFERENCE" --repair-round "$ROUND" \
  --data-jsonl "$RUN/data/train.jsonl" --ce-data-jsonl "$ROOT/data/spad_basin_closure_sft_v1_20260904/train.jsonl" \
  --output-dir "$RUN/train" >"$RUN/train.out" 2>"$RUN/train.err"
test -f "$RUN/train/_SUCCESS"
cp "$RUN/train/POLICY_PATH" "$RUN/POLICY_PATH"
touch "$RUN/_SUCCESS"
