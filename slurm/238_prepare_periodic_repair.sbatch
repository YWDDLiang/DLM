#!/usr/bin/env bash
#SBATCH --job-name=periodic-repair-prepare
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=periodic-repair-prepare-%j.out
#SBATCH --error=periodic-repair-prepare-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
EXPERIMENT="${PERIODIC_REPAIR_ROOT:?}"
ROUND="${PERIODIC_REPAIR_ROUND:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
BASE="$ROOT/runs/spad_state_path_train_39892/train/checkpoints/step-1020"
RUN="$EXPERIMENT/runs/prepare_${ROUND}_${SLURM_JOB_ID}"
mkdir -p "$EXPERIMENT/runs"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
if [[ "$ROUND" == 0 ]]; then
  paths=("$ROOT/runs/spad_state_paths_39872/sample/paths.jsonl" "$ROOT/runs/spad_state_paths_39878/sample/paths.jsonl")
  labels=("$ROOT/data/spad_state_labels_round0_consistent_20260906_v2/labels.jsonl")
  expected=4096
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
    "$CODE/src/scripts/preflight_periodic_repair.py" \
    --model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct --checkpoint-path "$BASE" \
    --paths-jsonl "${paths[@]}" --output-dir "$RUN/preflight" \
    >"$RUN/preflight.out" 2>"$RUN/preflight.err"
  test -f "$RUN/preflight/_SUCCESS"
elif [[ "$ROUND" == 1 ]]; then
  test -f "${PERIODIC_REPAIR_PREFLIGHT_RUN:?}/preflight/_SUCCESS"
  paths=("$ROOT/runs/spad_state_paths_39896/sample/paths.jsonl")
  labels=("$ROOT/runs/spad_state_labels_39911/labels/labels.jsonl" "$ROOT/runs/spad_state_labels_39916/labels/labels.jsonl")
  expected=8192
else
  exit 2
fi
"$PY" "$CODE/scripts/build_terminal_repair_dataset.py" --mode prepare \
  --parent-paths "${paths[@]}" --labels "${labels[@]}" --tokenizer "$BASE" \
  --expected-requests "$expected" --output "$RUN/targets" >"$RUN/prepare.out" 2>"$RUN/prepare.err"
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/targets/target_paths.jsonl" \
  --output-dir "$RUN/labels" --gpu-count 2 --workers-per-gpu 2 \
  >"$RUN/label.out" 2>"$RUN/label.err"
test -f "$RUN/labels/_SUCCESS"
touch "$RUN/_SUCCESS"
