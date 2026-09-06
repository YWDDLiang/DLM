#!/usr/bin/env bash
#SBATCH --job-name=periodic-path-collect
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=periodic-path-collect-%j.out
#SBATCH --error=periodic-path-collect-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
ROUND="${PERIODIC_POSTTRAIN_ROUND:?}"
POLICY="${PERIODIC_POSTTRAIN_POLICY:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPERIMENT="$ROOT/experiments/periodic_self_repair_20260906/from_original_llada_v1"
CONDITIONS="$ROOT/runs/spad_state_train_conditions_39857/conditions/plans_for_dlm.jsonl"
test -f "$ROOT/runs/spad_state_path_train_39938/_SUCCESS"
test -f "$ROOT/runs/spad_state_train_conditions_39857/_SUCCESS"
test -f "$POLICY/raw_periodic_initialization.json"
if [[ "$ROUND" == 0 ]]; then
  test -f "${PERIODIC_RAW_ADMISSION_DIR:?}/_SUCCESS"
  candidates=4
  seed=20260907
elif [[ "$ROUND" == 1 ]]; then
  test -f "${PERIODIC_RAW_K4_TRAIN_RUN:?}/_SUCCESS"
  candidates=8
  seed=20260908
else
  exit 2
fi
RUN="$EXPERIMENT/path_posttraining/runs/collect_r${ROUND}_${SLURM_JOB_ID}"
mkdir -p "$EXPERIMENT/path_posttraining/runs"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
common=(--model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
  --checkpoint-path "$POLICY" --conditions "$CONDITIONS" --condition-stop 1024
  --candidates "$candidates" --batch-size 4 --world-size 2 --purpose train
  --collection-round "$ROUND" --seed "$seed" --full-cell-repair --output-dir "$RUN/sample")
pids=()
for rank in 0 1; do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" \
    "${common[@]}" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" "${common[@]}" --merge-only \
  >"$RUN/merge.out" 2>"$RUN/merge.err"
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/sample/paths.jsonl" \
  --output-dir "$RUN/labels" --gpu-count 2 --workers-per-gpu 2 \
  >"$RUN/label.out" 2>"$RUN/label.err"
extra=()
if "$PY" - "$RUN/labels/labels.jsonl" <<'PY'
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
raise SystemExit(0 if any(str(row["group_id"])=="13087" and row.get("verified") is True
                          for row in rows) else 1)
PY
then
  extra+=(--uniform-reference-group 13087
    --uniform-reference-reason "Unresolved train-label energy-scale uncertainty; retain protocol verification and all occurrences; q=u is not physical disproof.")
fi
"$PY" "$CODE/src/scripts/build_programmed_path_teacher.py" \
  --paths-jsonl "$RUN/sample/paths.jsonl" --labels-jsonl "$RUN/labels/labels.jsonl" \
  --expected-conditions 1024 --candidates "$candidates" "${extra[@]}" \
  --output-dir "$RUN/teacher" >"$RUN/teacher.out" 2>"$RUN/teacher.err"
"$PY" "$CODE/scripts/diagnose_programmed_basins.py" --teacher-json "$RUN/teacher/teacher.json" \
  --output-json "$RUN/teacher/BASIN_DIAGNOSTIC.json" >"$RUN/basins.out" 2>"$RUN/basins.err"
test -f "$RUN/teacher/_SUCCESS"
touch "$RUN/_SUCCESS"
