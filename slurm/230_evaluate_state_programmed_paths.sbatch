#!/usr/bin/env bash
#SBATCH --job-name=spad-state-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=spad-state-eval-%j.out
#SBATCH --error=spad-state-eval-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
ROLE="${SPAD_EVAL_ROLE:?reference or method}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
CONFIG="$PROJECT/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json"
CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
RUN="$ROOT/runs/spad_state_eval_${ROLE}_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
extra=(--purpose evaluation)
if [[ "$ROLE" == reference ]]; then
  POLICY="$ROOT/runs/spad_basin_closure_ce_39700/train/checkpoints/step-1696"
  extra+=(--reference-closure)
elif [[ "$ROLE" == method ]]; then
  POLICY="${SPAD_EVAL_POLICY_PATH:?}"
  "$PY" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); r=json.loads((p.parent.parent/"TRAIN_FINAL.json").read_text()); assert r["eligible_policy"] is True; assert pathlib.Path(r["policy_path"]).resolve()==p.resolve()' "$POLICY"
else
  exit 2
fi
CONDITIONS="$ROOT/cohorts/spad_prospective_seed23_256_v1_20260903/plans_for_dlm.jsonl"
common=(--model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct --checkpoint-path "$POLICY"
  --conditions "$CONDITIONS" --condition-stop 256 --candidates 1 --batch-size 4 --world-size 2
  --seed 20260905 --output-dir "$RUN/sample")
pids=()
for rank in 0 1; do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" \
    "${common[@]}" "${extra[@]}" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" "${common[@]}" "${extra[@]}" --merge-only >"$RUN/merge.out" 2>"$RUN/merge.err"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/sample/paths.jsonl" \
  --output-dir "$RUN/native" --crysllmgen-dir "$CRYS" >"$RUN/artifacts.out" 2>"$RUN/artifacts.err"
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/native/paths.jsonl" \
  --output-dir "$RUN/labels" --purpose evaluation --gpu-count 2 --workers-per-gpu 2 >"$RUN/label.out" 2>"$RUN/label.err"
"$PY" "$CODE/scripts/evaluate_programmed_paths.py" --paths-jsonl "$RUN/native/paths.jsonl" \
  --labels-jsonl "$RUN/labels/labels.jsonl" --frozen-config "$CONFIG" \
  --official-cache "$ROOT/runs/spad_basin_closure_official_20260904_v1/official_mp_cache" \
  --output-dir "$RUN/evaluation" --expected-requests 256 --endpoint native --cohort-role fixed_development \
  >"$RUN/evaluate.out" 2>"$RUN/evaluate.err"
test -f "$RUN/evaluation/_SUCCESS"
touch "$RUN/_SUCCESS"
