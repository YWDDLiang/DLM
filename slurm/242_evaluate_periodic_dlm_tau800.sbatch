#!/usr/bin/env bash
#SBATCH --job-name=periodic-dlm-tau800
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=periodic-dlm-tau800-%j.out
#SBATCH --error=periodic-dlm-tau800-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
NATIVE="${PERIODIC_RAW_NATIVE_RUN:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
CONFIG="$PROJECT/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json"
CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
EXPERIMENT="$ROOT/experiments/periodic_self_repair_20260906/from_original_llada_v1"
test -f "$NATIVE/_SUCCESS"
test -f "$NATIVE/native/proposal_graphs.pt"
RUN="$EXPERIMENT/runs/eval_tau800_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$CODE/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "$NATIVE/native/proposal_graphs.pt" \
  --checkpoint "$CRYS/out/mp_20/22042026/203930/model_494.pt" \
  --crysllmgen-dir "$CRYS" --output-dir "$RUN/refine" --batch-size 1 \
  --seed-from-graph-field refiner_seed --diff-steps 800 --num-evals 1 --max-proposals 256 \
  >"$RUN/refine.out" 2>"$RUN/refine.err"
PAYLOAD="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "$RUN/refine/refinement_metrics.json")"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$NATIVE/native/paths.jsonl" \
  --refined-pt "$PAYLOAD" --output-dir "$RUN/tau800" >"$RUN/artifacts.out" 2>"$RUN/artifacts.err"
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/tau800/paths.jsonl" \
  --output-dir "$RUN/labels" --purpose evaluation --gpu-count 2 --workers-per-gpu 2 \
  >"$RUN/label.out" 2>"$RUN/label.err"
"$PY" "$CODE/scripts/evaluate_programmed_paths.py" --paths-jsonl "$RUN/tau800/paths.jsonl" \
  --labels-jsonl "$RUN/labels/labels.jsonl" --frozen-config "$CONFIG" \
  --official-cache "$ROOT/runs/spad_basin_closure_official_20260904_v1/official_mp_cache" \
  --output-dir "$RUN/evaluation" --expected-requests 256 --endpoint tau800 \
  --cohort-role fixed_development --policy-stage final >"$RUN/evaluate.out" 2>"$RUN/evaluate.err"
"$PY" "$CODE/scripts/compare_programmed_path_evaluations.py" \
  --reference-eval-dir "$ROOT/runs/spad_state_reference_verification_39893/tau800-evaluation" \
  --method-eval-dir "$RUN/evaluation" \
  --reference-paths-jsonl "$ROOT/runs/spad_state_tau800_39890/tau800/paths.jsonl" \
  --method-paths-jsonl "$RUN/tau800/paths.jsonl" --output-dir "$RUN/comparison" \
  --comparison-purpose final >"$RUN/compare.out" 2>"$RUN/compare.err"
test -f "$RUN/evaluation/_SUCCESS"
test -f "$RUN/comparison/_SUCCESS"
"$PY" "$CODE/scripts/check_periodic_raw_posttrain_admission.py" \
  --new-native-eval-dir "$NATIVE/evaluation" \
  --new-tau-eval-dir "$RUN/evaluation" \
  --current-k4-native-eval-dir "$ROOT/runs/spad_state_eval_method_39910/evaluation" \
  --current-k4-tau-eval-dir "$ROOT/runs/spad_state_tau800_39910/evaluation" \
  --output-dir "$RUN/posttrain-admission" >"$RUN/admission.out" 2>"$RUN/admission.err"
touch "$RUN/_SUCCESS"
