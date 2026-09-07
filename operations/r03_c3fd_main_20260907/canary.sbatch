#!/usr/bin/env bash
#SBATCH --job-name=r03-interface-canary
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=200G
#SBATCH --time=01:20:00
set -Eeuo pipefail
SOURCE="${R03_SOURCE_ROOT:?}"
RUNROOT="${R03_RUN_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
ASSETS="$RUNROOT/pointer_40395/assets"
RUN="$RUNROOT/canary_$SLURM_JOB_ID"
mkdir "$RUN"
export PYTHONPATH="$SOURCE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
IFS=, read -r -a VISIBLE <<< "${CUDA_VISIBLE_DEVICES:?}"
test "${#VISIBLE[@]}" -eq 2
run_arm() (
 set -Eeuo pipefail
 role="$1"
 export CUDA_VISIBLE_DEVICES="$2"
 exec >"$RUN/$role.out" 2>"$RUN/$role.err"
 trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/$role.FAILURE.txt"; exit "$rc"' ERR
 extra=(--batch-size 4)
 if [[ "$role" == I ]]; then extra+=(--c3fd-domain "$ASSETS/C3FD_DOMAIN.json"); fi
 "$PY" "$SOURCE/src/scripts/sample_r03_h1a2_plans.py" \
  --model-path /public/home/jiaosz/ywliang/models/Meta-Llama-3-8B \
  --checkpoint-path "$PROJECT/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final" \
  --native-prompt-file "$ASSETS/P0_NATIVE_PROMPT.txt" \
  --output-dir "$RUN/$role/planner" --num-samples 16 --seed 202609070 \
  --sample-index-offset 0 "${extra[@]}"
 "$PY" "$SOURCE/src/scripts/run_r03_integrated_body.py" \
  --frozen-runtime-root "$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1" \
  --base-model /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct \
  --b0-checkpoint "$PROJECT/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final" \
  --plans-jsonl "$RUN/$role/planner/plans_for_dlm.jsonl" \
  --output-dir "$RUN/$role/body" --expected-requests 16 --seed 202609170 --batch-size 8 \
  --crysllmgen-dir "$PROJECT/reference/crysllmgen"
 "$PY" "$SOURCE/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "$RUN/$role/body/proposal_graphs.pt" \
  --checkpoint /public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt \
  --crysllmgen-dir /public/home/jiaosz/hengzhang/Code/crysllmgen-main \
  --output-dir "$RUN/$role/tau800" --batch-size 1 --num-evals 1 \
  --max-proposals 16 --diff-steps 800 --seed-by-sample-index --seed 202609270
 touch "$RUN/$role/_SUCCESS"
)
run_arm R "${VISIBLE[0]}" & p0=$!
run_arm I "${VISIBLE[1]}" & p1=$!
set +e
wait "$p0"; r0=$?
wait "$p1"; r1=$?
set -e
printf '{"R_exit_code":%s,"I_exit_code":%s,"role":"engineering_canary_separate_from_development_and_main","requests_per_arm":16}\n' "$r0" "$r1" >"$RUN/STAGE_FINAL.json"
test "$r0" -eq 0
test "$r1" -eq 0
touch "$RUN/_SUCCESS"
