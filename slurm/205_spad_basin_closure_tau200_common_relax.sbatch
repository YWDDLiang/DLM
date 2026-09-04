#!/usr/bin/env bash
#SBATCH --job-name=spad-t200-rshard
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=spad-t200-rshard-%A_%a.out
#SBATCH --error=spad-t200-rshard-%A_%a.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly TAU_RUN="${ROOT}/runs/spad_basin_closure_tau200_s17_39732"
readonly GENERATION="${TAU_RUN}/generation/generation.jsonl"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly RESUMABLE="${PROJECT}/reference/a100_eval_sun/eval_sun_resumable.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RANK="${SLURM_ARRAY_TASK_ID:?}"
readonly RUN="${ROOT}/runs/spad_basin_closure_tau200_common_relax_${SLURM_ARRAY_JOB_ID}"
readonly INPUT="${RUN}/frozen_input_rank${RANK}"
readonly SHARDS="${RUN}/relax_cache_shards"

mkdir -p "${RUN}/logs" "${INPUT}" "${SHARDS}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE_rank${RANK}.tsv"; touch "${RUN}/_FAILED_rank${RANK}"; exit "$rc"' ERR
[[ -f "${TAU_RUN}/_SUCCESS" && -f "${GENERATION}" ]]
[[ "$(wc -l <"${GENERATION}")" -eq 256 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 1 ]]

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${R03E}/runtime" \
  "${PYTHON}" - "${GENERATION}" "${INPUT}" <<'PY'
import pathlib,sys
from crystal_dlm.wqcodiff.crysllmgen import a100_sun
generation=pathlib.Path(sys.argv[1]).resolve(strict=True); output=pathlib.Path(sys.argv[2])
manifest=a100_sun.prepare_a100_input(generation_jsonl=generation,
 output_pt=output/'all_attempts.pt', output_manifest=output/'input_manifest.json',
 expected_attempts=256)
assert manifest['total_attempts']==256 and manifest['reconstructed_structures']==256
PY

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/scripts/relax_spad_basin_closure_shard.py" \
  --generation-jsonl "${GENERATION}" --generated-pt "${INPUT}/all_attempts.pt" \
  --input-manifest "${INPUT}/input_manifest.json" --base-cache "${CACHE}" \
  --frozen-resumable-py "${RESUMABLE}" --output-jsonl "${SHARDS}/cache_rank${RANK}.jsonl" \
  --report-json "${SHARDS}/report_rank${RANK}.json" --denominator 256 \
  --shard-rank "${RANK}" --shard-count 2 --device cuda \
  >"${RUN}/logs/shard${RANK}.out" 2>"${RUN}/logs/shard${RANK}.err"

trap - ERR
touch "${RUN}/_SUCCESS_rank${RANK}"
