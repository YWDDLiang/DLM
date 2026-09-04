#!/usr/bin/env bash
#SBATCH --job-name=spad-basin-rshard
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --array=0-1
#SBATCH --output=spad-basin-rshard-%A_%a.out
#SBATCH --error=spad-basin-rshard-%A_%a.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly GENERATION_RUN="${SPAD_BASIN_CLOSURE_GENERATION_RUN:?export completed native generation run}"
readonly RAW_SCREEN_RUN="${SPAD_BASIN_CLOSURE_RAW_SCREEN_RUN:?export completed raw screen run}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly RESUMABLE="${PROJECT}/reference/a100_eval_sun/eval_sun_resumable.py"
readonly A100_ADAPTER="${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/a100_sun.py"
readonly SHARDER="${SOURCE}/scripts/relax_spad_basin_closure_shard.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly GENERATION="${GENERATION_RUN}/closure_ce/stream17/raw_generation/generation.jsonl"
readonly RANK="${SLURM_ARRAY_TASK_ID:?array rank is required}"
readonly ARRAY_RUN="${ROOT}/runs/spad_basin_closure_common_relax_${SLURM_ARRAY_JOB_ID}"
readonly SHARDS="${ARRAY_RUN}/relax_cache_shards"
readonly FROZEN_INPUT="${ARRAY_RUN}/frozen_input_rank${RANK}"

mkdir -p "${ARRAY_RUN}" "${ARRAY_RUN}/logs" "${SHARDS}" "${FROZEN_INPUT}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${ARRAY_RUN}/ENGINEERING_FAILURE_rank${RANK}.tsv"
  touch "${ARRAY_RUN}/_FAILED_rank${RANK}"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ "${RANK}" == 0 || "${RANK}" == 1 ]]
for path in "${GENERATION_RUN}/_SUCCESS" "${RAW_SCREEN_RUN}/_SUCCESS" \
  "${GENERATION}" "${CACHE}" "${RESUMABLE}" "${A100_ADAPTER}" "${SHARDER}"; do
  [[ -f "${path}" ]]
done
[[ "$(wc -l <"${GENERATION}")" -eq 256 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${R03E}/runtime" \
  "${PYTHON}" - "${GENERATION}" "${FROZEN_INPUT}" <<'PY'
import pathlib
import sys

from crystal_dlm.wqcodiff.crysllmgen import a100_sun

generation = pathlib.Path(sys.argv[1]).resolve(strict=True)
output = pathlib.Path(sys.argv[2])
manifest = a100_sun.prepare_a100_input(
    generation_jsonl=generation,
    output_pt=output / "all_attempts.pt",
    output_manifest=output / "input_manifest.json",
    expected_attempts=256,
)
assert manifest["total_attempts"] == 256
assert manifest["reconstructed_structures"] == 256
assert len(manifest["attempt_records"]) == 256
PY

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SHARDER}" --generation-jsonl "${GENERATION}" \
  --generated-pt "${FROZEN_INPUT}/all_attempts.pt" \
  --input-manifest "${FROZEN_INPUT}/input_manifest.json" \
  --base-cache "${CACHE}" --frozen-resumable-py "${RESUMABLE}" \
  --output-jsonl "${SHARDS}/cache_rank${RANK}.jsonl" \
  --report-json "${SHARDS}/report_rank${RANK}.json" \
  --denominator 256 --shard-rank "${RANK}" --shard-count 2 --device cuda \
  >"${ARRAY_RUN}/logs/shard${RANK}.out" \
  2>"${ARRAY_RUN}/logs/shard${RANK}.err"

"${PYTHON}" - "${SHARDS}/report_rank${RANK}.json" "${RANK}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
rank = int(sys.argv[2])
assert report["schema"] == "spad_basin_closure_relax_cache_shard_v1"
assert report["shard_rank"] == rank
assert report["shard_count"] == 2
assert report["fixed_denominator"] == 256
assert report["selection"] is False
PY

trap - ERR
touch "${ARRAY_RUN}/_SUCCESS_rank${RANK}"
printf '%s\n' "${ARRAY_RUN}"
