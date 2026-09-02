#!/usr/bin/env bash
#SBATCH --job-name=spad-bs-valid
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=spad-bs-valid-%j.out
#SBATCH --error=spad-bs-valid-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly BODY_RUN="${SPAD_BS_BODY_RUN:?}"
readonly FULL_DIRECT_RUN="${SPAD_FULL_DIRECT_RUN:-${ROOT}/runs/spad_all_raw_direct_39518}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly SNAPSHOT="${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_bs_fast_validity_${SLURM_JOB_ID}"
mkdir "${RUN}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR

# First certify predicted-only validity against one completed full Direct cell.
env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  PYTHONPATH="${SOURCE}/src" "${PYTHON}" "${SOURCE}/scripts/run_crysllmgen_validity_fast.py" \
  --generation-jsonl "${FULL_DIRECT_RUN}/BC/stream17/generation/generation.jsonl" \
  --snapshot-root "${SNAPSHOT}" --output-dir "${RUN}/equivalence"
"${PYTHON}" - "${RUN}/equivalence/attempt_metrics.jsonl" \
  "${FULL_DIRECT_RUN}/BC/stream17/direct/attempt_metrics.jsonl" <<'PY'
import json,sys
left=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
right=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
keys=('attempt_id','comp_valid','struct_valid','valid')
assert [{k:r[k] for k in keys} for r in left] == [{k:r[k] for k in keys} for r in right]
PY

run_stream() {
  local stream="$1" source="${BODY_RUN}/BS/stream${stream}/generation/generation.jsonl"
  [[ -f "${BODY_RUN}/BS/stream${stream}/generation/_SUCCESS" ]]
  env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    PYTHONPATH="${SOURCE}/src" "${PYTHON}" "${SOURCE}/scripts/run_crysllmgen_validity_fast.py" \
    --generation-jsonl "${source}" --snapshot-root "${SNAPSHOT}" \
    --output-dir "${RUN}/stream${stream}"
}
run_stream 17 & p0=$!
run_stream 18 & p1=$!
wait "${p0}"; wait "${p1}"
"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); cells=[]
for s in (17,18):
 x=json.load(open(r/f'stream{s}'/'report.json'))
 cells.append({'stream':s,'attempts':x['attempts'],'comp_valid':x['comp_valid_count'],
               'struct_valid':x['struct_valid_count'],'direct_joint':x['valid_count']})
(r/'FAST_VALIDITY_FINAL.json').write_text(json.dumps({'equivalent_to_full_direct':True,'cells':cells},indent=2,sort_keys=True)+'\n')
print(json.dumps(cells,sort_keys=True))
PY
trap - ERR
touch "${RUN}/_SUCCESS"
