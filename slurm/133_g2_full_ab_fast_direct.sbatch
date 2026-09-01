#!/usr/bin/env bash
#SBATCH --job-name=g2-ab-fast-direct
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=g2-ab-fast-direct-%j.out
#SBATCH --error=g2-ab-fast-direct-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export root}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export code root}"
readonly SCRIPT="${CODE_ROOT}/scripts/run_direct_validity_fast.py"
readonly SCRIPT_SHA="${G2_FAST_DIRECT_SHA256:?pin script SHA}"
readonly SOURCE="${ROOT}/runs/g2_full_ab_raw_screen_39168"
readonly BASE_GEN="${ROOT}/runs/compact_v2_periodic_relation_g2_final_eval_39139/BASE/raw/generation/generation.jsonl"
readonly BASE_REPORT="${ROOT}/runs/compact_v2_periodic_relation_g2_final_eval_39139/BASE/raw/evaluation/direct/report.json"
readonly R03E="${G2_FULL_R03E:?export frozen R03E}"
readonly RUN="${ROOT}/runs/g2_full_ab_fast_direct_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
[[ "$(sha256sum "${SCRIPT}"|awk '{print $1}')" == "${SCRIPT_SHA}" ]]
[[ -f "${SOURCE}/A/generation/_SUCCESS" && -f "${SOURCE}/B/generation/_SUCCESS" ]]
mkdir "${RUN}"
on_error(){ set +e; touch "${RUN}/_FAILED"; exit "$1"; }
trap 'on_error "$?"' ERR
run(){
 local name="$1" generation="$2" expected="$3"
 args=(--generation-jsonl "${generation}" --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" --output-dir "${RUN}/${name}")
 if [[ -n "${expected}" ]]; then args+=(--expected-report "${expected}"); fi
 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "${PYTHON}" "${SCRIPT}" "${args[@]}" >"${RUN}/${name}.out" 2>"${RUN}/${name}.err"
}
run BASE "${BASE_GEN}" "${BASE_REPORT}" & p0=$!
run A "${SOURCE}/A/generation/generation.jsonl" "" & p1=$!
run B "${SOURCE}/B/generation/generation.jsonl" "" & p2=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"
"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); result={}
for name in ('BASE','A','B'):
 d=json.load((root/name/'report.json').open()); result[name]={k:int(d[k]) for k in ('generation_succeeded','comp_valid_count','struct_valid_count','valid_count')}
result['A_directional']=result['A']['valid_count']>121; result['A_meaningful']=result['A']['valid_count']>=134
result['B_directional']=result['B']['valid_count']>121; result['B_meaningful']=result['B']['valid_count']>=134
(root/'G2_FULL_AB_FAST_DIRECT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
PY
trap - ERR
touch "${RUN}/_SUCCESS"
