#!/usr/bin/env bash
#SBATCH --job-name=g2-full-ab-direct
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=g2-full-ab-direct-%j.out
#SBATCH --error=g2-full-ab-direct-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export root}"
readonly SOURCE="${ROOT}/runs/g2_full_ab_raw_screen_39168"
readonly RUN="${ROOT}/runs/g2_full_ab_direct_recovery_${SLURM_JOB_ID}"
readonly DIRECT="${G2_FULL_DIRECT_PATH:?export frozen Direct evaluator}"
readonly DIRECT_SHA="${G2_FULL_DIRECT_SHA256:?pin Direct SHA256}"
readonly GT_CSV="${G2_FULL_GT_PATH:?export frozen MP20 test CSV}"
readonly GT_SHA="${G2_FULL_GT_SHA256:?pin GT SHA256}"
readonly R03E="${G2_FULL_R03E:?export frozen R03E root}"
readonly BASE_REPORT="${ROOT}/runs/compact_v2_periodic_relation_g2_final_eval_39139/BASE/raw/evaluation/direct/report.json"
readonly BASE_REPORT_SHA=9b4557298900ecf4d615b6d7bc08d98a1b571a74a35896e0aa0912695e37993e
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

mkdir "${RUN}"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
[[ -f "${SOURCE}/_FAILED" && ! -e "${SOURCE}/_RAW_SUCCESS" ]]
[[ "$(sha256sum "${DIRECT}"|awk '{print $1}')" == "${DIRECT_SHA}" ]]
[[ "$(sha256sum "${GT_CSV}"|awk '{print $1}')" == "${GT_SHA}" ]]
[[ "$(sha256sum "${BASE_REPORT}"|awk '{print $1}')" == "${BASE_REPORT_SHA}" ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	g2_full_ab_direct_recovery_v1
source_body_run	${SOURCE}
recovery_reason	body_complete_direct_thread_environment_only
methods	A,B
requested	256
thread_settings	4_equal_worker1
planner_resampled	false
body_resampled	false
EOF

for method in A B; do
 mkdir -p "${RUN}/${method}"
 ln -s "${SOURCE}/${method}/body" "${RUN}/${method}/body"
 ln -s "${SOURCE}/${method}/generation" "${RUN}/${method}/generation"
done

run_direct(){
 local method="$1" output="${RUN}/${1}"
 env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
  "${PYTHON}" "${DIRECT}" --generation-jsonl "${output}/generation/generation.jsonl" \
  --gt-csv "${GT_CSV}" --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "${output}/direct" --eval-model-name mp20 \
  >"${output}/direct.out" 2>"${output}/direct.err"
 [[ -f "${output}/direct/report.json" ]]
}
run_direct A & p0=$!
run_direct B & p1=$!
wait "${p0}"; wait "${p1}"

"${PYTHON}" - "${RUN}" "${BASE_REPORT}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); base=json.load(open(sys.argv[2]))
result={'schema':'g2_full_ab_same_plan_raw_final_v1','requested':256,
        'base':{'body':int(base['generation_succeeded']),'comp':int(base['comp_valid_count']),'direct':int(base['valid_count'])},'methods':{}}
for method in ('A','B'):
 report=json.load((root/method/'direct/report.json').open())
 assert int(report['attempts'])==256 and report['retry_or_replacement_used'] is False
 body=int(report['generation_succeeded']); comp=int(report['comp_valid_count']); direct=int(report['valid_count'])
 result['methods'][method]={'body':body,'comp':comp,'direct':direct,
   'technical_pass':body>=244 and comp>=244,'directional_pass':direct>121,'meaningful_pass':direct>=134}
a=result['methods']['A']; b=result['methods']['B']
result['B_direct_promotion_pass']=b['body']>=a['body'] and b['direct']>=a['direct']+8
(root/'G2_FULL_AB_RAW_FINAL.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
PY
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_RAW_SUCCESS"
printf '%s\n' "${RUN}"
