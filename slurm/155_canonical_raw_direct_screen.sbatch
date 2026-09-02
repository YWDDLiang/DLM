#!/usr/bin/env bash
#SBATCH --job-name=canonical-raw-direct
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=280G
#SBATCH --time=04:00:00
#SBATCH --output=canonical-raw-direct-%j.out
#SBATCH --error=canonical-raw-direct-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly CODE="${H1A2_CODE_ROOT:?}"
readonly CANON_DLM="${CANONICAL_DLM_POLICY:?}"
readonly CANON_G2="${CANONICAL_G2_POLICY:?}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PLANS="${ROOT}/runs/compact_v2_periodic_relation_g2_final_generation_39137/plans_for_body.jsonl"
readonly OLD_ROOT="${ROOT}/runs/g2_full_ab_refine_eval_39172/A/raw"
readonly OLD_REPORT="${OLD_ROOT}/evaluation/direct/report.json"
readonly OLD_ATTEMPTS="${OLD_ROOT}/evaluation/direct/attempt_metrics.jsonl"
readonly SAMPLER="${CODE}/src/scripts/sample_sgtc_l6.py"
readonly ASSEMBLER="${ROOT}/scripts/assemble_raw_body_repeat.py"
readonly DIRECT="${CODE}/scripts/run_direct_validity_fast.py"
readonly SNAPSHOT="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream"
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/canonical_raw_direct_screen_${SLURM_JOB_ID}"

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ "$(wc -l < "${PLANS}")" -eq 256 ]]
[[ -f "${CANON_DLM}/adapter_model.safetensors" ]]
[[ ! -e "${CANON_DLM}/periodic_relation_adapter.pt" ]]
for file in adapter_model.safetensors periodic_relation_adapter.pt periodic_relation_config.json; do
  [[ -f "${CANON_G2}/${file}" ]]
done
[[ -f "${OLD_REPORT}" && -f "${OLD_ATTEMPTS}" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	canonical_raw_direct_screen_v1
plans	reuse_final_prospective_256
old_control	reuse_G2_PBC_R_A_raw_same_plan_noise
new_arms	canonical_DLM,canonical_G2_PBC_R
stream	17
dlm_seed	91117
temperature	0.7
schedule	exact_axis
requested_per_arm	256
trajectory_per_plan	1
planner_resampled	false
retry_filter_replacement_rerank_best_of_n	false
model494	false
chgnet	false_until_direct_screen
EOF

run_body() {
  local arm="$1" policy="$2" device="$3" relation="$4" output="${RUN}/${1}"
  mkdir "${output}"
  local -a sampler_args=(
    --model-path "${MODEL}"
    --checkpoint-path "${policy}"
  )
  if [[ "${relation}" == true ]]; then
    sampler_args+=(--periodic-relation-checkpoint "${policy}" --periodic-relation-rank 64)
  fi
  sampler_args+=(
    --prompt-jsonl "${PLANS}"
    --output-dir "${output}/body"
    --crysllmgen-dir "${CRYS}"
    --num-samples 256
    --expected-denominator 256
    --temperature 0.7
    --seed 91117
  )
  env CUDA_VISIBLE_DEVICES="${device}" PYTHONPATH="${CODE}/src" \
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    "${PYTHON}" "${SAMPLER}" "${sampler_args[@]}" \
    >"${output}/body.out" 2>"${output}/body.err"
  [[ -f "${output}/body/_SUCCESS" ]]
}
run_body CANON_DLM "${CANON_DLM}" "${allocated[0]}" false & p0=$!
run_body CANON_G2 "${CANON_G2}" "${allocated[1]}" true & p1=$!
wait "${p0}"
wait "${p1}"

run_direct() {
  local arm="$1" output="${RUN}/${1}"
  "${PYTHON}" "${ASSEMBLER}" \
    --body-dir "${output}/body" \
    --output-dir "${output}/generation" \
    --seed 17 \
    --dlm-seed 91117 \
    --denominator 256 \
    --source-arm full_axis \
    --arm-label candidate
  env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    "${PYTHON}" "${DIRECT}" \
    --generation-jsonl "${output}/generation/generation.jsonl" \
    --snapshot-root "${SNAPSHOT}" \
    --output-dir "${output}/direct" \
    --expected-denominator 256 \
    >"${output}/direct.out" 2>"${output}/direct.err"
  [[ -f "${output}/direct/_SUCCESS" ]]
}
run_direct CANON_DLM & d0=$!
run_direct CANON_G2 & d1=$!
wait "${d0}"
wait "${d1}"

"${PYTHON}" - "${RUN}" "${OLD_REPORT}" "${OLD_ATTEMPTS}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
reports={"OLD_G2":json.load(open(sys.argv[2]))}
attempts={
  "OLD_G2":{
    int(r.get("ordinal",str(r["attempt_id"]).rsplit("-",1)[-1])):r
    for r in (json.loads(x) for x in open(sys.argv[3]) if x.strip())
  }
}
for arm in ("CANON_DLM","CANON_G2"):
 reports[arm]=json.load((root/arm/"direct/report.json").open())
 attempts[arm]={int(r["ordinal"]):r for r in (json.loads(x) for x in (root/arm/"direct/attempt_metrics.jsonl").open() if x.strip())}
for arm in reports:
 assert reports[arm]["attempts"]==256 and set(attempts[arm])==set(range(256))
def counts(r):
 return {"body":int(r["generation_succeeded"]),"composition":int(r["comp_valid_count"]),"structure":int(r["struct_valid_count"]),"direct":int(r["valid_count"])}
def flips(a,b):
 i2v=sum(not attempts[a][i]["valid"] and attempts[b][i]["valid"] for i in range(256))
 v2i=sum(attempts[a][i]["valid"] and not attempts[b][i]["valid"] for i in range(256))
 return {"invalid_to_valid":i2v,"valid_to_invalid":v2i,"net":i2v-v2i}
result={"schema":"canonical_raw_direct_final_v1","requested":256,"arms":{a:counts(r) for a,r in reports.items()}}
result["paired"]={"CANON_DLM_minus_OLD_G2":flips("OLD_G2","CANON_DLM"),"CANON_G2_minus_OLD_G2":flips("OLD_G2","CANON_G2"),"CANON_G2_minus_CANON_DLM":flips("CANON_DLM","CANON_G2")}
old=result["arms"]["OLD_G2"]; g2=result["arms"]["CANON_G2"]
result["supports_raw_chgnet"]=bool(g2["body"]>=old["body"]-2 and g2["composition"]>=old["composition"]-2 and g2["direct"]>=old["direct"]-2)
(root/"RAW_DIRECT_FINAL.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,sort_keys=True))
PY

trap - ERR
touch "${RUN}/_RAW_SUCCESS"
printf '%s\n' "${RUN}"
