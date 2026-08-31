#!/usr/bin/env bash
#SBATCH --job-name=fused-body-direct-cpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=fused-body-direct-cpu-%j.out
#SBATCH --error=fused-body-direct-cpu-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly BODY="${ROOT}/runs/fused_planner_compact_v2_body_direct_39096/body"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly ASSEMBLER="${ROOT}/scripts/assemble_raw_body_repeat.py"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly RUN="${ROOT}/runs/fused_planner_body_direct_downstream_${SLURM_JOB_ID}"

readonly BODY_RAW_SHA=0cba0a009bcdb317eac768184518f481df5b09324341c928853b9224e03a299a
readonly BODY_MANIFEST_SHA=cb2e22e4feec3f347e7a3e72cc011842f0d55454b9a617e59b68937b22be933e
readonly BODY_GRAPHS_SHA=f0b6f21daa5dac4a22860d00f02bab06375f61f7abc202d23e96abd8c24942b6
readonly ASSEMBLER_SHA=8e9e9380eb2b2ca9e833faacb242e04ad0b34f897db69f5c3fca3cce3557d6ec
readonly DIRECT_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c

mkdir "${RUN}"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'exit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for spec in "${BODY_RAW_SHA}:${BODY}/raw_generations.jsonl" \
  "${BODY_MANIFEST_SHA}:${BODY}/SGTC_BODY_MANIFEST.json" \
  "${BODY_GRAPHS_SHA}:${BODY}/proposal_graphs.pt" \
  "${ASSEMBLER_SHA}:${ASSEMBLER}" "${DIRECT_SHA}:${DIRECT}" "${GT_SHA}:${GT_CSV}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done
[[ -f "${BODY}/_SUCCESS" ]]
cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	fused_planner_body_direct_downstream_v1
source_body_job	39096
requested	256
parsed	248
stream	17
dlm_seed	91117
body_rerun	false
model494	false
chgnet	false
official_query	false
EOF
sha256sum "${BODY}/raw_generations.jsonl" "${BODY}/SGTC_BODY_MANIFEST.json" \
  "${BODY}/proposal_graphs.pt" "${ASSEMBLER}" "${DIRECT}" "${GT_CSV}" >"${RUN}/INPUTS.sha256"

"${PYTHON}" "${ASSEMBLER}" --body-dir "${BODY}" \
  --output-dir "${RUN}/generation" --seed 17 --dlm-seed 91117 \
  --denominator 256 --source-arm full_axis --arm-label candidate

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 CRYSLLMGEN_METRICS_NUM_CPUS=8 \
  "${PYTHON}" "${DIRECT}" --generation-jsonl "${RUN}/generation/generation.jsonl" \
  --gt-csv "${GT_CSV}" \
  --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "${RUN}/direct" --eval-model-name mp20 \
  >"${RUN}/direct.out" 2>"${RUN}/direct.err"
[[ -f "${RUN}/direct/report.json" ]]

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); r=json.load(open(root/'direct/report.json'))
result={'schema':'fused_planner_body_direct_final_v1','requested':256,
 'generation_succeeded':int(r['generation_succeeded']),
 'comp_valid':int(r['comp_valid_count']),'struct_valid':int(r['struct_valid_count']),
 'direct_joint':int(r['valid_count']),'retry_or_replacement_used':bool(r['retry_or_replacement_used'])}
assert r['denominator']==256 and r['attempts']==256 and not result['retry_or_replacement_used']
(root/'BODY_DIRECT_FINAL.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
PY
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"

