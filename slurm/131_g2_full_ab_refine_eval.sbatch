#!/usr/bin/env bash
#SBATCH --job-name=g2-full-ab-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=420G
#SBATCH --time=06:00:00
#SBATCH --output=g2-full-ab-eval-%j.out
#SBATCH --error=g2-full-ab-eval-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:?export deployed git commit}"
readonly RAW="${G2_FULL_AB_RAW_RUN:?export completed matched raw run}"
readonly A_BODY_SHA="${G2_FULL_A_BODY_SHA256:?pin A body SHA256}"
readonly A_GRAPH_SHA="${G2_FULL_A_GRAPH_SHA256:?pin A graph SHA256}"
readonly B_BODY_SHA="${G2_FULL_B_BODY_SHA256:?pin B body SHA256}"
readonly B_GRAPH_SHA="${G2_FULL_B_GRAPH_SHA256:?pin B graph SHA256}"
readonly RUN="${ROOT}/runs/g2_full_ab_refine_eval_${SLURM_JOB_ID}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly REFINE_SCRIPT="${CODE_ROOT}/src/scripts/refine_dlm_with_crysllmgen.py"
readonly RAW_ASSEMBLER="${CODE_ROOT}/scripts/assemble_raw_body_repeat.py"
readonly REFINED_ASSEMBLER="${CODE_ROOT}/scripts/assemble_grounding_repeat.py"
readonly FULL="${CODE_ROOT}/eval_runtime/run_full_reconstructed_eval.py"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly REFINE_SCRIPT_SHA="${G2_FULL_REFINE_SCRIPT_SHA256:?pin refiner script SHA256}"
readonly REFINER_SHA="${G2_FULL_REFINER_SHA256:?pin model494 SHA256}"
readonly RAW_ASSEMBLER_SHA="${G2_FULL_RAW_ASSEMBLER_SHA256:?pin raw assembler SHA256}"
readonly REFINED_ASSEMBLER_SHA="${G2_FULL_REFINED_ASSEMBLER_SHA256:?pin refined assembler SHA256}"
readonly FULL_SHA="${G2_FULL_FULL_EVAL_SHA256:?pin full evaluator SHA256}"
readonly DIRECT_SHA="${G2_FULL_DIRECT_SHA256:?pin Direct evaluator SHA256}"
readonly GT_SHA="${G2_FULL_GT_SHA256:?pin MP20 test SHA256}"
readonly CACHE_SHA="${G2_FULL_CHGNET_CACHE_SHA256:?pin CHGNet cache SHA256}"
readonly CONFIG_SHA="${G2_FULL_CONFIG_SHA256:?pin official config SHA256}"

mkdir "${RUN}"; mkdir "${RUN}/logs"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.${BASHPID}.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ "$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD)" == "${CODE_COMMIT}" ]]
[[ -f "${RAW}/_RAW_SUCCESS" && ! -e "${RAW}/_FAILED" ]]
for spec in \
 "${A_BODY_SHA}:${RAW}/A/body/raw_generations.jsonl" "${A_GRAPH_SHA}:${RAW}/A/body/proposal_graphs.pt" \
 "${B_BODY_SHA}:${RAW}/B/body/raw_generations.jsonl" "${B_GRAPH_SHA}:${RAW}/B/body/proposal_graphs.pt" \
 "${REFINE_SCRIPT_SHA}:${REFINE_SCRIPT}" "${REFINER_SHA}:${REFINER}" \
 "${RAW_ASSEMBLER_SHA}:${RAW_ASSEMBLER}" "${REFINED_ASSEMBLER_SHA}:${REFINED_ASSEMBLER}" \
 "${FULL_SHA}:${FULL}" "${DIRECT_SHA}:${DIRECT}" "${GT_SHA}:${GT_CSV}" \
 "${CACHE_SHA}:${CACHE}" "${CONFIG_SHA}:${V3_SOURCE}/CONFIG.json"; do
 expected="${spec%%:*}"; path="${spec#*:}"
 [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose GPUs}"
[[ "${#gpu[@]}" -eq 4 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	g2_full_ab_refine_eval_v1
raw_run	${RAW}
arms	A,B
endpoints	raw,refined
requested_per_cell	256
model494_tau	800
refiner_seed	101117_seed_by_sample_idx
chgnet	all_four_cells
official_cache	reuse_existing_same_chemsys_no_query
missing_policy	preserve_sample_idx
selection_retry_topup	false
claim_scope	post_outcome_matched_development
EOF
sha256sum "${RUN}/scientific_contract.tsv" \
 "${RAW}/A/body/raw_generations.jsonl" "${RAW}/A/body/proposal_graphs.pt" \
 "${RAW}/B/body/raw_generations.jsonl" "${RAW}/B/body/proposal_graphs.pt" \
 "${REFINE_SCRIPT}" "${REFINER}" "${RAW_ASSEMBLER}" "${REFINED_ASSEMBLER}" \
 "${FULL}" "${DIRECT}" "${GT_CSV}" "${CACHE}" "${V3_SOURCE}/CONFIG.json" >"${RUN}/INPUTS.sha256"

refine(){
 local method="$1" device="$2" output="${RUN}/${1}/refine"
 mkdir -p "$(dirname "${output}")"
 env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${REFINE_SCRIPT}" --proposal-graphs "${RAW}/${method}/body/proposal_graphs.pt" \
  --checkpoint "${REFINER}" --output-dir "${output}" --crysllmgen-dir "${CRYS}" \
  --batch-size 1 --seed 101117 --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
  >"${RUN}/logs/${method}-refine.out" 2>"${RUN}/logs/${method}-refine.err"
 metrics="${output}/refinement_metrics.json"; [[ -f "${metrics}" ]]
 path="$(${PYTHON} -c 'import json,sys;print(json.load(open(sys.argv[1]))["output_file"])' "${metrics}")"
 [[ -f "${path}" ]]
}
refine A "${gpu[0]}" & r0=$!
refine B "${gpu[1]}" & r1=$!
wait "${r0}"; wait "${r1}"

assemble(){
 local method="$1" label="$2" body="${RAW}/${1}/body" refine="${RUN}/${1}/refine"
 "${PYTHON}" "${RAW_ASSEMBLER}" --body-dir "${body}" --output-dir "${RUN}/${method}/raw/generation" \
  --seed 17 --dlm-seed 91117 --denominator 256 --source-arm full_axis --arm-label "${label}"
 "${PYTHON}" "${REFINED_ASSEMBLER}" --body-dir "${body}" --refine-dir "${refine}" \
  --output-dir "${RUN}/${method}/refined/generation" --arm "${label}" --repeat 0 \
  --dlm-seed 91117 --refiner-seed 101117 --denominator 256
}
assemble A control
assemble B candidate

run_cell(){
 local slot="$1" method="$2" endpoint="$3" label="$4" cell="${RUN}/${2}/${3}"
 mkdir "${cell}/evaluation"; cp --reflink=auto "${CACHE}" "${cell}/relax_cache.jsonl"
 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
  "${PYTHON}" "${DIRECT}" --generation-jsonl "${cell}/generation/generation.jsonl" --gt-csv "${GT_CSV}" \
  --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" --output-dir "${cell}/evaluation/direct" \
  --eval-model-name mp20 >"${RUN}/logs/${method}-${endpoint}-direct.out" 2>"${RUN}/logs/${method}-${endpoint}-direct.err" & dpid=$!
 env CUDA_VISIBLE_DEVICES="${gpu[${slot}]}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${CODE_ROOT}/eval_runtime:${R03E}/runtime" "${PYTHON}" "${FULL}" \
  --arm "${label}" --repeat 0 --generation-jsonl "${cell}/generation/generation.jsonl" \
  --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" --working-relax-cache "${cell}/relax_cache.jsonl" \
  --output-dir "${cell}/evaluation/full_reconstructed" >"${RUN}/logs/${method}-${endpoint}-full.out" 2>"${RUN}/logs/${method}-${endpoint}-full.err"
 wait "${dpid}"; [[ -f "${cell}/evaluation/direct/report.json" && -f "${cell}/evaluation/full_reconstructed/_SUCCESS" ]]
}
run_cell 0 A raw control & p0=$!
run_cell 1 A refined control & p1=$!
run_cell 2 B raw candidate & p2=$!
run_cell 3 B refined candidate & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${RUN}"

