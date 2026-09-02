#!/usr/bin/env bash
#SBATCH --job-name=canonical-raw-chgnet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=canonical-raw-chgnet-%j.out
#SBATCH --error=canonical-raw-chgnet-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly RAW="${CANONICAL_RAW_RUN:?}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly FULL="${ROOT}/eval_runtime/run_full_reconstructed_eval.py"
readonly OLD_LABELS="${ROOT}/runs/g2_full_ab_refine_eval_39172/A/raw/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/canonical_raw_chgnet_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ -f "${RAW}/_RAW_RECOVERED_SUCCESS" ]]
[[ -f "${RAW}/RAW_DIRECT_FINAL.json" ]]
for arm in CANON_DLM CANON_G2; do
  [[ -f "${RAW}/${arm}/generation/generation.jsonl" ]]
  [[ "$(wc -l < "${RAW}/${arm}/generation/generation.jsonl")" -eq 256 ]]
done
for path in "${V3_SOURCE}/CONFIG.json" "${CACHE}" "${FULL}" "${OLD_LABELS}"; do
  [[ -f "${path}" ]]
done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	canonical_raw_chgnet_v1
raw_run	${RAW}
arms	CANON_DLM,CANON_G2
old_control	reuse_G2_PBC_R_A_raw_labels
requested_per_arm	256
stage	raw_pre_model494
working_cache	frozen_global_copy_per_arm
official_query	false
selection_filter_rerank	false
EOF

run_arm() {
  local arm="$1" gpu="$2" cell="${RUN}/${1}"
  mkdir "${cell}"
  cp --reflink=auto "${CACHE}" "${cell}/relax_cache.jsonl"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 H1_ACTIVE_DENOMINATOR=256 \
    PYTHONPATH="${ROOT}/eval_runtime:${R03E}/runtime" \
    "${PYTHON}" "${FULL}" \
    --arm candidate \
    --repeat 0 \
    --generation-jsonl "${RAW}/${arm}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" \
    --r03e-root "${R03E}" \
    --working-relax-cache "${cell}/relax_cache.jsonl" \
    --output-dir "${cell}/full_reconstructed" \
    >"${RUN}/logs/${arm}.out" 2>"${RUN}/logs/${arm}.err"
  [[ -f "${cell}/full_reconstructed/_SUCCESS" ]]
}
run_arm CANON_DLM "${allocated[0]}" & p0=$!
run_arm CANON_G2 "${allocated[1]}" & p1=$!
wait "${p0}"
wait "${p1}"

"${PYTHON}" - "${RUN}" "${OLD_LABELS}" <<'PY'
import json,math,pathlib,random,statistics,sys
root=pathlib.Path(sys.argv[1])
paths={"OLD_G2":pathlib.Path(sys.argv[2]),"CANON_DLM":root/"CANON_DLM/full_reconstructed/attempt_labels_preofficial.jsonl","CANON_G2":root/"CANON_G2/full_reconstructed/attempt_labels_preofficial.jsonl"}
rows={a:{int(r["ordinal"]):r for r in (json.loads(x) for x in p.open() if x.strip())} for a,p in paths.items()}
assert all(set(v)==set(range(256)) for v in rows.values())
def energies(arm): return [float(r["chgnet_energy_per_atom"]) for r in rows[arm].values() if r.get("chgnet_relaxation_known") and r.get("chgnet_energy_per_atom") is not None and math.isfinite(float(r["chgnet_energy_per_atom"]))]
def summary(v):
 return {"known":len(v),"mean":statistics.fmean(v),"median":statistics.median(v),"q10":sorted(v)[max(0,round(.1*(len(v)-1)))],"q90":sorted(v)[round(.9*(len(v)-1))]}
def paired(a,b):
 common=[i for i in range(256) if rows[a][i].get("chgnet_relaxation_known") and rows[b][i].get("chgnet_relaxation_known") and rows[a][i].get("chgnet_energy_per_atom") is not None and rows[b][i].get("chgnet_energy_per_atom") is not None]
 d=[float(rows[b][i]["chgnet_energy_per_atom"])-float(rows[a][i]["chgnet_energy_per_atom"]) for i in common]
 rng=random.Random(20260902); boots=[]
 for _ in range(5000): boots.append(statistics.fmean(d[rng.randrange(len(d))] for _ in d))
 boots.sort()
 return {"common":len(d),"mean_delta":statistics.fmean(d),"median_delta":statistics.median(d),"lower_fraction":sum(x<0 for x in d)/len(d),"mean_ci95":[boots[124],boots[4874]]}
result={"schema":"canonical_raw_chgnet_final_v1","arms":{a:summary(energies(a)) for a in rows},"paired":{"CANON_DLM_minus_OLD_G2":paired("OLD_G2","CANON_DLM"),"CANON_G2_minus_OLD_G2":paired("OLD_G2","CANON_G2"),"CANON_G2_minus_CANON_DLM":paired("CANON_DLM","CANON_G2")}}
result["promotion"]={a:bool(result["paired"][f"{a}_minus_OLD_G2"]["common"]>=240 and result["paired"][f"{a}_minus_OLD_G2"]["median_delta"]<=-0.01 and result["paired"][f"{a}_minus_OLD_G2"]["lower_fraction"]>=0.55) for a in ("CANON_DLM","CANON_G2")}
(root/"RAW_CHGNET_FINAL.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,sort_keys=True))
PY

trap - ERR
touch "${RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${RUN}"
