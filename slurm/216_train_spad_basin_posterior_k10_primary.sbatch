#!/usr/bin/env bash
#SBATCH --job-name=spad-k10-primary
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=spad-k10-primary-%j.out
#SBATCH --error=spad-k10-primary-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly LABELLED="${ROOT}/runs/spad_basin_preflight_values_39752/labelled_groups.jsonl"
readonly DECISION="${ROOT}/runs/spad_basin_preflight_decision_20260904_v1"
readonly AUTH="${DECISION}/PRELIGHT_TRAINING_AUTHORIZED"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly POLICY="${ROOT}/runs/spad_basin_closure_ce_39700/train/checkpoints/step-1696"
readonly SFT="${ROOT}/data/spad_basin_closure_sft_v1_20260904"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_posterior_k10_primary_${SLURM_JOB_ID}"

for p in "${LABELLED}" "${AUTH}" "${POLICY}/adapter_model.safetensors" \
 "${POLICY}/spad_basin_closure_capability.json" "${SFT}/train.jsonl"; do [[ -f "$p" ]]; done
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?}"; [[ "${#gpu[@]}" -eq 2 ]]
mkdir "${RUN}"; mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
cat >"${RUN}/scientific_contract.tsv" <<'EOF'
field	value
schema	spad_basin_posterior_k10_primary_v1
primary_route	K10_basin_consistent
groups	128_x_4_passes
objectives	256_clean_CE_plus_256_posterior_interleaved
world_size	2
learning_rate	5e-6
checkpoint_selection	none_final_only
EOF

CUDA_VISIBLE_DEVICES="${gpu[0]},${gpu[1]}" OPENBLAS_NUM_THREADS=4 \
 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
 TOKENIZERS_PARALLELISM=false PYTHONPATH="${SOURCE}/src" \
 "${PYTHON}" -m torch.distributed.run --standalone --master-port 37218 \
 --nproc_per_node=2 "${SOURCE}/src/scripts/train_spad_basin_posterior.py" \
 --model-path "${MODEL}" --checkpoint-path "${POLICY}" \
 --labelled-groups "${LABELLED}" --sft-data-dir "${SFT}" \
 --output-dir "${RUN}/k10" --authorization-marker "${AUTH}" \
 --route-name k10 --value-field terminal_relax_k10_energy_eV_per_atom \
 --seed 99018 >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); q=json.load(open(r/'k10'/'TRAIN_FINAL.json'))
assert q['status']=='success' and q['route_name']=='k10'
assert q['optimizer_updates']==512 and q['each_group_seen_exactly']==4
assert q['objective_counts']=={'clean_ce':256,'transaction_posterior':256}
assert q['checkpoint']['only_final_policy_saved'] is True
(r/'POLICY_PATH').write_text(q['checkpoint']['policy_adapter_path']+'\n')
(r/'CAPABILITY_PATH').write_text(q['checkpoint']['capability_json']+'\n')
(r/'TRAIN_FINAL.json').write_text(json.dumps(q,indent=2,sort_keys=True)+'\n')
PY
trap - ERR; touch "${RUN}/_SUCCESS"
