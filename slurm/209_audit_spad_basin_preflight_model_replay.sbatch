#!/usr/bin/env bash
#SBATCH --job-name=spad-replay
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=spad-replay-%j.out
#SBATCH --error=spad-replay-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly STATES="${ROOT}/data/spad_basin_preflight_states_v1_20260904"
readonly TRAIN_RUN="${ROOT}/runs/spad_basin_closure_ce_39700"
readonly POLICY="$(<"${TRAIN_RUN}/POLICY_PATH")"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_preflight_model_replay_${SLURM_JOB_ID}"

mkdir "${RUN}"; mkdir "${RUN}/logs"; mkdir "${RUN}/replay"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
for p in "${STATES}/_SUCCESS" "${STATES}/states.jsonl" "${POLICY}/adapter_model.safetensors"; do [[ -f "$p" ]]; done
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?}"; [[ "${#gpu[@]}" -eq 2 ]]

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
 PYTHONPATH="${SOURCE}/src" "${PYTHON}" -m torch.distributed.run \
 --standalone --nproc_per_node=2 \
 "${SOURCE}/src/scripts/audit_spad_basin_preflight_model_replay.py" \
 --states-jsonl "${STATES}/states.jsonl" --model-path "${MODEL}" \
 --checkpoint-path "${POLICY}" --output-dir "${RUN}/replay" --original-batch-size 8 \
 >"${RUN}/logs/replay.out" 2>"${RUN}/logs/replay.err"

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); attempts=[]; reports=[]
for rank in range(2):
 reports.append(json.load(open(r/'replay'/f'report_rank{rank}.json')))
 attempts += [json.loads(x) for x in open(r/'replay'/f'attempts_rank{rank}.jsonl') if x.strip()]
attempts.sort(key=lambda x:x['sample_idx'])
assert len(attempts)==128 and [x['sample_idx'] for x in attempts]==list(range(128))
mismatch=[x for x in attempts if not x['matches_final']]
report={'schema':'spad_basin_preflight_model_replay_final_v1','attempts':128,
 'matches':128-len(mismatch),'mismatches':len(mismatch),
 'cell':sum(x['state_type']=='cell' for x in attempts),
 'xyz':sum(x['state_type']=='xyz' for x in attempts),'outcomes_read':False,
 'bit_exact_required_before_value_labels':True,'rank_reports':reports}
(r/'REPLAY_FINAL.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if mismatch:
 (r/'MISMATCHES.json').write_text(json.dumps(mismatch,indent=2,sort_keys=True)+'\n')
 raise SystemExit(f'{len(mismatch)} model continuation replays mismatched')
PY
trap - ERR; touch "${RUN}/_SUCCESS"
