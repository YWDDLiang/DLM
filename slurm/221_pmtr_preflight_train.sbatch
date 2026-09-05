#!/usr/bin/env bash
#SBATCH --job-name=pmtr-train384
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=320G
#SBATCH --time=01:30:00
#SBATCH --output=pmtr-train384-%j.out
#SBATCH --error=pmtr-train384-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly CERT_RUN="${PMTR_CERT_RUN:-${ROOT}/runs/pmtr_preflight_certify_39824}"
readonly BASE_RUN="${ROOT}/runs/spad_basin_closure_ce_39700"
readonly BASE_POLICY="$(<"${BASE_RUN}/POLICY_PATH")"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/pmtr_preflight_train_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR

for path in \
  "${CERT_RUN}/_SUCCESS" "${CERT_RUN}/data_fit/data.jsonl" \
  "${BASE_POLICY}/adapter_model.safetensors" \
  "${SOURCE}/src/scripts/train_pmtr.py"; do
  [[ -f "${path}" ]]
done
[[ "$(wc -l <"${CERT_RUN}/data_fit/data.jsonl")" -eq 384 ]]
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?four allocated GPUs are required}"
[[ "${#gpu[@]}" -eq 4 ]]
gpu_csv="$(IFS=,; printf '%s' "${gpu[*]}")"

env CUDA_VISIBLE_DEVICES="${gpu_csv}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" -m torch.distributed.run --standalone --master-port=38221 \
  --nproc_per_node=4 "${SOURCE}/src/scripts/train_pmtr.py" \
  --data-jsonl "${CERT_RUN}/data_fit/data.jsonl" \
  --model-path "${MODEL}" --checkpoint-path "${BASE_POLICY}" \
  --output-dir "${RUN}/train" --epochs 2 --expected-rows 384 --limit 384 \
  --batch-size 4 --num-workers 1 --max-length 382 \
  --learning-rate 1e-4 --weight-decay 1e-2 --gradient-clip 1.0 \
  --seed 20260905 --probe-batches 5 --head-width 128 \
  --radial-basis-count 16 --radial-cutoff-A 8.0 \
  --max-metric-tangent 0.20 --max-cartesian-step-A 0.75 \
  --transport-gain 6.0 --image-radius 2 \
  --lattice-target-scale 0.20 --cartesian-target-scale-A 0.75 \
  --step-regularization 1e-3 \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

[[ -f "${RUN}/train/pmtr_final.pt" ]]
"${PYTHON}" - "${RUN}" <<'PY'
import json, pathlib, sys, torch
root=pathlib.Path(sys.argv[1]); checkpoint=root/"train/pmtr_final.pt"
payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
if payload.get("schema") != "pmtr_repair_head_v1":
    raise RuntimeError("PMTR preflight checkpoint schema changed")
events=[json.loads(line) for line in (root/"logs/train.out").open() if line.lstrip().startswith("{")]
complete=[row for row in events if row.get("event")=="complete"]
if len(complete)!=1 or complete[0]["source_rows"]!=384:
    raise RuntimeError("PMTR preflight completion report is incomplete")
if complete[0]["certified_rows"]!=382 or complete[0]["fallback_rows"]!=2:
    raise RuntimeError("PMTR preflight fit coverage changed")
(root/"TRAIN_FINAL.json").write_text(json.dumps(complete[0],indent=2,sort_keys=True)+"\n")
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
