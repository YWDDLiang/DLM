#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-typed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --output=c3fd-llama-typed-%j.out
#SBATCH --error=c3fd-llama-typed-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly DATA="${ROOT}/data/c3fd_llama_fused_typed_v2_20260831"
readonly C3FD="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly VOCAB="${ROOT}/data/c3fd_semantic_v21_step1b_20260828/vocabulary.json"
readonly TRAINER="${ROOT}/src/scripts/train_c3fd_llama_typed_planner.py"
readonly TYPED_CORE="${ROOT}/src/crystal_dlm/c3fd_llama_typed_planner.py"
readonly RUN="${ROOT}/runs/c3fd_llama_typed_planner_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/train_seed85017"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly C3FD_SHA=87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53
readonly VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly MODEL_TREE_SHA="${C3FD_LLAMA_MODEL_TREE_SHA256:?pin model tree SHA256}"
readonly DATA_MANIFEST_SHA="${C3FD_LLAMA_FUSED_MANIFEST_SHA256:?pin data manifest SHA256}"
readonly DATA_TRAIN_SHA="${C3FD_LLAMA_FUSED_TRAIN_SHA256:?pin data train SHA256}"
readonly DATA_VAL_SHA="${C3FD_LLAMA_FUSED_VAL_SHA256:?pin data val SHA256}"
readonly TRAINER_SHA="${C3FD_LLAMA_TYPED_TRAINER_SHA256:?pin trainer SHA256}"
readonly TYPED_CORE_SHA="${C3FD_LLAMA_TYPED_CORE_SHA256:?pin typed core SHA256}"

mkdir "${RUN}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose a GPU}"
[[ "${#allocated[@]}" -eq 1 ]]
for spec in "${C3FD_SHA}:${C3FD}" "${VOCAB_SHA}:${VOCAB}" \
  "${DATA_MANIFEST_SHA}:${DATA}/manifest.json" "${DATA_TRAIN_SHA}:${DATA}/train.jsonl" \
  "${DATA_VAL_SHA}:${DATA}/val.jsonl" "${TRAINER_SHA}:${TRAINER}" \
  "${TYPED_CORE_SHA}:${TYPED_CORE}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done
[[ -f "${DATA}/_SUCCESS" ]]
actual_model_tree_sha="$("${PYTHON}" - "${MODEL}" <<'PY'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]); digest=hashlib.sha256()
files=sorted(path for path in root.rglob("*") if path.is_file())
if not files: raise SystemExit("empty model tree")
for path in files:
    relative=path.relative_to(root).as_posix().encode()
    file_digest=hashlib.sha256(path.read_bytes()).hexdigest().encode()
    digest.update(relative); digest.update(b"\0"); digest.update(file_digest); digest.update(b"\n")
print(digest.hexdigest())
PY
)"
[[ "${actual_model_tree_sha}" == "${MODEL_TREE_SHA}" ]]
sha256sum "${C3FD}" "${VOCAB}" "${DATA}/manifest.json" "${DATA}/train.jsonl" \
  "${DATA}/val.jsonl" "${TRAINER}" "${TYPED_CORE}" >"${RUN}/INPUTS.sha256"
CUDA_VISIBLE_DEVICES="${allocated[0]}" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  "${PYTHON}" "${TRAINER}" --model-path "${MODEL}" \
  --model-tree-sha256 "${MODEL_TREE_SHA}" --data-dir "${DATA}" \
  --data-manifest-sha256 "${DATA_MANIFEST_SHA}" --c3fd-checkpoint "${C3FD}" \
  --c3fd-checkpoint-sha256 "${C3FD_SHA}" --vocabulary "${VOCAB}" \
  --vocabulary-sha256 "${VOCAB_SHA}" --output-dir "${OUTPUT}" \
  --seed 85017 --epochs 1 --batch-size 2 --grad-accum 8 --lr 2e-5 \
  --warmup-steps 100 --num-workers 4 >"${RUN}/train.out" 2>"${RUN}/train.err"
[[ -f "${OUTPUT}/_SUCCESS" && -f "${OUTPUT}/final/_SUCCESS" ]]
[[ -f "${OUTPUT}/final/llama_adapter/adapter_model.safetensors" ]]
[[ -f "${OUTPUT}/final/typed_residual_config.json" ]]
[[ -f "${OUTPUT}/final/typed_residual_state.pt" && ! -e "${OUTPUT}/checkpoints" ]]
"${PYTHON}" -c 'import json,sys; c=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); assert c["seed"]==85017 and c["epochs"]==1 and c["effective_batch"]==16; assert c["eligible_checkpoint"]=="final_only" and c["checkpoint_selection"]=="none"; assert c["c3fd_frozen"] is True; assert m["epochs_completed"]==1 and m["c3fd_frozen_verified"] is True' "${OUTPUT}/run_config.json" "${OUTPUT}/train_metrics.json"
find "${RUN}" -type f ! -name OUTPUTS.sha256 ! -name adapter_model.safetensors ! -name typed_residual_state.pt -print0 | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
