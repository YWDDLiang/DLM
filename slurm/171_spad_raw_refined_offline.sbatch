#!/usr/bin/env bash
#SBATCH --job-name=spad-offline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=spad-offline-%j.out
#SBATCH --error=spad-offline-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly BC_BODY_RUN="${SPAD_BC_BODY_RUN:?}"
readonly BR_BODY_RUN="${SPAD_BR_BODY_RUN:?}"
readonly BS_BODY_RUN="${SPAD_BS_BODY_RUN:?}"
readonly REFINE_RUN="${SPAD_REFINE_RUN:?}"
readonly RAW_DIRECT_RUN="${SPAD_RAW_DIRECT_RUN:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly FULL="${SOURCE}/eval_runtime/run_full_reconstructed_eval.py"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly ASSEMBLER="${SOURCE}/scripts/assemble_raw_body_repeat.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_raw_refined_offline_${SLURM_JOB_ID}"
readonly EXIT_GRACE=300
readonly TERM_GRACE=30
mkdir "${RUN}"; mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#gpu[@]}" -eq 6 ]]
[[ -f "${REFINE_RUN}/_SUCCESS" ]]

body_dir() {
  local arm="$1" stream="$2"
  case "${arm}" in
    BC) printf '%s\n' "${BC_BODY_RUN}/BC/stream${stream}/body" ;;
    BR) printf '%s\n' "${BR_BODY_RUN}/BR/stream${stream}/body" ;;
    BS) printf '%s\n' "${BS_BODY_RUN}/BS/stream${stream}/body" ;;
    *) return 2 ;;
  esac
}

prepare_cell() {
  local arm="$1" stream="$2" endpoint="$3"
  local cell="${RUN}/${arm}/stream${stream}/${endpoint}"
  mkdir -p "${cell}"
  if [[ "${endpoint}" == raw ]]; then
    local body seed dlm_seed
    body="$(body_dir "${arm}" "${stream}")"
    [[ "${stream}" == 17 ]] && { seed=17; dlm_seed=91117; } || { seed=18; dlm_seed=92117; }
    "${PYTHON}" "${ASSEMBLER}" --body-dir "${body}" \
      --output-dir "${cell}/generation" --seed "${seed}" --dlm-seed "${dlm_seed}" \
      --denominator 256 --source-arm full_axis --arm-label candidate
  else
    local source_generation
    source_generation="${REFINE_RUN}/${arm}/stream${stream}/generation"
    [[ -f "${source_generation}/_SUCCESS" ]]
    cp -a "${source_generation}" "${cell}/generation"
  fi
  [[ "$(wc -l <"${cell}/generation/generation.jsonl")" -eq 256 ]]
}

for arm in BC BR BS; do
  for stream in 17 18; do
    prepare_cell "${arm}" "${stream}" raw
    prepare_cell "${arm}" "${stream}" refined
  done
done

# Reuse already completed full Direct for raw BC/BR. BS raw Direct is run once
# below because only its certified predicted-only screen existed.
for arm in BC BR; do
  for stream in 17 18; do
    destination="${RUN}/${arm}/stream${stream}/raw/evaluation"
    mkdir -p "${destination}"
    cp -a "${RAW_DIRECT_RUN}/${arm}/stream${stream}/direct" "${destination}/direct"
  done
done

process_alive() {
  kill -0 -- "-$1" 2>/dev/null || kill -0 "$1" 2>/dev/null
}

wait_completed() {
  local pid="$1" artifact="$2" seen=-1 deadline rc=0
  while process_alive "${pid}"; do
    if [[ -e "${artifact}" ]]; then
      (( seen < 0 )) && seen="${SECONDS}"
      if (( SECONDS - seen >= EXIT_GRACE )); then
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        deadline=$((SECONDS + TERM_GRACE))
        while process_alive "${pid}" && (( SECONDS < deadline )); do sleep 2; done
        process_alive "${pid}" && { kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true; }
        break
      fi
    fi
    sleep 2
  done
  wait "${pid}" || rc="$?"
  if [[ ! -e "${artifact}" ]]; then
    return 1
  fi
  (( rc == 0 )) || return "${rc}"
  return 0
}

run_direct() {
  local arm="$1" stream="$2" endpoint="$3"
  local root="${RUN}/${arm}/stream${stream}/${endpoint}"
  mkdir -p "${root}/evaluation"
  setsid env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${DIRECT}" --generation-jsonl "${root}/generation/generation.jsonl" \
    --gt-csv "${GT}" --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${root}/evaluation/direct" --eval-model-name mp20 \
    >"${RUN}/logs/${arm}_${stream}_${endpoint}_direct.out" \
    2>"${RUN}/logs/${arm}_${stream}_${endpoint}_direct.err" &
  local pid="$!"
  wait_completed "${pid}" "${root}/evaluation/direct/report.json"
}

run_full() {
  local slot="$1" arm="$2" stream="$3" endpoint="$4"
  local root="${RUN}/${arm}/stream${stream}/${endpoint}"
  mkdir -p "${root}/evaluation"
  cp --reflink=auto "${CACHE}" "${root}/relax_cache.jsonl"
  setsid env CUDA_VISIBLE_DEVICES="${gpu[${slot}]}" OPENBLAS_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" \
    "${PYTHON}" "${FULL}" --arm candidate --repeat 0 \
    --generation-jsonl "${root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${root}/relax_cache.jsonl" \
    --output-dir "${root}/evaluation/full_reconstructed" \
    >"${RUN}/logs/${arm}_${stream}_${endpoint}_full.out" \
    2>"${RUN}/logs/${arm}_${stream}_${endpoint}_full.err" &
  local pid="$!"
  wait_completed "${pid}" "${root}/evaluation/full_reconstructed/_SUCCESS"
}

run_cell() {
  local slot="$1" arm="$2" stream="$3" endpoint="$4"
  local direct_pid=""
  if [[ "${endpoint}" == refined || "${arm}" == BS ]]; then
    run_direct "${arm}" "${stream}" "${endpoint}" & direct_pid="$!"
  fi
  run_full "${slot}" "${arm}" "${stream}" "${endpoint}"
  [[ -z "${direct_pid}" ]] || wait "${direct_pid}"
}

run_wave() {
  local endpoint="$1" index=0 status=0 pid
  local -a pids=()
  for arm in BC BR BS; do
    for stream in 17 18; do
      run_cell "${index}" "${arm}" "${stream}" "${endpoint}" & pids+=("$!")
      index=$((index + 1))
    done
  done
  for pid in "${pids[@]}"; do wait "${pid}" || status="$?"; done
  return "${status}"
}

run_wave raw
run_wave refined

for arm in BC BR BS; do
  for stream in 17 18; do
    for endpoint in raw refined; do
      root="${RUN}/${arm}/stream${stream}/${endpoint}/evaluation"
      [[ -f "${root}/direct/report.json" ]]
      [[ -f "${root}/full_reconstructed/_SUCCESS" ]]
      [[ -f "${root}/full_reconstructed/attempt_labels_preofficial.jsonl" ]]
      [[ -f "${root}/full_reconstructed/worker_cleanup.json" ]]
    done
  done
done
trap - ERR
touch "${RUN}/_OFFLINE_SUCCESS"
