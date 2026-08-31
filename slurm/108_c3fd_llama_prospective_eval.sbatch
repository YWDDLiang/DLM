#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=c3fd-llama-eval-%j.out
#SBATCH --error=c3fd-llama-eval-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly GEN_RUN="${C3FD_LLAMA_DLM_RUN:?export completed DLM generation run}"
readonly EVAL_RUN="${ROOT}/runs/c3fd_llama_prospective_eval_${SLURM_JOB_ID}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly ASSEMBLE_REFINED_SHA=e4bd9c4eb8c659aaf3fe49f96899e5805a892771826b605a08dab528919e48d4
readonly ASSEMBLE_RAW_SHA=8e9e9380eb2b2ca9e833faacb242e04ad0b34f897db69f5c3fca3cce3557d6ec
readonly FULL_EVAL_SHA=b8f44499072ca8a7ca5e67847ed352e492dd660a8e846ea865d80ca1371db972
readonly RAW_REUSE_SHA=5eb3654cc443327985506f6915728cbbb1ed3ce2821b5a23f1838bb2c6173528
readonly CONFIG_SHA=a79a574209b9ebc4432778dc89ca65b5c18d337d7d7f20408a607f4ab25004ab
readonly DIRECT_METRICS_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c
readonly CACHE_SHA=9d29489fcf61544ed2420fee7f78c6d59a0dca4bc55f989223bb7000f4961c71
readonly PROCESS_EXIT_GRACE_SECONDS=300
readonly PROCESS_TERM_GRACE_SECONDS=30
readonly PROCESS_POLL_SECONDS=5

mkdir "${EVAL_RUN}"
mkdir "${EVAL_RUN}/logs"
printf 'utc\tlabel\tpid\tevent\n' >"${EVAL_RUN}/logs/process_shutdown_events.tsv"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${EVAL_RUN}/ENGINEERING_FAILURE.tsv"
  touch "${EVAL_RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ -f "${GEN_RUN}/_SUCCESS" ]]
command -v setsid >/dev/null
for spec in \
  "${ASSEMBLE_REFINED_SHA}:${ROOT}/scripts/assemble_grounding_repeat.py" \
  "${ASSEMBLE_RAW_SHA}:${ROOT}/scripts/assemble_raw_body_repeat.py" \
  "${FULL_EVAL_SHA}:${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
  "${RAW_REUSE_SHA}:${EVAL_RUNTIME}/exact_raw_reuse.py" \
  "${CONFIG_SHA}:${V3_SOURCE}/CONFIG.json" \
  "${DIRECT_METRICS_SHA}:${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
  "${GT_SHA}:${GT_CSV}" "${CACHE_SHA}:${BASE_RELAX_CACHE}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done

cat >"${EVAL_RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_prospective_eval_v1
routes	F,M
streams	17,18
denominator	256
order	raw_then_refined
planner_and_DLM_failures	preserve_by_sample_idx
evaluator_labels	F:control,M:candidate_labels_only
exact_raw_reuse_pair_scope	within_stream_F_M
exact_raw_reuse_identity	canonical_lossless_structure_sha256
exact_raw_reuse_near_equivalence	false
refined_exact_reuse	false
selection	false
official_query	false
EOF

assemble_cell() {
  local stream="$1" route="$2" label="$3"
  local body="${GEN_RUN}/stream${stream}/${route}/body"
  local refine="${GEN_RUN}/stream${stream}/${route}/refine"
  local dlm_seed refiner_seed
  if [[ "${stream}" == 17 ]]; then dlm_seed=91117; refiner_seed=101117; else dlm_seed=92117; refiner_seed=102117; fi
  "${PYTHON}" "${ROOT}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${body}" --refine-dir "${refine}" \
    --output-dir "${EVAL_RUN}/stream${stream}/${route}/generation" \
    --arm "${label}" --repeat "${stream}" --dlm-seed "${dlm_seed}" \
    --refiner-seed "${refiner_seed}" --denominator 256
  "${PYTHON}" "${ROOT}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${body}" \
    --output-dir "${EVAL_RUN}/stream${stream}/raw_${route}/generation" \
    --seed "${stream}" --dlm-seed "${dlm_seed}" --denominator 256 \
    --source-arm full_axis --arm-label "${label}"
}

for stream in 17 18; do
  assemble_cell "${stream}" F control
  assemble_cell "${stream}" M candidate
done

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
[[ "${#allocated_gpus[@]}" -eq 4 ]]

process_group_alive() {
  local pid="$1"
  kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null
}

write_process_shutdown_diagnostic() {
  local label="$1" pid="$2" completion_artifact="$3" event="$4"
  local diagnostic="${EVAL_RUN}/logs/${label}-process_shutdown.log"
  {
    printf 'event\t%s\n' "${event}"
    printf 'utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'pid\t%s\n' "${pid}"
    printf 'completion_artifact\t%s\n' "${completion_artifact}"
    if [[ -e "${completion_artifact}" ]]; then
      printf 'completion_sha256\t%s\n' "$(sha256sum "${completion_artifact}" | awk '{print $1}')"
    fi
    ps -o pid,ppid,pgid,sid,stat,etime,pcpu,pmem,args --forest \
      --pgroup "${pid}" || true
    if command -v pstree >/dev/null; then
      pstree -ap "${pid}" || true
    fi
  } >>"${diagnostic}" 2>&1
  printf '%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${label}" "${pid}" "${event}" \
    >>"${EVAL_RUN}/logs/process_shutdown_events.tsv"
}

wait_for_completed_process() {
  local pid="$1" label="$2" completion_artifact="$3"
  local completion_seen_at=-1 forced_cleanup=0 rc=0 deadline
  # Active science has no wall-clock timeout. The grace starts only after the
  # process has committed its scientific completion artifact.
  while process_group_alive "${pid}"; do
    if [[ -e "${completion_artifact}" ]]; then
      if (( completion_seen_at < 0 )); then
        completion_seen_at="${SECONDS}"
        write_process_shutdown_diagnostic \
          "${label}" "${pid}" "${completion_artifact}" completion_artifact_seen
      elif (( SECONDS - completion_seen_at >= PROCESS_EXIT_GRACE_SECONDS )); then
        forced_cleanup=1
        write_process_shutdown_diagnostic \
          "${label}" "${pid}" "${completion_artifact}" exit_grace_expired
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        deadline=$((SECONDS + PROCESS_TERM_GRACE_SECONDS))
        while process_group_alive "${pid}" && (( SECONDS < deadline )); do
          sleep "${PROCESS_POLL_SECONDS}"
        done
        if process_group_alive "${pid}"; then
          write_process_shutdown_diagnostic \
            "${label}" "${pid}" "${completion_artifact}" term_grace_expired
          kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
        fi
        break
      fi
    fi
    sleep "${PROCESS_POLL_SECONDS}"
  done
  if wait "${pid}"; then
    rc=0
  else
    rc="$?"
  fi
  if (( forced_cleanup == 1 )); then
    write_process_shutdown_diagnostic \
      "${label}" "${pid}" "${completion_artifact}" "forced_cleanup_exit_${rc}"
    return 0
  fi
  return "${rc}"
}

run_cell() {
  local slot="$1" stream="$2" arm="$3" label="$4"
  local root="${EVAL_RUN}/stream${stream}/${arm}"
  local device="${allocated_gpus[${slot}]}" direct_pid full_pid full_rc direct_rc
  local exact_role="" exact_peer="" exact_shared="" exact_failure=""
  local -a exact_args=()
  if [[ "${arm}" == raw_F || "${arm}" == raw_M ]]; then
    exact_role="${arm#raw_}"
    if [[ "${exact_role}" == F ]]; then exact_peer=M; else exact_peer=F; fi
    exact_shared="${EVAL_RUN}/stream${stream}/raw_exact_reuse"
    exact_failure="${exact_shared}/${exact_role}.FAILED"
    exact_args=(
      --exact-raw-reuse-role "${exact_role}"
      --exact-raw-reuse-peer-generation-jsonl \
        "${EVAL_RUN}/stream${stream}/raw_${exact_peer}/generation/generation.jsonl"
      --exact-raw-reuse-peer-output-dir \
        "${EVAL_RUN}/stream${stream}/raw_${exact_peer}/evaluation/full_reconstructed"
      --exact-raw-reuse-dir "${exact_shared}"
    )
  fi
  mkdir -p "${root}/evaluation"
  cp --reflink=auto "${BASE_RELAX_CACHE}" "${root}/relax_cache.jsonl"
  setsid env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "${root}/generation/generation.jsonl" --gt-csv "${GT_CSV}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${root}/evaluation/direct" --eval-model-name mp20 \
    >"${EVAL_RUN}/logs/s${stream}-${arm}-direct.out" 2>"${EVAL_RUN}/logs/s${stream}-${arm}-direct.err" &
  direct_pid="$!"
  setsid env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 H1_ACTIVE_DENOMINATOR=256 \
    PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${EVAL_RUNTIME}/run_full_reconstructed_eval.py" \
    --arm "${label}" --repeat 0 --generation-jsonl "${root}/generation/generation.jsonl" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${root}/relax_cache.jsonl" \
    --output-dir "${root}/evaluation/full_reconstructed" "${exact_args[@]}" \
    >"${EVAL_RUN}/logs/s${stream}-${arm}-full.out" 2>"${EVAL_RUN}/logs/s${stream}-${arm}-full.err" &
  full_pid="$!"

  if wait_for_completed_process \
    "${full_pid}" "s${stream}-${arm}-full" \
    "${root}/evaluation/full_reconstructed/_SUCCESS"; then
    full_rc=0
  else
    full_rc="$?"
  fi
  if (( full_rc != 0 )); then
    if [[ -n "${exact_failure}" ]]; then
      mkdir -p "${exact_shared}"
      printf 'role\t%s\nexit_code\t%s\n' "${exact_role}" "${full_rc}" \
        >"${exact_failure}"
    fi
    return "${full_rc}"
  fi

  if wait_for_completed_process \
    "${direct_pid}" "s${stream}-${arm}-direct" \
    "${root}/evaluation/direct/report.json"; then
    direct_rc=0
  else
    direct_rc="$?"
  fi
  return "${direct_rc}"
}

run_four() {
  local prefix="$1"
  run_cell 0 17 "${prefix}F" control & p0=$!
  run_cell 1 17 "${prefix}M" candidate & p1=$!
  run_cell 2 18 "${prefix}F" control & p2=$!
  run_cell 3 18 "${prefix}M" candidate & p3=$!
  wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"
}

run_four raw_
run_four ""

for stream in 17 18; do for arm in F M raw_F raw_M; do
  [[ -f "${EVAL_RUN}/stream${stream}/${arm}/evaluation/direct/report.json" ]]
  [[ -f "${EVAL_RUN}/stream${stream}/${arm}/evaluation/full_reconstructed/_SUCCESS" ]]
  [[ -f "${EVAL_RUN}/stream${stream}/${arm}/evaluation/full_reconstructed/worker_cleanup.json" ]]
  "${PYTHON}" -m json.tool \
    "${EVAL_RUN}/stream${stream}/${arm}/evaluation/direct/report.json" >/dev/null
done; done
for stream in 17 18; do for role in F M; do
  [[ -f "${EVAL_RUN}/stream${stream}/raw_exact_reuse/${role}.results.json" ]]
done; done
find "${EVAL_RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${EVAL_RUN}/OUTPUTS.sha256"
trap - ERR
touch "${EVAL_RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${EVAL_RUN}"
