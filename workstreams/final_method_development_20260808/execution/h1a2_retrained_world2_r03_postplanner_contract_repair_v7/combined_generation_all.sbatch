#!/bin/bash
#SBATCH --job-name=h1a2-postplan-v7
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v7/logs/combined-all-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v7/logs/combined-all-%j.err

set -Eeuo pipefail
umask 077
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_postplanner_contract_repair_v7"
UPSTREAM="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_sun_recovery_contract_repair_v5"
SOURCE="$RUN/source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_RECOVERY_SOURCE_SHA256:?source manifest SHA is required}"

EXIT_CODE="$RUN/status/combined_all_exit_code.txt"
test ! -e "$EXIT_CODE"
on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then touch "$RUN/status/combined_all_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT

test "${SLURM_JOB_PARTITION:-}" = gpu
test "${SLURM_CPUS_PER_TASK:-0}" -eq 32
test -f "$RUN/status/preparation_SUCCESS"
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_RECOVERY_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY
"$PYTHON" - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 4 or any("A800" not in name for name in names):
    raise SystemExit(f"expected four A800s, observed {names}")
print({"combined_visible_gpus": names})
PY

echo "STAGE upstream_planner_evidence_audit"
test "$(sha256sum "$UPSTREAM/source/SOURCE_SHA256.txt" | cut -d' ' -f1)" = 29ca85d57d311f455a4f91d83e6d1c4c896d62b47835958dd37719fdd4d9f162
test "$(sha256sum "$UPSTREAM/planner_terminal_report.json" | cut -d' ' -f1)" = 5a9676dcc10a4c4938d29aa39bbe509a4635f08f2215d49f756ea165729ca966
test "$(sha256sum "$UPSTREAM/planner_distribution_deep_audit.json" | cut -d' ' -f1)" = 649eddbc90148f74fb11cab865e59a21282750f134d2345522880754f61a9953
test "$(sha256sum "$UPSTREAM/planner_topology_match_audit.json" | cut -d' ' -f1)" = dce0b7cdcb2ad24b3aa7f418e9ede3e3a32689c61e1a7b3cbe34470ac4120b3a
for index in 0 1 2 3 4; do
  test "$(cat "$UPSTREAM/status/planner_${index}_exit_code.txt")" = 0
  test -f "$UPSTREAM/status/planner_${index}_SUCCESS"
  test ! -e "$UPSTREAM/status/planner_${index}_FAILED"
done
test -f "$UPSTREAM/status/combined_planner_stage_SUCCESS"
test -f "$UPSTREAM/status/planner_assembly_SUCCESS"
touch "$RUN/status/upstream_planner_evidence_SUCCESS"

run_fresh_r03() {
  local -a pids=() labels=()
  local index
  for index in 0 1 2 3; do
    env CUDA_VISIBLE_DEVICES="$index" \
      SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID" \
      SLURM_ARRAY_TASK_ID="$index" \
      SLURM_CPUS_PER_TASK=8 \
      H1_RETRAINED_RECOVERY_SOURCE_SHA256="$H1_RETRAINED_RECOVERY_SOURCE_SHA256" \
      /bin/bash "$SOURCE/body_refine.sbatch" \
      >"$RUN/logs/combined-${SLURM_JOB_ID}-fresh-${index}.out" \
      2>"$RUN/logs/combined-${SLURM_JOB_ID}-fresh-${index}.err" &
    pids+=("$!")
    labels+=("fresh_${index}")
  done
  local failed=0 position rc
  set +e
  for position in "${!pids[@]}"; do
    wait "${pids[$position]}"
    rc=$?
    printf '%s\n' "$rc" > "$RUN/status/combined_${labels[$position]}_child_exit_code.txt"
    echo "STAGE fresh_r03_terminal label=${labels[$position]} rc=$rc"
    if [[ "$rc" -ne 0 ]]; then failed=1; fi
  done
  set -e
  test "$failed" -eq 0
}

echo "STAGE fresh_r03_begin"
run_fresh_r03
touch "$RUN/status/combined_fresh_r03_SUCCESS"

run_h1a2_and_topology_body() {
  local -a pids=() labels=()
  env CUDA_VISIBLE_DEVICES=0,1 SLURM_CPUS_PER_TASK=16 \
    H1_RETRAINED_RECOVERY_SOURCE_SHA256="$H1_RETRAINED_RECOVERY_SOURCE_SHA256" \
    /bin/bash "$SOURCE/h1a2_dlm_control.sbatch" \
    >"$RUN/logs/combined-${SLURM_JOB_ID}-h1a2-b0d1.out" \
    2>"$RUN/logs/combined-${SLURM_JOB_ID}-h1a2-b0d1.err" &
  pids+=("$!")
  labels+=("h1a2_b0d1")
  env CUDA_VISIBLE_DEVICES=2 SLURM_CPUS_PER_TASK=8 \
    H1_RETRAINED_RECOVERY_SOURCE_SHA256="$H1_RETRAINED_RECOVERY_SOURCE_SHA256" \
    /bin/bash "$SOURCE/topology_body.sbatch" \
    >"$RUN/logs/combined-${SLURM_JOB_ID}-topology-body.out" \
    2>"$RUN/logs/combined-${SLURM_JOB_ID}-topology-body.err" &
  pids+=("$!")
  labels+=("topology_body")

  local failed=0 position rc
  set +e
  for position in "${!pids[@]}"; do
    wait "${pids[$position]}"
    rc=$?
    printf '%s\n' "$rc" > "$RUN/status/combined_${labels[$position]}_child_exit_code.txt"
    echo "STAGE h1a2_topology_body_terminal label=${labels[$position]} rc=$rc"
    if [[ "$rc" -ne 0 ]]; then failed=1; fi
  done
  set -e
  test "$failed" -eq 0
}

echo "STAGE h1a2_and_topology_body_begin"
run_h1a2_and_topology_body
touch "$RUN/status/combined_h1a2_and_topology_body_SUCCESS"

run_topology_refiners() {
  local -a pids=() labels=()
  local repeat
  for repeat in 0 1 2 3; do
    env CUDA_VISIBLE_DEVICES="$repeat" \
      SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID" \
      SLURM_ARRAY_TASK_ID="$repeat" \
      SLURM_CPUS_PER_TASK=8 \
      H1_RETRAINED_RECOVERY_SOURCE_SHA256="$H1_RETRAINED_RECOVERY_SOURCE_SHA256" \
      /bin/bash "$SOURCE/topology_refine.sbatch" \
      >"$RUN/logs/combined-${SLURM_JOB_ID}-topology-refine-${repeat}.out" \
      2>"$RUN/logs/combined-${SLURM_JOB_ID}-topology-refine-${repeat}.err" &
    pids+=("$!")
    labels+=("topology_refine_${repeat}")
  done
  local failed=0 position rc
  set +e
  for position in "${!pids[@]}"; do
    wait "${pids[$position]}"
    rc=$?
    printf '%s\n' "$rc" > "$RUN/status/combined_${labels[$position]}_child_exit_code.txt"
    echo "STAGE topology_refine_terminal label=${labels[$position]} rc=$rc"
    if [[ "$rc" -ne 0 ]]; then failed=1; fi
  done
  set -e
  test "$failed" -eq 0
}

echo "STAGE topology_refiners_begin"
run_topology_refiners
touch "$RUN/status/combined_topology_refiners_SUCCESS"

echo "STAGE generation_assembly"
export PYTHONPATH="$SOURCE"
"$PYTHON" "$SOURCE/assemble_generation.py" \
  --run-root "$RUN" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_RECOVERY_SOURCE_SHA256"
test -f "$RUN/status/generation_assembly_SUCCESS"
touch "$RUN/status/combined_all_SUCCESS"
echo "STAGE combined_all_complete postplanner_slurm_job_count=1 total_slurm_jobs_since_v5=2 max_a800=4 post_cells=9 pre_refine_evaluated=false"
