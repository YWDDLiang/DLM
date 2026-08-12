#!/bin/bash
#SBATCH --job-name=h1a2-v8-sunfinal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=500G
#SBATCH --time=18:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4/logs/%x-%j.err

set -Eeuo pipefail
umask 077
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_combined_v4"
SOURCE="$RUN_ROOT/source"
UPSTREAM="$PROJECT/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_RETRAINED_SUN_SOURCE_SHA256:?source manifest SHA is required}"

EXIT_CODE="$RUN_ROOT/status/combined_official_exit_code.txt"
test ! -e "$EXIT_CODE"
on_exit() {
  rc=$?
  set +e
  printf '%s\n' "$rc" > "$EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then touch "$RUN_ROOT/status/combined_official_FAILED"; fi
  return "$rc"
}
trap on_exit EXIT

test "${SLURM_JOB_PARTITION:-}" = gpu
test "${SLURM_CPUS_PER_TASK:-0}" -eq 32
test -f "$RUN_ROOT/status/preparation_SUCCESS"
test -f "$RUN_ROOT/status/precompleted_official_cache_SUCCESS"
test -f "$RUN_ROOT/precompleted_official_mp_cache/completion_SUCCESS"
test -f "$RUN_ROOT/status/combined_submission_SUCCESS"
test -f "$UPSTREAM/status/generation_assembly_SUCCESS"
test -f "$UPSTREAM/status/combined_all_SUCCESS"
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)" = "$H1_RETRAINED_SUN_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
for index in $(seq 0 8); do
  test ! -e "$RUN_ROOT/status/preliminary_cell_${index}_exit_code.txt"
  test ! -e "$RUN_ROOT/status/preliminary_cell_${index}_SUCCESS"
  test ! -e "$RUN_ROOT/status/preliminary_cell_${index}_FAILED"
done

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX="$RUN_ROOT/.pycache/combined_official_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY MATERIALS_PROJECT_API_KEY
"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 4 or any("A800" not in name for name in names):
    raise SystemExit(f"expected four A800 GPUs, observed {names}")
if metadata.version("smact") != "3.1.0":
    raise SystemExit("SMACT version changed")
print({"gpu_count": len(names), "gpu": names, "torch": torch.__version__, "smact": metadata.version("smact")})
PY

run_wave() {
  local -a indices=("$@")
  local -a pids=()
  local position index gpu
  for position in "${!indices[@]}"; do
    index="${indices[$position]}"
    gpu="$position"
    echo "STAGE launch_post_cell index=$index gpu=$gpu"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export SLURM_CPUS_PER_TASK=8
      export SLURM_ARRAY_TASK_ID="$index"
      export SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID"
      export H1_RETRAINED_SUN_SOURCE_SHA256
      /bin/bash "$SOURCE/sun_cell.sbatch"
    ) >"$RUN_ROOT/logs/combined-cell-${SLURM_JOB_ID}_${index}.out" \
      2>"$RUN_ROOT/logs/combined-cell-${SLURM_JOB_ID}_${index}.err" &
    pids+=("$!")
  done

  local failed=0 rc
  set +e
  for position in "${!pids[@]}"; do
    wait "${pids[$position]}"
    rc=$?
    echo "STAGE post_cell_terminal index=${indices[$position]} rc=$rc"
    if [[ "$rc" -ne 0 ]]; then failed=1; fi
  done
  set -e
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more frozen post-only cells failed" >&2
    return 1
  fi
}

echo "STAGE post_cells_wave_1 indices=0,1,2,3"
run_wave 0 1 2 3
echo "STAGE post_cells_wave_2 indices=4,5,6,7"
run_wave 4 5 6 7
echo "STAGE post_cells_wave_3 index=8"
run_wave 8
for index in $(seq 0 8); do
  test "$(cat "$RUN_ROOT/status/preliminary_cell_${index}_exit_code.txt")" = 0
  test -f "$RUN_ROOT/status/preliminary_cell_${index}_SUCCESS"
  test ! -e "$RUN_ROOT/status/preliminary_cell_${index}_FAILED"
done
touch "$RUN_ROOT/status/all_preliminary_cells_SUCCESS"

export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32
for mode in audit freeze; do
  args=()
  if [[ "$mode" = audit ]]; then args+=(--audit-only); fi
  echo "STAGE official_inputs_$mode"
  "$PYTHON" "$SOURCE/collect_official_inputs.py" \
    --source-dir "$SOURCE" \
    --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
    --run-root "$RUN_ROOT" "${args[@]}"
done
echo "STAGE preliminary_assembly"
"$PYTHON" "$SOURCE/assemble_preliminary.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$RUN_ROOT"
test -f "$RUN_ROOT/status/preliminary_assembly_SUCCESS"

echo "STAGE adopt_precompleted_official_cache"
"$PYTHON" "$SOURCE/adopt_precompleted_cache.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$RUN_ROOT"
test -f "$RUN_ROOT/status/official_cache_completion_SUCCESS"
test -f "$RUN_ROOT/official_mp_cache/completion_SUCCESS"

echo "STAGE official_finalization"
set +e
"$PYTHON" "$SOURCE/finalize_postonly.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_RETRAINED_SUN_SOURCE_SHA256" \
  --run-root "$RUN_ROOT"
final_rc=$?
set -e
printf '%s\n' "$final_rc" > "$RUN_ROOT/status/finalization_exit_code.txt"
if [[ "$final_rc" -ne 0 ]]; then
  touch "$RUN_ROOT/status/finalization_FAILED"
  exit "$final_rc"
fi
test -f "$RUN_ROOT/status/finalization_SUCCESS"
test -f "$RUN_ROOT/official_results/_SUCCESS"
test -f "$RUN_ROOT/terminal_report.json"
test -f "$RUN_ROOT/RESULTS_COMPLETE.md"
touch "$RUN_ROOT/status/combined_official_SUCCESS"
echo "STAGE combined_official_complete cells=9 pre_refine_evaluated=false slurm_jobs=1 max_a800=4 max_cpu=32"
