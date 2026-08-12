#!/bin/bash
#SBATCH --job-name=h1-r03-h1a2-arch256v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1_r03_h1a2_archived_first256_once_v2/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1_r03_h1a2_archived_first256_once_v2/logs/%x-%j.err

set -Eeuo pipefail
umask 077
ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/final_method_development_20260808/execution/h1_r03_h1a2_archived_first256_once_v2"
V1_SOURCE="$ROOT/workstreams/final_method_development_20260808/execution/h1_r03_h1a2_archived_first256_once_v1"
RUN="$ROOT/runs/20260812_h1_r03_h1a2_archived_first256_once_v2"
SCI_CONFIG="$V1_SOURCE/CONFIG.json"
BODY_SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
REF_SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
REF_CONFIG="$REF_SOURCE/config.json"
LEDGER="$ROOT/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1/data/attempt_ledger.jsonl"
REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
RUNTIME="$REF_SOURCE/runtime"
BODY_SOURCE_SHA=6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f
REF_SOURCE_SHA=7beaf38b7d378ecf8fb31627f195f5fe6095350b60d99882c0a11b39cbf211a4
: "${ARCHIVED_ONCE_V2_SOURCE_SHA256:?v2 wrapper source manifest SHA is required}"

failure_marker() {
  code=$?
  printf '%s\n' "$code" > "$RUN/status/exit_code"
  touch "$RUN/status/PIPELINE_FAILURE"
  exit "$code"
}
trap failure_marker ERR

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]]; then
  echo "resource contract changed" >&2
  exit 2
fi
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')" = "$ARCHIVED_ONCE_V2_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt > "$RUN/status/job_wrapper_source_sha_check.txt")

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$BODY_SOURCE/runtime:$BODY_SOURCE:$RUNTIME:$REF_SOURCE"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CRYSLLMGEN_METRICS_NUM_CPUS=1
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

python - <<'PY'
import json, os, torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise RuntimeError(f"expected exactly one A800, observed {names}")
print(json.dumps({"gpu": names, "torch": torch.__version__, "job": os.environ["SLURM_JOB_ID"]}, sort_keys=True))
PY

cd "$ROOT"
echo "STAGE archived_body_pair_start"
python "$BODY_SOURCE/run_schedule256.py" --config "$BODY_SOURCE/CONFIG.json" --output-dir "$RUN/body"

echo "STAGE archived_body_byte_identity_gate"
python "$V1_SOURCE/verify_body_gate.py" \
  --config "$SCI_CONFIG" --body-dir "$RUN/body" --output "$RUN/status/body_identity_gate.json"
touch "$RUN/status/BODY_IDENTITY_SUCCESS"

for ARM in control candidate; do
  if [[ "$ARM" == control ]]; then
    BODY_ATTEMPTS="$RUN/body/control_body_attempts.jsonl"
    PROPOSAL_GRAPHS="$RUN/body/control_proposal_graphs.pt"
  else
    BODY_ATTEMPTS="$RUN/body/candidate_body_attempts.jsonl"
    PROPOSAL_GRAPHS="$RUN/body/candidate_proposal_graphs.pt"
  fi
  ARM_ROOT="$RUN/arms/$ARM"
  REFINE_DIR="$ARM_ROOT/refinement"
  GENERATION_DIR="$ARM_ROOT/generation"
  EVALUATION_DIR="$ARM_ROOT/evaluation"

  echo "STAGE arm=$ARM model494_refine800"
  python "$REF_SOURCE/refine_safeaxis_arm.py" \
    --arm "$ARM" --repeat 0 --config "$REF_CONFIG" \
    --body-attempts "$BODY_ATTEMPTS" --proposal-graphs "$PROPOSAL_GRAPHS" \
    --attempt-ledger "$LEDGER" --checkpoint "$REFINER" \
    --checkpoint-sha256 573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e \
    --output-dir "$REFINE_DIR" --crysllmgen-dir "$RUNTIME/reference/crysllmgen"

  echo "STAGE arm=$ARM finalize_all_attempt"
  python "$REF_SOURCE/finalize_generation.py" \
    --arm "$ARM" --repeat 0 --config "$REF_CONFIG" --source-dir "$REF_SOURCE" \
    --source-manifest-sha256 "$REF_SOURCE_SHA" --refinement-dir "$REFINE_DIR" \
    --output-dir "$GENERATION_DIR"
  test -f "$GENERATION_DIR/_SUCCESS"
  mkdir "$EVALUATION_DIR"

  echo "STAGE arm=$ARM crysllmgen_direct"
  python "$RUNTIME/scripts/a800/run_crysllmgen_metrics.py" \
    --generation-jsonl "$GENERATION_DIR/generation.jsonl" \
    --gt-csv reference/crysllmgen/data/mp_20/test.csv \
    --snapshot-root "$RUNTIME/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "$EVALUATION_DIR/crysllmgen_metrics" --eval-model-name mp20

  echo "STAGE arm=$ARM archived_frozen_cache_sun"
  python "$RUNTIME/scripts/a800/run_crysllmgen_a100_sun.py" \
    --generation-jsonl "$GENERATION_DIR/generation.jsonl" --expected-attempts 256 \
    --eval-sun-py reference/a100_eval_sun/eval_sun.py \
    --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
    --train-csv reference/crysllmgen/data/mp_20/train.csv \
    --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
    --mp-hull-cache data/a100_eval_sun_cache/mp_hull_entries_cache_merged_slim_plus_mpapi.jsonl \
    --chgnet-relax-cache data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl \
    --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
    --chgnet-runtime-checkpoint /public/home/jiaosz/miniconda3/envs/diff_meets_diff/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
    --base-source-bundle-sha256 "$BODY_SOURCE_SHA" --execution-patch-sha256 "$REF_SOURCE_SHA" \
    --output-dir "$EVALUATION_DIR/r5c_a100_sun"

  echo "STAGE arm=$ARM validate_all_attempt"
  python "$REF_SOURCE/validate_evaluation.py" \
    --arm "$ARM" --repeat 0 --config "$REF_CONFIG" --source-dir "$REF_SOURCE" \
    --source-manifest-sha256 "$REF_SOURCE_SHA" --generation-dir "$GENERATION_DIR" \
    --output-dir "$EVALUATION_DIR"
  test -f "$EVALUATION_DIR/_SUCCESS"
  touch "$RUN/status/${ARM}_SUCCESS"
done

echo "STAGE assemble_terminal"
python "$V1_SOURCE/assemble_once.py" \
  --config "$SCI_CONFIG" --run-root "$RUN" \
  --output-json "$RUN/terminal_report.json" --output-md "$RUN/RESULTS_COMPLETE.md"
touch "$RUN/_SUCCESS" "$RUN/status/PIPELINE_SUCCESS"
printf '0\n' > "$RUN/status/exit_code"
trap - ERR
echo "STAGE complete"
