#!/usr/bin/env bash
#SBATCH --job-name=spad-basin-rfinal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=spad-basin-rfinal-%j.out
#SBATCH --error=spad-basin-rfinal-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly GENERATION_RUN="${SPAD_BASIN_CLOSURE_GENERATION_RUN:?export completed native generation run}"
readonly RAW_SCREEN_RUN="${SPAD_BASIN_CLOSURE_RAW_SCREEN_RUN:?export completed raw screen run}"
readonly ARRAY_RUN="${SPAD_BASIN_CLOSURE_RELAX_ARRAY_RUN:?export completed two-shard array run}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly FULL="${SOURCE}/eval_runtime/run_full_reconstructed_eval.py"
readonly RESUMABLE="${PROJECT}/reference/a100_eval_sun/eval_sun_resumable.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly GENERATION="${GENERATION_RUN}/closure_ce/stream17/raw_generation/generation.jsonl"
readonly BASELINE_FULL="${ROOT}/runs/spad_prospective_offline_s17_39542/BS/stream17/raw/evaluation/full_reconstructed"
readonly RUN="${ROOT}/runs/spad_basin_closure_common_relax_finalize_${SLURM_JOB_ID}"
readonly CELL="${RUN}/closure_ce_raw_s17/evaluation"

mkdir "${RUN}"
mkdir "${RUN}/logs"
mkdir -p "${CELL}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for path in "${GENERATION_RUN}/_SUCCESS" "${RAW_SCREEN_RUN}/_SUCCESS" \
  "${GENERATION}" "${CACHE}" "${FULL}" "${RESUMABLE}" \
  "${ARRAY_RUN}/_SUCCESS_rank0" "${ARRAY_RUN}/_SUCCESS_rank1" \
  "${ARRAY_RUN}/relax_cache_shards/cache_rank0.jsonl" \
  "${ARRAY_RUN}/relax_cache_shards/cache_rank1.jsonl" \
  "${ARRAY_RUN}/relax_cache_shards/report_rank0.json" \
  "${ARRAY_RUN}/relax_cache_shards/report_rank1.json" \
  "${ARRAY_RUN}/frozen_input_rank0/all_attempts.pt" \
  "${BASELINE_FULL}/_SUCCESS" \
  "${BASELINE_FULL}/attempt_labels_preofficial.jsonl"; do
  [[ -f "${path}" ]]
done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

"${PYTHON}" - "${ARRAY_RUN}/relax_cache_shards" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
reports = [json.load((root / f"report_rank{rank}.json").open()) for rank in range(2)]
assert [report["shard_rank"] for report in reports] == [0, 1]
assert all(report["shard_count"] == 2 for report in reports)
assert sum(report["assigned_occurrences"] for report in reports) == 256
PY

cp --reflink=auto "${CACHE}" "${CELL}/relax_cache.jsonl"
cat "${ARRAY_RUN}/relax_cache_shards/cache_rank0.jsonl" >>"${CELL}/relax_cache.jsonl"
cat "${ARRAY_RUN}/relax_cache_shards/cache_rank1.jsonl" >>"${CELL}/relax_cache.jsonl"
env PYTHONPATH="${SOURCE}/scripts" "${PYTHON}" - \
  "${ARRAY_RUN}/frozen_input_rank0/all_attempts.pt" "${RESUMABLE}" \
  "${CELL}/relax_cache.jsonl" "${RUN}/CACHE_COVERAGE.json" <<'PY'
import json
import pathlib
import sys

from relax_spad_basin_closure_shard import cache_keys, load_frozen_resumable

frozen = load_frozen_resumable(pathlib.Path(sys.argv[2]))
cache = cache_keys(pathlib.Path(sys.argv[3]))
structures, total = frozen.load_generated_structures(pathlib.Path(sys.argv[1]))
assert int(total) == 256 and len(structures) == 256
required = {str(frozen.structure_cache_key(structure)) for structure in structures}
assert not (required - cache)
pathlib.Path(sys.argv[4]).write_text(
    json.dumps(
        {
            "schema": "spad_basin_closure_relax_cache_coverage_v1",
            "attempts": 256,
            "unique_frozen_keys": len(required),
            "all_frozen_keys_present": True,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" \
  "${PYTHON}" "${FULL}" --arm candidate --repeat 0 \
  --generation-jsonl "${GENERATION}" --config "${V3_SOURCE}/CONFIG.json" \
  --r03e-root "${R03E}" --working-relax-cache "${CELL}/relax_cache.jsonl" \
  --output-dir "${CELL}/full_reconstructed" \
  >"${RUN}/logs/full.out" 2>"${RUN}/logs/full.err"

readonly OUTPUT="${CELL}/full_reconstructed"
for path in "${OUTPUT}/_SUCCESS" "${OUTPUT}/summary.json" \
  "${OUTPUT}/attempt_labels_preofficial.jsonl"; do
  [[ -f "${path}" ]]
done

"${PYTHON}" - "${RUN}" "${OUTPUT}" "${BASELINE_FULL}" \
  "${GENERATION_RUN}" "${RAW_SCREEN_RUN}" "${ARRAY_RUN}" <<'PY'
import json
import pathlib
import statistics
import sys

run = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2]).resolve(strict=True)
baseline = pathlib.Path(sys.argv[3]).resolve(strict=True)
generation_run = pathlib.Path(sys.argv[4]).resolve(strict=True)
raw_screen_run = pathlib.Path(sys.argv[5]).resolve(strict=True)
array_run = pathlib.Path(sys.argv[6]).resolve(strict=True)
summary = json.load((output / "summary.json").open(encoding="utf-8"))
shards = [
    json.load((array_run / "relax_cache_shards" / f"report_rank{rank}.json").open())
    for rank in range(2)
]
assert summary["raw_attempts"] == 256
assert summary["generation_succeeded"] == 256
assert summary["reconstructed"] == 256
assert summary["working_relax_cache_sha256_before"] == summary["working_relax_cache_sha256_after"]
candidate_rows = [json.loads(line) for line in (output / "attempt_labels_preofficial.jsonl").open() if line.strip()]
baseline_rows = [json.loads(line) for line in (baseline / "attempt_labels_preofficial.jsonl").open() if line.strip()]
assert len(candidate_rows) == len(baseline_rows) == 256
candidate = {int(row["ordinal"]): row for row in candidate_rows}
control = {int(row["ordinal"]): row for row in baseline_rows}
assert set(candidate) == set(control) == set(range(256))
deltas = []
for ordinal in range(256):
    left, right = candidate[ordinal], control[ordinal]
    if not (left["chgnet_relaxation_known"] and right["chgnet_relaxation_known"]):
        continue
    assert left["chgnet_composition"] == right["chgnet_composition"]
    deltas.append(float(left["chgnet_energy_per_atom"]) - float(right["chgnet_energy_per_atom"]))
paired = {
    "definition": "closure_ce_minus_frozen_BS_CHGNet_relaxed_eV_per_atom",
    "paired_known": len(deltas),
    "median": None if not deltas else float(statistics.median(deltas)),
    "mean": None if not deltas else float(statistics.fmean(deltas)),
    "lower": sum(value < 0 for value in deltas),
    "equal": sum(value == 0 for value in deltas),
    "higher": sum(value > 0 for value in deltas),
}
report = {
    "schema": "spad_basin_closure_common_relax_preofficial_v1",
    "endpoint": "native_closure_ce_raw",
    "stream": 17,
    "attempts": 256,
    "reconstructed": summary["reconstructed"],
    "novel_unique": summary["novel_unique"],
    "chgnet_relaxation_known": summary["chgnet_relaxation_known"],
    "chgnet_relaxation_unknown": 256 - summary["chgnet_relaxation_known"],
    "parallel_relaxation_shards": 2,
    "base_cache_hits": sum(item["base_cache_hits"] for item in shards),
    "unique_missing_relaxed": sum(item["unique_missing_relaxed"] for item in shards),
    "failed_relaxations": sum(item["failed_relaxations"] for item in shards),
    "frozen_evaluator_cache_misses": 0,
    "frozen_cache_coverage": json.load((run / "CACHE_COVERAGE.json").open()),
    "paired_relaxed_energy": paired,
    "candidate_full_reconstructed": str(output),
    "frozen_BS_full_reconstructed": str(baseline),
    "generation_run": str(generation_run),
    "raw_screen_run": str(raw_screen_run),
    "array_run": str(array_run),
    "direct_run": False,
    "official_query": False,
    "model494": False,
    "selection_retry_replacement": False,
}
(run / "PREOFFICIAL_FINAL.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
