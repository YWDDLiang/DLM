#!/usr/bin/env bash
#SBATCH --job-name=pmtr-cert512
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --output=pmtr-cert512-%j.out
#SBATCH --error=pmtr-cert512-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT}"
readonly COHORT="${PMTR_PREFLIGHT_COHORT:-${ROOT}/cohorts/pmtr_preflight_v2_20260905}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/pmtr_preflight_certify_${SLURM_JOB_ID}"
readonly WORLD_SIZE=4

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR

for path in \
  "${COHORT}/_SUCCESS" \
  "${COHORT}/fit_sources.jsonl" "${COHORT}/fit_pointers.jsonl" \
  "${COHORT}/holdout_sources.jsonl" "${COHORT}/holdout_pointers.jsonl" \
  "${SOURCE}/scripts/certify_pmtr_corruptions.py" \
  "${SOURCE}/scripts/build_pmtr_preflight_data.py"; do
  [[ -f "${path}" ]]
done

"${PYTHON}" - "${COHORT}" "${RUN}/sources.jsonl" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1]); output = pathlib.Path(sys.argv[2])
with output.open("x", encoding="utf-8", newline="\n") as target:
    for name in ("fit_sources.jsonl", "holdout_sources.jsonl"):
        with (root / name).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    target.write(line.rstrip("\n") + "\n")
assert sum(1 for line in output.open(encoding="utf-8") if line.strip()) == 512
PY

IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?four allocated GPUs are required}"
[[ "${#gpu[@]}" -eq "${WORLD_SIZE}" ]]
declare -a child_pids=()
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  env CUDA_VISIBLE_DEVICES="${gpu[${rank}]}" OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/scripts/certify_pmtr_corruptions.py" \
    --input-jsonl "${RUN}/sources.jsonl" \
    --output-dir "${RUN}/cert_rank${rank}" \
    --device cuda --seed 20260905 --max-proposals 4 --batch-size 16 \
    --probe-fraction 0.10 --shard-rank "${rank}" --shard-count "${WORLD_SIZE}" \
    >"${RUN}/logs/cert_rank${rank}.out" 2>"${RUN}/logs/cert_rank${rank}.err" &
  child_pids+=("$!")
done
status=0
for pid in "${child_pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]]

"${PYTHON}" - "${RUN}" "${WORLD_SIZE}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); world=int(sys.argv[2]); rows=[]; manifests=[]
for rank in range(world):
    cell=root/f"cert_rank{rank}"
    manifests.append(json.loads((cell/"manifest.json").read_text()))
    rows.extend(json.loads(line) for line in (cell/"certificates.jsonl").open() if line.strip())
rows.sort(key=lambda row: (int(row["source_row_idx"]), int(row["proposal_index"])))
if len(rows) != 2048 or len({(int(r["source_row_idx"]), int(r["proposal_index"])) for r in rows}) != 2048:
    raise RuntimeError("distributed PMTR certificate merge changed 512x4 support")
with (root/"certificates.jsonl").open("x", encoding="utf-8", newline="\n") as output:
    for row in rows: output.write(json.dumps(row, separators=(",", ":"))+"\n")
report={
    "schema":"pmtr_preflight_certification_final_v1",
    "sources":512,
    "proposals":len(rows),
    "certified_proposals":sum(bool(row["certified"]) for row in rows),
    "sources_with_certificate":len({int(row["source_row_idx"]) for row in rows if row["certified"]}),
    "world_size":world,
    "selection":"none_during_certification_first_certified_in_builder",
    "rank_manifests":manifests,
}
(root/"CERTIFICATION_FINAL.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,sort_keys=True))
PY

for split in fit holdout; do
  env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/scripts/build_pmtr_preflight_data.py" \
    --input-jsonl "${COHORT}/${split}_sources.jsonl" \
    --pointer-jsonl "${COHORT}/${split}_pointers.jsonl" \
    --certificates-jsonl "${RUN}/certificates.jsonl" \
    --output-dir "${RUN}/data_${split}" --seed 20260905 \
    >"${RUN}/logs/build_${split}.out" 2>"${RUN}/logs/build_${split}.err"
done

[[ "$(wc -l <"${RUN}/data_fit/data.jsonl")" -eq 384 ]]
[[ "$(wc -l <"${RUN}/data_holdout/data.jsonl")" -eq 128 ]]
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
