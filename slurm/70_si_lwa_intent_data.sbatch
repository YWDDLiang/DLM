#!/usr/bin/env bash
#SBATCH --job-name=h1a2-si-intent
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-si-intent-%j.out

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly MP20="${H1A2_MP20_DIR:?export H1A2_MP20_DIR}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly OUTPUT="${ROOT}/data/si_lwa_intent_v1_20260830"
readonly RUN="${ROOT}/runs/si_lwa_intent_data_${SLURM_JOB_ID}"
readonly TRAIN_SHA=9b8031cf4ea7bb62709c74735da7ec11d00e367c5eaa05658fad5b5e7a530dde
readonly VALIDATION_SHA=ae7b87064fb63cdd9b2e3a69b4695046edede96c96b652c5aae7fcf195aff398

mkdir -p "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for path in "${MP20}/train.csv" "${MP20}/val.csv" \
  "${ROOT}/scripts/build_si_lwa_intent_data.py"; do
  [[ -f "${path}" ]] || exit 2
done
[[ ! -e "${OUTPUT}" && ! -e "${OUTPUT}.preparing" && ! -e "${OUTPUT}.FAILED" ]] || exit 2
[[ "$(sha256sum "${MP20}/train.csv" | awk '{print $1}')" == "${TRAIN_SHA}" ]] || exit 2
[[ "$(sha256sum "${MP20}/val.csv" | awk '{print $1}')" == "${VALIDATION_SHA}" ]] || exit 2

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
method	SI-LWA-v1 train-only structural-intent label build
train_csv	${MP20}/train.csv
train_sha256	${TRAIN_SHA}
validation_csv	${MP20}/val.csv
validation_sha256	${VALIDATION_SHA}
VPA	train-only Q8 of log(volume/N)
CN	train-only 8 representative CrystalNN effective-CN histograms
fit_seed	82000
workers	48
gpu_jobs_used	0
test_or_holdout_input	unsupported
output	${OUTPUT}
EOF
sha256sum "${MP20}/train.csv" "${MP20}/val.csv" \
  "${ROOT}/scripts/build_si_lwa_intent_data.py" >"${RUN}/inputs.sha256"

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 "${PYTHON}" \
  "${ROOT}/scripts/build_si_lwa_intent_data.py" \
  --train-csv "${MP20}/train.csv" \
  --validation-csv "${MP20}/val.csv" \
  --output-dir "${OUTPUT}" \
  --workers 48 >"${RUN}/builder.out" 2>"${RUN}/builder.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
sha256sum -c <(
  "${PYTHON}" -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "SI_LWA_INTENT_MANIFEST.json").read_text())
for split, digest in sorted(manifest["output_hashes"].items()):
    print(f"{digest}  {root / (split + chr(46) + chr(106) + chr(115) + chr(111) + chr(110) + chr(108))}")
' "${OUTPUT}"
)
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"
