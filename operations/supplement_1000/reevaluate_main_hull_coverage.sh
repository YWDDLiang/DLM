#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
MAIN="${SPAD_MAIN_RUN:?}"
COHORT="${SPAD_MAIN_COHORT_RUN:?}"
OFFICIAL="${SPAD_COHORT_OFFICIAL_RUN:?}"
OPS="$ROOT/experiments/periodic_self_repair_20260906/supplement_1000/operations"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
CONFIG=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json
CACHE="$OFFICIAL/official_mp_cache"
FIX="$MAIN/official-coverage-repair"
test -f "$MAIN/_SUCCESS"
test -f "$OFFICIAL/QUERY_SUCCESS"
test -f "$CACHE/completion_SUCCESS"
mkdir -p "$FIX"
export PYTHONPATH="$SOURCE/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" "$OPS/check_hull_coverage.py" --plans "$COHORT/cohort/plans_for_dlm.jsonl" --cache "$CACHE" \
  >"$FIX/COHORT_COVERAGE.json"
for endpoint in native tau800; do
  if ! test -f "$MAIN/$endpoint-evaluation-hull-complete/_SUCCESS"; then
    "$PY" "$SOURCE/scripts/evaluate_programmed_paths.py" \
      --paths-jsonl "$MAIN/$endpoint/paths.jsonl" --labels-jsonl "$MAIN/$endpoint-labels/labels.jsonl" \
      --frozen-config "$CONFIG" --official-cache "$CACHE" \
      --selection-json "$MAIN/selection/SELECTION_FINAL.json" \
      --output-dir "$MAIN/$endpoint-evaluation-hull-complete" --expected-requests 1200 \
      --endpoint "$endpoint" --cohort-role independent_main --policy-stage final \
      >"$FIX/$endpoint.out" 2>"$FIX/$endpoint.err"
  fi
done
"$PY" - "$MAIN" "$FIX" "$CACHE" <<'PY'
import hashlib, json, pathlib, sys
main, fix, cache = map(pathlib.Path, sys.argv[1:])
read = lambda p: [json.loads(line) for line in p.open() if line.strip()]
coverage = json.loads((fix/'COHORT_COVERAGE.json').read_text())
assert coverage['coverage_accounted'] is True
selection = json.loads((main/'selection/SELECTION_FINAL.json').read_text())
chosen = selection['selected_source_ordinals']
assert len(chosen) == 1000 and chosen == sorted(set(chosen))
unchanged = ('sample_idx','trajectory_id','group_id','reconstructed','native_execution_success',
             'endpoint_execution_success','novel','unique_representative','novel_unique',
             'terminal_verified','terminal_status','raw_energy_eV_atom','terminal_energy_eV_atom',
             'gap_eV_atom','raw','terminal','actual_relaxation_steps')
report = {'coverage_accounted': True, 'requests': 1200, 'selected': 1000, 'resampled': False,
          'labels_reused': True, 'old_reports_preserved': True, 'non_hull_columns_unchanged': True,
          'official_cache': str(cache), 'cohort_coverage': coverage, 'endpoints': {}}
for endpoint in ('native','tau800'):
    old, new = main/f'{endpoint}-evaluation', main/f'{endpoint}-evaluation-hull-complete'
    a, b = read(old/'attempt_results.jsonl'), read(new/'attempt_results.jsonl')
    assert len(a) == len(b) == 1200
    assert all(all(x[k] == y[k] for k in unchanged) for x,y in zip(a,b))
    final = json.loads((new/'EVALUATION_FINAL.json').read_text())
    assert final['counts']['requests'] == 1200
    assert final['conditional_1000']['counts']['requests'] == 1000
    assert final['conditional_1000']['selection']['selected_source_ordinals'] == chosen
    assert final['hull_statuses'].get('official_cache_not_covered', 0) == 0
    report['endpoints'][endpoint] = {'directory': str(new), 'counts': final['counts'],
        'conditional_1000_counts': final['conditional_1000']['counts'],
        'hull_statuses': final['hull_statuses'],
        'source_paths_sha256': hashlib.sha256((main/endpoint/'paths.jsonl').read_bytes()).hexdigest(),
        'source_labels_sha256': hashlib.sha256((main/f'{endpoint}-labels'/'labels.jsonl').read_bytes()).hexdigest()}
(fix/'REPAIR_FINAL.json').write_text(json.dumps(report,indent=2)+'\n')
(fix/'_SUCCESS').touch()
print(json.dumps(report))
PY
