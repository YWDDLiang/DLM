# Restored R5-C Experiment Index

All paths below are relative to `baseline/`.  The launchers are preserved
historical entry points.  They are evidence and implementation references; they
must be adapted to the current one-A800/eight-CPU and single-environment rules
before submission.

## Canonical R5-C

| Purpose | Restored entry |
| --- | --- |
| Exact-length data | `scripts/build_r5_exact_length_sft_data.py` |
| Exact-length training/sampling | `scripts/sample_llada_r5_exact_length.py` |
| Plan/body sampling | `scripts/sample_llada_r5c_plan_body.py` |
| CrysLLMGen refinement | `scripts/refine_dlm_with_crysllmgen.py` |
| CrysLLMGen metrics | `scripts/run_crysllmgen_metrics.py` |
| R5-C gate | `scripts/evaluate_r5c_gate.py` |
| Full-1000 legacy pipeline | `launchers/pre_wyckoff/a800/run_r5c_full1000_sun.sh` |
| Original A100-script comparison | `launchers/pre_wyckoff/a800/run_a100_eval_sun_compare.sh` |

## De Novo Planner Sequence

| Experiment | Restored entry | Historical status |
| --- | --- | --- |
| R5-C DN1-DN5 | `launchers/pre_wyckoff/a800/run_r5c_de_novo_plan_body.sh` | completed negative/diagnostic sequence |
| R5-D11 count fields | `launchers/pre_wyckoff/a800/run_r5d11_countfields_all_direct256.sh` | best clean count-field representation signal |
| R5-D13 strict TraceRL | `launchers/pre_wyckoff/a800/run_r5d13_plan_tracerl_strict_direct256.sh` | chemistry signal, all-metal unresolved |
| H1-A2 rich planner | `launchers/pre_wyckoff/a800/run_h1a2_rich_planner_retrain.sh` | best usable de novo branch |
| H1-A3 joint planner | `launchers/pre_wyckoff/a800/run_h1a3_joint_planner.sh` | epoch1/2 not promoted; epoch3 cancelled |
| Free-geometry executor | `launchers/pre_wyckoff/a800/run_h1_free_geometry_full1000_sun.sh` | default ablation retained for analysis |
| H1-G1 robust exact DLM | `launchers/pre_wyckoff/a800/run_h1g1_robust_exact_dlm.sh` | valid/covered but no S.U.N. gain |
| H2-P1 plain text | `launchers/pre_wyckoff/a800/run_h2p1_plaintext_dlm_proposal.sh` | graph-yield collapse; do not resume |
| H1-A4 joint basin | `launchers/pre_wyckoff/a800/run_h1a4_joint_basin_planner.sh` | completed two epochs; epoch2 hybrid gate passed, but strict/meta did not promote |

H1-A4 terminal values recovered from the A800 run
`20260604_h1a4_joint_basin_planner_clean`:

| Epoch | comp-valid | struct-valid | strict adjusted | meta-like adjusted | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 84.6% | 100.0% | 8.55% | 44.92% | acceptable planner/hybrid, not promoted |
| 2 | 86.4% | 99.9% | 8.35% | 47.32% | hybrid target passed, not promoted |

The historical sequence is therefore complete through H1-A4.  The next work is
a new post-H1-A4 continuation, not a recovery rerun of an unfinished job.

## Canonical Evidence

| Evidence | Path |
| --- | --- |
| Full reproduction report | `evidence/reports/20260531_r5c_full_pipeline_reproducibility_and_technical_evolution.md` |
| Baseline vs R5-C bundle | `evidence/reports/20260531_crysllmgen_baseline_vs_r5c_dlm/` |
| A100/CHGNet evaluator run | `evidence/runs/20260531_2200-a100-eval-sun-mpapi-cache-final2/` |
| Final post-R5-C diagnosis | `evidence/reports/20260604_r5c_vs_next_round_plan_dlm_sun_comparison.md` |

## Current Adaptation Rule

Do not execute the historical launchers unchanged.  In particular, several
request two GPUs or switch to a separate `crysllm` environment.  Reactivated
jobs must:

- request one A800 and no more than eight CPUs;
- run generation, refinement, CrysLLMGen metrics, CHGNet, and the preserved A100
  scripts inside `diff_meets_diff`;
- use new run IDs and output paths;
- avoid retry, per-sample filtering, repair, and reranking;
- record a single terminal report instead of per-poll audit files.
