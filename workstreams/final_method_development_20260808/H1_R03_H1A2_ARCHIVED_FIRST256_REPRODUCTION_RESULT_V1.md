# Archived R03 versus original H1-A2 first256 reproduction

Status: `TERMINAL_COMPLETE_ARCHIVED_CACHE_COVERAGE_LIMITED`

## Outcome

The archived first256 pipeline completed end to end. Slurm job `31963`
finished `COMPLETED 0:0` in `04:10:02` on one A800 with eight CPUs. It reused
the body artifacts produced by job `31931`; all four body artifacts are
byte-identical to the successful 2026-08-02 archive. The downstream repair
only corrected Python import precedence. It did not change the cohort,
seeds, schedules, model-494 checkpoint, 800-step refinement, evaluators, or
scientific denominator.

The two arms are the original H1-A2 `D1` exact-plan control and R03 `D2`
safe-axis candidate. Each contains exactly 256 terminal all-attempt records.
There was no planner resampling, retry, replacement, repair, filtering,
reranking, training, or RL.

| arm | generated | comp/joint valid | structure valid | COV-P | COV-R | density W1 | elements W1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original H1-A2 D1 | 246/256 | 212/256 (82.81%) | 246/256 (96.09%) | 96.09% | 84.31% | 0.7698 | 0.0468 |
| R03 D2 safe-axis | 248/256 | 213/256 (83.20%) | 248/256 (96.88%) | 96.48% | 83.76% | 0.7826 | 0.0702 |

## Archived frozen-cache S.U.N.

The registered primary values below use all 256 attempts as denominator.
They are lower bounds because the archived MP cache does not cover roughly
half of each arm's novel-and-unique chemical systems.

| arm | reconstructed | novel+unique | hull evaluated | hull unknown | strict S.U.N. | meta-S.U.N. |
|---|---:|---:|---:|---:|---:|---:|
| original H1-A2 D1 | 246 | 223 | 122/223 | 101/223 | 12/256 (4.69%) | 70/256 (27.34%) |
| R03 D2 safe-axis | 248 | 224 | 120/224 | 104/224 | 14/256 (5.47%) | 73/256 (28.52%) |

The evaluator's report-only coverage-adjusted estimates are 8.92% strict and
52.01% meta for H1-A2, and 10.54% strict and 54.95% meta for R03. These are
diagnostics, not replacements for fixed-denominator counts.

Paired exact McNemar gives R03-only/control-only `4/2` for strict S.U.N.
(`p=0.6875`) and `14/11` for meta-S.U.N. (`p=0.690038`). The single repeat
therefore does not establish a significant R03 improvement.

## Reproduction interpretation

The archived body/cohort and execution contract were reproduced exactly, and
the complete Direct/refine chain succeeded. The historical absolute S.U.N.
counts were not exactly reproduced under this bundled frozen cache: historical
repeat-0 expectations were H1-A2 `27/256` strict and `133/256` meta, and R03
`28/256` strict and `122/256` meta. The observed deficit is dominated by
`101` and `104` explicit hull-unknown records, not by failed reconstruction or
invalid structures. R03's coverage-adjusted strict estimate (10.54%) is close
to its historical fixed-all value (10.94%), but this agreement is descriptive
because coverage adjustment is not an exact-label result.

This run uses the archived R5-C/A100 frozen-cache evaluator, not the separate
official-clean MP protocol. It therefore demonstrates reproducibility of the
archived generation/refinement machinery while also demonstrating that the
bundled cache snapshot is insufficient for exact recovery of the historical
absolute S.U.N. count. It does not overturn the completed official/current
cache replay, which reproduced the four historical R03 labels exactly.

## Evidence

- Run root: `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1_r03_h1a2_archived_first256_downstream_repair_v4`
- Terminal report SHA-256: `1357008340603c741ac4cf96e0f29114e6d050e593c7aa5503f2a636fa600fd3`
- Results Markdown SHA-256: `11e6d538ce7fb986349ab40a2508d86030f9dd8ce88debda015026506b9689ed`
- Returned evidence archive SHA-256: `a55828140f41fbfb369736037173c31dc0a4b07d7763e1926fbd7cf0a3abca35`
- Local evidence: `evidence/h1_r03_h1a2_archived_first256_downstream_repair_v4/`
