# Retrained H1-A2 world2 / R03 official S.U.N. recovery

Status: `TERMINAL_COMPLETE_HISTORICAL_BEST_NOT_REPRODUCED`

## Outcome

The requested end-to-end recovery is terminal. The exact epoch-2 H1-A2
retrain completed, five immutable planner cohorts were frozen, nine
post-`model_494` cells completed, and all nine were evaluated under the clean
official Materials Project stability contract. No pre-refine result was
evaluated.

The engineering pipeline is reproducible, but the historical four-repeat R03
best is not reproduced scientifically. On the fixed 256-attempt denominator,
the four topology-matched R03 repeats obtained strict S.U.N. counts
`[13, 15, 12, 15]`, versus the registered historical clean counts
`[28, 32, 30, 30]`. Meta-S.U.N. was `[115, 113, 107, 114]`, versus historical
`[122, 125, 126, 127]`. The strict deficits are `[-15, -17, -18, -15]` and the
meta deficits are `[-7, -12, -19, -13]`.

This rules out a refine-step mismatch as the explanation: every current cell
used CrysLLMGen `model_494` with exactly 800 reverse refinement updates, the
same registered refine800 endpoint used by the historical panel. The stronger
diagnostic is the planner/generation trajectory. Even the retrained seed-17
topology cohort is not byte-identical to the historical seed-17 cohort and has
substantially different formula and chemical-system support.

## Frozen method

- Training: exact H1-A2 epoch-2 retrain, job `31856`, `COMPLETED 0:0`, final
  `eval_loss=0.3186934978`. The registered adapter identity is reused; the
  multi-gigabyte artifact was not rehashed during recovery.
- Planner: historical sampler implementation, implicit torch seed plus rank,
  `raw1200`, world size 2, batch 4, rank concatenation, then frozen first 256
  rows (`sample_idx=0,2,...,510`). Fresh seeds were `52021`, `62023`, `72031`,
  and `82037`; the topology cohort used seed `17`.
- Body: R03 with `D2_SAFE_AXIS`; the control used original H1-A2 B0 with
  `D1_EXACT_PLAN` on the same retrained seed-17 planner cohort.
- Refiner: CrysLLMGen `model_494`, 1,000-step diffusion timetable and exactly
  800 reported reverse updates for all nine cells.
- Evaluation: post-`model_494` only, fixed all-attempt denominator 256; no
  retry, replacement, scientific repair, filter, rerank, training continuation,
  or RL.

## Complete post-only cells

`Joint` is the frozen Direct joint-valid count. Strict and meta use fixed
all-256 denominators. MP-unknown attempts remain failures in these columns.

| cell | generated | comp | structure | joint | COV-P | COV-R | novel+unique | MP evaluated / unknown | strict | meta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh seed 52021 | 246 | 206 | 245 | 205 | 95.3125 | 84.7999 | 225 | 221 / 4 | 16/256 | 111/256 |
| fresh seed 62023 | 242 | 217 | 242 | 217 | 94.1406 | 85.2200 | 217 | 212 / 5 | 13/256 | 106/256 |
| fresh seed 72031 | 250 | 215 | 250 | 215 | 97.2656 | 84.5014 | 224 | 210 / 14 | 7/256 | 106/256 |
| fresh seed 82037 | 245 | 208 | 243 | 206 | 95.7031 | 85.3416 | 221 | 217 / 4 | 18/256 | 110/256 |
| topology repeat 0 | 242 | 207 | 241 | 207 | 94.5312 | 83.3960 | 213 | 207 / 6 | 13/256 | 115/256 |
| topology repeat 1 | 242 | 207 | 241 | 206 | 94.5312 | 84.0150 | 212 | 207 / 5 | 15/256 | 113/256 |
| topology repeat 2 | 242 | 207 | 242 | 207 | 94.5312 | 82.3126 | 209 | 204 / 5 | 12/256 | 107/256 |
| topology repeat 3 | 242 | 207 | 241 | 207 | 94.5312 | 84.3025 | 216 | 211 / 5 | 15/256 | 114/256 |
| H1-A2 B0/D1 control | 244 | 208 | 243 | 207 | 95.3125 | 85.5958 | 212 | 207 / 5 | 16/256 | 113/256 |

The four fresh cohorts have mean strict rate `5.2734%` with 50,000-draw
hierarchical 95% interval `[3.2227%, 7.4219%]`; their mean meta rate is
`42.2852%`, interval `[39.1602%, 45.4102%]`. The four topology processes have
mean strict rate `5.3711%`, interval `[3.9062%, 6.8359%]`, and mean meta rate
`43.8477%`, interval `[40.6250%, 47.1680%]`.

## Same-cohort H1-A2 control

The single H1-A2 B0/D1 control has strict/meta `16/113`. Against the four R03
topology processes, strict R03-minus-H1-A2 differences are `-3`, `-1`, `-4`,
and `-1` counts; exact paired p-values are `0.375`, `1`, `0.34375`, and `1`.
Meta differences are `+2`, `0`, `-6`, and `+1`, with exact p-values
`0.88299591`, `1`, `0.42959051`, and `1`. Because the same left-hand H1-A2
vector is reused four times, no pooled independent-block claim is made.

## Why the historical best was not recovered

All current refinement reports agree on `diffusion_steps=800`; step count is
therefore not the differentiator. The seed-17 planner audit instead shows:

| cohort | parsed / 256 | planner failed | unique formulas | unique chemical systems | duplicate chemical-system rows |
|---|---:|---:|---:|---:|---:|
| historical seed-17 | 252 | 4 | 251 | 247 | 5 |
| retrained seed-17 | 251 | 5 | 250 | 236 | 15 |

The retrained versus historical seed-17 distribution has formula total
variation `0.79365` and chemical-system total variation `0.75790`, while
coarser anion-framework and lattice-system total variations are only
`0.05801` and `0.07450`. Thus broad structural families look similar, but the
actual formulas and chemical systems entering generation are different. The
four fresh cohorts are farther from the historical sample: formula total
variation is `0.98824--0.99608` and chemical-system total variation is
`0.93324--0.96471`.

The best historical result was therefore produced by the exact archived
planner cohort and downstream stochastic trajectory together with R03 D2 and
model-494 refine800. The separate archived first256 replay already proved
that point by recovering R03 strict `28/256` after official E_hull completion.
Recreating only the nominal seed, world-size/batch geometry, R03 schedule,
checkpoint family, and refine800 count is insufficient when the planner bytes
and generated cohort differ.

## Official stability contract

The finalizer used
`mp_api.client.MPRester.get_entries_in_chemsys(..., compatible_only=True)`
with `GGA_GGA+U`. A fresh-clean cache superset covered 1,076 requested systems;
822 were newly queried, of which 798 resolved and 24 remained unresolved. No
historical or August polluted-cache row was reused. For the nine-cell union,
983 systems were required, 952 resolved, and 31 remained explicit
`hull_unknown`. Unknowns are excluded only from columns explicitly labelled
`skip MP unknown`; they are never silently classified as unstable.

Reconstruction, novelty, uniqueness, structures, CHGNet relaxed energies, and
Direct outputs were frozen before official stability replacement. No
credential value is present in source, logs, Slurm state, reports, archives,
or Git.

## Engineering lineage and resource accounting

| identity | Slurm job | terminal state | reusable terminal output |
|---|---:|---|---|
| exact epoch-2 H1-A2 retrain | 31856 | `COMPLETED 0:0` in `02:05:32` | frozen adapter; `eval_loss=0.3186934978` |
| V3 resource-cap cancellation | 31895 | `CANCELLED` while pending in `00:00:00` | none; never allocated |
| V4 CPU-cap cancellation | 31897 | `CANCELLED` while pending in `00:00:00` | none; never allocated |
| V5 planner/body bundle | 31900 | `FAILED 1:0` | all five planner cohorts complete and frozen |
| V7 post-planner repair | 31965 | `FAILED 1:0` | four fresh R03 body256 outputs complete and frozen |
| V8 refine-import repair | 31983 | `COMPLETED 0:0` in `05:51:14` | all nine post-model494 cells complete |
| official combined V4 | 32049 | `FAILED 1:0` in `01:40:55` | all nine preliminary cells and clean cache complete |
| finalization continuation V5 | none | `SUCCESS` | official terminal report and Markdown complete |

Jobs 31895 and 31897 were cancelled before allocation: the former requested
eight GPUs and the latter requested 64 CPUs. They were never run or reused.
V5 failed before body model loading because the immutable body CONFIG lacked
required adapter fields. V7 preserved its completed bodies but failed before
the first refinement because its refiner import path selected the old runtime.
V8 changed import precedence only and completed. Official job 32049 completed
all nine GPU cells, then hit an empty Bash-array `set -u` error during input
freeze. The zero-Slurm continuation reused those nine cells and the completed
clean cache, then performed only input freeze, assembly, cache adoption, and
finalization. Official job 32049 was the technically necessary fourth Slurm ID
after the three generation-lineage IDs; the continuation added no job. Every
allocated/running job stayed at or below four A800 GPUs and 32 CPUs.

## Decision

- Engineering status: complete.
- Scientific status: report all results; historical best not reproduced by
  the retrained world2 route.
- Historical protected result: retain the archived R03 first256 official
  strict `28/256` evidence; do not replace it with any current cell.
- Current retrained route: no promotion and no automatic retry or additional
  generation. Any future attempt must be a new hypothesis about exact planner
  bytes/checkpoint trajectory, not a refine-step sweep.
- RL remains unauthorized.

## Evidence

- V8 generation run:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1a2_retrained_world2_r03_refine_import_contract_repair_v8`
- V8 job: `31983`, `COMPLETED 0:0`.
- V8 source manifest SHA-256:
  `79f1aa3a198e1b9cb7133735f59c75d58902fdfb6e4fd8976d34d561e07852b8`.
- Final official run:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260813_h1a2_retrained_world2_r03_sun_official_v8_finalization_continuation_v5`
- Finalization source manifest SHA-256:
  `e4b8ff9ae369ec449b8ce70b713caf60bc2bdd8a3ef7eeca5cbb7a5c9c312695`.
- Terminal report SHA-256:
  `c74a4fdc0b6f51117316263384166c9ca82672a42514904076e356e9b700527c`.
- Results Markdown SHA-256:
  `204ef06bf2cb1f8a29499f38c29e6e90a2c9e4be580ee5f8ea1f2b27f272fd55`.
- All completion markers are present, `official_results/_SUCCESS` exists,
  and the final immutable continuation contains no failure or submission
  marker.
