# Historical frozen R03 refined256 re-evaluated with current S.U.N./MP cache

## Interpretation contract

The active H1A2 and current seven-line h1_rich_plan_v1 prompt branch is byte-identical. The actual sampling change is the random-stream/cohort design: legacy used one global RNG stream at seed 17029, while the current study uses stateless ordinals across three independent base seeds and three disjoint first-1000 parse-success cohorts.

This replay does not run Plan, body/DLM, diffusion refinement, or CHGNet. It reuses the byte-frozen refined256 generation ledgers and frozen CHGNet relax-energy caches, then reruns the current exact novelty/uniqueness and MP-hull evaluation against a cohort-complete cache. Therefore old-to-current changes isolate the MP snapshot/completion effect under the same S.U.N. implementation.

Headline S.U.N. uses reconstructed structures (248) exactly as the legacy evaluator. The all-256 rate is a conservative secondary denominator; evaluated-only is diagnostic.

## Per-repeat S.U.N. and meta-S.U.N.

| Repeat | Snapshot | Evaluated | Unknown | strict S.U.N. /248 | strict /256 | meta-S.U.N. /248 | meta /256 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | historical | 218 | 9 | 28/248 = 11.29% | 28/256 = 10.94% | 122/248 = 49.19% | 122/256 = 47.66% |
| 0 | current_cohort_complete | 218 | 9 | 28/248 = 11.29% | 28/256 = 10.94% | 122/248 = 49.19% | 122/256 = 47.66% |
| 1 | historical | 215 | 9 | 31/248 = 12.50% | 31/256 = 12.11% | 123/248 = 49.60% | 123/256 = 48.05% |
| 1 | current_cohort_complete | 215 | 9 | 31/248 = 12.50% | 31/256 = 12.11% | 123/248 = 49.60% | 123/256 = 48.05% |
| 2 | historical | 217 | 9 | 29/248 = 11.69% | 29/256 = 11.33% | 125/248 = 50.40% | 125/256 = 48.83% |
| 2 | current_cohort_complete | 217 | 9 | 29/248 = 11.69% | 29/256 = 11.33% | 125/248 = 50.40% | 125/256 = 48.83% |
| 3 | historical | 218 | 9 | 29/248 = 11.69% | 29/256 = 11.33% | 126/248 = 50.81% | 126/256 = 49.22% |
| 3 | current_cohort_complete | 218 | 9 | 29/248 = 11.69% | 29/256 = 11.33% | 126/248 = 50.81% | 126/256 = 49.22% |

## Full S.U.N. components

| Repeat | Snapshot | Novel | Unique representatives | Novel-unique | Hull evaluated | Hull unknown |
|---:|---|---:|---:|---:|---:|---:|
| 0 | historical | 227 | 248 | 227 | 218 | 9 |
| 0 | current_cohort_complete | 227 | 248 | 227 | 218 | 9 |
| 1 | historical | 224 | 248 | 224 | 215 | 9 |
| 1 | current_cohort_complete | 224 | 248 | 224 | 215 | 9 |
| 2 | historical | 226 | 248 | 226 | 217 | 9 |
| 2 | current_cohort_complete | 226 | 248 | 226 | 217 | 9 |
| 3 | historical | 227 | 248 | 227 | 218 | 9 |
| 3 | current_cohort_complete | 227 | 248 | 227 | 218 | 9 |

## Paired cache-snapshot transitions

| Repeat | Endpoint | Old only | Current only | Discordant | Exact McNemar p |
|---:|---|---:|---:|---:|---:|
| 0 | strict_full_sun | 0 | 0 | 0 | 1 |
| 0 | meta_full_sun | 0 | 0 | 0 | 1 |
| 1 | strict_full_sun | 0 | 0 | 0 | 1 |
| 1 | meta_full_sun | 0 | 0 | 0 | 1 |
| 2 | strict_full_sun | 0 | 0 | 0 | 1 |
| 2 | meta_full_sun | 0 | 0 | 0 | 1 |
| 3 | strict_full_sun | 0 | 0 | 0 | 1 |
| 3 | meta_full_sun | 0 | 0 | 0 | 1 |

## Cache audit

- Historical refined256 union: 224 chemical systems.
- Resolved before new queries: 132.
- Current MP completion queries: 92; all resolved.
- Final cohort cache rows: 224; all populated.
- MP API was not available inside Slurm evaluation jobs.
- Four repeats are CUDA process realizations on the same frozen first256 cohort; pooled 1024 values are descriptive only and are not treated as independent samples.
