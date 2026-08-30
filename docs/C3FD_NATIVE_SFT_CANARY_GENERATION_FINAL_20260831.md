# C3FD-native SFT canary generation final

Date: 2026-08-31

Slurm job: `38745`

Status: **SUCCESS**

The four fixed development-canary cells completed without retry, reranking,
replacement, policy selection, or official query. Job38745 finished
`COMPLETED 0:0` in `01:00:21` on `4 A800 / 32 CPU`, for `4.0233 A800-hours`.

The frozen cohort contains 128 MP20-train and 128 MP20-standard-validation
unique exact compositions. Stream17 uses C3FD Planner seed17; stream18 uses
Planner seed18. Both fresh DLM policies are evaluated in both streams with
temperature0.7, exact-axis generation, and model494 tau800 refinement.

| Planner/stream | Policy | Requested | Parsed/graphs/refined | Missing |
|---:|---:|---:|---:|---:|
| 17 | 82017 | 256 | 248/248/248 | 8 |
| 17 | 82018 | 256 | 252/252/252 | 4 |
| 18 | 82017 | 256 | 254/254/254 | 2 |
| 18 | 82018 | 256 | 254/254/254 | 2 |

Aggregate body execution is `502/512 = 98.05%` for policy82017 and
`506/512 = 98.83%` for policy82018. All parsed graph sample indices are unique
and preserved exactly by refinement. Fixed-denominator failures remain attached
to their original sample indices.

This is strong evidence that fresh teacher-rich SFT restores execution under
the C3FD-predicted V2 interface. It is not yet an energy-stability or S.U.N.
result. The next step is the single raw-first, then refined offline evaluation.

Run report SHA-256:
`d25baacbab66238a3380632674093644b71eb7195a7690605b0cd652fb92ce5f`.
Positive archive:
`archive/native_sft/canary_generation_success_38745/_ARCHIVE_SUCCESS`.
