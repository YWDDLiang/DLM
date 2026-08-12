# Archived H1-A2 vs R03 first256 — one-shot reproduction

- Cohort: archived P0 first256; no planner resampling.
- Body: original H1-A2 B0, shared seeds; fresh body artifacts passed the archived byte-SHA gate.
- Arms: original H1-A2 D1 exact-plan and R03 D2 safe-axis.
- Refiner: model_494, 800 reverse steps, repeat 0 only.
- S.U.N.: archived frozen-cache protocol, denominator = all 256 generation attempts.

| Arm | Generated | Comp-valid | Struct-valid | Joint-valid | Novel | Unique | Novel+unique | strict S.U.N. | meta-S.U.N. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_H1A2_DLM | 246/256 | 212/256 | 246/256 | 212/256 | 223/256 | 246/256 | 223/256 | 12/256 (4.69%) | 70/256 (27.34%) |
| R03_safe_axis_DLM | 248/256 | 213/256 | 248/256 | 213/256 | 224/256 | 248/256 | 224/256 | 14/256 (5.47%) | 73/256 (28.52%) |

## Paired exact McNemar

| Endpoint | H1-A2 only | R03 only | Discordant | Two-sided exact p |
|---|---:|---:|---:|---:|
| strict S.U.N. | 2 | 4 | 6 | 0.6875 |
| meta-S.U.N. | 11 | 14 | 25 | 0.690038 |

Full evaluator counts/rates and artifact SHA-256 identities are retained in `terminal_report.json`.
