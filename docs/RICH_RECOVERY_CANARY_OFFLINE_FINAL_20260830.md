# Rich-recovery development canary: offline terminal result

Date: 2026-08-30

This report records the development-only matched canary on the frozen C3FD
seed-19 cohort. It is not an official MP evaluation and it is not a prospective
paper result. All rates retain the requested denominator of 256 and preserve
failed body rows by sample index.

## Arms and estimands

- `M0`: current minimal prompt plus minimal DLM.
- `RCF`: historical rich-compatible DLM plus a jointly permuted
  `(metric lattice, separately supervised SG, volume)` tuple.
- `R0`: the same rich-compatible DLM plus the composition-aligned predicted
  tuple.
- `R0-RCF` estimates alignment of the three soft fields within the frozen rich
  package. `R0-M0` is a package comparison, not a single-field Planner effect.

## Terminal evidence

| Stage | Stream | M0 Direct | RCF Direct | R0 Direct |
|---|---:|---:|---:|---:|
| raw | 17 | 150/256 | 118/256 | 111/256 |
| raw | 18 | 167/256 | 119/256 | 110/256 |
| refined | 17 | 254/256 | 248/256 | 246/256 |
| refined | 18 | 254/256 | 247/256 | 252/256 |

Composition-paired, stream-averaged continuous effects (candidate minus
reference; lower energy is better):

- raw `R0-M0`: **+0.911996 eV/atom**, 95% composition-bootstrap CI
  `[+0.552966, +1.281389]`, fraction lower `0.3792`;
- raw `R0-RCF`: `+0.782252 eV/atom`, CI
  `[-0.041234, +2.124605]`, fraction lower `0.4672`;
- refined `R0-M0`: `+1.292 meV/atom`, CI
  `[-6.631, +9.993]`, fraction lower `0.5144`;
- refined `R0-RCF`: `-5.784 meV/atom`, CI
  `[-19.971, +5.977]`, fraction lower `0.4850`.

The offline job completed all 12 raw/refined cells in 7,134 seconds and used
11.8900 A800-hours. Accounting and sample-index preservation passed. The first
finalizer invocation failed before output because a dynamically imported
dataclass module was not registered in `sys.modules`; commit `b11c2c7` fixed
that Python 3.10 engineering issue, and the parameter-identical finalization
then succeeded. No scientific generation or evaluation was repeated.

## Interpretation and route decision

The current rich-compatible checkpoint/package does **not** recover raw DLM
quality on this exact-composition cohort. Its aligned arm loses 39 and 57 raw
Direct structures in streams 17 and 18 relative to `M0`, and its paired raw
energy is substantially worse with a confidence interval entirely above zero.
The refiner restores near-saturated Direct validity but leaves no replicated
energy advantage over `M0`. The noisy refined `R0-RCF` direction is insufficient
to claim that predicted soft-field alignment improves stable DLM generation.

This is a negative result for restoring the historical rich package as the
main Stable-DLM executor. It does not establish that structural context is
useless: the package changes checkpoint, prompt schema, and conditioning
distribution together, and its predicted fields have only 60--70% marginal
accuracy. The development result therefore supports retaining the corrected
rich Planner as an interface diagnostic while selecting the predeclared
minimal-DLM, same-composition continuous listwise plus raw-safety route for the
single prospective experiment. No rich-conditioned candidate-pool training is
opened from this result.

Immutable remote artifacts:

- generation: `runs/rich_recovery_generation_final_20260830_v1`;
- offline evaluation: `runs/rich_recovery_canary_eval_38420`;
- offline final: `runs/rich_recovery_offline_final_20260830_v1`.

