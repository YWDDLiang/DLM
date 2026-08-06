# H1 R03E Refiner-Repeat Protocol V1

State: `preregistered_before_candidate_refined_metrics`

Date: 2026-08-02

R03E evaluates the only R03 candidate that passed the body-only safety ladder:
`P0+B0+D2-safe-axis`.  Its paired control is the frozen H1 path
`P0+B0+D1`.  No Planner, body checkpoint, Plan, prompt, ordinal, body sample,
body seed, proposal graph, refiner checkpoint, refiner seed, metric, cache, or
threshold is changed.

## Frozen inputs

- Source body run: `20260802_h1_body_safeaxis256_v1`, job 29862.
- Control body attempts:
  `7f486fd61dd4d73ebdf10a065e344a208a7dd274a499f40f9b8a9502cf6908c1`.
- Candidate body attempts:
  `c030f5548e94f1bf4cdaaecf3614417acc998b9c320bb1c6a6436d740767364f`.
- Control proposal graphs:
  `b7506859563b1282fd41cde19d740a5d7fb9f3bafd61ec3665c9718e13469e78`.
- Candidate proposal graphs:
  `023fdd56dd786c788b8219f66d91eea2a9933991c522bbddfc04218ede8a8e8e`.
- Frozen H1 attempt ledger:
  `24295854aac87f3eb9ad7cc293f2bf2d2eb1d8c292b7f05aeaad8348b6665c8f`.
- Refiner: CrysLLMGen `model_494`,
  `573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e`,
  exact 800 reverse steps and effective batch size one.

The raw denominator is 256 for both arms. Planner failures 86 and 211 and all
body failures remain failed attempts. The control has 246 body-success
proposals and the candidate has 248. There is no retry, replacement, repair,
filter, rerank, or MP API call.

## Common repeat ledger

Four independent process realizations are run. A repeat does not create a new
scientific seed: every repeat reuses the exact frozen per-ordinal
`refiner_noise_seed` from the H1 attempt ledger. The repeat ID records the CUDA
process realization only.

The packed arm order is fixed before any R03E refined metric is observed:

| Repeat | First | Second |
|---:|---|---|
| 0 | control | candidate |
| 1 | candidate | control |
| 2 | candidate | control |
| 3 | control | candidate |

Each repeat is one Slurm array element on one A800. Both arms are evaluated
inside that element. Results are never selected, discarded, or replaced by
repeat outcome.

## Endpoints and analysis

All endpoints use the raw 256-attempt denominator in every arm and repeat.

Primary endpoints:

1. direct joint validity;
2. frozen-cache meta S.U.N. at 0.1 eV/atom.

Secondary endpoints:

- generation/refiner completion;
- composition and structure validity;
- strict S.U.N. at 0.0 eV/atom;
- novel, unique-representative, and novel-unique counts.

The terminal report must include every repeat, candidate-minus-control paired
effects, exact McNemar tests, a deterministic hierarchical paired bootstrap
that resamples repeat blocks and ordinals, and sign stability. The pooled
1024-attempt counts are descriptive; they are not treated as 1024 independent
scientific samples.

## Frozen decision rule

`safe_axis_refined_signal_passed=true` only when all of the following hold:

1. all four repeats and both arms complete with exact frozen input identities;
2. every body-success proposal is processed by the frozen 800-step refiner;
3. joint-valid candidate-minus-control is positive in at least three of four
   repeats and its four-repeat mean is positive;
4. meta-S.U.N. candidate-minus-control is non-negative in at least three of
   four repeats and its four-repeat mean is positive;
5. the mean candidate loss is no worse than 1 percentage point for strict
   S.U.N. and no worse than 2 points for structure validity;
6. no new failure class is introduced after the frozen body stage.

This is a diagnostic single-factor signal gate, not formal G3 or promotion.
Failure stops safe-axis as an ICLR headline improvement; it does not alter the
H1 fallback. Passing permits only a separately registered next H1-preserving
single-factor experiment.

`formal_g3=false`, `automatic_promotion=false`,
`automatic_training=false`, and `automatic_downstream=false`.
