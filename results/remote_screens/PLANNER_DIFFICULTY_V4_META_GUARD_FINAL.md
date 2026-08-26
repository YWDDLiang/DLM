# Planner Meta-guard V4 — final two-seed screen

Route-B screen useful: **False**

| Seed | Arm | Planner parsed | Body | Refined | Reconstructed | Direct C/S/J | N/U/N∩U | Hull K/U | Strict (attempt; known) | Meta (attempt; known) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | control | 254/256 | 253/256 | 253 | 253 | 218/253/218 | 225/253/225 | 242/11 | 17/256=6.64%; 17/242=7.02% | 108/256=42.19%; 108/242=44.63% |
| 17 | candidate | 255/256 | 254/256 | 254 | 254 | 225/254/224 | 224/254/224 | 248/6 | 15/256=5.86%; 15/248=6.05% | 123/256=48.05%; 123/248=49.60% |
| 18 | control | 255/256 | 253/256 | 253 | 253 | 218/253/216 | 220/253/220 | 241/12 | 22/256=8.59%; 22/241=9.13% | 119/256=46.48%; 119/241=49.38% |
| 18 | candidate | 255/256 | 255/256 | 255 | 255 | 229/255/229 | 226/255/226 | 253/2 | 16/256=6.25%; 16/253=6.32% | 114/256=44.53%; 114/253=45.06% |

## Pooled 512-attempt comparison

| Arm | Planner parsed | Body/refined/reconstructed | Direct C/S/J | N/U/N∩U | Hull K/U | Strict (attempt; known) | Meta (attempt; known) |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 509/512 | 506/506/506 | 436/506/434 | 445/506/445 | 483/23 | 39/512=7.62%; 39/483=8.07% | 227/512=44.34%; 227/483=47.00% |
| candidate | 510/512 | 509/509/509 | 454/509/453 | 450/509/450 | 501/8 | 31/512=6.05%; 31/501=6.19% | 237/512=46.29%; 237/501=47.31% |

## Pooled deltas (candidate minus control)

- planner_parse_rate: `+0.1953%`
- body_rate: `+0.5859%`
- direct_joint_rate: `+3.7109%`
- novel_rate: `+0.4640%`
- unique_rate: `+0.0000%`
- novel_unique_rate: `+0.4640%`
- strict_known_rate: `-1.8869%`
- meta_known_rate: `+0.3075%`
- strict_attempt_rate: `-1.5625%`
- meta_attempt_rate: `+1.9531%`

## Decision criteria

- strict_direction: `False`
- meta_noninferiority_1pp: `True`
- planner_parse_noninferiority_1pp: `True`
- body_noninferiority_1pp: `True`
- direct_joint_noninferiority_1pp: `True`
- novelty_noninferiority_1pp: `True`
- candidate_strict_at_least_52_of_512: `False`
- candidate_meta_at_least_256_of_512: `False`
- route_b_screen_useful: `False`

## Absolute 10/50 target

- Candidate Strict: `31/512` (`6.05%`), target `>=52/512`.
- Candidate Meta: `237/512` (`46.29%`), target `>=256/512`.

## Seed stability and statistics

- seed 17: Strict attempt delta `-0.7812%`; Meta attempt delta `+5.8594%`.
- seed 18: Strict attempt delta `-2.3438%`; Meta attempt delta `-1.9531%`.
- Pooled known-both exact McNemar: Strict candidate-only/control-only=28/37, p=0.3211; Meta candidate-only/control-only=122/125, p=0.8988.

## Planner proposal and rich-field drift

| Seed | Family TVD | Arity TVD | N-bin TVD | All-metal delta | Volume-bin TVD | Lattice TVD | SG-bucket TVD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 9.25% | 10.37% | 9.28% | -2.83pp | 7.20% | 5.67% | 9.24% |
| 18 | 8.24% | 6.27% | 3.92% | -2.35pp | 6.67% | 9.41% | 7.45% |

The proposal shift is substantive despite `alpha=0`. In seed 17, oxide rises `55→66`, halide falls `25→22`, all-metal falls `57→50`, N=`1–4` falls `38→24`, and N=`13–20` rises `83→107`. In seed 18, oxide rises `44→59`, halide falls `23→16`, all-metal falls `66→60`, N=`1–4` falls `41→34`, and N=`13–20` changes `86→83`. Volume, lattice-system, and space-group-bucket shifts are also nonzero in both seeds. Complete family, arity, N-bin, all-metal, volume, lattice, and SG count maps at Planner, body, reconstructed, and novel-unique stages are retained in the companion JSON.

This matters mechanistically: historical audit rates were strongest for halides, all-metal plans, and small-N plans, while oxides and large-N plans were harder. V4 therefore improved execution validity while moving proposal mass partly toward regions with lower Strict conversion. Setting the explicit proposal-shift coefficient to zero did not preserve the P0 marginal because the replay buffer's own stratum mass differs from P0.

## Reviewer-style assessment

Strengths:

- original P0 is the control; no drifting 800-update control is used;
- sparse Planner failures remain failures in the 256 denominator;
- DLM, refiner, exact-plan policy, temperature, seeds, evaluator, and fresh official MP contract are frozen;
- candidate improves pooled body `506→509`, Direct joint `434→453`, novel-unique `445→450`, hull-known coverage `483→501`, and Meta `227→237`.

Weaknesses and decision:

- Strict drops in both seeds (`17→15`, `22→16`) and pooled `39→31` (`-1.56pp`);
- Meta is seed-unstable (`+15` then `-5`) and pooled `237/512=46.29%` remains 19 successes below the 50% target;
- Strict `31/512=6.05%` remains 21 successes below the 10% target;
- known-both McNemar is non-significant for Strict (`p=0.3211`) and Meta (`p=0.8988`), and two Planner seeds remain a screen;
- proposal-mix drift prevents a fixed-composition realization interpretation.

Therefore V4 is **not** retained as contribution point 2. It is useful diagnostic evidence that non-RL weighted Planner replay can improve realizability and all-attempt Meta yield, but the scalar `2×Meta+Strict` objective does not protect the rarer Strict subset and does not hit the requested absolute operating point.

## Frozen V5 recommendation (non-RL)

The next experiment should change the replay estimator rather than merely increase replay strength:

1. calibrate replay sampling so every `family×arity×N-bin×all-metal` stratum has the original P0 mass;
2. learn field preference only within each preserved stratum, including formula, volume-per-atom, lattice system, and SG bucket;
3. separate Meta and Strict advantages, and permit a Strict bonus only when the local Meta advantage is nonnegative;
4. cap per-stratum influence and report effective sample size, preventing rare easy strata from becoming shortcuts;
5. screen two seeds against P0, then combine the surviving Planner with the grounded DLM only after the Planner-alone gate passes.

This remains weighted supervised replay / preference fitting, not reinforcement learning.

Proposal-mix changes and downstream conversion are reported separately. Because the Planner arms sample different compositions, ordinal pairing is only an end-to-end common-random-number comparison and is not evidence of a fixed-composition realization effect.

V4 uses the original P0 Planner as control and 20% non-RL weighted replay with reward 2×Meta+Strict. Alpha=0 removes the explicit proposal-shift multiplier, but the replay buffer does not match P0 stratum mass, so V4 must not be described as composition-preserving.

The public 105/1000 Strict and 488/1000 Meta headline remains unchanged pending user confirmation.
