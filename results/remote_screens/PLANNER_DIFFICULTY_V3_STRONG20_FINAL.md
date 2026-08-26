# Difficulty-Decomposed Self-Improving Planner strong20 V3 — final two-seed screen

Route-B screen useful: **False**

| Seed | Arm | Planner parsed | Body | Refined | Reconstructed | Direct C/S/J | N/U/N∩U | Hull K/U | Strict (attempt; known) | Meta (attempt; known) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | control | 255/256 | 254/256 | 254 | 254 | 219/254/218 | 222/254/222 | 237/17 | 21/256=8.20%; 21/237=8.86% | 112/256=43.75%; 112/237=47.26% |
| 17 | candidate | 255/256 | 253/256 | 253 | 253 | 222/253/221 | 231/252/230 | 248/5 | 22/256=8.59%; 22/248=8.87% | 107/256=41.80%; 107/248=43.15% |
| 18 | control | 254/256 | 250/256 | 250 | 250 | 219/250/219 | 215/250/215 | 241/9 | 13/256=5.08%; 13/241=5.39% | 104/256=40.62%; 104/241=43.15% |
| 18 | candidate | 254/256 | 253/256 | 253 | 253 | 224/253/224 | 212/253/212 | 251/2 | 15/256=5.86%; 15/251=5.98% | 106/256=41.41%; 106/251=42.23% |

## Pooled 512-attempt comparison

| Arm | Planner parsed | Body/refined/reconstructed | Direct C/S/J | N/U/N∩U | Hull K/U | Strict (attempt; known) | Meta (attempt; known) |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 509/512 | 504/504/504 | 438/504/437 | 437/504/437 | 478/26 | 34/512=6.64%; 34/478=7.11% | 216/512=42.19%; 216/478=45.19% |
| candidate | 509/512 | 506/506/506 | 446/506/445 | 443/505/442 | 499/7 | 37/512=7.23%; 37/499=7.41% | 213/512=41.60%; 213/499=42.69% |

## Pooled deltas (candidate minus control)

- planner_parse_rate: `+0.0000%`
- body_rate: `+0.3906%`
- direct_joint_rate: `+1.5625%`
- novel_rate: `+0.8431%`
- unique_rate: `-0.1976%`
- novel_unique_rate: `+0.6454%`
- strict_known_rate: `+0.3019%`
- meta_known_rate: `-2.5029%`
- strict_attempt_rate: `+0.5859%`
- meta_attempt_rate: `-0.5859%`

## Decision criteria

- strict_direction: `True`
- meta_noninferiority_1pp: `False`
- planner_parse_noninferiority_1pp: `True`
- body_noninferiority_1pp: `True`
- direct_joint_noninferiority_1pp: `True`
- novelty_noninferiority_1pp: `True`
- route_b_screen_useful: `False`

## Interpretation

- Correct weighting produced a coherent positive realization signal: pooled body `+2`, Direct joint `+8`, novel `+6`, N∩U `+5`, and Strict `+3` attempts.
- Strict moved upward in both seeds (`+1` and `+2` attempts). Meta was mixed (`-5` and `+2`) and pooled at `-3/512 = -0.59pp`, which satisfies the all-attempt 1pp non-inferiority margin.
- Hull coverage improved substantially (`478/504 → 499/506` known; unknown `26 → 7`). Because the newly known denominator expanded, Meta among hull-known structures fell `-2.50pp` and failed the preregistered second Meta gate.
- Therefore strong20 V3 is a promising scoped Planner improvement, but not a full positive result under the frozen conjunctive criterion.

## Seed stability and statistics

- seed 17: Strict attempt delta `+0.3906%`; Meta attempt delta `-1.9531%`.
- seed 18: Strict attempt delta `+0.7812%`; Meta attempt delta `+0.7812%`.
- Pooled known-both exact McNemar: Strict candidate-only/control-only=28/28, p=1; Meta candidate-only/control-only=106/116, p=0.5459.

Proposal-mix changes and downstream conversion are reported separately. Because the Planner arms sample different compositions, ordinal pairing is only an end-to-end common-random-number comparison and is not evidence of a fixed-composition realization effect.

This is the corrected strong20 treatment: dedicated replacement weighted sampling, 20% self-improvement probability, and 800 matched control/candidate updates.

The public 105/1000 Strict and 488/1000 Meta headline remains unchanged pending user confirmation.
