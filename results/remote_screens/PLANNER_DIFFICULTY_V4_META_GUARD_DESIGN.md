# Planner V4: Meta guard

## Audit finding

The strong20 V3 treatment improved body, Direct, novelty and Strict, but its
800-update matched control drifted away from the original H1-A2 Planner toward
more oxide, larger-N Plans. V3 also used an equal Meta/Strict reward, which gives
a strict-positive example twice the reward of a Meta-only example.

Historical hull-known attempts contain 576/1219 Meta and 94/1219 Strict
positives. Within the frozen family×arity×N-bin×all-metal strata, a
`2×Meta + Strict` reward yields an approximately 84.3% Meta and 23.7% Strict
weighted buffer before the 20% replay mixture.

## Single V4 treatment

- control: the original H1-A2 rich-Plan checkpoint P0, with no extra SFT;
- candidate: P0 plus 400 optimizer updates;
- replay probability: 20%, using the corrected dedicated weighted sampler;
- reward: `2×I(Meta S.U.N.) + I(Strict S.U.N.)`;
- proposal-shift coefficient: zero;
- within-stratum advantage coefficient: one;
- seeds: Planner 17/18, sampling 17017/18018;
- 256 rich Plans per arm and seed.

This version targets within-measured-stratum exact-formula/coarse-Plan residuals
while preserving the original Planner as the absolute baseline. It does not
claim that 10% Strict and 50% Meta are guaranteed; those correspond to at least
52 and 256 successes per 512 attempts, versus V3's 37 and 213.
