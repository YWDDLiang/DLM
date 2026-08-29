# SGTC-DLM-v1 L7 confirmation contract

This contract is frozen before viewing any SGTC L7 body, Direct, CHGNet, or
official-hull outcome. SGTC L6 authorized exactly one requested-1000
confirmation; it did not authorize checkpoint, seed, temperature, refiner, or
threshold search.

## Fixed matched design

- Plans: `CTV_DLM_L7_PLANS.jsonl`, all `1000` seed18 requests, SHA-256
  `62bf1017b17f696db95b026e7bfe3eed8284a7ea3743332e121c11098e8e46d5`.
- Arms: frozen minimal-spec step696 base, G0 all-MP20 geometry-only
  continuation, and G1 strict-stable-MP20 geometry-only continuation.
- All arms use DLM seed `92117`, temperature `0.7`, exact-axis special-token
  generation, refiner seed `102117`, model494 at tau `800`, and exactly one
  attempt per Plan.
- There is no reranking, replacement, checkpoint selection, or arm selection.
  G0 and G1 are both reported; training/validation NLL does not choose an arm.
- The visible prompt remains the C³FD minimal composition specification. No
  stability, energy, or hull field is visible at generation time.
- Generation and offline evaluation are separate three-A800 jobs, each with
  eight CPUs per GPU. MatterSim remains paused.

## Official evaluation

The three refined cells are evaluated by the existing Direct and full CHGNet
protocols at denominator `1000`. Their union of reconstructed chemical systems
is frozen before one fresh official Materials Project query. Historical caches
are not reused. Unresolved hull systems are missing, never relabeled unstable.

## Terminal gate

G1 is the predeclared SGTC candidate. L7 passes only if all of the following
hold:

1. G1 Strict S.U.N. is at least `100/1000` and Meta S.U.N. is at least
   `500/1000`.
2. Relative to matched base, body and Direct joint rates are each at least
   `-3 pp`; structural uniqueness and novelty are each at least `-5 pp`;
   Strict and Meta stable-to-S.U.N. retention are each at least `-10 pp`.
3. Relative to G0, G1 improves Strict or Meta S.U.N. and the other is no worse
   than `-1 pp`. This preserves the strict-stable-selection mechanism claim.

All pairwise counts, known/unknown accounting, paired rate-difference 95%
intervals, and exact McNemar tests are reported. Threshold attainment is a
benchmark gate, not confidence-bound significance. The existing public
`105/1000` Strict and `488/1000` Meta headline remains unchanged during this
internal confirmation.
