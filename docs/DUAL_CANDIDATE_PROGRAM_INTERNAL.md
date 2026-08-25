# Proposal/realization dual-candidate program

## Frozen scientific center

> In generative materials discovery, to what extent do gains in discovery
> yield arise from changing the distribution of material specifications being
> explored, versus improving structural realization conditional on an
> explored specification?

`Specification = composition + N`.  Rich lattice/SG/volume fields are soft
realization hints.  The inherited continuous refiner is fixed and is not a
method contribution.

## Frozen contribution 1

The specification-compiled Crystal DLM instantiates exactly `7+4N` typed
positions, anchors `N` and the element multiset, and generates `6+3N` geometry
tokens with the standard lattice→X→Y→Z schedule and existing selected support.

## Candidate A: counterfactual grounding

- Keep formula, `N`, elements, anion and charge identical.
- Replace lattice+SG+volume atomically with a donor matched on family, arity and
  N-bin.
- Use the same target and corruption mask for factual and counterfactual views.
- Rank length-normalized geometry-token denoising scores with a pairwise
  logistic loss.
- Initialize from frozen B0; inference is unchanged.

## Candidate B: difficulty-decomposed Planner

- Deduplicate evaluator replays before using historical labels.
- Reward is `I(Meta S.U.N.) + I(Strict S.U.N.)`; hull unknown is missing, not
  negative.
- Estimate a cross-fitted, shrunk difficulty baseline from family, arity,
  N-bin and all-metal indicator.
- Factor each training weight into stratum easiness and within-stratum
  advantage; cap weights and preserve a minimum effective sample size.
- Keep original MP-20 Planner data as the dominant anchor.

## Minimal execution

1. Train/screen A and B independently against H1-A2.
2. Use fixed 256 attempts and the standard schedule.
3. If only one passes, confirm that candidate on 1,000 attempts.
4. If both pass, screen exactly one combined cell; confirm it only if the
   interaction is non-adverse.
5. If neither passes, retain H1-A2 and stop method expansion.

The public `105/1000` Strict and `488/1000` Meta result is unchanged until a
candidate completes confirmation.  Failed candidate results remain internal.
