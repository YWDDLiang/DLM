# CTV-DLM-v1 final decision

Date: 2026-08-28

Disposition: **NO-GO before L6.** No gamma was selected, no guided L6 was run,
and public `105/488` is unchanged.

## Complete denominator

- Branch train: 128 Plans, 256 states, 2048 forced-action completions;
- Branch validation: 32 disjoint Plans, 64 states, 1024 completions from two
  common-noise continuations per action;
- all 3072 branches parsed, reconstructed and received finite CHGNet terminal
  energy;
- Direct valid was 2040/2048 train and 1024/1024 validation; feasibility
  remained a separate diagnostic and unknown energy was never a negative.

The generator, exact-axis schedule, C3FD-certified composition, model494
tau800, action set and continuation/refiner random keys were frozen before
terminal labels.

## Frozen feature and head audit

The frozen output-head input had dimension 4096 and was projected to 256 using
seed 73017. All 320 states and 202 legal geometry tokens were extracted. The
maximum difference between reproduced and rollout-time selected-action base
probability was `8.88e-16` against a `1e-5` gate, so feature extraction did not
change the generator.

Two heads used disjoint 64-Plan train groups and the frozen architecture and
512-update contract. Validation was not used for fitting or early stopping.

## Correct state-centered result

| Metric | Result | Gate |
|---|---:|---:|
| Plan-bootstrap state-centered Spearman | 0.0353 | 95% LCB > 0 |
| Spearman 95% interval | [-0.0563, 0.1234] | fail |
| Within-state action pairwise AUC | 0.5053 / 1319 comparisons | >0.60; fail |
| Raw two-continuation sign agreement | 0.4915 / 1060 | >0.60; fail |
| Mean supported legal probability mass | 0.1613 | diagnostic |
| Guided-state coverage | 0.0781 | >=0.60; fail |
| Projected fallback | 0.9219 | <=0.40; fail |
| Oxide pairwise AUC | 0.4914 | direction fail |
| N=13--20 pairwise AUC | 0.4708 | direction fail |

Validation Direct had no negative class, so feasibility AUROC was correctly
reported non-estimable rather than assigned a perfect value. Symmetry testing
was not run after the earlier hard gates failed.

The first finalizer version reported a positive absolute-energy Spearman
because composition/state baselines dominated pooled values. Commit `3261893`
fixed the estimand by centering prediction and target within every state before
the Plan bootstrap. The original output is retained as a statistical
implementation error and is not scientific evidence; `gate_centered_v2` is the
only official gate.

## Mechanism conclusion

The heads learned substantial cross-composition absolute-energy information,
but not a reproducible token action advantage. One forced lattice/coordinate
token changes terminal energy, yet its direction is overwhelmed by the
remaining DLM continuation and model494@800 refinement. This rules out the
registered single-token CTV guidance policy; it does not rule out the exact
special-token executor or a parameter-level stable-geometry curriculum.

The frozen fallback therefore moves to SGTC-DLM-v1: N/elements remain anchored
while only real MP-20 lattice and XYZ tokens are masked and supervised, with a
matched all-MP20 versus strict-stable training comparison. RL and MatterSim
remain deferred.
