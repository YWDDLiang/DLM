# CTV-DLM contribution stack V1

Date: 2026-08-28

The new stability work extends the frozen Crystal DLM; it does not reset the
paper to a generic structure generator.

## Layer 1 — proposal versus realization

The scientific framing separates:

- which material specifications the proposal model explores; and
- how well a fixed specification is realized as a crystal structure.

Composition and `N` are the fixed specification for the matched realization
question. Cohort uniqueness remains a nonlinear cohort-level endpoint rather
than an additive per-request rate.

## Layer 2 — exact-cardinality typed masked executor

For a composition with `N` atoms, the Crystal DLM instantiates exactly:

```text
7 + 4N positions
= 1 count + 6 lattice + N x (element + X + Y + Z)
```

`N` and the `N` element tokens are anchored, so the free realization problem
contains exactly `6+3N` geometry tokens. This layer includes:

- no padded max-length crystal tail;
- typed per-position legal token support;
- exact atom-count and element-multiset preservation;
- selected lattice-volume and periodic duplicate-coordinate support;
- an explicit, predeclared legal commitment trajectory (the primary run uses
  the current lattice -> all X -> all Y -> all Z exact-axis policy);
- raw-body and fixed-refiner stagewise attribution.

This remains the core DLM method contribution. CTV may not alter its
composition anchors, free-token count, legal support, schedule order or
terminal accounting.

The exact-axis/safe-axis policy is not required by the `7+4N` representation
and is not claimed as a separate contribution. CTV is implemented against
schedule groups so another legal policy could be tested as a later ablation;
the primary comparison freezes one policy to avoid schedule/guidance
confounding.

## Layer 3 — C³FD-v2.5 composition certification

C³FD-v2.5 makes de novo Planner composition generation constructively valid
with a joint family/N/charge/arity/Pauling witness. Its established claim is
composition correctness and diversity, not stability.

It feeds Layer 2 a minimal certified composition. Rich lattice, space-group,
volume and prototype hints are optional ablations rather than requirements of
the primary pipeline.

## Layer 4 — CTV stability-aware generation (candidate)

CTV asks whether the same exact-cardinality DLM can prefer lower-cost legal
geometry tokens *during* masked generation. It adds:

- real forced-token counterfactual continuations at two fixed milestones;
- terminal proxy-energy action values under common random numbers;
- a separate value head that leaves generator logits unchanged;
- normalized full-support discrete guidance;
- compute-matched gamma-zero controls and no terminal reranking.

If the Branch, MatterSim, L6 and L7 gates pass, the DLM contribution becomes
stronger than validity alone: it is an exact-cardinality masked executor whose
trajectory is stability-aware. If they fail, Layers 1--3 remain intact and CTV
is reported as a negative registered extension.

## No double counting

| Evidence | Claim owner |
|---|---|
| Exact `7+4N`, anchored composition, typed decoding | Crystal DLM executor |
| Comp-valid and composition diversity | C³FD-v2.5 |
| Lower terminal cost and improved Strict/Meta S.U.N. | CTV-DLM, only if gated |
| Continuous cleanup at tau800 | inherited model494, not a standalone contribution |
| Proposal/realization and pre/post-stage accounting | evaluation/attribution framework |

The existing public `105/488` result remains unchanged until a complete
replacement pipeline passes its frozen requested-1000 evaluation.
