# Proposal and realization in generative materials discovery

## Main research question

> **In generative materials discovery, to what extent do gains in discovery
> yield arise from changing the distribution of material specifications being
> explored, versus improving structural realization conditional on an
> explored specification?**

The paper asks a scientific attribution question, not whether one decoding
trick is universally superior. Its empirical scope is de novo inorganic
crystal generation.

## Crystal instantiation

For this study, a material specification is the composition together with atom
count `N`. The learned Planner determines which specifications are explored.
The Crystal DLM and the fixed continuous refiner determine how each explored
specification is structurally realized.

The remaining rich-Plan fields—lattice family, space-group bucket and volume
bin—are realization hints. They do not redefine the material specification and
are not hard guarantees on the final structure.

```text
learned Planner
  -> composition + N
  -> exact-cardinality Crystal DLM
  -> fixed identity-preserving continuous refiner
```

## Technical contribution 1

> **Specification-compiled exact-cardinality Crystal DLM.** A sampled formula
> is compiled into a typed state of exactly `7+4N` positions. `N` and the
> element multiset are anchored; the masked DLM completes the `6+3N` lattice
> and coordinate tokens.

The standard realization schedule is:

```text
lattice -> all X -> all Y -> all Z
```

Selected state-dependent support is activated only when its prerequisites are
available. The schedule and support are mechanisms inside the executor, not
separate headline contributions.

The historical evidence motivates the complete interface rather than a
single-factor claim: a fixed padded canvas admitted length/template shortcuts,
changing `N` without matching slot occupancy failed, and exact length alone was
not sufficient before composition anchoring and compatible generation support.

The learned Planner is the paper's de novo composition source. The method does
not claim that a learned Planner is the only possible source of a fixed
composition.

## Technical contribution 2

Contribution 2 remains unclaimed until one of the two preregistered candidates
passes its fixed-256 screen and 1,000-attempt confirmation:

1. composition-matched counterfactual Plan grounding for the realization
   route; or
2. difficulty-decomposed self-improvement for the proposal route.

Both are disabled by default in the released H1-A2 path. A failed candidate is
not promoted into the paper claim. If both pass independently, their combined
system is tested once before confirmation.

## Evidence answering the research question

Aggregate S.U.N. is accompanied by:

- the attempted composition distribution;
- broad family, arity, atom-count-bin and all-metal strata;
- Strict and Meta S.U.N. rates with uncertainty in every supported stratum;
- common-support proposal-mix and conditional-realization accounting;
- the complete Planner -> body -> refinement -> evaluation funnel;
- fixed-Plan analysis for realization interventions.

Exact formulas, exact element sets and individual halogens are exploratory
views because their supported sample sizes are too sparse for headline claims.
Evaluator replays are deduplicated and do not count as new generated cohorts.

## Role of continuous diffusion

The inherited continuous model keeps atom count and atom identities fixed while
refining lattice and fractional coordinates. It is a fixed downstream module,
does not read the Plan, and is not claimed as a new algorithmic contribution.

## Frozen result

Until a candidate completes confirmation, the public main-table result remains:

| Entries | Strict S.U.N. | Meta S.U.N. |
|---:|---:|---:|
| 1,000 | 105/1,000 = 10.50% | 488/1,000 = 48.80% |

## Claim boundaries

The paper does not claim:

- universal superiority of DLMs over autoregressive models;
- that `7+4N` alone caused the endpoint gain;
- that rich Plan hints are hard enforced;
- that selected support guarantees complete physical validity;
- that the fixed refiner is a new method;
- that a higher aggregate score alone proves better conditional realization.
