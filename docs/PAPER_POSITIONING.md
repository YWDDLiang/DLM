# Constraint timing and commitment in crystal generation

## Main research question

> **When different crystal-validity checks can only be evaluated after
> different information has been generated, do restricting invalid choices
> whenever the prerequisite information is available and choosing which
> geometric variables are eligible for commitment at each stage affect how
> reliably a model-proposed composition is realized as a periodic crystal?**

Scope:

> **The composition and atom count are held fixed, and the question is
> evaluated over eligible Plans sampled by the learned source.**

In plain language:

> Some crystal errors can only be detected after particular fields have been
> generated. We ask whether restricting an invalid choice once the required
> information is available, and restricting which related geometry fields may
> compete next before committing one selected field per model call, actually
> makes periodic-crystal realization more reliable.

The question does not assume a positive answer. Early restrictions may help,
have no effect, or push probability mass toward worse candidates. A grouped
commitment policy may likewise outperform, match, or underperform a fixed
positional policy.

The policy treatment is the bundle of **group restriction and confidence-
adaptive position selection within each group**. It is not an isolated test of
grouping alone.

## Why this is a scientific ML question

The prerequisite information for selected crystallographic checks appears at
different partial states. This is an objective property of the representation;
whether generation should use that information immediately is an empirical
question. The experimental variables are:

1. whether selected invalid token choices are restricted when their
   prerequisites are already visible; and
2. which unresolved geometry positions are eligible to compete for the next
   commitment at each stage.

The primary outcome is not the pass rate of an individual hard mask. It is the
all-request yield of bodies that are reconstructable and pass the same
independent post-hoc selected checks. Generation, parsing, and reconstruction
failures remain in the denominator.

## Problem evolution

The paper question was built from a sequence of increasingly complete,
falsifiable questions. Each level adds one concept and preserves the earlier
question as a special case.

| Level | Added concept | Question answered | Claim boundary |
|---|---|---|---|
| Q1 | one intervention | Does a duplicate-Z token restriction improve reconstructable-and-duplicate-free yield? | A local causal sub-question, not the paper claim. |
| Q2 | constraint prerequisites | Does the implemented selected-support bundle help when its required information is visible? | Only three implemented checks; no general constraint guarantee. |
| Q3 | learned-Plan scope | Does the Q2 effect hold over the eligible Plan distribution sampled by the learned source, and is it heterogeneous? | The Plan is the statistical unit; this is not an end-to-end effect. |
| Q4 | commitment policy | Does a grouped confidence-adaptive policy differ from a fixed positional policy under one frozen masked checkpoint? | It does not compare DLM with AR or prove an optimal order. |
| Q5 | continuous conversion | Does a fixed identity-preserving refiner improve successful discrete bodies, and does the proposal-stage policy gap persist? | The refiner is a downstream consequence, not the primary mechanism. |

Two proposed branches were explicitly rejected during review:

- hard anchors were not promoted as a causal performance contribution; they
  define the realization task and paired controls; and
- the selected checks were not claimed to derive the complete lattice-to-X-to-
  Y-to-Z order or a universal constraint system.

## Three sub-questions

### 1. Prerequisite-aware selected support

With checkpoint and commitment policy fixed, does restricting a selected
invalid choice when its required information is visible change joint body
realization yield?

The implemented scope is limited to:

- nonzero lattice-length tokens;
- an opportunistic gamma-degeneracy restriction when alpha and beta are
  already visible; and
- a discrete PBC duplicate-Z restriction after the relevant X/Y information
  is visible.

This does not cover minimum atomic distance, approximate continuous overlap,
exact space-group constraints, Plan-volume enforcement, or global physical
satisfiability.

Operationally, the three post-hoc predicates use the same mathematical rules
as the generation-time restrictions but are invoked independently of the mask
state:

- a lattice length must be finite and strictly positive;
- for parsed integer-degree angles, the lattice angle factor
  `1 + 2 cos(a) cos(b) cos(g) - cos(a)^2 - cos(b)^2 - cos(g)^2` must be finite
  and greater than `1e-4`; and
- two active sites are discrete PBC duplicates when all three 100-bin
  fractional-coordinate indices are equal modulo 100, independent of species.

Generation-time masks cannot serve as outcome labels.

### 2. Commitment policy and its interaction with support

Under the same masked checkpoint, Plans, attempt ordering, sampling settings,
number of model calls, and call-indexed random stream, does a group-restricted
confidence-adaptive policy differ from a fixed positional policy? Does the
selected-support effect depend on the policy?

The grouped policy opens lattice fields, then all X, all Y, and all Z, while
using confidence inside each group. The positional control commits unfilled
positions in the fixed representation order. The treatment is therefore the
whole group-restriction plus confidence-selection policy, not grouping alone.

The paired execution contract fixes the policy details:

- both policies skip prefilled count/element anchors and commit exactly one
  unresolved position per denoising call for `6+3N` calls;
- the grouped policy selects the highest current `low_confidence`-remasking
  confidence inside the active group, with lowest position index as the fixed
  tie-break;
- the positional policy uses the unique order `LA, LB, LC, alpha, beta,
  gamma, X1, Y1, Z1, ..., XN, YN, ZN`;
- both arms use the same call-indexed full noise tensors and model-call budget;
  and
- empty token support is a recorded generation failure with no retry,
  backtracking, mask relaxation or replacement.

### 3. Fixed-refiner downstream consequence

For every reconstructable discrete body, how does a fixed identity-preserving
continuous refiner change the same body's structure and physical evaluation?
Is a proposal-stage policy gap preserved, attenuated, or reversed after
refinement?

This is a downstream system question. It is not part of the primary mechanism
and does not make the inherited refiner a contribution.

## Analysis populations and denominators

The documents use three distinct populations:

1. **Source-request population:** every Planner request, including raw outputs
   that cannot form an eligible Plan. This is the end-to-end denominator.
2. **Eligible-Plan mechanism population:** complete Plan records whose parsed
   formula, `N`, elements and counts define a realization task, with
   `1 <= N <= 20` and counts summing to `N`. This is the Plan-level paired
   mechanism scope.
3. **Successful-body conversion population:** every body that completes the
   common typed parse, periodic-Structure construction and graph conversion.
   This is the fixed-refiner pre/post population; unsuccessful bodies remain in
   end-to-end rates but do not have a pre/post structure pair.

These populations must never be given the same unlabeled rate. Hull coverage,
hull-known conditional rates and all-request lower-bound yields are reported
separately.

## Short glossary

- **Plan:** one complete record sampled by the learned source. Formula-derived
  composition and cardinality are hard anchors; lattice, space-group and volume
  fields are soft prompt context.
- **Selected support:** the three implemented token restrictions listed above,
  not a general constraint solver.
- **Commitment policy:** the rule controlling which unresolved positions may be
  submitted after each model call.
- **S.U.N.:** the cohort-level intersection of stable, unique and novel
  candidates; Strict and Meta use different frozen stability thresholds.

The exact Strict/Meta numerical thresholds and StructureMatcher settings are
part of the still-pending public evaluator contract. Until those settings are
published, the headline counts are reference results rather than a complete
standalone evaluation specification.

## Plan-level mechanism estimand

For eligible Plan `P_j`, repeat `r`, support setting `s` and policy `p`, let
`Y_jr(s,p)` be the all-request binary body-realization outcome. With `K`
paired repeats, define the Plan-level cell mean:

```text
mu_j(s,p) = average_r Y_jr(s,p)
```

The paired design contains four cells: selected support on/off crossed with
grouped/positional policy. The support main effect, policy main effect and
interaction are computed within each Plan before averaging over the `J` Plan
records. In particular, the interaction is:

```text
[mu_j(on, grouped) - mu_j(off, grouped)]
- [mu_j(on, positional) - mu_j(off, positional)]
```

Plan records, not individual body repeats, are the statistical units. The
mechanism estimand is conditional on the eligible Plan cohort; source-level
ineligible outputs remain part of the separate end-to-end denominator.

## Implemented factorization

Let `P` be the complete learned Plan record, including formula and soft coarse
fields. Let `A(P)` contain the hard formula-derived anchors: atom count `N` and
the element multiset/counts. Let `G` contain the generated geometry, `B` the
complete discrete body, and `M` the refined crystal.

```text
P ~ p_phi(P)
G ~ p_theta(G | P, A(P), commitment policy, selected support)
B = (A(P), G)
M ~ p_psi(M | B)
```

The refiner reads `B`, not `P`; the final factor is `p_psi(M|B)`, not
`p_psi(M|B,P)`.

The complete state has length `7+4N`:

```text
1 count + 6 lattice/angle + N * (1 element + X + Y + Z)
```

Count and element positions are prefilled in the standard route. The masked
executor freely generates `6+3N` lattice/coordinate tokens. The exact state
size is a deterministic consequence of known `N`, not a novel variable-length
algorithm.

## Why each component appears

### Learned Plan source

The learned source closes the fully de novo loop and defines the actual Plan
distribution. It is not claimed as a new planning algorithm. Ineligible raw
outputs are upstream attrition and are not hidden inside downstream metrics.

### Anchors and typed state

Composition, `N`, and element counts are held constant so every paired arm
solves the same realization task. Soft lattice, space-group, and volume fields
remain prompt context; they are not hard-enforced invariants.

### Masked partial-state executor

The executor keeps unresolved positions masked, conditions on the current
partial state, applies the implemented selected support when its prerequisites
are visible, and permits an explicit commitment policy to choose the next
eligible positions. This makes masked completion a natural concrete answer to
the research question, but not the only theoretically possible answer.

### Fixed continuous refiner

The inherited `model_494` preserves atom count and the ordered atom-type array,
while changing lattice and fractional coordinates. “Identity-preserving” in
this document means only those atom-identity invariants; it does not mean that
the structural prototype or every material property is preserved. The refiner
tests the discrete-to-continuous interface and the downstream persistence of
proposal-stage effects. It is not a new algorithmic contribution and does not
read the Plan.

## Exactly three contributions

1. **Problem formulation.** We formulate crystal realization from
   model-sampled Plans as a partial-state problem in which selected validity
   checks require different prerequisite information and geometry commitment
   is an explicit, testable decision.
2. **Core executor.** We develop a composition-anchored, exact-cardinality,
   typed masked executor with state-conditional selected support and an
   explicitly testable grouped confidence-adaptive commitment policy.
3. **Plan-level paired evaluation protocol.** We specify how to estimate
   selected support, commitment policy, their interaction and heterogeneity
   over eligible Plans, and how to trace downstream conversion under a fixed
   identity-preserving refiner.

The third item is currently a paired evaluation design. It becomes an empirical
contribution only after the strict paired wiring and results are complete.

## Plan-source taxonomy

| Name | Source | Role | Fully de novo |
|---|---|---|---:|
| `A_learned` | learned H1-A2 Plan source | main system and mechanism scope | yes |
| `C_gold / R5-C` | held-out MP-20-derived gold Plan | conditional executor reference | no |
| `C_replay` | frozen generated Plans | downstream replay control | no |

R5-C is neither Planner-free nor a mathematical upper bound. It remains a
conditional reference rather than the source of the fully de novo claim.

## Falsification contract

| Result | Required claim revision |
|---|---|
| Selected support has zero effect | Remove the claim that the implemented restrictions improve realization; retain only the interface. |
| Selected support is harmful | Do not recommend the bundle; report that early restriction distorted the proposal distribution. |
| Grouped and positional policies are equivalent | Remove commitment-policy performance claims; retain only reorderability of the interface. |
| Grouped policy is worse | Do not present it as the preferred policy. |
| Effects vary strongly across Plan strata | Report heterogeneity; do not claim a uniform effect over the Plan domain. |
| The refiner attenuates the proposal-stage gap | Restrict the policy claim to discrete proposals. |
| Support and policy effects are both absent | The primary method claim is unsupported even if the end-to-end score is competitive. |

## Related-work boundary

- DDPD learns which positions to denoise or revise; H1-A2 compares fixed,
  crystal-specific policies and does not reopen committed tokens.
- ADLM learns important anchors; H1-A2's hard anchors are formula-derived and
  are not learned importance predictions.
- DINGO enforces formal regular-language constraints with stronger guarantees;
  H1-A2 implements only three selected crystal checks.
- CrysLLMGen already establishes language-model proposal followed by
  identity-preserving continuous refinement. The hybrid split and refiner are
  not H1-A2 contributions.
- CrystalDiT is a strong unified-generation counterexample; H1-A2 does not
  claim that modular generation is inherently superior.

## End-to-end context

The future paper table retains H1-A2 at `105/1000` Strict S.U.N. and
`488/1000` Meta S.U.N. The locally reproduced CrysLLMGen result is the closest
external hybrid-system context; its exact public comparison contract remains
to be frozen. End-to-end values do not establish the causal effect of selected
support, commitment policy, or refinement.

## Claims explicitly out of scope

- joint species-site and geometry generation;
- DLM superiority over autoregression;
- a universally optimal commitment order;
- support-consistent training or a legal-mass objective;
- backtracking, revision, or reopening committed tokens;
- exact Plan-volume, lattice-family, or space-group enforcement;
- permutation invariance, exact symmetry, or global satisfiability;
- algorithmic novelty of the learned Plan source or `model_494` refiner.

## Current review status

The proposer-reviewer process approved the problem-method-contribution logic
at a concept score of approximately `7/10`. The strongest remaining risk is
empirical: if the three selected checks and the grouped policy do not produce
clear Plan-level effects that remain meaningful after refinement, the work may
be judged a hand-designed decoding heuristic rather than a consequential ML
method.

The remaining implementation gap is narrow but real: the strict positional
control, call-indexed paired randomness, stable Plan/attempt metadata and
uniform pre/post evaluation must be wired before contribution 3 is claimed as
evidence. No checkpoint retraining is required for that mechanism test.

The core H1-A2 method and selected restrictions already exist. What remains
unfinished is the strict causal-comparison and evaluation wiring; this document
does not claim that the paired evidence already exists.
