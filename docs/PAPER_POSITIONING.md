# Serialization is not commitment order

## Plain-language story

> A formula tells us which atoms a crystal contains and how many there are.
> H1-A2 lets a masked model fill an exactly sized table of quantized periodic
> geometry instead of writing that geometry irreversibly from left to right;
> a continuous diffusion model then polishes the geometry without changing the
> chemistry.

## Research question

> **How should a masked discrete language model realize a model-sampled
> composition as an exact-cardinality periodic crystal body when different
> crystallographic legality checks become evaluable at different partial
> states, without letting serialization order dictate commitment order?**

This is an inductive-bias and interface question. It does not claim that an
autoregressive model is inexpressive or that masked generation is universally
faster, more diverse, or more stable.

## Implemented factorization

Let `P` be the learned source's formula and coarse fields. The formula defines
the anchored state `A(P)`: atom count `N` and the ordered element sequence.
Let `G` contain the fields the standard body executor actually generates: six
quantized lattice/angle fields and `3N` quantized fractional coordinates. Let
`B=(A(P),G)` be the complete discrete proposal and `M` the refined crystal.

```text
P ~ p_phi(P)
G ~ p_theta(G | P, A(P))
M ~ p_psi(M | B)
```

The full state has length `7+4N`: one count field, six lattice/angle fields,
and one element plus X/Y/Z per atom. Because count and element fields are
prefilled in the standard route, the DLM freely generates `6+3N` fields. The
continuous refiner consumes the proposal graph rather than the Plan, so the
implemented last factor is `p_psi(M|B)`, not `p_psi(M|B,P)`.

## What is implemented

- a learned rich-Plan source for the fully de novo route;
- exact-cardinality typed body states indexed by formula-derived `N`;
- hard count and composition anchors;
- field-specific token vocabularies;
- non-prefix masked prediction from the whole current partial state;
- an effective commitment order of lattice, all X, all Y, then all Z;
- lightweight inference-time checks for nonzero lattice lengths, selected
  degenerate-angle combinations, and exact PBC duplicate coordinates once
  X/Y prerequisites are visible;
- an equivariant continuous refiner that preserves atom count and atom types
  while modifying lattice and fractional coordinates.

The learned Plan's coarse lattice, space-group, and volume fields are prompt
conditions. Their actual effect must be established by counterfactual Plan
tests; they are not currently hard-enforced invariants.

## Exactly three contributions

1. **Problem and interface.** We formulate fully de novo generation as
   model-sampled global chemistry followed by composition-anchored,
   exact-cardinality typed realization, while separating learned-Plan inference
   from gold-Plan and frozen-Plan controls.
2. **Crystal DLM executor.** We develop a typed masked completion interface
   over an exact `7+4N` state, with non-prefix context, field-specific support,
   and a dependency-respecting commitment schedule coupled to selected lattice
   and periodic-coordinate legality checks.
3. **Attribution and evaluation.** We separate condition-source effects,
   discrete realization, and continuous refinement through gold-Plan R5-C
   controls, pre/post-refiner analysis, and chemistry-standardized
   decomposition of aggregate stability into composition-mix and
   within-chemistry conversion.

The third item is an evaluation contribution, not a claim that the inherited
refiner or hybrid pipeline is new.

## R5-C in the paper

Three Plan sources have different scientific meanings:

| Name | Source | Role | Fully de novo |
|---|---|---|---:|
| `A_learned` | learned H1-A2 Plan source | main system | yes |
| `C_gold / R5-C` | held-out MP-20-derived gold Plan | conditional executor reference | no |
| `C_replay` | frozen generated Plans | downstream replay control | no |

R5-C is not Planner-free and is not a mathematical upper bound. It is named
**Gold Plan (R5-C; conditional executor reference)**. Historical adjusted R5-C
numbers are legacy context until rerun under the same raw-attempt, evaluator,
and selection contract as `A_learned`.

## Chemistry-aware attribution

Aggregate stability can be written as

```text
P_m(Y=1) = sum_h p_m(h) mu_m(h),
```

where `p_m(h)` is the chemistry mix sampled by method `m`, and `mu_m(h)` is
the outcome rate within chemistry stratum `h`. H1-A2 reports both parts rather
than treating a higher aggregate stability rate as proof of better structural
generation. The primary strata are composition family, arity, atom-count bin,
and all-metal/unary shortcut status.

This analysis is motivated by reward-guided materials work that explicitly
changes elemental distributions or can concentrate on safe regions. It is not
an argument that reinforcement learning only changes composition: fixed-
composition guidance can also improve structure.

## Claims explicitly out of scope

- joint generation of species-site assignments and geometry;
- a novel variable-length DLM—the state is fixed after `N` is sampled;
- support-consistent training or a legal-mass objective;
- violation-guided remasking, revision, or dead-end recovery;
- exact Plan-volume, lattice-family, or space-group enforcement;
- permutation invariance or symmetry equivariance from non-prefix decoding;
- global satisfiability guarantees from local support masks;
- universal quality or speed superiority over autoregression.

These are future method candidates, not current contributions.

## Five-paragraph introduction arc

1. Crystals are serialized for computation, but composition, cardinality,
   lattice, and periodic coordinates form a globally coupled object.
2. Crystal LMs establish text generation; geometry-native models establish
   periodic continuous generation; CrysLLMGen and FlowLLM establish proposal
   followed by refinement. The hybrid split itself is not new.
3. The remaining question is how to realize model-sampled chemistry on an
   exact-size heterogeneous state without making storage order the only
   commitment order.
4. H1-A2 anchors composition and count, completes quantized geometry with a
   typed masked executor and dependency-aware field order, and refines only
   continuous geometry.
5. A learned-vs-gold Plan decomposition, chemistry-standardized outcomes, and
   pre/post-refiner analysis test whether gains arise from condition selection,
   discrete realization, or continuous conversion.

## Suggested abstract

> Crystal structures are commonly serialized as text although composition,
> cardinality, lattice, and periodic coordinates are globally coupled. We
> study composition-anchored crystal realization with a masked language model.
> A learned condition source first samples a formula and coarse structural
> fields. The formula fixes atom count and species, defining an exact
> `7+4N` typed state. A Crystal DLM then completes six quantized lattice fields
> and `3N` fractional-coordinate fields from non-prefix context, using a
> dependency-respecting commitment order and selected state-dependent lattice
> and periodic-coordinate legality checks. An equivariant continuous model
> subsequently refines lattice and coordinates while preserving atom count and
> composition. We distinguish learned de novo Plans, MP-20-derived gold Plans,
> and frozen replay Plans, and decompose aggregate stability into chemistry-mix
> and within-chemistry structural conversion. This formulation treats
> serialization as an interface rather than a mandatory commitment order and
> makes condition quality, discrete proposal quality, and continuous
> refinement separately auditable.
