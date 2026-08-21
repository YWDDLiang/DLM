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
3. **Stage-aware evaluation.** We report the condition source, discrete body,
   and continuous refiner as separate stages, so an end-to-end gain is not
   automatically attributed to the DLM alone.

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

## Evaluation perspective

The final score depends on the chemistry sampled by the Planner, the geometry
proposed by the DLM, and the correction performed by the refiner. These stages
should be inspected separately at a high level. In particular, a stability
gain should not be described as better structure generation without checking
whether the generated chemistry distribution also changed.

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
5. Stage-aware evaluation asks whether observed gains arise from condition
   selection, discrete realization, or continuous refinement.

## What is still missing

At a high level, the project still needs:

- the final release assets and an end-to-end public run;
- stronger matched baselines and a small number of decisive ablations;
- clearer evidence for what the rich Plan contributes;
- broader seed/statistical support and final evaluator documentation.

The exact experiment matrix is intentionally left open for now.

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
> and frozen replay Plans, and evaluate condition generation, discrete
> realization, and continuous refinement separately. This formulation treats
> serialization as an interface rather than a mandatory commitment order and
> makes condition quality, discrete proposal quality, and continuous
> refinement separately auditable.
