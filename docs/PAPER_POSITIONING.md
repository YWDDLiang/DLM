# From serialization to completion

## One-sentence thesis

> Fully de novo crystal generation must generate both what to explore and how
> to realize it. H1-A2 first samples an underdetermined global Plan, then forms
> a compatible discrete crystal by typed non-prefix completion, and finally
> refines its geometry in continuous periodic space.

H1-A2 realizes this thesis with a learned prior over compact Plans, an
exact-cardinality masked discrete language model, and an equivariant continuous
diffusion refiner. A crystal may be serialized left to right, but that storage
order need not determine when its variables are generated.

## The small observation and the larger principle

The small observation is that changing the order of crystal tokens does not
change the crystal. The larger principle is:

> **Serialization order is an interface choice, not a scientific dependency
> order.**

This distinction matters when a scientific object has four properties:

1. a global condition is available before local details are known;
2. the number of local variables is part of that condition;
3. discrete fields constrain one another across distant sequence positions;
4. a continuous relaxation stage follows the discrete proposal.

Crystals are a concrete instance: composition and atom count are global;
species, lattice fields, and sites are mutually constrained; coordinates are
periodic; and final geometry is continuous. H1-A2 uses this instance to study a
broader modeling question without claiming that every structured domain must
use the same architecture.

## Starting from CrysLLMGen

CrysLLMGen established an important division of labor: a language model is
strong at proposing discrete chemistry, while an equivariant diffusion model
is strong at refining lattice and coordinates. It retains the proposed atom
types and sends geometry to a continuous refiner.

H1-A2 accepts that decomposition and asks two next questions:

> How should a fully de novo model first sample global chemistry and
> cardinality, and should the resulting discrete crystal realization still be
> generated as an irreversible text prefix?

Our answer factorizes the full de novo distribution. A learned Planner samples
an underdetermined global problem; a masked model completes an
exact-cardinality typed crystal state; and the continuous refiner receives a
discrete structural hypothesis whose composition and cardinality are fixed.
The novelty claim is therefore not “an LLM plus diffusion,” and not a backbone
swap in isolation. It is a **Plan-generation-to-realization interface** that
separates global sampling, discrete commitment order, and continuous physical
refinement.

## Research question

> How can fully de novo crystal generation be factorized into learning a prior
> over underdetermined global Plans, sampling compatible variable-cardinality
> discrete realizations through non-prefix masked completion, and refining
> those realizations in continuous periodic geometry?

This question contains two coupled hypotheses: global chemistry and atom count
should be sampled before body realization, while the body should not make its
serialization prefix the irreversible decision schedule. It is an
inductive-bias claim, not an expressivity claim. Any joint distribution can be
autoregressively factorized, and an AR system can add grammar constraints,
search, or repair.

## Why the Planner is required for fully de novo generation

An exact-cardinality body model needs `N` before it can instantiate its state,
and composition must be defined before it can be anchored. In the main H1-A2
route these variables are not supplied by a dataset row or a user: they are
sampled from a learned Plan prior.

Training crystals may be deterministically converted into Plan labels for
supervision. At inference, however, the fully de novo route samples
`P ~ p_phi(P)` from the learned Planner. Replaying a Plan extracted from MP-20,
using a frozen Plan file, or accepting a user Plan is useful for controlled
downstream evaluation, but it is conditional at the Plan level and does not by
itself close the fully de novo loop.

Let `P` denote the global Plan, `B` the discrete body, and `M` the final
continuous crystal. H1-A2 models the hierarchy

```text
p(M) = sum_P p_phi(P) sum_B p_theta(B | P) p_psi(M | B, P).
```

The Planner answers what to explore, the DLM answers how that Plan can be
realized, and the refiner answers whether the realization can be improved in
continuous geometry. The learned Planner architecture is not claimed as a
standalone novelty; its role is to provide the global generative prior required
by the fully de novo task.

## The Plan-to-realization interface

Let `P` denote a Plan and let `N(P)` be its atom count. H1-A2 constructs a
crystal body of length

```text
L(P) = 7 + 4 N(P),
```

with seven lattice fields and, for every atom, one element field plus three
fractional-coordinate fields. The Plan anchors composition and count. During
generation, unresolved positions are predicted from the entire partial state
and `P`, with each position restricted to a state-dependent legal token set.

Conceptually, each reverse step applies

```text
underdetermined Plan
        -> exact-cardinality partial crystal state
        -> legal support for each unresolved field
        -> compatible discrete realization
        -> continuous geometric refinement.
```

This formulation makes three distinctions explicit:

- **cardinality before realization:** atom count chooses the state dimension;
- **type before value:** lattice, element, and coordinate slots have different
  categorical support;
- **constraint availability before commitment:** an unresolved field can wait
  for information outside its serialization prefix.

The method does not rely on the claim that masked diffusion has a unique notion
of continuous diffusion time. Its scientific role is iterative, non-prefix
masked generation with a probabilistic corruption objective and programmable
support. It is also not sold as an automatic speed advantage over AR.

## Contribution 1: a hierarchical fully de novo formulation

H1-A2 factorizes fully de novo crystal generation into a learned prior over
global Plans, Plan-conditioned structured completion, and continuous physical
refinement. The sampled Plan is deliberately underdetermined: it specifies
composition, atom count, and coarse structural intent, while exact lattice
values, site realization, fractional coordinates, and the compatible
structural hypothesis remain for the body model.

The modeled object is not “text with unusual tokens.” It is a variable-size,
heterogeneously typed state whose dependencies do not follow its stored order.

## Contribution 2: a crystal-specific masked generator

A generic masked language model is insufficient. H1-A2 combines:

- an exact `7 + 4N` state selected by the planned cardinality;
- crystal-specific token families for elements, lattice fields, and periodic
  coordinates;
- composition and count anchoring;
- generation-time schema, count, volume, periodic-coordinate, and
  duplicate-site support restrictions;
- non-prefix masked completion conditioned on the full partial state.

These restrictions make generation constraint-aware rather than proving that
every locally legal partial state has a globally valid completion. Exact length
is not presented as a standalone novelty; it enables the complete
Plan-to-variable-cardinality-state interface.

## Contribution 3: a typed discrete--continuous factorization

H1-A2 assigns different scientific variables to different inductive biases:

| Stage | Responsibility | Interface contract |
|---|---|---|
| Planner | samples global chemistry, atom count, and coarse structural intent | produces an underdetermined Plan rather than replaying a dataset row |
| masked DLM | discrete, typed, mutually constrained realization | obeys planned composition and count |
| equivariant refiner | continuous periodic lattice and coordinates | preserves composition and count |

This factorization is intentionally diagnostic. The Planner determines what
problem is posed, the DLM determines a discrete structural hypothesis, and the
refiner determines whether that hypothesis can be improved in continuous
geometry. Stability is interpreted as proposal-to-physical-refinement
conversion, while Unique and Novel candidates describe the supply presented to
that conversion stage.

## Why a DLM here?

Masked discrete generation contributes four native capabilities:

1. **Non-prefix context.** An unresolved field conditions on information
   revealed anywhere in the current state.
2. **Delayed commitment.** Fields may remain masked until relevant context is
   available; this does not imply unlimited revision of revealed tokens.
3. **State-dependent support.** Legal token families and crystal constraints
   can change as the partial state evolves.
4. **Randomized information order.** Training does not bind every dependency
   to one fixed left-to-right factorization.

The claim is not that these properties guarantee stable crystals or that AR
cannot emulate them with additional machinery. The claim is that they directly
match the conditional completion problem defined above.

## What the paper does not claim

- H1-A2 is not atom-permutation invariant merely because decoding is
  non-prefix; invariant tokenization is an orthogonal improvement.
- A space-group range in the Plan is not exact symmetry enforcement; H1-A2 is
  not a replacement for symmetry-native generators.
- Local legal support is not a proof of global constraint satisfiability.
- The continuous refiner cannot repair an incorrect formula if composition is
  frozen.
- High Unique/Novel supply is a system-level operating point, not a theorem of
  masked diffusion.
- The hybrid use of language and continuous diffusion is inherited from prior
  work; the contribution is the structured interface between global intent,
  discrete realization, and continuous refinement.
- MP-20-derived, frozen, or user-provided Plans are conditional controls. They
  can test downstream realization, but are not presented as the fully de novo
  Plan source.

## Five-paragraph introduction arc

1. **Scientific mismatch.** Many scientific objects are stored as sequences
   even when their variables are globally coupled and have no physical token
   order. A crystal is a globally compatible realization, not a sentence.
2. **What prior work solved.** Language generators model discrete chemistry;
   continuous, flow, Bayesian, and symmetry-aware models handle periodic
   geometry; CrysLLMGen and FlowLLM show that proposal and refinement are
   complementary. The underdesigned component is a hierarchy that generates
   both the global specification and its compatible realizations.
3. **The missing abstraction.** Fully de novo generation must learn a prior
   over underdetermined Plans; once a Plan is sampled, storage order need not
   become the body's commitment schedule.
4. **Method.** H1-A2 samples a global Plan, turns it into an
   exact-cardinality typed state, completes that state by constrained masked
   generation, and refines only continuous geometry with an equivariant
   diffusion model.
5. **Scientific payoff.** The decomposition exposes the conversion from
   discrete hypotheses to low-energy structures and illustrates a broader
   principle for serialized mixed-variable scientific objects.

## Suggested abstract

> Scientific objects are often serialized for computation even when their
> variables have no causal left-to-right order. Crystal generation makes this
> mismatch explicit: global composition, atom count, lattice fields, species,
> and periodic coordinates are mutually constrained, while final geometry is
> continuous. Building on the discrete-proposal/continuous-refinement insight
> of hybrid crystal generators, we factorize the fully de novo distribution
> through an underdetermined global Plan. H1-A2 first samples chemistry,
> cardinality, and coarse structural intent from a learned Plan prior;
> instantiates an exact-length
> `7+4N` typed crystal state, and uses a masked discrete language model to
> complete unresolved fields from non-prefix context under evolving crystal
> support constraints. An equivariant diffusion model then refines lattice and
> coordinates while preserving composition and cardinality. This
> Planner--completion--refinement factorization separates global-plan
> generation,
> discrete realizability, and continuous physical relaxation. More broadly, it
> treats serialization as an interface rather than a generative causal graph,
> and provides a diagnostic view of crystal discovery as the conversion of a
> diverse discrete proposal set into physically stable structures.
