# Concept-only area-chair review

This review evaluates research framing and novelty after comparison with
recent ICLR, ICML, NeurIPS, and AAAI work. It assumes that supporting evidence
will be supplied separately and does not score reproducibility or empirical
completeness.

## Verdict

**Story score: 7/10 — credible weak accept at the concept level.**

| Criterion | Score (1–10) |
|---|---:|
| Importance | 7 |
| Research-question precision | 8 |
| Novelty | 6 |
| Conceptual soundness | 7 |
| Positioning clarity | 8 |
| Overall story | 7 |

The earlier story was vulnerable to a 5/10 summary: “CrysLLMGen with its AR
proposer replaced by a masked LM.” The revised story reaches 7/10 because its
central object is now the **Plan-to-realization interface**:

```text
underdetermined global intent
        -> exact-cardinality partial crystal state
        -> set of compatible discrete realizations
        -> continuous geometric refinement.
```

The DLM is essential to the implementation of this interface, but the paper is
not merely about choosing a different decoder backbone. The main task remains
fully de novo because the global Plan is sampled from a learned prior rather
than replayed from MP-20.

## Strongest defensible claim

> Fully de novo crystal generation can be factorized into a learned prior over
> intentionally underdetermined global Plans, constrained completion of typed
> variable-cardinality realizations, and continuous periodic refinement.
> H1-A2 samples global chemistry and cardinality, realizes each Plan without
> tying commitment order to serialization order, and lets an equivariant
> refiner act only on continuous geometry.

This claim is narrower and more defensible than “crystals require DLMs.” It
states which problem has been defined, why masked completion matches it, and
what each module may change.

## Why the positioning survives recent work

- **CrysLLMGen and FlowLLM** already establish discrete proposal plus
  continuous refinement. H1-A2 claims the next step: an explicit conditional
  completion contract for the discrete proposal.
- **Mat2Seq, WyFormer, SymmCD, and SGEquiDiff** solve canonicalization,
  permutation, or symmetry. H1-A2 does not conflate those properties with
  non-prefix information flow.
- **DiffCSP, FlowMM, CrysBFN, TGDMat, and CrystalDiT** show that unified or
  geometry-native generation is viable. H1-A2 presents factorization as a
  deliberate, diagnosable design choice rather than a universally superior
  theorem.
- **Masked-DLM theory** supports randomized information orders and
  state-dependent masking, while also warning against universal efficiency
  claims. The revised story is about the completion interface, not speed or
  diffusion mystique.

## Three fatal wording risks

1. **“Crystals are not naturally ordered sequences” as the novelty claim.**
   Representation and symmetry papers already address ordering ambiguity. Use
   “serialization order need not determine commitment order.”
2. **“DLM natively guarantees diversity, consistency, or correctness.”**
   It permits non-prefix context and delayed commitment; none of the stronger
   properties follows automatically.
3. **“Planner--DLM--refiner is the contribution.”** A three-module pipeline is
   not enough after CrysLLMGen, FlowLLM, and SGEquiDiff. The contribution is the
   semantics of the interfaces: what the Plan fixes, what remains uncertain,
   what the DLM completes, and what the refiner preserves.

## Contribution verdicts

1. **Plan-conditioned structured completion:** strongest intellectual
   contribution and the paper's central research question.
2. **Exact-cardinality crystal completion interface:** method contribution only
   when exact length, typed support, anchors, and evolving constraints are
   presented as one coherent object.
3. **Discrete-to-continuous contract:** meaningful because it defines
   invariants and diagnoses proposal versus refinement, not because a hybrid
   architecture is new.

## What blocks an 8+

Four conceptual points still require precise treatment in the paper:

1. define the learned Plan prior and the Plan as an intentionally
   underdetermined specification, including exactly which degrees of freedom
   remain and why replayed Plans are only controls;
2. formalize the partial state and state-dependent support operator, rather
   than listing masks as engineering rules;
3. state refiner invariants explicitly and distinguish them from soft Plan
   semantics;
4. keep non-prefix information flow, atom-permutation invariance, and
   space-group equivariance separate throughout the manuscript.

If these remain precise, the story is coherent at roughly a 7/10 level. If the
paper falls back to “DLM instead of AR,” it returns to 5/10.

## Recommended title

> **Serialization Is Not Generation Order: Discrete Crystal Completion and
> Continuous Refinement**
