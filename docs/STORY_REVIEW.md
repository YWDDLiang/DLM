# Concept-only proposer–reviewer verdict

## Final disposition

> **APPROVED — approximately 7/10 at the concept level.**

The approved paper-level story is **Proposal versus Realization**. This
approves the scientific question, H1-A2 method hypothesis, and mechanism
hierarchy. It does not pre-approve positive empirical conclusions.

## Approved main scientific question

> **In generative materials discovery, to what extent do gains in discovery
> yield arise from changing the distribution of material specifications being
> explored, versus improving structural realization conditional on an
> explored specification?**

The general question is evaluated only through a de novo crystal instantiation:

> **For de novo crystal generation, can composition-anchored masked completion
> improve structural realization across model-sampled chemistries beyond gains
> explained by measured changes in the proposed-chemistry distribution,
> without collapsing cohort-level diversity?**

For H1-A2, an explored specification is the composition, atom count, and
element multiset fixed before body generation. The question does not treat a
changed proposal distribution as wrong; it asks which source of improvement
explains an aggregate result.

## Approved H1-A2 hypothesis

> **Given a model-proposed composition and atom count, anchoring chemical
> identity and cardinality while using masked discrete completion for periodic
> geometry, followed by identity-preserving continuous refinement, defines a
> system in which we test for a positive standardized within-stratum outcome
> difference across prespecified chemical regimes and retained cohort-level
> uniqueness.**

This matches the implemented factorization: the Planner samples composition
and `N`; anchors fix `N` and the element slots; the Crystal DLM generates
`6+3N` lattice/coordinate tokens; and the refiner changes continuous geometry
without changing `N` or ordered atom types.

The hypothesis can be tested as a system-level comparison. A causal claim that
masked completion itself improves realization requires a matched executor such
as an AR control; the internal support/policy factorial does not provide that
architecture comparison.

## Approved mechanism question

> **At fixed composition and atom count, do prerequisite-aware restrictions on
> selected invalid token choices and a dependency-aware commitment policy
> improve discrete periodic-body realization and downstream conversion under
> an unchanged identity-preserving refiner?**

The question of when selected restrictions become active and which positions
may compete for commitment is therefore retained as the mechanism RQ rather
than the paper-level motivation.

## Why the story passes review

- It identifies a general ambiguity in aggregate discovery metrics: what
  chemistry is attempted versus how well attempted chemistry is realized.
- It leads naturally to H1-A2 without naming H1-A2 in the Main RQ.
- It is falsifiable by chemical-distribution, standardized-conversion, fixed-
  condition, and pre/post-refiner evidence.
- It makes cohort-level diversity part of the hypothesis instead of treating
  uniqueness as an independent per-body label.
- It keeps selected support and commitment policy as concrete mechanisms while
  respecting their narrow implemented scope.

## Approved contribution hierarchy

1. **Scientific and evaluation formulation:** separate changes in the
   explored-specification distribution from structural realization conditional
   on a proposal, instantiated for crystals by composition and cardinality.
2. **Core method:** a composition-anchored, exact-cardinality typed masked
   executor with selected partial-state support and an explicit commitment
   policy, followed by an identity-preserving continuous refiner.
3. **Attribution framework:** combine complete chemical-distribution and
   stagewise conversion reporting, common-mix standardization, a matched
   executor comparison, fixed-condition mechanism analysis, and pre/post-
   refiner conversion.

Contribution 3 remains a framework until the corresponding results are
complete.

## Evidence roles

The evidence chain has five non-interchangeable parts:

1. Full compound distributions and within-stratum conversion rates establish
   breadth and rule out concentration in one obvious high-success family.
2. Common-mix standardization and proposal/within-stratum accounting answer
   the Main RQ for additive per-request endpoints at the preregistered-stratum
   resolution within measured overlapping support; they do not remove exact-
   formula residual selection or linearly decompose cohort-level S.U.N.
3. A matched AR-versus-DLM executor comparison identifies the scoped
   architecture contrast required by the masked-completion crystal RQ.
4. Fixed-condition paired analysis identifies the selected-support bundle,
   grouped-confidence policy, and their interaction under one frozen masked
   checkpoint. It does not identify masked DLM architecture versus AR.
5. Pre/post-refiner conversion measures conditional conversion and whether a
   proposal-stage gap is preserved, attenuated, or reversed. Aggregate refiner
   attribution additionally requires the all-request funnel.

“Almost every compound type improves” is not a premise. A defensible result
must report the exact number of adequately supported preregistered strata with
positive, inconclusive, and negative effects, plus their reference-distribution
coverage.

## Strongest rejection

> Even if oxide, halide, arity, and atom-count strata all improve, the Planner
> may still select easier exact formulas within every coarse stratum; final
> gains may also be dominated by the inherited refiner. Broad tables alone do
> not identify composition-conditioned realization or the DLM mechanism.

The defense is common-support standardization, explicit scope language, a
matched executor comparison, fixed-condition mechanism evidence, and pre/post-
refiner attribution. It is not stronger prose.

## Defensible boundaries

The paper may claim, when supported, that:

- aggregate S.U.N. conflates attempted-chemistry selection and conditional
  structural realization;
- the gain remains positive after standardization over preregistered measured
  chemical strata;
- improvements are distributed across most adequately supported strata rather
  than concentrated in one family;
- a scoped matched executor comparison favors masked completion, if that
  experiment supports the claim; and
- fixed-condition selected support and commitment policy affect realization
  under the same refiner.

The paper may not claim:

- that competing methods “cheat” or only select easy chemistry;
- uniform improvement over every compound type or all chemical space;
- causal mediation from the descriptive accounting decomposition;
- universal DLM superiority over AR beyond the matched scoped comparison;
- complete rich-Plan execution or general constraint satisfaction;
- revision or reopening of committed tokens; or
- algorithmic novelty of the Planner backbone or continuous refiner.

## Evidence status

The future paper table retains `105/1000` Strict S.U.N. and `488/1000` Meta
S.U.N. as aggregate headline results. They do not by themselves answer the
Main RQ. The remaining evidence work is the complete preregistered chemistry
audit, common-support standardized accounting, fixed-condition mechanism
analysis, matched executor comparison, and pre/post-refiner conversion. No new
claim is assumed positive in advance.
