# Connecting the proposal--realization story to experiments

## Research hierarchy

The paper uses three nested questions:

1. **Field-level Main RQ:** in generative materials discovery, does higher
   discovery yield come from changing which material specifications are
   explored, improving structural realization conditional on a specification,
   or both?
2. **Crystal instantiation:** can composition-anchored masked completion
   improve structural realization across model-sampled chemistries beyond gains
   explained by measured proposal-distribution changes, without collapsing
   cohort-level diversity?
3. **Mechanism RQ:** at fixed composition and atom count, do selected
   prerequisite-aware support restrictions and an explicit commitment policy
   improve body realization under the same refiner?

The field-level principle is proposed generally but empirically evaluated only
for de novo inorganic crystals.

## Paper narrative

The Introduction should follow one logical chain rather than present three
modules as three unrelated ideas:

1. Aggregate discovery yield conflates the distribution of explored material
   specifications with realization conditional on a specification.
2. In crystals, composition and cardinality can be proposed before periodic
   geometry, creating a concrete proposal--realization boundary.
3. H1-A2 preserves de novo proposal through a learned Planner, then anchors
   composition/`N` so the masked executor cannot evade a proposed chemistry by
   changing it; the fixed refiner changes only continuous geometry.
4. Chemical-distribution and standardized-conversion analyses answer the
   crystal instantiation at system level; fixed-condition and pre/post analyses
   explain the executor and refiner mechanisms.
5. Conclusions are stated at the resolution supported by common chemical
   support and are not generalized empirically beyond crystals.

Recommended title:

> **Proposal or Realization? Composition-Anchored Masked Completion for De Novo
> Crystal Discovery**

A more conservative title is:

> **Disentangling Proposal and Realization in De Novo Crystal Generation**

## Story-to-evidence map

| Story statement | Primary evidence | What it does not prove |
|---|---|---|
| The systems explore different chemistry | Attempted-composition distributions at every pipeline stage | Which distribution is scientifically preferable |
| Improvement is broad across chemistry | Complete within-stratum funnels and absolute risk differences | Exact-formula conditioning or DLM causality |
| Measured proposal reweighting does not explain a per-request endpoint gap at the chosen stratum resolution | Common-support standardization and proposal/within-stratum accounting | Causal mediation, exact-specification balance, or cohort-level S.U.N. decomposition |
| The selected execution policy affects realization | Fixed-condition paired support/policy analysis under one checkpoint | Masked DLM superiority over AR |
| The final result is not only an inherited-refiner effect | Pre/post conversion combined with the all-request funnel | A refiner effect for bodies that never reached refinement |
| Stability was not bought by diversity collapse | Fixed-size cohort-level uniqueness and S.U.N. recomputation | Per-body independent uniqueness probabilities |

## Required analysis from existing outputs

The proposal--realization analysis requires a traceable raw cohort with one
specification and outcome record per requested attempt. A rounded or normalized
aggregate headline may remain in the main table, but it cannot be assigned
sample-level chemical strata. If the headline is not itself a raw cohort, the
paper must name the separate raw analysis cohort or generate one.

Before computing a gain, freeze the primary comparator, requested-attempt
denominator, retry/refinement contract, evaluator thresholds, novelty and
StructureMatcher settings, one primary per-request endpoint, one primary
reference mix, common-support rule, and trimming rule. Without that contract,
“gain” remains an undefined future estimand.

### Complete proposal and attrition audit

Use the requested-attempt population and report:

```text
requested
-> parsed/eligible specification
-> body success
-> Direct composition/structure/joint validity
-> refined/reconstructed
-> hull known/unknown
-> stable and stable--novel
-> cohort-level Unique and S.U.N.
```

The primary chemical strata are defined before inspecting outcomes:

- mutually exclusive compound family;
- unary, binary, ternary, and quaternary-or-higher arity; and
- atom-count bins over the supported range.

Element presence, atom-weighted element frequency, exact element set,
training-set density, and pretreatment Plan fields are sensitivity analyses.
Final generated symmetry or geometry must not be used as a pretreatment
standardization variable.

### Composition-standardized accounting

For method `m` and preregistered stratum `h`, report:

```text
p_m(h): proposed-specification distribution
r_m(h): additive per-request outcome rate inside h
theta_m = sum_h p_m(h) * r_m(h)
```

Standardize both systems to prespecified shared mixes, report common-support
coverage and effective sample size, and decompose the observed difference into
proposal-distribution and within-stratum residual components. This is
descriptive accounting, not causal mediation. Coarse-stratum estimates remain
vulnerable to residual exact-formula selection and must not be labeled exact-
specification realization without matched specifications.

The additive accounting is applied to per-request endpoints such as body
success, Direct joint validity, stable all-request yield, novelty, or
stable--novel yield. `Stable among hull-known` is reported with hull missingness
and does not alone carry the stability claim. Full S.U.N. is excluded from the
linear decomposition.

### Cohort-dependent outcomes

Uniqueness and full S.U.N. must be recomputed on equal-size standardized
cohorts. Uncertainty should resample the correct independent unit and preserve
equivalence clusters; duplicated bootstrap records must not create artificial
non-uniqueness. These are nonlinear cohort comparisons, not additive
proposal/realization components.

“Without diversity collapse” is a non-inferiority statement. Its margin,
reference mix, cohort size, and interval decision rule must be frozen before
analysis; failure to reject a difference from zero is not evidence of retained
diversity.

### Proposal-to-refiner conversion

Report discrete-body and refined outcomes separately, including:

- identity and atom-count invariance;
- valid-to-valid, invalid-to-valid, and valid-to-invalid transitions;
- lattice and coordinate displacement;
- hull coverage and known-both energy changes; and
- whether a proposal-stage gap persists, attenuates, or reverses.

## Additional inference evidence

No checkpoint retraining is required for the minimum mechanism test.

1. Freeze one eligible learned-Plan cohort and one attempt seed per Plan.
2. Cross selected support on/off with grouped/positional commitment policy.
3. Hold checkpoint, Plan, anchors, model-call budget, sampling settings,
   call-indexed randomness, parser, and evaluator fixed.
4. Freeze one primary eligible-Plan body-request endpoint, risk-difference
   main effects, simple effects, interaction, smallest effect of interest, and
   an equivalence/inconclusive rule before sampling.
5. Treat the Plan--seed block as the paired unit, cluster by Plan, and retain
   every eligible-Plan body request failure. A one-seed design estimates an
   average intervention effect but not Plan-specific seed variance or seed
   robustness.
6. Preselect the refiner comparison rather than choosing the best-looking
   control after body results. Refine all body successes for that comparison
   with the same inherited identity-preserving refiner.

Because the frozen crystal instantiation explicitly names masked completion, a
matched AR-versus-DLM executor is required for the intended strong-method
paper. It must share specification, representation, anchors, training/parameter
budget, refiner, evaluator, and attempted denominator. If this experiment is
omitted, the crystal RQ, title, and conclusions must be downgraded to the H1-A2
system rather than masked-architecture improvement.

Independent Planner seeds strengthen population-level generalization but are
not replaced by multiple body attempts from one Planner cohort.

Because the executor reads soft rich-Plan fields in addition to hard
composition/`N` anchors, a low-cost paired context ablation should compare the
full Plan with an anchors-only or neutral-soft-field context under the standard
support/policy. If soft fields matter, they must enter the conditioning scope;
their effect cannot be silently counted as composition-conditioned
realization.

A small repeated-sampling subset under the standard H1-A2 policy can test
whether one proposed composition/Plan supports multiple valid structural
realizations. StructureMatcher clusters and local-environment fingerprints
should be reported per Plan. This diagnoses conditional multiplicity; it is not
a substitute for the support/policy intervention or the headline S.U.N. cohort.

## Gold-Plan and replay roles

- A held-out MP-20-derived Gold Plan is a conditional executor reference. If
  its soft fields are derived from the target structure, label it an
  **oracle-rich-Plan-conditioned reference**. It estimates realization
  capacity under supplied context, not fully de novo discovery.
- Frozen generated Plans are fixed-condition mechanism controls.
- Neither source replaces the learned Plan source in the headline system.

## Claim gates

The paper may say “broadly improved across chemical regimes” only after
reporting the number of adequately supported preregistered strata with
positive, inconclusive, and negative effects and the reference probability
mass they cover.

The paper may say measured proposal reweighting does not explain most of the
gain only under a prespecified decision rule supported by joint uncertainty.
If accounting components have opposite signs, do not summarize them as simple
percentages of one total. The paper may attribute a mechanism effect only to
the intervention actually paired under fixed conditions.

## Minimal paper presentation

1. **Figure 1:** proposal--realization problem and H1-A2 interface.
2. **Table 1:** aggregate headline and complete all-request funnel.
3. **Figure 2:** proposed-composition distribution, within-family effects, and
   standardized proposal/within-stratum residual accounting.
4. **Table 2:** matched AR-versus-DLM executor comparison under the same
   specification and downstream contract.
5. **Table 3:** fixed-condition execution-policy result and pre/post-refiner
   conversion.

Detailed family-by-arity-by-atom-count tables, element maps, common-support
diagnostics, Gold-Plan references, and negative strata belong in the appendix.
