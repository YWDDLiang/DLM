# Standalone prompt for ChatGPT Web

The prompt below is self-contained and does not depend on Codex skills, local
tools, or access to this workspace. Relevant documents may optionally be
uploaded with it.

---

I am developing a top-conference research story for a fully de novo inorganic
crystal generator called H1-A2. Act simultaneously as a research strategist,
a skeptical ICLR/NeurIPS reviewer, and an area chair. If web browsing is
available, research the latest relevant 2024–2026 papers using primary official
sources, with CrysLLMGen as the closest foundation. Also compare against
CrystaLLM, Mat2Seq, DiffCSP, FlowMM, FlowLLM, CrysBFN, TGDMat, SymmCD,
Wyckoff Transformer, SGEquiDiff, CrystalDiT, and recent masked discrete
diffusion-language-model theory. Cite the sources used.

The frozen paper-level theme is **Proposal versus Realization**: aggregate
generative-materials yield may improve because a model changes which material
specifications it explores, because it better realizes an explored
specification as a structure, or both. The paper evaluates this general
distinction only in de novo inorganic crystals. Do not replace it with a
solution-first decoding question or extrapolate the crystal evidence to other
scientific domains.

## GitHub repository and source of truth

Repository:

`https://github.com/YWDDLiang/DLM`

The current paper-facing H1-A2 source of truth is this branch:

`codex/h1a2-paper-de-novo`

Direct branch URL:

`https://github.com/YWDDLiang/DLM/tree/codex/h1a2-paper-de-novo`

Do not treat `main`, `codex/evidence-first-sun-msun`, or other branches as the
current paper method. They contain historical experiments and earlier framing.
Use them only if you explicitly need historical context and clearly label that
context as non-current.

Before evaluating the story, browse and read the following files from
`codex/h1a2-paper-de-novo` completely:

1. `README.md`
2. `docs/PAPER_POSITIONING.md`
3. `docs/PROPOSAL_REALIZATION_EVIDENCE.md`
4. `docs/DE_NOVO_SCOPE.md`
5. `docs/RELATED_WORK.md`
6. `docs/STORY_REVIEW.md`
7. `docs/PLANNER_PROMPT.md`
8. `docs/SEEDS.md`
9. `REPRODUCTION.md`

Then inspect the relevant implementation rather than relying only on the
narrative documents:

- `src/crystal_dlm/h1_llm_planner.py`
- `src/scripts/sample_llama_h1_formula_plans.py`
- `src/crystal_dlm/r5_plan_state.py`
- `src/crystal_dlm/r5_plan_body.py`
- `src/crystal_dlm/r5_dynamic_length.py`
- `src/crystal_dlm/generation_schedule.py`
- `src/scripts/sample_llada_r5_exact_length.py`
- `src/scripts/refine_dlm_with_crysllmgen.py`
- `scripts/submit_h1a2.sh`
- `results/reference_results.json`

Treat code and executable contracts as evidence about what is currently
implemented. If a narrative claim is not supported by the branch, identify the
mismatch instead of silently accepting it. Do not modify the GitHub repository,
open issues, or create pull requests; this task is analysis and story design
only. If you cannot access the branch or a required file, explicitly list what
you could not read and ask me to upload it before reaching a strong conclusion.

## Review protocol: keep the research story separate from audits

The primary task is to evaluate and improve the **research story at the
conceptual level**. Assume that missing empirical support can be added later.
Do not lower the research-story score because of missing checkpoints,
placeholder assets, incomplete environment files, undocumented seeds,
unfinished launch scripts, reproducibility gaps, repository organization, or
ordinary implementation bugs.

Use the repository to understand the intended method and detect mismatches,
but report engineering findings in separate audit sections. Do not mix them
into the conceptual novelty and story verdict.

Maintain three independent tracks:

1. **Research-story review — primary.** Evaluate the problem definition,
   scientific motivation, novelty relative to prior work, conceptual
   coherence, falsifiability, contribution structure, and whether the proposed
   interfaces form a convincing method. Score this while assuming suitable
   evidence and a correct implementation will be supplied.
2. **Evidence and reproducibility audit — secondary.** Separately list missing
   experiments, weak attribution, statistical concerns, unavailable assets,
   seed/checkpoint gaps, and reproducibility risks. Give this its own readiness
   score, but do not use it to lower the research-story score.
3. **Code and documentation audit — secondary.** Separately list code/story
   mismatches, incomplete implementations, dead paths, configuration problems,
   or claims not yet represented by executable code. Give this its own
   readiness score, but do not use it to lower the research-story score.

Conceptual problems still belong in the primary review. Examples include an
RQ that silently changes from fully de novo to conditional generation, a Plan
that leaves no meaningful realization freedom, contributions already subsumed
by prior work, an internally inconsistent factorization, or an unfalsifiable
claim. Report these as story problems even if they are discovered while reading
code. By contrast, “the intended method is coherent but this branch has not yet
implemented it correctly” is an implementation audit finding, not a reason to
downgrade the intended story.

Never collapse the three tracks into one overall score. Report at least:

- concept-only research-story score;
- evidence/reproducibility readiness score;
- code/artifact readiness score.

## Current H1-A2 method

The current system contains:

1. A learned Planner that samples an underdetermined global crystal Plan:
   composition/formula, atom count `N`, chemistry family, coarse lattice mode,
   space-group range, and volume range.
2. A masked diffusion language model that realizes the sampled Plan as an
   exact-cardinality typed crystal body of length `7+4N`: one count token, six
   lattice/angle tokens, and `N` element/X/Y/Z groups. Count and all element
   slots are prefilled, so the DLM freely generates only `6+3N` geometry
   tokens. It uses non-prefix information flow, typed schema, a
   lattice→X→Y→Z schedule, and selected local lattice/PBC checks.
3. An equivariant continuous diffusion refiner that preserves composition and
   atom count while refining lattice and fractional coordinates.

The intended hierarchical model is:

`P ~ p_phi(P); A=A(P); G ~ p_theta(G|P,A); B=(A,G); M ~ p_psi(M|B)`

where the Planner answers “what to explore,” the Crystal DLM answers “how the
Plan can be realized,” and the refiner asks whether that realization can become
reasonable continuous periodic geometry.

The frozen Main Scientific RQ is:

> In generative materials discovery, to what extent do gains in discovery
> yield arise from changing the distribution of material specifications being
> explored, versus improving structural realization conditional on an
> explored specification?

The frozen crystal instantiation is:

> For de novo crystal generation, can composition-anchored masked completion
> improve structural realization across model-sampled chemistries beyond gains
> explained by measured changes in the proposed-chemistry distribution,
> without collapsing cohort-level diversity?

For H1-A2, an explored material specification is the composition, atom count
`N`, and element multiset fixed before body generation. H1-A2's method
hypothesis is that anchoring those variables and using masked discrete
completion for periodic geometry, followed by identity-preserving continuous
refinement, defines a system whose standardized within-stratum outcomes and
cohort-level uniqueness are tested across prespecified chemical regimes.

The former Main RQ is retained as the Mechanism RQ:

> At fixed composition and atom count, do prerequisite-aware restrictions on
> selected invalid token choices and a dependency-aware commitment policy
> improve discrete periodic-body realization and downstream conversion under
> an unchanged identity-preserving refiner?

The learned Plan source defines fully de novo scope. Selected support ×
commitment policy explains the masked-executor mechanism; the fixed refiner is
the downstream conversion stage. Do not promote the mechanism question back to
the paper-level scientific motivation.

The future paper main table is locked to `105/1000 = 10.50%` Strict S.U.N. and
`488/1000 = 48.80%` Meta S.U.N. These are aggregate headline results, not a
substitute for proposal-versus-realization evidence. Historical
`94/1000 = 9.40%` and `474/1000 = 47.40%` values are compatibility views and
must not replace the headline.

## Critical de novo boundary

During training, MP-20 crystals may be deterministically converted into Plan
labels. During fully de novo inference, however, Plans must be generated by a
model rather than replayed from MP-20. A frozen, empirical, or user-provided
Plan can still produce a structurally novel realization, but it is conditional
at the Plan level.

## Frozen route roles and counterfactual alternatives

Route A is the frozen paper method. Route C is a conditional diagnostic. Route
B is an unimplemented future alternative. Compare them to expose boundaries,
but do not reopen the paper configuration or replace the frozen Main RQ unless
you find a genuine conceptual contradiction.

### Route A — separate learned Planner

`learned Planner -> exact-cardinality masked DLM -> continuous refiner`

- Generates the global Plan and remains fully de novo.
- Makes global chemistry and cardinality explicit.
- Risks making the Planner an extra bottleneck or looking like a structured
  prompt in front of CrysLLMGen.
- This is the currently implemented H1-A2 route.

### Route B — no separate Planner model, but self-planning DLM

`DLM pass 1 generates formula/N/coarse header -> instantiate 7+4N body -> DLM
pass 2 completes body -> continuous refiner`

- Can remain fully de novo because global variables are still generated.
- Removes the separate Planner backbone but not the planning function.
- May produce a cleaner unified-DLM contribution.
- Requires a new two-pass model/training design and is not currently
  implemented.

### Route C — no learned Planner, use MP-20/R5C/frozen/user Plans

`empirical or fixed Plan -> exact-cardinality masked DLM -> continuous refiner`

- Cleanly isolates the Crystal DLM and refiner.
- Can generate new structures under an existing specification.
- Is not fully de novo at the Plan level.
- Is currently available as a downstream control.

The paper presents Route A as the fully de novo system and may use Route C as a
separately labeled conditional reference. Route B remains future work.

## Scientific boundaries

- Do not claim that autoregressive models are inexpressive.
- Do not claim masked diffusion guarantees diversity, consistency, stability,
  or faster sampling.
- Non-prefix generation is not atom-permutation invariance.
- A coarse space-group Plan is not exact symmetry equivariance.
- Generation-time masks provide typed schema, nonzero lengths, selected
  angle-degeneracy checks, and duplicate-Z checks after X/Y are known. They
  do not enforce Plan volume bins, exact space groups, minimum distance, or
  global satisfiability.
- “Language model plus continuous diffusion” is already established by
  CrysLLMGen and FlowLLM.
- Exact length alone is engineering; its scientific role must be explained as
  cardinality-before-realization within a complete interface.
- Default to standard H1-A2. Do not use Safe-axis as part of the paper story.
- Evidence will be expanded later. For this task, prioritize conceptual
  positioning, falsifiability, and a coherent story rather than
  reproducibility engineering.
- Do not characterize reward-guided, Wyckoff, or symmetry-aware methods as
  cheating. The academically defensible statement is that aggregate metrics
  do not identify selection, conditional realization, or both.
- Treat full compound distributions and within-stratum stability as breadth
  evidence; common-mix standardization as anti-shortcut evidence; fixed-
  condition comparisons as selected-support/policy evidence under one masked
  checkpoint; and pre/post-refiner conversion as conditional refiner evidence.
  These roles are not interchangeable.
- Uniqueness is cohort-level. Do not analyze it as an independent per-body
  Bernoulli outcome.
- Do not linearly decompose full S.U.N. with `sum_h p(h)r(h)`. Use additive
  per-request endpoints for accounting and equal-size standardized cohorts for
  nonlinear U/S.U.N. comparison.
- The executor reads soft rich-Plan context in addition to composition/N.
  Require a full-Plan-versus-anchors-only ablation before calling all residual
  differences composition-only realization.
- A matched executor is required to attribute a system-level difference to
  masked completion rather than to the H1-A2 package.

## Questions you must answer

1. Why is proposal-versus-realization a substantive generative-materials
   question rather than a post-hoc metric decomposition, and why is the
   current empirical claim limited to crystals?
2. Does frozen Route A naturally identify what chemistry is attempted and how
   attempted chemistry is realized, or are important variables still mixed?
3. What evidence would distinguish a useful learned `p_phi(P)` from training-
   Plan replay, memorization, or coarse-stratum selection effects?
4. Is the Plan sufficiently underdetermined to support a distribution of
   realizations, or is the DLM only filling a template?
5. How is frozen Route A scientifically different from CrysLLMGen, FlowLLM,
   SGEquiDiff, CrysBFN, and CrystalDiT without claiming DLM superiority?
6. What can Route C diagnose without being mistaken for the fully de novo
   system, and what remains future-only in Route B?
7. How should the partial crystal state, state-dependent support operator,
   Plan entropy/coverage, compatible-realization multiplicity, and refiner
   invariants be formalized?
8. What is the strongest “easier exact formulas within every coarse stratum”
   rejection, and which measured conclusions survive it?
9. What is the strongest “the refiner caused the final gain” rejection?
10. What is the broad principle beyond crystals, and what conditions limit its
    generalization?
11. Which preregistered chemistry strata, standardization targets, and
    common-support diagnostics are needed to separate proposal-distribution
    components from within-stratum residuals without claiming causal mediation
    or exact-specification balance?
12. What exact evidence would justify “broadly improved across chemical
    regimes,” and what result would falsify that conclusion?

## Required output

Return a structured report containing:

1. A stress test of the frozen Main RQ and H1-A2 method hypothesis, preserving
   them unless a genuine conceptual contradiction is found.
2. A compact scope table for frozen Route A, conditional Route C, and future
   Route B; do not reopen the paper configuration.
3. Two or three sub-questions under the frozen Main RQ, including the existing
   support/commitment Mechanism RQ.
4. Exactly three contribution statements suitable for the Introduction of
   frozen Route A.
5. A formal factorization and explicit contract stating what every stage may
   condition on, generate, preserve, and modify.
6. A Planner evaluation protocol covering validity, coverage, novelty versus
   training Plans, memorization, and downstream-support alignment.
7. The strongest “this is only CrysLLMGen plus a masked decoder” rejection and
   a concrete rebuttal that does not overclaim.
8. A 1–10 concept-only reviewer score for every route and the changes needed
   to reach at least 7/10.
9. A recommended title, five-paragraph Introduction arc, and approximately
   150-word abstract for the final recommended configuration.
10. A prioritized list of conceptual and empirical gaps to resolve next.
11. A minimal main-text figure/table plan for complete compound distributions,
    within-stratum conversion, common-mix standardization, fixed-condition
    mechanism evidence, and pre/post-refiner attribution.

After the primary report, add two clearly separated appendices:

- **Appendix A — Evidence and reproducibility audit.** List findings without
  changing any concept-only score.
- **Appendix B — Code and documentation audit.** List implementation and
  narrative mismatches without changing any concept-only score.

For every recommendation, label it as one of `STORY`, `EVIDENCE`, `REPRO`, or
`CODE` so that engineering cleanup cannot accidentally rewrite the scientific
positioning.

Be critical. If the Planner cannot support the fully de novo claim, say so
directly. If the no-Planner route changes the scientific task, make that change
explicit rather than hiding it with terminology. Do not optimize the story by
making claims that the stated method cannot support.

---
