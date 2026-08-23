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
3. `docs/DE_NOVO_SCOPE.md`
4. `docs/RELATED_WORK.md`
5. `docs/STORY_REVIEW.md`
6. `docs/PLANNER_PROMPT.md`
7. `docs/SEEDS.md`
8. `REPRODUCTION.md`

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

The current main research question is:

> When different crystal-validity checks can only be evaluated after different
> information has been generated, do restricting invalid choices whenever the
> prerequisite information is available and choosing which geometric variables
> are eligible for commitment at each stage affect how reliably a model-proposed
> composition is realized as a periodic crystal?

Composition and atom count are fixed; the scope is eligible Plans sampled by
the learned source. The primary mechanism is selected support × commitment
policy. The learned Plan source defines fully de novo scope, and the fixed
continuous refiner is a downstream consequence rather than part of the Main
RQ. The wording has passed a proposer–reviewer process; future analysis should
stress-test evidence and boundaries rather than replace it with a broader
composition-to-structure question.

## Critical de novo boundary

During training, MP-20 crystals may be deterministically converted into Plan
labels. During fully de novo inference, however, Plans must be generated by a
model rather than replayed from MP-20. A frozen, empirical, or user-provided
Plan can still produce a structurally novel realization, but it is conditional
at the Plan level.

## Three architectural routes to compare

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

I may eventually present A and C together, present only A, or reformulate the
paper around C. Route B should be recommended only if its additional method and
experiments are worth the cost.

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

## Questions you must answer

1. Is a separate learned Planner scientifically justified, or does it merely
   satisfy the implementation requirement that `N` be known before creating a
   `7+4N` body?
2. What evidence would distinguish a useful learned `p_phi(P)` from training-
   Plan replay or memorization?
3. Is the Plan sufficiently underdetermined to support a distribution of
   realizations, or is the DLM only filling a template?
4. Is Route A genuinely different from CrysLLMGen, FlowLLM, SGEquiDiff,
   CrysBFN, and CrystalDiT?
5. Does Route B create a stronger paper, or merely hide the Planner inside the
   DLM?
6. Could Route C support a cleaner and more defensible paper if the claim is
   changed from fully de novo generation to specification-conditioned crystal
   completion?
7. Should the paper present A plus C, only A, or only C? Give an explicit
   recommendation and explain the trade-off.
8. How should the partial crystal state, state-dependent support operator,
   Plan entropy/coverage, compatible-realization multiplicity, and refiner
   invariants be formalized?
9. What is the strongest rejection argument against each route?
10. What is the broad principle beyond crystals, and what conditions limit its
    generalization?

## Required output

Return a structured report containing:

1. A decision table comparing Routes A, B, and C on task scope, novelty,
   scientific cleanliness, implementation cost, reviewer risk, and required
   evidence.
2. Your recommended paper configuration: A+C, only A, only C, or a justified
   transition to B.
3. The strongest main research question and 2–3 sub-questions for each viable
   paper configuration.
4. Exactly three contribution statements suitable for the Introduction for
   each viable configuration.
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
