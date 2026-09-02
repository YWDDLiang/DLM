# Scientific question and contributions

## Central question

> **How can a generative language model carry scientific feasibility from
> discrete chemical decisions into periodic geometric realization and stable
> crystal formation—without outsourcing correctness to post-hoc filtering?**

Crystal generation crosses two qualitatively different spaces. Composition is
a discrete, long-horizon scientific decision governed by atom conservation,
charge/valence compatibility and chemical-family reachability. Structure is a
periodic, coupled geometric object in which lattice and coordinates jointly
determine every local relation. A flat language model must learn both spaces
through one token stream, while a hard enumerator has no learned materials
prior. Our method resolves this mismatch with one tightly coupled hierarchy:
scientific support shapes the LLM action distribution, the resulting Plan is
the exact interface into a crystal diffusion language model, and periodic
relations are injected inside denoising before a fixed terminal diffusion
transition.

## Main result and supporting profiles

The main reported H1-A2 result is Strict/Meta S.U.N. `105/488` per 1,000.
Three newer profiles support specific parts of the method without being merged
into that row: a fresh 256-sample G2 comparison, a full-epoch geometry mechanism
ablation, and a Plan1200 scale validation.

## Contribution 1 — Science-Constrained LLM Planning

**Problem.** Unconstrained LLM composition generation can violate basic
chemistry; validating or repairing samples afterward wastes probability mass
and makes the language model scientifically incidental.

**Coupled mechanism.** C3FD constructs the reachable typed action support at
every composition step. Llama scores exactly those actions from the partial
scientific state. The sampled distribution is a unit-weight product of experts:

\[
\log p_\phi(a_t\mid s_t)
=\log p_{\mathrm{C3FD}}(a_t\mid s_t)
+\log p_{\mathrm{LLM}}(a_t\mid s_t),
\qquad a_t\in\mathcal A_{\mathrm{reachable}}(s_t).
\]

The same LLM then predicts the Compact-V2 structural hints. Scientific support
therefore shapes generation *inside* the LLM decision process rather than
accepting or rejecting completed formulae.

**Technical novelty.** The Planner combines a formal, prefix-dependent
chemical support with a pretrained language prior while preserving one sampled
trajectory. Zero-initialized typed residual heads make the initial fused model
exactly recover C3FD, after which learning can reweight every proposal,
species/count and soft structural action without weakening the reachable set.

**Evidence.** C3FD-v2.5 reaches `2000/2000` composition-valid proposals versus
`1724/2000` for its predecessor. The learned fused Planner reaches `256/256` on
the fresh prospective cohort and `1200/1200` in the independent scale run.
Across the registered diagnostic, mean fused-vs-C3FD KL is `0.06819` and
`87.05%` of typed decisions have nonzero KL, demonstrating that Llama is
causally active rather than a decorative formatter.

## Contribution 2 — Plan-Conditioned Crystal Diffusion Language

**Problem.** Direct crystal-text generation entangles global composition,
sequence length, lattice and coordinates. Local token errors can change the
number or identity of atoms, while geometry learning is diluted by repeated
symbolic chemistry decisions.

**Coupled mechanism.** The Planner emits a canonical latent Plan `z` containing
exact N/elements/counts and compact global structural hints. A byte-identical
train/serve serializer converts `z` into the DLM condition. The crystal body is
represented by dynamic `7+4N` tokens: seven global lattice tokens followed by
exactly one element/X/Y/Z tuple for each atom. N and composition are visible
throughout masked denoising, so the DLM learns

\[
p_\theta(x\mid z),\qquad |x|=7+4N(z),
\]

rather than rediscovering chemical intent from scratch.

**Technical novelty.** The Plan is a learned sufficient interface between
scientific language planning and parallel crystal denoising. It turns variable
crystal cardinality into an exact conditional language while retaining the
non-autoregressive advantages of a masked DLM.

**Evidence.** Compact-V2 uses the full MP20 `27136/9047` train/validation split.
Train and inference renderers are byte-identical. The G0 audit obtains `248/248`
strict `7+4N` decode/re-encode matches, exact species order and zero validity
flips. The independent scale profile obtains `1139/1159 = 98.27%` valid CIFs,
showing that the exact interface remains operational beyond the 256-sample
mechanism studies.

## Contribution 3 — Periodic-Relational Denoising

**Problem.** Tokenwise denoising does not expose the coupled physical state of
a crystal: the distance between two coordinate tokens depends on the complete
lattice, periodic images and both species. Consequently a DLM can model each
token plausibly while producing catastrophic periodic collisions.

**Coupled mechanism.** At every denoising step, G2 converts q0 soft lattice and
coordinate distributions into a periodic species relation graph. Strict
triclinic minimum-image distances, metric/RDF/coordination statistics and
species-aware margins produce pair messages. A zero-initialized relation
adapter returns those messages as a residual to q1 logits:

\[
q_1=q_0+\mathcal R_{\mathrm{G2}}
\left(q_0,\,G_{\mathrm{PBC}}(\mathbb E[x\mid q_0],z)\right).
\]

The residual is an internal DLM inference path—not a geometry repair step.
model494 then serves as the fixed terminal diffusion transition of the same
coarse-to-fine realization process.

**Technical novelty.** Periodic relations are reconstructed from uncertain
masked-DLM states and injected back into token probabilities with exact
step-zero identity. The promoted G2-PBC-R implementation uses an audited
125-image triclinic operator and species-aware packing while preserving the
pretrained DLM function at initialization.

**Evidence.** On the fresh prospective cohort, G2 changes BASE refined
Strict/Meta S.U.N. from `19/111` to `24/117` and shifts paired official hull by
`-16.43 meV/atom`, with the entire bootstrap CI below zero. The full-epoch
mechanism study raises raw Direct from BASE `118/256` to G2-PBC-R `128/256`;
the uncertainty-gated alternative does not improve energy and is rejected. On
the independent main1000 scale profile, the complete tau800 pipeline reaches
Strict/Meta `81/486`, supporting the same learned realization path at scale.

## Inherited terminal module — model494

model494 is necessary to define the complete coarse-to-fine generator, but it
is frozen and is not claimed as a newly learned contribution. Its role is to
separate raw DLM realization from a fixed fine-scale basin transition. In a
matched 512-row study, raw→tau800 changes Direct `188→457`, Strict S.U.N.
`10→48` and Meta S.U.N. `66→230`. The Plan1200 profile uses the same tau800
transition and reaches `81/486` on main1000.

## Why the complete method is necessary

- Remove scientific support: the LLM spends mass on unreachable chemistry.
- Remove Llama: composition remains legal but loses the learned materials
  distribution that chooses among legal paths.
- Remove the Plan contract: chemistry and variable structure length become
  entangled again inside the body generator.
- Remove periodic relational denoising: exact composition survives, but lattice
  and coordinates can remain jointly inconsistent.
- Remove the terminal diffusion transition: the system loses its final
  coarse-to-fine basin realization; raw and refined evidence show that this
  transition is substantial but distinct from learned DLM gains.

The contribution is therefore the coordinated flow of scientific information
across scales, not a bag of independent modules.
