# Concept-only story review

Evidence, code readiness, and reproducibility are audited separately. The score
below evaluates the intended research story while requiring its method
description to match the implemented scientific object.

## Verdict

| Story | Concept-only score | Status |
|---|---:|---|
| current audited H1-A2 | 5.5–6/10 | clear interface/mechanism paper; not yet a new general DLM method |
| support-consistent training added and validated | about 7/10 | credible near-term method upgrade |
| exact-multiset species assignment plus physical revision | about 7.5/10 | higher ceiling, but a new project rather than current H1-A2 |

The current story cannot be raised to a stable ICLR weak accept by prose alone.
Its strongest honest positioning is **composition-anchored quantized geometry
completion**, not joint species-geometry generation.

## Strongest current thesis

> A model-sampled formula fixes crystal cardinality and composition but not
> periodic geometry. H1-A2 completes the remaining exact-size typed geometry
> with a masked model whose commitment order respects the prerequisites of
> selected legality checks, followed by continuous refinement that preserves
> the discrete identity.

## Strongest rejection

> This is CrysLLMGen with its autoregressive proposer replaced by a masked,
> composition-prefilled quantized-geometry decoder. Exact length follows from
> known `N`; species are not generated; the rules are hand-written inference
> masks; revealed tokens cannot be revised; and the continuous refiner is
> inherited. Without a representation-, constraint-, and compute-matched AR
> baseline, no gain can be attributed to masked factorization.

The rejection is not fully answerable with the current evidence. The paper
must therefore avoid claims of universal DLM superiority and present matched AR
as the principal deferred experiment.

## What is currently defensible

- learned-Plan fully de novo inference is distinct from gold/frozen Plan
  controls;
- cardinality selects an exact typed state before geometry realization;
- masked fields condition on the whole current partial state;
- selected legality checks require a dependency-respecting field order;
- the refiner preserves atom count/species and changes geometry;
- stagewise reporting can keep chemistry selection, body realization, and
  refinement from being conflated.

## What is not currently defensible

- free species-site generation;
- rich Plan compliance without counterfactual evidence;
- support-consistent training;
- general constrained-DLM novelty over DINGO, DDPD, or domain-validity losses;
- revisable commitments or violation-guided repair;
- exact symmetry, permutation invariance, or global satisfiability;
- an empirical superiority claim over a matched constrained AR executor.

## Closest novelty boundaries

- CrysLLMGen and FlowLLM establish discrete/text proposal followed by
  continuous refinement.
- DINGO establishes constrained dLLM inference for formal languages.
- DDPD and planned diffusion establish denoising-order planning.
- PepTune establishes domain-dependent corruption and invalid-loss design.
- Mat2Seq, Wyckoff Transformer, and SGEquiDiff address representation,
  permutation, and symmetry axes that non-prefix decoding does not solve.

The remaining current contribution is a crystal-specific composition-anchored
completion interface plus a transparent stage-aware evaluation framing.

## Separate evidence/repro/code audit

The following findings do not alter the concept-only score but must be fixed
before release:

- the 105/488 method-family main-table value and its underlying cohort-level
  audit views must remain explicitly separated;
- execution stages and assets remain incomplete in the release scaffold;
- previous documents overstated refiner conditioning and support semantics;
- first-success selection and N-sorted processing can create chemistry survivor
  bias unless an immutable raw-ordinal ledger is retained;
- the current hull evaluation is a shared ML-potential/reference proxy, not
  candidate-level DFT and cannot be mixed with published DFT S.U.N. rankings.
