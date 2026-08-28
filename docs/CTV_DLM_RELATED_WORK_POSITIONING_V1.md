# CTV-DLM related-work positioning V1

Date: 2026-08-28

This note bounds the contribution before any positive result is observed.

## Closest mechanisms

### Discrete diffusion guidance

Schiff et al., *Simple Guidance Mechanisms for Discrete Diffusion Models*
(ICLR 2025; arXiv:2412.10193), derives classifier-based and classifier-free
guidance for discrete diffusion and explicitly notes that continuous guidance
does not transfer directly.

CTV-DLM uses the same high-level principle of a normalized discrete
distribution, but its scientific object is different: a legal crystal token's
cost is supervised by real forced-action continuations through the frozen
structure/refinement pipeline. It does not copy continuous CFG, use a terminal
classifier, or claim a generic discrete-guidance derivation.

### Generator plus energy predictor

Wu et al., *Siamese Foundation Models for Crystal Structure Prediction*
(arXiv:2503.10471v2), combines a crystal generator and an energy predictor and
uses the predictor to relax/guidance unstable configurations.

CTV-DLM differs in representation and estimand: composition is fixed by
C³FD, geometry is a masked discrete token trajectory, the value target is a
counterfactual token action under common continuations, and the generator
weights stay frozen during the primary guidance test. DAO is therefore a
direct conceptual prior, not an identical method.

### Masked-diffusion path planning

Peng et al., *Path Planning for Masked Diffusion Model Sampling*
(arXiv:2502.03540v5), learns or designs which tokens to update and allows
unmasked tokens to be revised.

CTV-DLM deliberately freezes the safe-axis position schedule and never
remasks. It changes only the legal-token distribution at two predeclared
milestones. This sacrifices generality to isolate stability-value guidance
from path-planning effects.

### DLM reinforcement learning

Diffu-GRPO/d1 (arXiv:2504.12216), ESPO (arXiv:2512.03759), GDPO
(arXiv:2510.08554), and StableDRL (arXiv:2603.06743) optimize DLM policies with
estimated sequence/token likelihood ratios or ELBO surrogates. The newer work
documents high estimator variance and reward-collapse risks when AR-style
GRPO is moved directly to DLMs.

CTV-DLM is intentionally pre-RL: it freezes the generator, measures real
action costs, and reweights normalized legal token probabilities at inference.
RL remains a final fallback only after the value-guidance design is tested.

## Permitted claim if all gates pass

The intended claim is not “the base DLM learns true thermodynamic stability.”
It is:

> Certified de novo compositions are realized by a masked crystal DLM whose
> generation trajectory is made stability-aware using counterfactual terminal
> action values, improving compute-matched Strict and Meta S.U.N. without
> terminal reranking.

The claim requires the frozen Branch-Q, MatterSim transfer, L6 and L7 gates.
Until then, CTV-DLM is a registered method candidate, not a result.
