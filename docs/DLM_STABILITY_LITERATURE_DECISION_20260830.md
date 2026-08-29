# DLM stability literature decision — 2026-08-30

## Decision

Keep the paper's causal boundary unchanged:

```text
C³FD composition + N -> masked DLM structural realization -> model494 refinement
```

Run the frozen same-composition shared-noise D3PO experiment first. It is the
closest low-resource extension supported by published work and by the existing
H1-A2 energy-pair assets. Do not mix special-token, CFG, refiner, or Planner
changes into that causal test.

## Primary published precedents

- [Preference-Based Alignment of Discrete Diffusion Models](https://arxiv.org/abs/2503.08295)
  derives direct preference optimization for a discrete diffusion process while
  preserving a reference distribution. Our uniform-mask dynamic-crystal loss is
  reported as a task-specific shared-noise variant, not claimed to be its exact
  CTMC objective.
- [Diffusion Model Alignment Using DPO](https://arxiv.org/abs/2311.12908)
  establishes the broader denoising-likelihood DPO construction and motivates
  the fixed reference adapter.
- [Improving Classifier-Free Guidance in Masked Diffusion](https://arxiv.org/abs/2507.08965)
  finds that strong early guidance can harm masked-diffusion generation while
  late guidance is safer. Therefore CFG cannot be added opportunistically to
  the current run.
- [Fine-Tuning Discrete Diffusion Models](https://openreview.net/pdf/16e43842dab8730ca80ab853ebbbb3cfc84af913.pdf)
  develops policy-gradient fine-tuning for discrete diffusion. Its online
  reward model/actor-critic complexity is not justified before the lower-
  variance offline D3PO test.
- [Diffusion LAIR](https://arxiv.org/abs/2605.26491) uses all candidates for a
  prompt, continuous centered reward advantages, a reference-model implicit
  reward, and an explicit quadratic magnitude penalty. Because the existing
  data naturally contains 3--8 structures per composition, this is the most
  relevant low-variance objective fallback to audit if pairwise D3PO is
  negative; it precedes online RL or an invasive representation change.
- [Latent-Augmented Discrete Diffusion Models](https://openreview.net/forum?id=0sZ4DiHn76)
  and [Train for the Worst, Plan for the Best](https://openreview.net/pdf?id=DjJmre5IkP)
  support internal latent structure and deliberate reveal ordering. They
  motivate, but do not yet authorize, a structural-intent extension.
- [CrysLLMGen](https://arxiv.org/abs/2510.23040) preserves atom types from an
  LLM proposal and refines continuous geometry with equivariant diffusion,
  validating the hybrid story. It does not answer our harder exact-novel-
  composition attribution question.
- [DiffCSP++](https://arxiv.org/abs/2402.03992) demonstrates that space-group
  and Wyckoff constraints can improve crystal generation. This motivates
  symmetry auxiliaries only if they remain predicted inside the DLM.

## Only authorized fallback design

If both D3PO training seeds fail to shift raw energy left, the next candidate
is an internal latent structural-intent channel, not an external rich Plan:

- tokens/heads: crystal system, volume-per-atom bin, and coarse packing or
  coordination bin;
- labels derived from training-only MP20 structures;
- all intent states masked at inference and predicted jointly by the DLM;
- geometry denoising conditioned on the model's own intent posterior;
- exact composition/N remain visible and immutable;
- no energy value, hull label, prototype lookup, or C³FD-generated structure
  enters inference.

This keeps the DLM scientifically necessary. A C³FD rich Plan may be measured
only as an information upper bound with a matched C³FD-direct-CIF baseline.

If D3PO shifts raw energy left but model494 erases it, do not retrain the DLM.
Reuse the identical frozen raw bodies for one separately authorized legal-
forward-noise bridge comparison. If D3PO works but is weak, a single late-only
weak policy/reference guidance setting may be preregistered; early or swept CFG
is prohibited.

## Paper-standard claims

- Primary estimand: paired candidate-minus-base raw and refined CHGNet energy
  and official e-hull under two training seeds and two common RNG streams.
- Threshold outputs: Strict/Meta stable and S.U.N.; no hard stopping gate.
- Integrity: one attempt per Plan, unknown hull is missing, no survivor filter,
  reranking, replacement, seed/checkpoint selection, or adaptive denominator.
- Negative interpretations are predeclared: validation-only improvement,
  raw-only improvement erased by refinement, or one-seed-only threshold gains.
