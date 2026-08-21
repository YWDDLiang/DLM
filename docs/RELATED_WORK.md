# Related-work map and positioning

This map is organized by the scientific layer each method addresses. It is not
a single-metric leaderboard: evaluation protocols, relaxation procedures, and
sample counts differ across papers.

## Crystal language models and representations

| Work | Venue | Main contribution | Relation to H1-A2 |
|---|---|---|---|
| [Fine-Tuned Language Models Generate Stable Inorganic Materials as Text](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5dc6d1c81754c32008c9667339b3fdd3-Abstract-Conference.html) | ICLR 2024 | Autoregressive text generation, infilling, and prompting for crystals. | Establishes the LM route; H1-A2 changes body generation from causal continuation to plan-conditioned masked completion. |
| [Mat2Seq](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e23133d34964a0a09f6d076fc4b922a4-Abstract.html) | NeurIPS 2024 | Canonical, SE(3)- and periodic-invariant crystal tokenization. | Solves representation invariance; H1-A2 addresses information-revelation order and typed constraints. The two are complementary. |

## Continuous, joint, and symmetry-native generators

| Work | Venue | Main contribution | Boundary for H1-A2 |
|---|---|---|---|
| [DiffCSP](https://proceedings.neurips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract-Conference.html) | NeurIPS 2023 | Joint equivariant diffusion in crystal space. | H1-A2 keeps categorical chemistry and cardinality in a native discrete state. |
| [FlowMM](https://proceedings.mlr.press/v235/miller24a.html) | ICML 2024 | Riemannian flow matching respecting crystal symmetries. | Targets continuous manifold modeling and efficient integration, not a discrete completion interface. |
| [CrysBFN](https://proceedings.iclr.cc/paper_files/paper/2025/file/1f09e1ee5035a4c3fe38a5681cae5815-Paper-Conference.pdf) | ICLR 2025 Spotlight | Periodic Bayesian flow with entropy conditioning and efficient sampling. | Models the periodic process end to end; H1-A2 factorizes typed discrete realization from continuous refinement. |
| [TGDMat](https://proceedings.iclr.cc/paper_files/paper/2025/hash/24e4e3234178a836b70e0aa48827e0ff-Abstract-Conference.html) | ICLR 2025 | Text-guided joint diffusion of types, coordinates, and lattice. | Uses text as guidance inside a joint denoiser; H1-A2 uses a Plan as a contract for a separate exact-cardinality body. |
| [SymmCD](https://proceedings.iclr.cc/paper_files/paper/2025/file/3a14ae9951e8153a8fc814b5f506b5b7-Paper-Conference.pdf) | ICLR 2025 | Symmetry-preserving diffusion in an asymmetric-unit representation. | H1-A2 does not claim exact space-group preservation. |
| [Wyckoff Transformer](https://proceedings.mlr.press/v267/kazeev25a.html) | ICML 2025 | Compressed Wyckoff representation and permutation-invariant AR generation. | Shows that AR can be structured and permutation invariant; H1-A2's distinction is non-prefix completion, not invariance. |
| [SGEquiDiff](https://proceedings.neurips.cc/paper_files/paper/2025/hash/697b2f31f99fb79f8a0a16e923b2471d-Abstract-Conference.html) | NeurIPS 2025 | Space-group-invariant likelihoods and equivariant coordinate diffusion. | Defines the symmetry-native frontier; a coarse Plan field must not be described as equivalent enforcement. |
| [CrystalDiT](https://ojs.aaai.org/index.php/AAAI/article/view/37121) | AAAI 2026 | Simple unified Transformer diffusion, periodic-table encoding, and balance-oriented training. | Provides the strongest unified-model counterpoint to H1-A2's modular factorization. |

## Hybrid proposal and refinement

| Work | Venue | Main contribution | H1-A2's incremental question |
|---|---|---|---|
| [FlowLLM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/51d317df78eded9eb3c9d3fb1091c279-Abstract-Conference.html) | NeurIPS 2024 | Uses an LLM distribution as the base for Riemannian flow refinement. | What should the discrete base expose as a contract, rather than as an unconstrained complete string? |
| [CrysLLMGen](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f789a628fca473e922c806657512a20f-Abstract-Conference.html) | NeurIPS 2025 | An LLM proposes atom types and geometry; equivariant diffusion retains species and refines geometry. | Can the proposal become plan-conditioned, exact-cardinality typed completion rather than an irreversible AR sequence? |

CrysLLMGen is the closest foundation. It establishes that the hybrid split is
useful, so H1-A2 must not claim the split itself. The current defensible gap is
narrower: a learned prior samples formula and coarse fields; formula-derived
cardinality and species instantiate an exact typed state; and a masked model
completes quantized lattice and coordinates before continuous refinement.
Replaying a Plan from MP-20 isolates downstream realization but cannot replace
the learned prior in the fully de novo claim.

## Masked discrete generation

| Work | Venue | Lesson used by H1-A2 |
|---|---|---|
| [Simple and Effective Masked Diffusion Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb0b13cc515724ab8015bc978fdde0ad-Abstract-Conference.html) | NeurIPS 2024 | Masked corruption and reverse prediction form a principled discrete generative model. |
| [Simplified and Generalized Masked Diffusion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html) | NeurIPS 2024 | State-dependent masking schedules are possible. |
| [Scaling up Masked Diffusion Models on Text](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ce1c1ff5d94079dea348a2317a889281-Abstract-Conference.html) | ICLR 2025 | Masked models scale and support bidirectional conditional reasoning. |
| [Masked Diffusion Models are Secretly Time-Agnostic Masked Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9e3b203e72c4e058de26d02a92a81844-Abstract-Conference.html) | ICLR 2025 | “Diffusion time” and categorical-sampling claims require care; H1-A2 relies on iterative masked completion, not mystique about diffusion. |
| [Diffusion Beats Autoregressive in Data-Constrained Settings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0f705a932553c08ebf0d1bc520b7cbc6-Abstract-Conference.html) | NeurIPS 2025 | Random masking trains across many information orders rather than one fixed factorization. |
| [Theoretical Benefit and Limitation of Diffusion Language Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2318d75a06437eaa257737a5cf3ab83c-Abstract-Conference.html) | NeurIPS 2025 | Efficiency depends on the target metric; H1-A2 should not claim automatic sampling-speed superiority. |

Constrained and ordered denoising are also established research areas:

- [DINGO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eb17a2030d1bd4a1bd29531bcd626705-Abstract-Conference.html)
  gives distribution-aware constrained inference for formal-language outputs.
- [DDPD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cbf883a744952d4a40591271a58ab9d0-Abstract-Conference.html)
  learns which positions to denoise or revisit.
- [PepTune](https://proceedings.mlr.press/v267/tang25n.html) combines a
  domain-dependent masking schedule with an invalid-loss objective.
- [Anchored DLM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8143b8c73073a9a23b9c18e400066471-Abstract-Conference.html)
  predicts important anchors before completing remaining tokens.

H1-A2 therefore does not claim constrained DLM inference, denoising order, or
anchoring in general. Its current contribution is their audited use in a
composition-anchored periodic-geometry executor.

## Reward guidance and distribution attribution

[Reward-guided crystal generation](https://www.nature.com/articles/s42256-026-01262-4)
reports both improved discovery metrics and a substantial shift in elemental
composition, including increased transition-metal and reduced non-metal
content. [PRO-MOF](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f7f47a73d631c0410cbc2748a8015241-Abstract-Conference.html)
shows that naive Pass@1 policy optimization can collapse onto a small set of
safe regions, while diversity-aware rewards mitigate that failure. Conversely,
[fixed-composition energy guidance](https://www.nature.com/articles/s41467-026-72362-3)
shows that structural improvement is possible without changing composition.

The correct conclusion is not that reward optimization only selects easy
chemistry. Aggregate stability must be decomposed into chemistry-mix change and
within-chemistry structural conversion. H1-A2 uses its explicit Plan/body/
refiner stages to make that attribution visible.

## 2026 horizon, not core accepted baselines

- [Crys-JEPA](https://arxiv.org/abs/2605.14759) explicitly formulates a
  stability--novelty trade-off and adds embedding screening plus generative
  refinement. H1-A2 may describe proposal-set quality and conversion, but
  should not claim that identifying this trade-off is itself new.
- [DynaCrys](https://arxiv.org/abs/2608.07401) jointly evolves space group,
  Wyckoff occupations, elements, and geometry through symmetry-aware symbolic
  diffusion. It sets a strong future boundary for any symmetry claim.

These are recent preprints and are treated as horizon work rather than
top-conference-accepted core comparisons.

## The open slot

Strong recent methods have addressed invariant representations, continuous
periodic manifolds, exact space-group symmetry, unified diffusion and flow,
hybrid proposal plus refinement, and stability-oriented feedback. H1-A2
targets a narrower slot:

> **Sample global chemistry, instantiate its exact-cardinality typed state,
> complete quantized periodic geometry by non-prefix masked generation, and
> audit separately what is gained by chemistry selection, discrete realization,
> and continuous refinement.**

This is interesting only if the interface is precise. “Replacing an AR model
with a DLM” is insufficient; the contribution is the coupling of Plan
semantics, exact cardinality, typed support, non-prefix information flow, and
refiner invariants.

## Recommended claim ladder

Safe claims:

- crystal serialization need not define the generation order;
- masked completion is a plausible inductive bias for composition-anchored
  periodic geometry;
- typed discrete and continuous variables benefit from explicit interfaces;
- H1-A2 exposes proposal quality and refinement conversion as separate stages.

Claims requiring direct causal evidence:

- masked completion is better than a matched AR body;
- the DLM itself causes higher Unique/Novel rates;
- the factorization dominates unified diffusion;
- the method is faster than AR or continuous generators.

Claims to avoid:

- AR cannot represent crystals;
- H1-A2 is permutation invariant or symmetry preserving;
- local support masks guarantee global validity;
- the hybrid LLM--diffusion architecture is new by itself.
