# DLM thermodynamic-stability research report

Date: 2026-08-27

Final API query: `thermodynamic stability crystal generation stability-conditioned fine-tuning discrete diffusion guidance`

Window: 2021–2026. The API query is intentionally shown in full before the
domain-specific synthesis. Several keyword matches are irrelevant to inorganic
crystal generation; retaining them makes the search audit reproducible.

Search errors (verbatim):

```text
[openreview] Error: openreview not installed. pip install openreview-py
[dblp] Error: 503 Server Error: Service Unavailable for url: https://dblp.org/search/publ/api?q=thermodynamic+stability+crystal+generation+stability-conditioned+fine-tuning+discrete+diffusion+guidance&format=json&h=5&f=0
```

The APIs also repeatedly reported `Rate limited. Waiting 3 seconds...` before
completing the remaining sources.

## Semantic Scholar (5 papers)

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [1](https://www.semanticscholar.org/paper/a74978f2cc7f4224830f58338a02dc74b6167d9a) | DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models | 2025 | AAAI | 5 |
| [2](https://www.semanticscholar.org/paper/f772d1ef93db9be7ed4a59dbf555c76094756635) | CrystalReasoner: Reasoning and RL for Property-Conditioned Crystal Structure Generation | 2026 | arXiv | 1 |
| [3](https://www.semanticscholar.org/paper/b4a9338a7f8f067e6c32f63186a45cc195641358) | Inverse Materials Design via Joint Generation of Crystal Structures and Local Electronic Descriptors | 2026 | — | 0 |
| [4](https://www.semanticscholar.org/paper/210d0a52c7de80b7c3d7dc72cb848eb41fdf15d3) | KungFuCap: monocular martial 3D reconstruction via Gaussian splatting and diffusion-guided view synthesis | 2026 | International Conference on Computer Vision and Image Computing | 0 |
| [5](https://www.semanticscholar.org/paper/0efd7dcc975cf2deea2ba5553330df7e60309768) | Bi-level Training of Latent Diffusion Model for Traffic Simulation | 2026 | IEEE Intelligent Vehicles Symposium | 0 |

## OpenAlex (5 papers)

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [1](https://doi.org/10.1002/aic.17815) | Online learning-based predictive control of crystallization processes under batch-to-batch parametric drift | 2022 | AIChE Journal | 50 |
| [2](https://doi.org/10.1021/jacs.1c06854) | Accumulated Lattice Strain as an Internal Trigger for Spontaneous Pathway Selection | 2021 | JACS | 14 |
| [3](https://doi.org/10.1038/s44221-025-00474-z) | Diffusion-driven selective crystallization of high-purity salt through simple and sustainable one-step evaporation | 2025 | Nature Water | 25 |
| [4](https://doi.org/10.1016/j.desal.2025.118888) | Solar-thermal gradient-driven evaporation for enhanced fractional crystallization | 2025 | Desalination | 7 |
| [5](https://doi.org/10.1016/j.partic.2024.10.018) | Optimization of batch cooling crystallization systems considering crystal growth, nucleation and dissolution. Part I: Simulation | 2024 | Particuology | 7 |

## arXiv (5 papers)

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [1](https://arxiv.org/abs/2412.10193) | Simple Guidance Mechanisms for Discrete Diffusion Models | 2024 | arXiv / ICLR 2025 | 0 in API |
| [2](https://arxiv.org/abs/2402.03701) | Unified Discrete Diffusion for Categorical Data | 2024 | arXiv | 0 in API |
| [3](https://arxiv.org/abs/2302.05737) | A Reparameterized Discrete Diffusion Model for Text Generation | 2023 | arXiv | 0 in API |
| [4](https://arxiv.org/abs/2510.27364) | Fine-Tuning Open Video Generators for Cinematic Scene Synthesis: A Small-Data Pipeline with LoRA and Wan2.1 I2V | 2025 | arXiv | 0 in API |
| [5](https://arxiv.org/abs/2506.07177) | Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models | 2025 | arXiv | 0 in API |

## OpenReview (0 papers)

No results were returned because `openreview-py` was not installed. The error
is recorded above. Primary OpenReview papers were subsequently opened directly
and are listed in the verified reading set below.

## Crossref (5 papers)

| # | Title | Date | Venue | Citations |
|---:|---|---:|---|---:|
| [1](https://doi.org/10.1098/rspb.2024.1827/v1/review1) | Review for “Fine-tuning the evolutionary stability of recombinant herpesviral transmissible vaccines” | 2024 | — | 0 |
| [2](https://doi.org/10.1098/rspb.2024.1827/v2/review1) | Review for “Fine-tuning the evolutionary stability of recombinant herpesviral transmissible vaccines” | 2024 | — | 0 |
| [3](https://doi.org/10.1098/rspb.2024.1827/v1/review2) | Review for “Fine-tuning the evolutionary stability of recombinant herpesviral transmissible vaccines” | 2024 | — | 0 |
| [4](https://doi.org/10.21203/rs.3.rs-10067094/v1) | Induction Circuit Stability Under Fine-Tuning: A Mechanistic Interpretability Study | — | Research Square | 0 |
| [5](https://doi.org/10.26434/chemrxiv-2024-nm8ks) | Thermodynamic stability and diffusion mechanism of LiMXCl4 superionic conductors | 2024 | ChemRxiv | 1 |

## DBLP (0 papers)

No results were returned because the endpoint returned HTTP 503. The exact
error is recorded above.

## Model knowledge and direct-primary-source completion (10 verified papers)

These entries were deduplicated against the final API tables and verified at
their primary paper pages.

| # | Title | Year | Venue | Notes |
|---:|---|---:|---|---|
| [M1](https://www.nature.com/articles/s43588-026-01037-2) | Enhancing materials discovery with valence-constrained design in generative modeling (CrysVCD) | 2026 | Nature Computational Science | Generated candidates are MLIP-labelled stable/unstable and used for stability-conditioned fine-tuning with CFG. |
| [M2](https://www.nature.com/articles/s41586-025-08628-5) | A generative model for inorganic materials design (MatterGen) | 2025 | Nature | Stable-data scaling, property adapters and classifier-free guidance; 607,683 stable structures in Alex-MP-20. |
| [M3](https://arxiv.org/html/2402.04379) | Fine-Tuned Language Models Generate Stable Inorganic Materials as Text | 2024 | arXiv | Crystal-text LoRA trained 21–65 epochs; temperature exposes a stability–coverage trade-off. |
| [M4](https://openreview.net/pdf?id=03RLpj-tc_) | Crystal Diffusion Variational Autoencoder for Periodic Material Generation | 2022 | ICLR | Stable crystal manifold and property optimization baseline. |
| [M5](https://openreview.net/pdf?id=DNdN26m2Jk) | Crystal Structure Prediction by Joint Equivariant Diffusion (DiffCSP) | 2023 | NeurIPS | Joint periodic-equivariant denoising trained on stable crystals. |
| [M6](https://arxiv.org/abs/2503.08295) | D3PO: Preference-Based Alignment of Discrete Diffusion Models | 2025 | arXiv | Non-RL reference-regularized preference objective for discrete diffusion; evidence is still a structured binary task. |
| [M7](https://arxiv.org/abs/2311.12908) | Diffusion Model Alignment Using Direct Preference Optimization | 2024 | CVPR | Derives diffusion DPO from a diffusion likelihood bound; continuous image-domain evidence. |
| [M8](https://arxiv.org/abs/2305.20009) | Protein Design with Guided Discrete Diffusion | 2023 | arXiv | NOS guides hidden states with a value model while using KL to preserve the generative prior. |
| [M9](https://openreview.net/forum?id=BTeWafMOyt) | Latent Conservative Objective Models for Data-Driven Crystal Structure Prediction | 2023 | AI4Mat | Shows ordinary energy surrogates can be exploited and worsen true energy; conservatism matters. |
| [M10](https://www.nature.com/articles/s42256-023-00716-3) | CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling | 2023 | Nature Machine Intelligence | Reports about 30 meV/atom test MAE for energy and provides a fast stability oracle. |

# Summary of searched results

## 1. Overview

The final API query covered 2021–2026 and returned 20 records from four working
sources; OpenReview and DBLP failed as documented. Ten additional primary
papers were verified directly. Keyword search alone had poor precision because
“diffusion”, “crystal” and “stability” are highly polysemous; the useful corpus
comes from the primary-source completion.

## 2. Trends

- 2021–2023 established stable-crystal generative priors and periodic geometry
  (CDVAE, DiffCSP), plus conservative offline energy optimization.
- 2024–2025 shifted toward scalable stable-data pretraining, property adapters,
  classifier-free guidance and preference alignment (MatterGen, discrete
  guidance, Diffusion-DPO and D3PO).
- 2026 has direct crystal evidence for concurrent stability-labelled training:
  CrysVCD reports thermodynamic and phonon gains after self-generated examples
  are labelled by MatterSim and fed back into a conditional generator.
- Across domains, stronger guidance or lower sampling temperature tends to
  improve target adherence at a cost to diversity/coverage, making a Pareto
  report essential rather than optional.

## 3. Key themes

1. **Stable-data density modelling:** learn the manifold from stable structures
   with sufficient data, epochs and symmetry augmentation (M2–M5).
2. **Stability-conditioned concurrent learning:** generate, score with an MLIP,
   retain both positive and negative labels, then train a conditional denoiser
   (M1).
3. **Discrete guidance:** train a stability predictor on corrupted states or a
   conditional/unconditional denoiser and alter reverse transitions (arXiv-1,
   M8).
4. **Non-RL preference learning:** pair preferred and rejected generations and
   preserve a reference-model prior (M6, M7).
5. **Surrogate robustness:** direct optimization can exploit predictor errors;
   use conservative models, uncertainty or oracle agreement (M9, M10).

## 4. Keywords frequency

Normalized paper-level title occurrence over the 30-record final corpus:

| Keyword group | Count |
|---|---:|
| diffusion | 14 |
| crystal / material | 12 |
| fine-tuning / training | 9 |
| guidance / conditioning | 6 |
| stability / thermodynamic | 5 |

## 5. Most cited by accepted paper

This mechanical API ranking is dominated by irrelevant crystallization-process
papers, illustrating the keyword-precision problem.

| Rank | Title | Year | Citations |
|---:|---|---:|---:|
| 1 | Online learning-based predictive control of crystallization processes under batch-to-batch parametric drift | 2022 | 50 |
| 2 | Diffusion-driven selective crystallization of high-purity salt through simple and sustainable one-step evaporation | 2025 | 25 |
| 3 | Accumulated Lattice Strain as an Internal Trigger for Spontaneous Pathway Selection | 2021 | 14 |
| 4 | Solar-thermal gradient-driven evaporation for enhanced fractional crystallization | 2025 | 7 |
| 5 | Optimization of batch cooling crystallization systems considering crystal growth, nucleation and dissolution | 2024 | 7 |

## 6. Most cited by first author

| Rank | Author | Papers in set | Total citations |
|---:|---|---:|---:|
| 1 | Yingzhe Zheng | 1 | 50 |
| 2 | Yang Liu | 1 | 25 |
| 3 | Hubiao Huang | 1 | 14 |
| 4 | Shiyuan Deng | 1 | 7 |
| 5 | Qilei Xu | 1 | 7 |

## 7. Recommendations for reading

1. **CDVAE → DiffCSP:** understand why periodic geometry and the stable-data
   manifold are the base prior before adding a reward.
2. **Fine-Tuned Language Models Generate Stable Inorganic Materials as Text:**
   closest evidence that crystal strings can learn stability through many-epoch
   stable-data LoRA and symmetry augmentation.
3. **MatterGen:** strongest evidence for stable-data scale, adapters and CFG.
4. **CrysVCD:** most direct blueprint for non-RL generated-negative feedback and
   stability-conditioned crystal generation.
5. **Simple Guidance + D3PO + NOS:** implementation choices for transferring
   property control to a masked discrete DLM, with D3PO treated as preliminary.

# H1-A2-specific technical synthesis

## What the current counterfactual loss actually learns

The current implementation compares the log probability of the **same factual
geometry target** under a factual Plan prompt and a structurally mismatched
counterfactual Plan prompt:

```text
margin = log p_theta(y_geometry | P_factual)
       - log p_theta(y_geometry | P_counterfactual)
```

It never supplies an unstable geometry as the rejected answer and never uses
energy. A positive margin therefore proves Plan-field grounding, not
thermodynamic preference. Its failure to improve stability is mechanistically
unsurprising.

## Training sufficiency audit

The original B0 audit shows:

- 27,136 exact-length training rows;
- one epoch, 1,696 optimizer steps, batch size 1 per rank, world size 2 and
  gradient accumulation 8;
- LoRA rank 8/alpha 32, cosine LR from `5e-5`, no origin-shift augmentation;
- validation CE still fell from `3.291` at step500 to `2.600` at step1000 and
  `1.967` at step1500;
- its historical 256 sampler produced only 29 graph successes under that old
  sampling contract.

The later checkpoint sweep initialized from B0. Its reported continuation
fractions 0.295/0.590/1.000 therefore correspond to approximately
**1.295/1.590/2.000 total epochs**, not absolute fractions of an epoch. By two
total epochs, validation CE reached about 1.289 and body validity was near
saturation, but stability remained non-monotonic.

For context, the crystal-text LLM study trained LoRA for 21–65 epochs and used
random translation augmentation. MatterGen used 607,683 stable structures and
reported large gains from data scaling. Epoch counts are not directly
comparable across architectures, but one pass over 27k structures is plainly
not evidence of convergence.

## Ranked methods for making this DLM learn stability

| Rank | Method | Evidence | Fit to this DLM | Main risk | Recommendation |
|---:|---|---|---|---|---|
| 1 | Longer stable-only SFT + origin-shift augmentation | Crystal-text LLM; MatterGen data scaling | Native current CE path | novelty/coverage contraction | Mandatory baseline repair, not a new contribution |
| 2 | Stability-labelled concurrent SFT + Plan-preserving CFG | CrysVCD; discrete CFG derivations | Only needs a stability field, condition dropout and dual prompts | label noise; condition shortcut | Best first new method |
| 3 | Same-exact-Plan energy preference loss | D3PO/Diffusion-DPO precedent | Reuses current pairwise denoising code | D3PO evidence is preliminary; pair scarcity | Best contribution add-on after rank 2 works |
| 4 | Noisy-state stability head / NOS-style classifier guidance | NOS and discrete classifier guidance | Technically possible through shared hidden states | extra inference cost and surrogate exploitation | Second-line ablation |
| 5 | Direct differentiable CHGNet loss via soft tokens/Gumbel | DRAKES-like relaxation, energy-guided SDE analogies | Poor: parser, lattice and coordinates are discrete | unstable gradients and reward hacking | Do not start here |
| 6 | RL fine-tuning | CrystalFormer-RL and later work | Possible | instability, complexity, hard credit assignment | Excluded by current preference |

## Recommended method: fixed-Plan thermodynamic feedback conditioning

### Data

1. Keep exact Plan/composition/N fixed.
2. Generate multiple bodies per train-only Plan with independent seeds.
3. Apply the same frozen model494 refiner.
4. Label the **input body** by its downstream refined energy/hull outcome. This
   explicitly learns “which discrete bodies enter a low-energy basin under the
   frozen refiner”.
5. Use CHGNet and MatterSim agreement where possible. CHGNet reports about
   30 meV/atom energy MAE, so pairwise supervision should require a conservative
   gap, initially at least 60–80 meV/atom, or agreement of two MLIPs.
6. Unknown hull remains missing and never becomes a negative label.
7. Split by exact formula/Plan before creating pairs to prevent leakage.

Within an exact composition, the phase-diagram reference is constant. Ranking
two structures by `E_hull` is therefore equivalent to ranking them by compatible
formation energy, which removes the proposal-composition shortcut.

### Conditioning objective

Add a condition such as a continuous clipped energy value or bins
`E0/E10/E50/E100/EHI` to the prompt. Train ordinary masked CE on both stable and
unstable generated bodies under their true labels, mixed with the original
stable MP-20 data. Drop only the stability condition for 10–20% of examples.

At inference, preserve the full Plan in both branches:

```text
logits_guided = logits(P, condition=null)
              + s * [logits(P, condition=stable)
                     - logits(P, condition=null)]
```

The current CFG implementation masks the entire prompt, including Plan, so it
must not be reused unchanged. It needs a paired conditional/null-stability
prompt while keeping composition, N and all Plan fields identical.

### Optional same-Plan preference regularizer

For a low-energy body `y+` and high-energy body `y-` under the same Plan and the
same corruption mask, define a denoising score as negative masked CE and use a
reference-corrected preference loss:

```text
L_pref = -log sigmoid(beta * [
    (s_theta(y+|P) - s_theta(y-|P))
  - (s_ref(y+|P)   - s_ref(y-|P))
])
```

Mix it with the ordinary stable-data CE. This is non-RL and prevents the model
from drifting solely to maximize an imperfect energy signal. It should be an
add-on, not the first experiment, because D3PO has not yet demonstrated
large-vocabulary crystal language modelling.

## Proposed evidence sequence before any large run

1. **Training-sufficiency control:** continue the retained plain-control
   checkpoint to approximately 3 and 5 total epochs with origin-shift
   augmentation and a decayed LR. Evaluate every frozen checkpoint; do not
   select by CE alone.
2. **Conditional pilot:** stability-conditioned CE only, same training budget,
   no preference term, guidance scales pre-frozen on a validation Plan set.
3. **Preference ablation:** add same-Plan reference-corrected ranking only if
   conditional CE improves stable counts without collapsing body/novelty.
4. **Held-out evaluation:** exact formulas and Plans unseen during feedback
   training, two seeds, fixed DLM/refiner streams, all-request denominator.
5. **Gates:** both Strict and Meta attempt yield positive; body, Direct joint,
   novelty and stable-to-S.U.N. retention each no worse than -1 pp; report raw
   stable counts, energy quantiles and MLIP/official agreement.

No new training job was started during this research pass.

## Contribution boundary

“Stability-conditioned generation” itself is no longer novel after CrysVCD.
The defensible contribution candidate is narrower:

> **fixed-Plan, same-composition thermodynamic feedback for a masked discrete
> crystal executor, trained without RL and evaluated as specification-conditioned
> stable conversion rather than proposal-mix change.**

That claim requires the same-Plan design and preferably the preference
regularizer; simply adding an `E_hull` prompt would be a useful engineering
baseline, not a standalone contribution.
