# DLM Noisy-State Energy Critic V1

Date: 2026-08-28

Status: implementation specification only. Do not train until the
conditioning/schedule and model494-timestep diagnostics are finalized.

## Scientific role

Teach the masked crystal DLM which partially denoised geometries are likely to
enter a lower-energy basin under the frozen downstream pipeline. The critic is
a sidecar model. It does not add `stable`, `E0`, `E_hull` or an energy target to
the Plan/DLM prompt, does not use policy-gradient RL and does not pick a final
sample from a generated pool.

The failed eight-stream DPO pilot is not the data gate for this method. That
pilot deliberately kept one extreme pair per Plan and required a 0.06 eV/atom
gap. The critic uses all compatible-energy-labelled bodies and continuous
within-Plan information.

## Frozen data contract

- Start from a new train-only rich-Plan cohort, exact-formula disjoint from L6
  and L7.
- Freeze formula-level train/validation split before body generation.
- Use the selected DLM conditioning/schedule and selected model494 injection
  timestep.
- Generate fixed independent sample-index streams; no replacement on failure.
- Keep a row only if body/refinement succeeds, Direct joint is valid and the
  compatible CHGNet energy is known.
- Unknown energy/hull is missing and never becomes a high-energy label.
- Novelty/uniqueness is evaluation-only and never enters the critic loss.
- Preserve body text, token ids, exact Plan, pre/post-refiner structures,
  compatible energy, stream seed and global ordinal.

The existing feasibility data contain 1,752 eligible structures across 256
Plans, with at least two eligible streams for 222 Plans. These numbers motivate
the architecture but cannot be silently redefined as the final critic cohort.

## Label construction

For Plan `P` with eligible compatible energies `E_i`, compute:

```text
center_P = median_i(E_i)
scale_P  = max(IQR_i(E_i), 0.02 eV/atom)
z_i      = clip((E_i - center_P) / scale_P, -5, 5)
r_i      = rank_i(E_i) / max(1, n_P - 1)
```

The primary regression target is `z_i`; `r_i` is an auxiliary ordinal target.
Because every comparison is within the exact same composition/Plan, the phase
diagram reference and composition-level energy offset cancel. Family, arity,
N and element identity cannot by themselves solve the task.

Do not train directly on the hard Strict threshold. Universal-MLIP error is of
the same order as the narrow near-hull band, while continuous ranking provides
denser and less brittle supervision. [CHGNet](https://arxiv.org/abs/2302.14231)
is trained on energies, forces, stresses and magnetic moments from roughly 1.5
million Materials Project trajectory structures; agreement with
[MatterSim](https://arxiv.org/abs/2405.04967) should be logged on the strongest
pairwise differences when available.

## Noisy-state construction

Use the exact DLM answer token sequence and its geometry-token mask. For each
body, draw the same corruption process as DLM training:

1. sample timestep/mask probability;
2. mask only free lattice and coordinate positions;
3. preserve N and element identities exactly;
4. retain the hard-anchor prompt;
5. record the visible-geometry fraction and timestep.

Oversample late/intermediate states where enough geometry is visible to make
energy ranking identifiable. Very high-mask states remain in the training set
with lower weight so the critic learns calibrated uncertainty rather than
guessing from composition.

## Model

Use a frozen copy of the selected LLaDA backbone plus:

- a small critic-only LoRA adapter;
- timestep and visible-fraction embeddings;
- attention pooling over answer-side lattice/coordinate hidden states only;
- one scalar energy-rank head and one uncertainty head.

Do not share the critic adapter with the generator in V1. This mirrors the
generator/predictor separation used by
[DAO-G/DAO-P](https://arxiv.org/abs/2503.10471) and prevents critic optimization
from silently changing the base sampling distribution.

## Loss

```text
L_reg  = confidence_weight * Huber(z_hat_i, z_i)
L_rank = mean_{same Plan pairs} softplus(
           -sign(E_j-E_i) * (z_hat_j-z_hat_i) / temperature
         )
L_unc  = Gaussian NLL using the predicted uncertainty

L = L_reg + lambda_rank * L_rank + lambda_unc * L_unc
```

Use all same-Plan comparisons but normalize each Plan to total weight one, so a
Plan with eight valid streams does not dominate one with three. Weight
comparisons smoothly by the energy gap and MLIP agreement; do not introduce a
new post-hoc hard gap to manufacture pair counts.

## Critic gate

Freeze thresholds before training. At minimum the held-out critic must:

- beat a family/arity/N/charge baseline in formula-grouped bootstrap samples;
- have positive held-out within-Plan Spearman correlation with a 95% bootstrap
  lower bound above zero;
- achieve low-versus-high within-Plan AUC above 0.60;
- retain direction in oxide, sulfide and long-N hard strata rather than only
  easy all-metal/halide Plans;
- remain calibrated as mask fraction changes;
- show no exact formula overlap with L6/L7.

Failure stops the critic route before generator training or guided sampling.

## Discrete trajectory guidance

Guidance acts only on the current masked transition. For a small, fixed number
of uncertain geometry positions, evaluate top-K generator proposals and use:

```text
guided_score(v at position j) =
    log p_DLM(v | current masked state, Plan)
    - gamma(t) * [E_critic(state with v) - E_critic(current state)]
```

Then commit according to guided score. Keep a fixed re-masking budget that can
reopen low-confidence or critic-conflicting geometry tokens. Formula, N,
elements and counts are immutable.

V1 constraints:

- one predeclared `gamma` calibrated on critic validation;
- fixed K and fixed number of critic-scored positions;
- a single evolving trajectory, not multiple completed structures;
- no official/CHGNet call at generation time;
- no final energy sort, rejection or replacement;
- report additional critic forward-pass cost.

Guidance for masked discrete diffusion should follow a discrete transition
derivation rather than reusing continuous CFG mechanically; see
[Simple Guidance Mechanisms for Discrete Diffusion
Models](https://arxiv.org/abs/2412.10193).

## L6 and L7 gates

First run two-seed L6 with the exact same Plans/noise streams against unguided
sampling. Report every attempt and paired McNemar tests.

Promotion requires:

- pooled Strict and Meta S.U.N. both positive;
- neither seed worse than -1 percentage point on Strict or Meta;
- body, Direct joint, novelty, uniqueness and stable-to-S.U.N. retention each
  noninferior by 1 percentage point;
- improved median and upper-tail `E_hull`, plus counts at
  `0/0.01/0.05/0.1 eV/atom`;
- no family/N distribution change because Plans are fixed.

Only then run requested-1000 L7. The terminal gate remains Strict S.U.N.
`>=10%` and Meta S.U.N. `>=50%`, with the public `105/488` result kept separate
until a complete replacement experiment passes.

## Fallback if discrete guidance is too expensive

Use the same labels to distil relaxed low-energy winners into the DLM:

- CE only on relaxed winners and stable MP-20 structures;
- high-energy generated bodies appear only in ranking/critic losses;
- frozen-reference KL protects body validity and novelty;
- one conservative update budget, two seeds and the same L6/L7 gates.

If both critic guidance and winner distillation fail after refiner calibration,
the remaining bottleneck is the text geometry representation. Replace that
part with a periodic E(3)-equivariant continuous decoder rather than extending
plain token CE again.
