# RRC-DLM-v1 proposed implementation contract

Status: **draft; no execution is authorized until explicit user confirmation**.

RRC means **Refiner-aware Relative-rank Conditioning**. The proposal keeps
C³FD-v2.5 fixed at `composition + N`, makes the masked DLM the only learned
stability intervention, and initially keeps model494 at tau 800 fixed.

## Why SGTC is not continued

SGTC selected strict-stable MP20 structures but trained the same geometry-token
denoising cross entropy. MP20 already contains only entries with
`e_above_hull <= 0.08 eV/atom`, so SGTC supplied neither same-composition hard
negatives nor an energy margin. Its lower teacher-forced validation loss did not
identify which polymorph is lower energy for a fixed composition. Official L7
confirmed the mismatch: G1 produced `53` Strict and `417` Meta S.U.N. versus
base `60/412` and G0 `55/421`, with no paired CHGNet or official-hull advantage.

## Frozen module boundary

- C³FD outputs only the exact composition and atom count.
- RRC-DLM chooses the discrete structural realization and is the only module
  trained with stability supervision.
- model494 performs continuous projection/refinement with fixed weights.
- No inference-time rejection, replacement, reranking, best-of-N selection, or
  test-outcome composition selection is allowed.
- External rich Plans are diagnostic information upper bounds only. A direct
  C³FD-to-CIF baseline is mandatory before claiming that such an interface needs
  a DLM.

## Data construction

Use training-split compositions only. Hold out complete chemical systems for
validation, and preserve the existing L6/L7 cohorts as untouched retrospective
tests.

Proposed first dataset:

- 512 training compositions and 128 chemical-system-held-out validation
  compositions, stratified to the frozen MP20-like N/arity/family marginals;
- four independently sampled base-DLM bodies per composition;
- the real MP20 structure when available plus controlled lattice, coordinate,
  and same-composition site-assignment perturbations as hard negatives;
- model494 tau-800 refinement under two fixed refiner seeds;
- `4096` training and `1024` validation body/refiner outcomes before optional
  hard-negative augmentation;
- composition-local labels: refined CHGNet energy-per-atom rank, mean rank over
  refiner seeds, rank variance, and official hull only where already available.

Absolute energies are never ranked across different compositions. The L7
attempts may be used only for frozen post-training evaluation, not to tune the
new loss or guidance scale.

## Model objective

The first implementation is deliberately lower risk than RL:

```text
L = L_masked_denoising
  + lambda_rank * L_same_composition_pairwise
  + lambda_quantile * L_energy_quantile
  + lambda_exact * L_composition_and_count_adherence
```

- Add five composition-local energy-rank quantile conditions plus a null
  condition.
- Predict a scalar rank score from the predicted-clean representation at random
  diffusion timesteps, not only from the final sequence.
- Use condition dropout during training. At inference, apply one predeclared
  classifier-free guidance scale toward the lowest-rank condition.
- Preserve all `7 + 4N` exact-axis invariants and one attempt per Plan.
- Select loss weights and the single guidance scale on the training/held-out
  validation split before any fresh C³FD evaluation cohort is generated.

If held-out pairwise AUC and centered rank correlation remain near chance, stop
interpreting token likelihood as an energy signal and move to an internal
coarse-to-fine structural latent. If the supervised rank is learnable but endpoint
guidance is weak, the next escalation is stepwise reward-tilted posterior
alignment, not high-variance policy-gradient search.

## DLM-to-refiner interface factorial

The current pipeline passes a clean DLM body directly as the tau state. Test a
matched 2x2 factorial on the same frozen compositions:

| DLM | Refiner input |
|---|---|
| frozen base | current direct-as-`x_tau` bridge |
| RRC-DLM | current direct-as-`x_tau` bridge |
| frozen base | valid forward-corrupted tau-800 state |
| RRC-DLM | valid forward-corrupted tau-800 state |

This separates a learned DLM effect from an interface correction. Report raw
body Direct metrics, pre-refiner rank evidence, tau-800 refined energy, and
official hull. Tau 900 is a predeclared secondary sensitivity only; it is not a
selection axis.

## Evaluation and interpretation

Report every arm on the requested denominator:

- body, composition/structure/joint Direct validity, N/U/NU;
- CHGNet-known rate and composition-matched energy deltas;
- official e_hull ECDF and q10/q25/q50/q75/q90;
- Strict and Meta stable/S.U.N., retention, paired intervals and exact McNemar;
- exact-composition and chemical-system seen/unseen strata;
- raw-to-refined sign preservation and effect attenuation.

There is no binary research hard stop at 10/50. Interpretation is mechanistic:

- raw and refined left shifts: RRC-DLM learned transferable stability;
- raw left shift lost after refinement: the refiner interface is the bottleneck;
- no held-out rank predictability: the proposed body representation lacks the
  needed signal and requires hierarchical internal structural intent;
- validity improves without energy movement: composition/execution remains
  solved, but stability remains unsolved.

## Composition policy

The main benchmark keeps the original outcome-blind C³FD distribution exactly.
A train-only realizability score and KL/maximum-entropy constrained composition
tilt may be studied separately, with fixed N/arity/family/element marginals and
full TVD/effective-sample-size disclosure. It cannot be used to define the RRC
main result or to select compositions after observing generated structures.

## Resource envelope after approval

- at most two simultaneous jobs and six A800 GPUs total;
- data-label construction: one six-A800 job or two non-overlapping three-A800
  jobs;
- matched base/RRC training: no more than four A800 GPUs total;
- factorial generation/evaluation: four A800 GPUs, eight CPUs per GPU;
- no MatterSim work until its separate dependency decision is reopened.

