# D3PO-256-Min proposed frozen contract

Status: **DLM-first direction confirmed; GPU execution waits for code/tests and
one final user notification.**

This contract replaces rank-conditioned scalar regression in the earlier RRC
draft. Zero-GPU analysis found exact within-composition CHGNet/official-hull
ordering and useful oracle headroom, but a chemsys-held-out handcrafted raw-body
rank probe remained at AUC `0.507`. The minimal remaining hypothesis is therefore
full-sequence preference optimization rather than a scalar critic.

The complete historical rationale is recorded in
[`DLM_EXPERIMENT_LINEAGE_FOR_D3PO_V1.md`](DLM_EXPERIMENT_LINEAGE_FOR_D3PO_V1.md).

## Scientific hypothesis

For the same fixed composition and N, increase the masked DLM probability of a
body whose **post-model494** energy is lower than another sampled body, without
selecting among completed samples at inference.

For winner `y+` and loser `y-`, sample one common mask probability `p`, timestep,
and geometry mask `M`. N and element tokens remain visible. Define the
reference-corrected geometry score

```text
S(y) = (1 / p) sum[j in M and geometry]
       (log pi_policy(y_j | y_t) - log pi_reference(y_j | y_t)).
```

The pair target is soft rather than binary:

```text
q = sigmoid((E_loser - E_winner) / 0.03 eV/atom)
L_pref = -q log sigmoid(beta * (S+ - S-))
         -(1-q) log sigmoid(-beta * (S+ - S-)).
L = L_pref + 0.2 * L_winner_denoising.
```

Freeze `beta=0.1`. Legal-token normalization is position-specific
(lattice/angle/X/Y/Z), log-softmax/gather/differences are FP32, and every score
uses the `1/p` masking-state correction. This is described as a
**shared-noise masked-D3PO variant**, not as the exact coupling used by the
original D3PO paper.

## Representation and module boundary

- Start from `ctv_minimal_base step696`; G1 is never the reference.
- Preserve dynamic exact-length `7 + 4N`; the historical fixed-slot
  107-token representation stays retired.
- Preserve exact/full-axis legality: lattice, all X, all Y, all Z.
- C³FD-v2.5 supplies the frozen minimal typed chemistry condition. No external
  rich Plan, stability token, energy value, or final structure is visible.
- model494 weights and tau800 remain fixed; current direct-as-`x_tau` bridge is
  unchanged in the primary comparison.
- One attempt per Plan; no rejection, repair, replacement, reranking,
  best-of-N, checkpoint choice, or survivor filtering.

## Reuse-first preference data

No new label-generation job is allowed.

1. Recompile the existing eight-stream energy-pair assets:
   - 1752 eligible outcomes over 222 Plans;
   - raw DLM body text as the target sequence;
   - post-model494/common-CHGNet energy as the preference label;
   - convert the 596-character historical rich prompt to the frozen minimal
     C³FD schema without changing body text;
   - include a historical Plan only if its composition can be represented by
     the current `all_metal`/`certified_neutral` minimal-spec contract. Count and
     report old `charge_fail`/unsupported rows; never relabel them as certified.
2. Retire completed SGTC L7 as an experiment and reuse its three matched bodies
   per known composition as additional training evidence. The immutable L7
   negative report remains archived, but L7 cannot subsequently be called a
   held-out D3PO test.
3. Canonicalize/cluster PBC-equivalent structures and identical body texts before
   pair construction.
4. Rebuild a salted chemsys-grouped train/validation split over the combined
   pool. All outcomes and pairs from one composition receive one split.
5. Use every non-tie pair, but multiply its loss by
   `min(1, abs(delta_E) / 0.06)`. Normalize all pair weights within each
   composition to sum to one. Balance source-arm winner/loser counts and remove
   source metadata from model input.
6. Report independent composition counts, not just pair rows. The existing
   eight-stream pool contributes about 164 train and 58 validation Plans under
   its old split; the exact new chemsys split is frozen before training.

The existing L6 two-seed cohort is excluded from training and remains the first
retrospective matched test. A fresh outcome-blind C³FD cohort is still required
for any paper-facing claim.

## Training implementation

Use a dedicated trainer; do not modify historical SFT/SGTC behavior.

- one 8B backbone with two named copies of the same starting crystal adapter:
  trainable `policy` and frozen `reference`;
- reference forward first under `no_grad`, retain only gathered target
  log-probabilities, then policy forward/backward;
- LoRA rank `8`, alpha `32`, dropout `0`;
- learning rate `5e-6`, independent training seeds `81017/81018`, exactly `348`
  optimizer updates for each;
- one pair/two sequences per microbatch, gradient accumulation `16`, BF16 plus
  FP32 score arithmetic, gradient checkpointing enabled from the start;
- run both seeds sequentially on one GPU and save only each scientific
  `step348`; no intermediate checkpoint, early stopping, or seed selection;
- validation preference margin/accuracy is diagnostic and cannot select a
  checkpoint or hyperparameter.

Before optimizer step 1, fail closed unless policy/reference produce zero
reference-corrected margin and `log(2)` hard-label loss on a deterministic test
batch, shared masks are bit-identical, N/elements are never masked, and swapping
winner/loser reverses the margin.

## Minimal GPU envelope

| Stage | Peak resources | Wall-time limit | Search space |
|---|---:|---:|---|
| CPU pair build/audit | 0 GPU, 8 CPU | 45 min | none |
| two D3PO-LoRA seeds, sequential | 1 A800, 8 CPU | 10 h | two mandatory step348 checkpoints |
| matched generation+tau800 refine | 2 A800, 16 CPU | 4 h | base plus both D3PO seeds |
| raw/refined evaluation | 2 A800, 16 CPU | 1.5 h | fixed cells only |
| official query/finalize | 0 GPU, 8 CPU | 45 min | one fresh query |

Run only one Slurm job at a time. Peak occupancy is two A800; the conservative
total ceiling is `18 A800-hours`. No requested1000, tau900, forward-noise factorial,
Planner change, rich-Plan arm, or second checkpoint is authorized.

## Two-seed evaluation

Historical H1-A2 and R03 high points were seed/process fragile, so a one-seed
increase is not accepted as clear evidence.

- Freeze 256 outcome-blind test Plans from unused C³FD seed17 attempts after
  exact-identity exclusion only; do not use stability or realizability scores.
- Report the test cohort's N/arity/family/element distance to full C³FD/MP20.
- Run the frozen base and both D3PO training seeds under two fixed common DLM
  sampling/refiner streams (six cells total) and paired per-ordinal RNG.
- Use one fixed refiner seed per DLM seed, model494 tau800, and the same Direct,
  CHGNet, official-hull and unknown-as-missing contracts.
- Evaluate raw body and refined body. Fresh base cells are run through the same
  code revision; a deterministic adapter-equality canary precedes full sampling.

## Evidence tiers, not an adaptive gate

All fixed256 cells finish regardless of scientific direction. No retries,
additional samples, or parameter changes follow a negative result.

The strongest credible positive would contain all of:

- both independent D3PO training seeds show negative candidate-minus-base
  paired refined-energy means under both common sampling streams;
- pooled paired CHGNet and official e_hull intervals lie predominantly left of
  zero, with raw evidence moving in the same direction;
- Meta S.U.N. improves on both seeds and pooled by roughly `1.5--3 pp`;
- Strict S.U.N. is non-decreasing, while Direct and NU remain materially intact.

Strict events are sparse: the existing evidence supports an expected Strict
gain of roughly `0--0.6 pp`, not a promise of statistical significance. The
K3/K6 oracle gains are ceilings, not expected D3PO effects. If energy shifts
left but threshold counts do not, report a continuous mechanistic positive. If
validation preference improves but raw/refined energy does not, report another
training-objective negative. If raw improves but refinement erases it, stop and
request separate authorization for a one-GPU bridge-only re-refinement of the
same raw bodies.

## Engineering-only recovery

One identical retry is allowed only for node preemption or transient filesystem
failure. Data/hash mismatch, composition mismatch, adapter inequality, NaN/Inf,
OOM, missing/duplicate ordinals, or false success markers stop the run. Recovery
cannot change batch, accumulation, beta, LR, LoRA rank, steps, seed, Plan,
checkpoint, tau, or denominator.
