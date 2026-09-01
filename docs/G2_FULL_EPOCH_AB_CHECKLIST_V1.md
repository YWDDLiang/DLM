# G2 full-epoch A/B checklist

Status: HOLD until the immutable geometry audit and final independent review
both pass. This is a post-outcome matched development experiment, not a fresh
prospective claim.

## Frozen methods

- **A — G2-PBC-R:** Compact-V2 step3392, one full 1696-update continuation,
  shared bounded 125-image triclinic PBC in the objective and residual graph,
  and normalized species-aware overlap
  `ReLU((m-d)/m)^2`, where
  `m=clamp(0.55(r_i+r_j),0.60 Å,1.40 Å)`.
- **B — G2-PBC-RU:** exactly A plus a deterministic detached q0 uncertainty
  gate with floor 0.25. A and B share a guarded circular coordinate mean;
  resultant below 0.25 falls back to the linear mean.

Both use Compact-V2 seed82017 step3392, MP20 teacher train27136, training
seed81017, LR `5e-6` cosine/warmup25/min0.1, LoRA r8/a32/d0.05, geometry loss
weights `0.1/0.1/0.2/0.05`, one eligible step1696 checkpoint, no selection.

## Geometry audit

- [x] Reject 27-image: 15/2,292,019 MP20 train+val pairs across four
  structures disagree with pymatgen; maximum error 1.149 Å. Final BASE raw
  also contains one mismatch.
- [x] Verify 125-image on all 36,183 MP20 train+val structures: zero mismatched
  pairs against pymatgen, maximum absolute error `1.78e-14 Å`.
- [x] Verify frozen radius semantics: pymatgen empirical atomic radius with
  calculated-radius fallback; all 88 used elements covered, no 1.5 Å fallback
  used.
- [x] Verify the frozen margin conflicts with zero of 2,292,019 teacher pairs.
- [ ] Immutable audit job39150 `_SUCCESS` and output hash.
- [x] Preserve pre-science import failure job39149 as a 0-GPU engineering
  negative.

## Complexity and resource contract

- Each method uses one training seed and 2 A800 / 16 CPU; two simultaneous
  jobs use 4 A800, below the hard six-GPU ceiling.
- Per-rank batch1 × world-size2 × grad-accum8 keeps effective batch16, so one
  MP20 epoch is exactly 1696 optimizer updates.
- For N=20, the largest 125-image coordinate tensor is about
  `20×20×125×3=150,000` float values per rank; this is negligible beside the
  8B backbone. The local PBC kernel is 4.63× the old 27-image kernel, but the
  language-model forward remains dominant.
- B adds no trainable uncertainty parameters. Entropy/variance work touches at
  most `6+3N≤66` geometry positions and its gate is stop-gradient.

## Frozen evaluation

- [x] Reuse Plan SHA
  `5f1ae510fb35d7bbe0b5da4b32b0302f49d78dae653c5c31493db8a2219a54cb`;
  do not call the Planner again.
- [x] Freeze A/B raw wrapper on stream17, DLM seed91117, temperature0.7,
  exact-axis, fixed256, one trajectory, same Plan/noise, no retry/rerank.
- [ ] Run A/B raw body+Direct after both training jobs complete.
- [ ] Run full model494/CHGNet and cached-official development S.U.N. for every
  technically valid G2 arm; do not issue a new official query.

## Pre-registered interpretation

- Technical floor: body and composition at least `244/256`.
- Directional improvement: raw Direct greater than current G2 `121/256`.
- Meaningful improvement: raw Direct at least `134/256` (+5 pp).
- B replaces A only if body is no lower and Direct is at least A+8, or if its
  paired raw-energy CI is clearly better; otherwise retain simpler A.
- Target remains valid plus Strict/Meta S.U.N., but neither target authorizes
  sample, Plan, seed, checkpoint, or failed-row selection.

Engineering priors: A directional raw-Direct improvement 60–75%; A ≥5 pp
45–60%; B>A 40–55%; at least one ≥5 pp 55–70%; final 10/50 probability
20–35%. These are planning priors, not result gates.

