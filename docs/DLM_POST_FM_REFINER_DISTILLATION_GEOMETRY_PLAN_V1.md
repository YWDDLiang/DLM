# Post-F/M refiner distillation and minimal-geometry DLM plan

Date: 2026-08-31

Status: planning-only until the fixed F/M prospective run is terminal

## Decision in one sentence

If Route M preserves C3FD formula validity and does not show a material raw
execution regression relative to Route F, retain M as the Planner mainline and
make the next method change inside the masked crystal DLM: add a minimal
gross-collision decoding mask, then train a fresh DLM with train-only
model494-relaxed geometry targets. Route F remains the disclosed formula-only
ablation.

This is not permission to select or hide an F/M result. Both routes, both
streams, and all failed rows remain in the F/M report.

## Historical audit: what was and was not run

No completed **model494-geometry-to-DLM distillation** experiment was found.
The repository-wide Git history and current tree contain no distillation data
builder, training entry point, Slurm job, checkpoint, or terminal report whose
CE target is the geometry emitted by model494. Relaxed-winner distillation was
specified in `DLM_SUN_STABILITY_MECHANISM_DEEP_DIVE_V2.md` and
`DUAL_TRACK_COMPOSITION_STABILITY_PLAN_V1.md`, but it remained a proposed
fallback.

Three nearby experiments can otherwise be mistaken for that method:

1. **model494 refinement/tau calibration was executed.** On the matched L6
   evidence, raw-to-tau800 changed Direct `188 -> 457`, Strict S.U.N.
   `10 -> 48`, and Meta S.U.N. `66 -> 230` over 512 attempts. This proves that
   model494 is a strong downstream converter, not that its correction was
   learned by the DLM.
2. **SGTC-DLM-v1 was executed.** It trained geometry-only CE on real MP20
   structures, comparing all MP20 rows with a strict-stable subset. Its L7
   base/G0/G1 Strict results were `60/55/53` and Meta `412/421/417` per 1000.
   The model494 output was not the CE target; the formal result was negative.
3. **masked-D3PO/listwise training used post-model494 energy labels.** Its
   target sequences were raw DLM bodies. It learned a preference over bodies;
   it did not serialize and imitate the model494-relaxed geometry. Raw behavior
   was adverse or non-replicated even when refined means were weakly favorable.

The proposed method below is therefore a new, narrower experiment, not a rerun
of SGTC or D3PO.

## What the decoder already constrains

The exact-dynamic DLM path already implements:

- position-specific token-schema masks;
- exact `N` and exact element-multiset prefill from the Plan;
- the exact-axis schedule: lattice, all X, all Y, then all Z;
- nonzero lattice lengths;
- a positive lattice-volume angle radicand;
- rejection of exact/PBC-equivalent duplicate coordinate triplets;
- strict body parsing and formula/N validation.

These guards explain why body execution can be high, but they do not impose a
finite interatomic separation. Two sites can be different coordinate bins and
still be an unphysical near-collision. The decoder also does not use periodic
species-pair distances, coordination, or a broad volume-per-atom envelope.

## Minimal new decoding constraint

Add exactly one new scientific guard first:

### Species-aware PBC gross-overlap mask

When a Z token is committed, X/Y, lattice, species, and any previously committed
Z values are known. For each candidate Z bin:

1. construct the minimum-image periodic distance to every committed site;
2. reject only candidates below a species-pair gross-overlap floor;
3. derive the floor once from MP20 train structures, with a covalent-radius
   fallback for unseen pairs;
4. freeze the table/scalar before any F/M prospective outcome is used;
5. record masked-token counts and no-legal-token failures.

The floor is a support constraint, not an energy score. It does not select a
completed sample, retry, repair, rerank, replace, or use a target structure.
If no legal token remains, the attempt fails and stays in the denominator.

Do not hard-enforce predicted space group or the exact M volume bin in this
first change. Those are uncertain Planner outputs. A broad train-derived
volume-per-atom envelope may be added only as a separately reported ablation
after the pair-distance guard is isolated.

## New training experiment: model494 basin distillation

Run this only after M is frozen as the Planner mainline.

### Train-only teacher construction

- Use MP20 train compositions only; keep chemical-system-held-out validation.
- Produce one M Plan and one DLM trajectory per source row under frozen seeds.
- Apply the frozen model494 at tau800 once to every parsed body.
- Keep all attempts in the audit. A row becomes a CE teacher only when the
  refined output preserves the exact atom multiset and passes structure parsing.
- Serialize the refined lattice and fractional coordinates back into the same
  dynamic `7+4N` token language with deterministic species-slot canonicalization.
- Use no energy threshold, best-of-K winner, survivor filter, retry, or outcome
  selection. The teacher is the single matched refiner output.

This transfers the correction `raw body -> model494-relaxed body`; it does not
teach the DLM to reproduce an official hull label.

### Fresh DLM training

Initialize new LoRA adapters from the shared pretrained LLaDA crystal base, not
from the old H1-A2 adapter or a selected F/M result. Use two fixed training
seeds and one frozen schedule.

Mix two supervised views with equal source-level weight:

- original MP20 teacher-rich Plan -> original MP20 crystal body;
- M-compatible rich Plan -> the single model494-relaxed on-policy body.

Use ordinary masked denoising CE with geometry-position supervision, plus a
frozen-reference bound. Apply deterministic periodic-origin translation and
species-preserving site permutation before serialization so token CE does not
overfit one arbitrary origin or site order. Do not add energy prompts,
listwise/alignment loss, RL, or extra epochs selected by validation.

The first experiment deliberately avoids a new graph network. If it improves
body parsing but not raw Direct/energy, the next architecture—not another CE
epoch—is a periodic pair-distance/metric-tensor auxiliary head at random mask
times.

## Matched evaluation after F/M

Use M and a train/validation-only fixed cohort for a small 2x2 causal screen:

| DLM weights | Decoder |
|---|---|
| fresh MP20-only control | existing guards |
| fresh MP20-only control | + PBC gross-overlap mask |
| model494-distilled DLM | existing guards |
| model494-distilled DLM | + PBC gross-overlap mask |

Hold Plan rows, DLM noise, temperature, exact-axis schedule, requested
denominator, and model494 evaluation seeds fixed. Report raw first, then
refined. The primary mechanism endpoints are raw body, raw Direct joint,
collision rate, and raw CHGNet energy. Refined and official S.U.N. are system
endpoints.

Only after the complete train/validation screen is frozen may one new
outcome-blind prospective cohort and one official query be run. No setting is
selected on the current F/M prospective cohort.

## M mainline rule

M is the preferred mainline on methodological grounds because it is the route
that actually carries frozen C3FD scientific state into the Rich Expander. It
is retained after F/M if all of the following hold:

- C3FD formula/N validity remains 100% and the Expander changes zero formulas;
- strict Rich-Plan parsing and DLM body execution remain at least 97%;
- aggregated raw Direct is not more than 5 percentage points below Route F;
- no gross adverse raw CHGNet shift is supported by a paired interval.

Compact V2's development values are context only, because that cohort contains
MP20 train/validation rows. Similar compact-V2-level refined S.U.N. supports
feasibility but is not a selection gate. If M misses the noninferiority rules,
retain F as the engineering fallback and report M as a negative ablation; do
not tune M on the same cohort.

## Paper interpretation

If the full path succeeds, the DLM-centered claim is:

> A C3FD-conditioned Rich Planner supplies hard-valid composition and a soft
> structural prior, while a minimally constrained, refiner-distilled masked
> DLM learns to realize physically plausible discrete crystal geometry before
> continuous refinement.

Required attribution boundaries:

- C3FD owns composition validity.
- M owns the learned Planner-conditioning interface.
- The masked DLM owns raw lattice/site realization.
- model494 is an inherited teacher and fixed downstream refiner.
- A refined-only gain is not evidence that the DLM learned stability.

## Ordered checklist

- [x] Audit Git history, code, jobs, and terminal reports for prior
  model494-to-DLM geometry distillation.
- [x] Enumerate the exact decoder guards already in production.
- [ ] Finish and disclose all F/M prospective raw/refined/official results.
- [ ] Apply the frozen M-mainline rule without deleting Route F or M outcomes.
- [ ] Build and unit-test the PBC gross-overlap mask.
- [ ] Run the fixed train/validation decoder A/B; archive all failures.
- [ ] Build immutable train-only model494-relaxed teacher data.
- [ ] Train two fresh distilled DLM seeds under one frozen schedule.
- [ ] Run the fixed 2x2 train/validation screen and decide the method before a
  new prospective cohort exists.
- [ ] Run one new prospective raw/refined evaluation and one official query.
- [ ] Update BUILD_STATUS/PAPER_STORY with SUPPORTED/CANDIDATE/UNSUPPORTED
  attribution for C3FD, M, DLM geometry guards, distillation, and model494.
