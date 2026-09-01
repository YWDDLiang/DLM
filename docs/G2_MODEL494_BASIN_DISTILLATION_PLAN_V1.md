# G2 model494 basin-distillation plan V1

Status: CANCELLED BEFORE EXECUTION on 2026-09-01. The estimated full-corpus
cost (`~106 A800-hours`) is not authorized. Keep this document as the negative
design record; do not submit D3-R or D3-J from this contract.

Replacement direction: preserve the same frozen Planner and G2 initialization,
but teach the residual with strict-PBC worst-pair/CVaR structure risk and dense
local correction directions from a fixed MP20-train-only subset. Prefer frozen
model494 short-step/score displacement targets; a frozen force-potential target
is admissible only with explicit leakage and circular-evaluation disclosure.
Full tau800 endpoints are not required. A fresh same-composition K=4
validity-first energy-ranking stage is optional and must not reuse the retired
mixed-policy 3,614 candidates.

## Objective

Move the fixed model494 basin correction upstream into the raw masked DLM.
The correction is not distilled only into the periodic residual: the residual
is the primary local carrier, while Compact-V2 LoRA receives a smaller joint
update for global conditional structure consistency.

```text
frozen C3FD+Llama Plan
        ↓
Compact-V2 LoRA (small joint LR)
        +
G2 periodic residual (main correction LR)
        ↓
raw 7+4N structure closer to the model494 basin
```

C3FD, the Llama Planner, the special-token vocabulary, and model494 remain
frozen. No prospective/dev/S.U.N. outcome may enter training.

## Immutable train-only data

1. Use all 27,136 MP20-train source rows. Retain repeated compositions and
   known polymorphs; do not apply a unique-composition or certificate filter.
2. Materialize the frozen Planner Plan once per source row.
3. Generate exactly one raw trajectory per Plan from the promoted A/B method;
   no retry, replacement, reranking, best-of-N, or survivor filtering.
4. Apply model494 tau800 once to every parsed raw structure. Preserve every
   parse/refinement failure in the denominator.
5. Encode both raw and refined structures in the same Compact-V2 `7+4N`
   vocabulary and record the periodic correction:

   - metric-tensor and log-volume delta;
   - periodic coordinate delta in `(-0.5, 0.5]`;
   - pair-distance/RDF delta;
   - collision-resolution direction.

No energy, hull, official MP, canary, or prospective label is required for the
distillation data. The teacher signal is the frozen model494 geometry only.

## Two-stage training

### D3-R: residual-only full-source pass

- Freeze Compact-V2 LoRA and all backbone parameters.
- Train only the periodic residual for one complete source-balanced MP20-train
  pass: `27136/16 = 1696 updates`. Rows without a successful raw→refined pair
  remain in the manifest and contribute no fabricated correction target.
- LR `5e-6`; sole step1696 checkpoint is a mechanistic ablation, not a result-
  selected checkpoint.

### D3-J: joint residual + LoRA distillation

- Initialize from D3-R step1696.
- Train residual at LR `5e-6` and existing Compact-V2 LoRA at LR `1e-6`.
- Use two deterministic source epochs. The first exposes the original MP20
  teacher target; the second exposes the model494-refined target when present
  and the original target when refinement is unavailable. Per-source weight is
  one in each epoch, so the joint stage is `2×27136/16 = 3392 updates`.
- Retain a frozen-reference preservation term. Save only step3392; no early
  stopping or checkpoint selection.

The registered loss family is:

```text
original MP20 token CE
+ model494-refined token CE
+ periodic displacement / metric / pair-distance losses
+ reference-preservation loss
```

Loss scales may receive one MP20-train-only gradient-norm calibration; no grid,
development outcome, or prospective tuning is allowed.

The full raw→model494 corpus is expected to require roughly `106 A800-hours`
at the observed 256-structure-per-GPU-hour scale. One six-GPU immutable job is
estimated at 18–24 wall-clock hours. This is expensive but is the appropriate
paper-scale dataset; a 4,096-row subset is not an eligible scientific output.

## Evaluation and promotion

- Reuse the existing frozen Plan SHA
  `5f1ae510fb35d7bbe0b5da4b32b0302f49d78dae653c5c31493db8a2219a54cb`
  and matched stream17/noise first; Planner is not sampled again.
- Report Compact-V2 base, promoted G2, D3-R, and D3-J raw Direct and raw
  CHGNet. model494/refined S.U.N. is evaluated only after raw accounting is
  immutable.
- D3-J is retained only if body/comp remains at least `244/256`, raw Direct is
  not below promoted G2, and paired raw energy moves in the favorable direction.
- D3-R remains a disclosed ablation even if D3-J is better. No sample, seed,
  checkpoint, or failure selection.
- Existing official cache may be reused for matched development S.U.N.; no new
  MP query is needed. A fresh cohort is required for any confirmatory claim.

## Expected role

Geometry-only G2 primarily targets validity. Basin distillation is the first
registered stage expected to improve raw energy and Meta S.U.N. directly by
teaching the DLM the same local correction currently supplied downstream by
model494. If D3-J remains raw-energy neutral, the next escalation is
same-composition validity-first energy preference—not additional CE epochs.
