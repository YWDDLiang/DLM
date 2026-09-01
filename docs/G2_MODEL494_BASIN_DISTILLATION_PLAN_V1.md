# G2 model494 basin-distillation plan V1

Status: FROZEN NEXT STAGE; do not start before the active G2 full-epoch A/B
experiment is terminal. This stage does not alter or resume jobs39164/39165.

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

1. Select 4,096 MP20-train unique exact compositions with an outcome-blind
   seed and disclose their train distribution.
2. Materialize the frozen Planner Plan once per composition.
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

### D3-R: residual-only warm-up

- Freeze Compact-V2 LoRA and all backbone parameters.
- Train only the periodic residual for one 4,096-pair source pass:
  `4096/16 = 256 updates`.
- LR `5e-6`; sole step256 checkpoint is a mechanistic ablation, not a result-
  selected checkpoint.

### D3-J: joint residual + LoRA distillation

- Initialize from D3-R step256.
- Train residual at LR `5e-6` and existing Compact-V2 LoRA at LR `1e-6`.
- Use two equal-weight views per source—original MP20 teacher target and
  model494-refined target—so one epoch is `8192/16 = 512 updates`.
- Keep per-source total weight one and retain a frozen-reference preservation
  term. Save only step512; no early stopping or checkpoint selection.

The registered loss family is:

```text
original MP20 token CE
+ model494-refined token CE
+ periodic displacement / metric / pair-distance losses
+ reference-preservation loss
```

Loss scales may receive one MP20-train-only gradient-norm calibration; no grid,
development outcome, or prospective tuning is allowed.

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

