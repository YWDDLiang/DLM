# CTV frozen-feature Q-head contract V1

Date: 2026-08-28

Status: frozen before Q-head fitting. This contract does not authorize L6.

## Frozen feature extraction

- generator: `ctv_minimal_base_36898/step-696`, eval mode and never updated;
- states: all 256 Branch-train states and 64 reduced-composition-disjoint
  validation states;
- representation: the exact state-token sequence captured at the frozen
  `0.60/0.80` milestones;
- state vector: input to the frozen output head at the intervened position;
- action vector: the corresponding frozen output-head weight row;
- both vectors use one shared Gaussian random projection to 256 dimensions,
  seed `73017`, scale `1/sqrt(256)`;
- exact constrained legal support and base probabilities are reproduced for
  every state; selected-action probability error must be <=`1e-5`;
- no terminal label is read during feature extraction.

The feature artifact stores complete legal token ids/probabilities for every
state. Loading or extracting features cannot modify generator logits.

## Frozen two-head model

Branch-train Plans are split by even/odd frozen Plan ordinal, yielding two
disjoint 64-Plan groups. Each head sees only its own group.

For state projection `h`, action projection `e`, frozen base probability `p`,
milestone and one-hot geometry-token family, the action input is:

```text
[h, e, h*e, clip(log(p), -30, 0), milestone, family_one_hot]
```

Each head contains a 256->64->1 state-baseline MLP and a
`(3*256+11)->128->1` action-advantage MLP. Predicted advantages are centered
over the eight observed actions while fitting. The fixed loss is equal-weight
Huber loss on absolute terminal energy and state-centered advantage. Training
uses AdamW, learning rate `1e-3`, weight decay `1e-3`, 512 full-group updates,
gradient clipping `1.0`, and seeds `74017/75017`. There is no architecture,
epoch, checkpoint or hyperparameter search and Branch validation is never used
for early stopping.

## Frozen gate

The pre-existing gates remain unchanged: Plan-bootstrap Spearman LCB>0,
within-state pairwise AUC>0.60, cross-continuation sign agreement>0.60,
feasibility AUROC>0.70 when both classes exist, symmetry rank agreement>0.90,
hard-strata direction, and the per-head support/disagreement/coverage rules in
`CTV_DLM_STABILITY_CONTRACT_V1.md`.

If a feasibility class is absent, AUROC is reported as non-estimable; it is not
silently assigned 1.0. Any failed estimable hard gate or non-estimable required
gate keeps L6 unauthorized. Gamma is selected from `{0,5,10}` only after all
non-gamma gates pass; otherwise gamma remains unset.
