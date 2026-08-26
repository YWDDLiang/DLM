# Difficulty Planner strong20 V3

## Why V3 exists

V2's per-row difficulty weights cancelled because training used
`batch_size=1` and normalized weights inside each microbatch. V2 therefore
tested buffer inclusion, not the intended weighted objective.

## Single treatment

- dedicated field: `difficulty_sampling_weight`;
- replacement weighted sampling, with loss weights forced to one to avoid
  applying the treatment twice;
- self-improvement target sampling probability: 20%;
- factorization unchanged: proposal shift × within-stratum-normalized advantage;
- alpha/beta=1, cap=5, minimum buffer ESS ratio=0.5;
- matched control and candidate: 800 optimizer updates, LR 2e-6;
- Planner seeds 17/18 and sampling seeds 17017/18018;
- 256 rich Plans per arm and seed.

This is one strong-dose treatment, not a hyperparameter sweep. V1 and V2 runs
remain frozen.

## Evaluation

If each candidate produces at least 250 parsed Plans without structural schema
damage, all four cells proceed through the same frozen CE-control DLM, model494,
D1 exact-plan schedule, Direct/N/U/CHGNet and fresh official hull evaluation.
Planner parse failures remain in the 256-attempt denominator.

Report both seeds and pooled results. A positive conclusion requires Strict
direction, Meta non-inferiority within 1pp, and no material body, Direct or
novelty collapse. Public 105/488 remains unchanged during the screen.
