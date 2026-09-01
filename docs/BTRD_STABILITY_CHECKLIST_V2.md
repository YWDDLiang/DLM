# BTRD geometry and stability checklist V2

Status: ACTIVE. Updated after structured multi-agent review on 2026-09-02.

## Locked mainline

`C3FD-constrained Llama Planner -> Compact-V2 Plan -> masked 7+4N DLM ->
G2 periodic residual -> frozen model494 endpoint refiner`.

The active experiment is **Basin-Target Residual Distillation, tau200
(BTRD-tau200)**. It trains only the existing G2 periodic residual against a
frozen model494 endpoint proxy; the base DLM, Compact-V2 LoRA, Planner and
inference architecture are unchanged.

If and only if BTRD-tau200 is scientifically complete and fails the registered
promotion requirements, the sole locked fallback is **BTRD-800T**: reuse the
same raw proposals, replace tau200 by a tau800 endpoint proxy, and use the
existing sitewise collision-tail mixture (`kappa=0.10`, `beta=0.50`). It is a
bundled fallback, adds no inference module, and is not yet executable.

## Fixed denominators

- Planner composition validity: all 1,200 Planner requests.
- DLM body/Direct/Strict S.U.N./Meta S.U.N.: 1,159 official-known requests.
- Development: first 256 official-known rows.
- Confirmation: remaining 903 official-known rows.
- Continuous CHGNet/hull: explicit known subsets plus missing counts.
- Valid-CIF metrics: secondary CrysLLMGen-compatible conditional denominator.

Frozen evaluation cohort:

- path: `cohorts/btrd_eval_official_known_dev256_confirm903_v1_20260902`;
- manifest SHA-256:
  `2a92913a019a0b7ab375bb76337bcddd142acb728a028f6c1a15d06c90e358b4`;
- development Plan SHA-256:
  `336dc76b6769a8bfbb8190a4891c623d027c27ec202f4da7ac13f15d337126de`;
- confirmation Plan SHA-256:
  `7debf9fa516d6ecdef73bdaea165937bd2d9fcaa6f5a2e48b4b9f93e098b3deb`;
- exact identities: 1,122 unique among 1,159 occurrences;
- outcome fields read while freezing: none.

## Terminal evidence

- [x] Planner1200 job39175: 1,200/1,200 parsed and composition-valid.
- [x] One official availability query: 1,104 unique chemsystems, 1,066
  resolved, 38 unresolved, zero transport retries; credential/process audits
  zero after the query.
- [x] Official-known split: 1,159 rows; 41 unavailable; main1000 plus
  remainder159; source manifest SHA `45a13ccf...1329`.
- [x] Baseline G2 job39183: 1,159 requested, 1,139 body/composition-valid
  (`98.27%`), fast Direct 554 (`47.80%`), 2.2817 A800-hours.
- [x] All 20 baseline non-constructible attempts are invalid lattice metric
  triples. They remain failures in 1,159 and are omitted only from expensive
  compute. Conditional Direct is 554/1,139 (`48.64%`).
- [x] BTRD train subset: 8,192 MP20-train rows disjoint from evaluation;
  6,144 tau200 endpoint requests plus 2,048 MP20 anchors; manifest SHA
  `707ad5a2...d345`.
- [x] Residual-only trainer commit `8073270`; teacher identity audit commit
  `6c031c2`; remote unittest 8+4+1 PASS, pycompile/bash PASS.
- [x] Structured review: Challenger REJECT -> revised; Constraint Guardian
  REVISE -> revised; Paper/User Advocate REVISE -> revised; final Arbiter
  **APPROVED**. Decision log:
  `docs/paper/GEOMETRY_STABILITY_FALLBACK_REVIEW.md`.

## Active execution

- [ ] Job39184, six A800 / 48 CPU:
  `runs/btrd_tau200_teacher_39184`.
- [x] 6,144 teacher Plans materialized into six fixed 1,024-row shards.
- [ ] Six G2 body shards complete with requested accounting.
- [ ] Combined proposal graph/accounting manifest complete.
- [ ] Frozen model494 tau200 endpoint proxy complete.
- [ ] Immutable 8,192-row BTRD SFT dataset complete.
- [ ] Record elapsed time, A800-hours, output SHA and failure counts.

## Immediate next steps

- [ ] Run `audit_btrd_teacher_order.py`; N, global index and ordered species
  mismatch must be zero. Missing row-level endpoints may use the registered
  MP20 anchor substitution; ledger corruption fails closed.
- [ ] Submit one BTRD residual-only job: two A800 / 16 CPU, seed81017, 512
  updates, effective batch16, LR1e-6 cosine/warmup10, sole step512.
- [ ] Generate BTRD once on all 1,159 frozen Plans using baseline shard/noise;
  first compute body and fast Direct.
- [ ] Compute raw CHGNet and cached official hull/S.U.N. on first256; do not run
  candidate model494 refinement before the raw promotion decision.
- [ ] Promote only if paired raw CHGNet and hull each improve at least
  10 meV/atom with CI upper below zero, raw Meta S.U.N. improves at least
  5/256, Strict S.U.N. does not decrease, raw Direct improves at least 1/256,
  body/composition remain at least 95%, and Planner composition loses zero.
- [ ] If promoted, report untouched confirmation903. If scientifically
  non-promoted, implement the approved BTRD-800T contract before its GPU run.
  Engineering failure never activates BTRD-800T.

## Resource and integrity rules

- At most six A800s, two Slurm jobs, and 4-8 CPU per GPU.
- Only existing tmux `ssha800`; never use or reconnect `ssha800_2`.
- No `nvidia-smi`, retry, replacement, top-up, rerank, best-of-N, Plan
  resampling, evaluation-label training or result-conditioned checkpoint/seed
  selection.
- Keep requested accounting, missing rows, hashes, Slurm exit, elapsed/A800h,
  and positive/negative archives.
