# C3FD–H1-A2 Fusion S.U.N. checklist v4

Date: 2026-08-31

Deadline: 2026-08-31 23:30 Asia/Shanghai

Status: **stopped and superseded by the approval-draft v5**

User stop record: Slurm job `38914` was cancelled on 2026-08-31 after
`00:16:13`. Its partial outputs are retained as a stopped engineering attempt;
they must not be resumed, completed, evaluated, or used for a S.U.N. claim.
The deterministic-completion method below is retired.

This checklist supersedes `C3FD_NATIVE_DLM_SUN50_CHECKLIST_36H_V3` after the
user stopped alignment training and prioritized immediate official S.U.N.
measurement plus a future fully trained H1-A2-style Planner.

## Method boundary

The active immediate method is:

`C3FD exact composition + predicted LS/SG/VP`

→ deterministic completion of the canonical H1-A2 rich JSON fields

→ old H1-A2 rich masked DLM

→ model494 tau800.

This is called **C3FD–H1-A2 fusion**. It tests whether the successful H1-A2
realization interface can be combined with C3FD proposal correctness. It is not
a newly trained complete Planner and must never be described as one.

The future method item is a separately retrained Planner that directly predicts
the complete H1-A2 rich Plan. It is out of the current critical path until the
available S.U.N. results are frozen.

## Frozen prospective cohort

- independent C3FD Planner sampling seed20, `1000/1000` source rows;
- source SHA-256: `1cbdb46128ea2ee62d924f4cec0c94fcd450e82b309535c317056ecbd63371df`;
- 256 unique exact compositions, 252 chemsys;
- exact overlap with MP20 train and every existing cohort excluded;
- first eligible source ordinal selected before policy outcomes;
- two matched views: canonical `H1A2_FULL` and compact `C3FD_V2`;
- manifest SHA-256: `4e612fcdcf76e3b5ac1f0750e4ec773877d58dd6f75e2a217cb73798cf71811e`;
- full/V2/ledger SHA-256: `bbadae4c...01cf8`, `97004621...af2a9`,
  `5189fb8e...f35df`;
- no policy, stability, hull, or S.U.N. outcome was read.

## Prospective comparison

Run three arms × streams17/18 on identical composition/order/noise:

- `h1a2_fusion`: H1A2_FULL prompt + old rich DLM;
- `v2_seed82017`: C3FD_V2 prompt + fresh SFT policy82017;
- `v2_seed82018`: C3FD_V2 prompt + fresh SFT policy82018.

All cells use temperature0.7, exact-axis generation, one Plan/trajectory, and
model494 tau800. No alignment, retry, replacement, reranking, best-of-N, or
checkpoint selection is allowed.

Primary headline: prospective fusion Strict/Meta S.U.N. Targets remain 10%/50%
as reporting objectives, never result-deletion rules.

## One combined official union

After prospective offline evaluation, issue one fresh MP query covering:

- prospective fusion/V2 raw+refined cells — final main result;
- faithful H0/R0S eval38603 — development full-schema evidence;
- fresh V2 SFT canary eval38768 — development evidence with MP20 train/val
  overlap explicitly disclosed.

Exclude the train-only alignment pool, malformed-schema canary38420, and D3PO
(already official). Keep development results separate from the headline.

## Execution checklist

- [x] Alignment stopped before any alignment weights were produced.
- [x] C3FD seed20 source job38903 terminal, `1000/1000`.
- [x] Matched H1A2-full/V2 prospective256 job38908 terminal.
- [ ] Six-cell generation/refinement job38914 terminal.
- [ ] Prospective raw-first/refined 12-cell offline evaluation terminal.
- [ ] Combined official union and SOURCE_SHA frozen.
- [ ] One fresh MP query terminal; credential unset and destroyed.
- [ ] Prospective and development S.U.N. reports terminal.
- [ ] Final paper story/RQs/contribution grades/resource ledger frozen.

## Current job

- `38914`, `c3fd-h1a2-fusion-gen`, `6 A800 / 48 CPU`;
- run: `$ROOT/runs/c3fd_h1a2_fusion_prospective_generation_38914`;
- duplicate submission forbidden.

## Credential lifecycle

Use handle `MP_API_KEY_FROM_USER_THREAD_CONTEXT`; never request it again or copy
its value into persistent artifacts. Inject only into the single combined
query's temporary non-ambient child environment, unset immediately, and verify
no process/runtime copy remains.

## Forbidden

- alignment/listwise/D3PO training;
- claiming deterministic completion is a newly trained Planner;
- historical 3,614 formal training;
- oracle/direct Planner CIF/prototype/Wyckoff targets;
- survivor/rerank/replacement/best-of-N/sweeps;
- AR/RL/GRPO/SMC/test-outcome training;
- changing public headline105/488 before complete prospective terminal.

After S.U.N. is frozen, design a separately trained complete H1-A2 rich Planner.
