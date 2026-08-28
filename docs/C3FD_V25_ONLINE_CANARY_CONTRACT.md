# C³FD-v2.5 Online-Canary Contract

Date: 2026-08-28

Status: frozen before v2.5 model or sampling outcomes. Contribution point 1
and public `105/488` remain unchanged.

## Rationale

The constructive witness is correct on frozen teacher trajectories: all
`24,558/24,558` train and `8,159/8,159` validation benchmark-supervised
sequences carry exact N/charge/arity/family and Pauling-consistent witnesses.
Full enumeration of all actions for every supported stratum is an overly
conservative performance benchmark and caused v2.2--v2.4 to stop without GPU.

v2.5 measures the actual learned proposal distribution without weakening any
composition rule. It uses the v2.4 bitset witness and a staged, outcome-blind
online canary.

## Frozen intervention

- model/data/training/calibration are identical to v2.1;
- sampler uses `reachability_mode=pauling_bitset`;
- each seed first generates global `sample_idx=0..31` exactly once;
- only if both canaries have `32/32` parsed, `32/32` independent comp-valid,
  zero failures, zero semantic dead ends, and sampling time <=600 seconds does
  the same checkpoint generate `sample_idx=32..255`;
- segments are concatenated by global sample index, not selected or replaced.

No repair, replacement, backtracking, reranking, RL, formula BPE, or outcome
label is used. The canary gate sees only syntax/certificate/runtime, never
novelty, family mix, stability, or downstream structure outcomes.

## requested-256 promotion gates

- both seeds requested exactly 256 with no missing/duplicate sample index;
- zero semantic dead ends;
- parse noninferior to frozen P0 within 1 percentage point;
- pooled/per-seed independent comp-valid deltas >0 and paired CI lower >0;
- ionic comp-valid improves;
- Novel × Unique noninferior within 1 percentage point;
- all-metal gap to full train <=3 percentage points; and
- family/N/arity TVD no worse than P0 +0.01.

Only an all-gate pass permits requested-1000 confirmation. No seed selection
or public-headline replacement is allowed.
