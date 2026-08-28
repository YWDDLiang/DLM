# C³FD-v2.2 Joint-Reachability Contract

Date: 2026-08-28

Status: frozen before any v2.2 GPU outcome. Contribution point 1 and public
`105/488` remain unchanged.

## Confirmed v2.1 failure mechanism

The v2.1 requested-256 pilot produced `52/512` `semantic_dead_end` failures.
Its decoder intersected two separate necessary tests:

1. exact `N`/charge/branch/arity continuation reachability; and
2. requested-family prefix reachability.

Those tests can both pass while no *single suffix* satisfies both. For
example, after `Li(+1)` at `N=2, arity=2`, `F(-1)` can close charge and `O`
exists as a future oxide witness, but neither continuation yields a neutral
oxide. The trajectory is therefore admitted and later dies.

The apparent all-metal excess is coupled to the same denominator loss, not an
independent collapse: v2.1 has `182/460=39.57%` all-metal among parsed plans;
restoring 52 predominantly ionic failures would make the unchanged numerator
`182/512=35.55%`, close to the full-train `34.91%` rate.

## Single v2.2 intervention

Replace only the sampler legality oracle with a family-aware viability kernel.
An action is legal iff a memoized exact suffix exists that jointly satisfies:

- locked atom count `N`;
- exact charge and alloy/ionic branch;
- exact requested arity;
- requested anion-family priority; and
- the independent terminal benchmark composition certificate.

Unchanged: training rows, frozen Planner context, model architecture, losses,
physics features, calibration, temperatures, top-p, `top_k=0`, pair-prior
weight `0`, seeds, one-request/one-trajectory denominator, and no repair,
replacement, reranking, RL, or formula BPE.

## Small-step ledger

### Step 1 — red/green invariant

- reproduce a split-reachability false positive synthetically;
- require the joint oracle to reject it while retaining a valid oxide path;
- require every returned action to retain a terminal witness.

### Step 2 — CPU vocabulary audit

- use the frozen v2.1 vocabulary and every supported train
  `(family,N,arity)` stratum;
- require 100% root reachability, zero deterministic-trajectory dead ends,
  100% exact-family benchmark terminals, and runtime <=300 seconds;
- do not load model weights or outcome/stability labels.

### Step 3 — requested-256 two-seed pilot

- retrain the unchanged v2.1 model because failed checkpoints were cleaned;
- compare against the frozen P0 first-256 controls;
- keep every failed request in the denominator.

Promotion requires all prior v2.1 effect gates plus:

- `semantic_dead_end == 0` in both seeds;
- parse noninferior to P0 within 1 percentage point;
- pooled and per-seed independent comp-valid deltas >0;
- paired 95% comp-valid CI lower bound >0;
- ionic comp-valid delta >0;
- Novel × Unique noninferior within 1 percentage point;
- all-metal absolute gap to full train <=3 percentage points;
- family/N/arity TVD no worse than P0 +0.01.

### Step 4 — requested-1000 confirmation

Run only after every Step-3 gate passes. Report both seeds and pooled; do not
select a seed or replace the public headline.

## Stop condition

If Step 2 fails after one engineering-only correction, or Step 3 retains any
semantic dead end / loses either seed's comp-valid direction, stop v2.2
without changing the model or relaxing gates.
