# C³FD-v2.3 Constructive-Witness Contract

Date: 2026-08-28

Status: frozen before v2.3 CPU or GPU outcomes. Contribution point 1 and
public `105/488` remain unchanged.

## Why v2.2 stopped

The exact full-suffix oracle was scientifically sound but unsuitable online.
The initial implementation exceeded its frozen 300-second CPU audit limit;
one semantics-preserving short-circuit recovery also exceeded 300 seconds.
No v2.2 GPU job was authorized. Its gates are not relaxed.

## v2.3 scientific representation

C³FD-v2.3 compiles a *constructive benchmark witness* into the finite decoder
state instead of enumerating complete future formulas.

For every semantic element/oxidation/count action it jointly tracks:

- remaining atoms and exact net charge;
- remaining distinct-element slots (exact arity);
- requested anion-family requirement and priority exclusions;
- alloy versus ionic branch;
- maximum Pauling electronegativity among selected cations;
- minimum Pauling electronegativity among selected anions; and
- whether every zero-valence element selected so far is a metal.

An ionic terminal is reachable only if charge is exactly zero and
`max(EN_cation) < min(EN_anion)`. A multi-element zero-valence terminal is
reachable only if all elements are metals; unary terminals retain the frozen
benchmark shortcut. Thus the emitted oxidation states themselves certify one
charge-neutral, Pauling-consistent SMACT witness. The independent legacy
composition validator still runs at EOS and remains the reported endpoint.

This is not formula BPE, repair, rejection sampling, replacement, reranking,
or RL. It is a typed chemistry state machine whose token legality depends on a
finite physical certificate.

## Frozen invariants

Unchanged from v2.1: data, frozen Planner context, architecture, losses,
physics features, per-head calibration, temperatures, top-p, `top_k=0`, pair
prior `0`, seeds, and one-request/one-trajectory denominators.

## Small-step ledger

### Step 1 — unit invariants

- reject the `Li(+1)` split-family/charge oxide false positive;
- retain a valid `O(-2),Fe(+2)` oxide path;
- reject a charge-neutral path with inverted Pauling ordering;
- preserve the exact N/charge/arity state invariants.

### Step 2 — frozen-data CPU audit

- require 100% constructive-witness coverage of all benchmark-supervised
  train and validation teacher trajectories;
- require every supported train `(family,N,arity)` stratum reachable;
- generate one deterministic full-mask trajectory per stratum with zero dead
  ends and 100% independent legacy comp-valid terminals;
- require total audit runtime <=120 seconds;
- load no Planner weights and use no stability/outcome labels.

### Step 3 — requested-256 two-seed pilot

Retrain the unchanged v2.1 head (cleaned checkpoints are not resurrected) and
change only `reachability_mode=pauling_witness`. Promote only if:

- zero semantic dead ends in both seeds;
- parse noninferior to P0 within 1 percentage point;
- pooled/per-seed independent comp-valid deltas >0 and paired CI lower >0;
- ionic comp-valid improves;
- Novel × Unique noninferior within 1 percentage point;
- all-metal gap to full train <=3 percentage points; and
- family/N/arity TVD no worse than P0 +0.01.

### Step 4 — requested-1000 confirmation

Run only after all Step-3 gates pass. Report both seeds and pooled without
seed selection or public-headline replacement.

## Stop condition

Stop v2.3 without GPU if teacher-witness coverage is below 100%, any supported
stratum is unreachable, any deterministic terminal fails the independent
validator, or the 120-second CPU limit is exceeded.
