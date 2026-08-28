# CTV-DLM implementation status V1

Date: 2026-08-28

Status: **prerequisite implementation in progress; no CTV scientific result.**

## Completed locally

- froze the minimal certified-composition prompt and removed rich soft fields;
- froze reduced-composition-disjoint canary, Branch train/validation, L6 and
  downstream-holdout identities;
- implemented exact C³FD certificate-source/hash binding for minimal SFT data;
- implemented the two `0.60/0.80` free-geometry milestones;
- implemented frozen-confidence intervention-position selection;
- implemented argmax plus seven cumulative-quantile legal actions with a hard
  distinctness assertion;
- implemented action-independent common continuation noise;
- implemented exact-schedule forced-action completion without changing
  composition tokens, remasking, position order or legal masks;
- implemented the 8 Plan x 2 milestone x 8 action x 2 continuation ledger and
  its 256-row hard assertion;
- implemented gamma-zero bit-identity checks;
- prepared a 4 x A800, 300G engineering-only canary job guarded by the
  minimal-data, identity, frozen-base and MatterSim Gate-A manifests.

Tests: `99` lightweight tests pass (`9` optional skips); the torch-enabled CTV
suite passes `7/7` in the local MLIP environment.

## Data-audit correction

The first minimal-spec audit reused the historical `plan_state.validator`
after the builder had correctly selected rows using the newer C³FD
certificate. It therefore rejected certified rows under a stale validator.
No training was launched from that audit.

The corrected audit binds every converted row to:

1. its exact C³FD `source_row_idx`;
2. aligned formula, N, elements and counts;
3. the frozen certificate split SHA-256;
4. the C³FD composition-supervision and valence witness.

The data must be rebuilt at a new immutable path and pass all rows before the
one-time base continuation is submitted.

## Remote gate

Both pre-existing A800 SSH connections are currently disconnected. The
frozen safety contract forbids creating or reconnecting an A800 session, so no
job has been submitted or duplicated. Once the user restores an existing
session, the execution order is:

1. verify the prior MatterSim dependency process and active-job ledger;
2. deploy the certificate-audit correction and rebuild the minimal dataset;
3. run the tokenizer/certificate audit;
4. freeze the one-time 696-update minimal-spec base;
5. complete MatterSim Gate A;
6. run the single authorized 256-completion resource canary;
7. return the ledger/resource evidence for a new formal-science arbitration.

Formal Branch-Q generation, Q-head training, L6, L7, distillation and RL are
still not authorized.
