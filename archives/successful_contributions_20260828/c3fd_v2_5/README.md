# C³FD-v2.5 successful contribution snapshot

Frozen scientific result date: 2026-08-28

Success commit: `6e5f827`

Archive purpose: preserve the exact code, tests, execution contracts,
documentation and result evidence supporting the successful C³FD-v2.5
composition-correctness contribution. The listed source files were verified
unchanged between `6e5f827` and the archive preparation commit.

## Result

On two requested-1000 seeds, pooled P0 versus C³FD-v2.5:

- parsed formulas: `1989/2000 -> 2000/2000`;
- independent comp-valid: `1724/2000 -> 2000/2000`;
- Novel: `1538 -> 1763`;
- Unique: `1961 -> 1985`;
- Novel x Unique: `1530 -> 1756`;
- per-seed comp-valid deltas: `+13.9 pp / +13.7 pp`;
- paired 95% CI: `[+12.29, +15.31] pp`;
- semantic dead ends: `0/2000`.

## Claim boundary

C³FD-v2.5 claims constructive composition correctness and composition-level
diversity. It does not claim thermodynamic stability, synthesizability, or
S.U.N. improvement. The selected Planner checkpoints remain in the remote
run `c3fd_v25_requested1000_36608`; checkpoint paths and hashes are recorded
separately in `CHECKPOINT_POINTERS.json` after remote verification.

## Layout

- `code/`: frozen implementation and training/sampling/finalization scripts;
- `tests/`: matching unit tests;
- `slurm/`: successful execution contracts;
- `docs/`: design and staged correction contracts;
- `results/`: machine-readable and human-readable terminal evidence;
- `MANIFEST.json` and `SHA256SUMS.txt`: provenance and integrity ledger.

This directory is an immutable backup. Future CTV-DLM work must not overwrite
it; a successful CTV stage receives its own sibling snapshot.
