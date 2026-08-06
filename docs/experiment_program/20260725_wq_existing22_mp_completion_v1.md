# Existing-22 MP hull completion v1

## Why this stage exists

Job 27631 completed the fixed all-22 CHGNet R5-C/S.U.N. transfer evaluation
without scientific or runtime errors.  Its lower bounds were `0/22` strict and
`3/22` meta-like, but eight of the 17 reconstructed structures were absent from
the frozen MP hull cache.  Those eight unknowns could move strict to `8/22` and
meta-like to `11/22`, so the correct terminal state was
`INCONCLUSIVE_MP_COVERAGE`, not effect failure and not pass.

This stage resolves only those eight frozen unknowns.  It does not rewrite the
historical formal survival failure (`17/22 < 20/22`); it completes the separate
user-accepted exploratory decision.

## Frozen inputs

- Denominator: the same 22 projected states.
- Reconstructed, novel and unique structures: 17.
- Frozen pre-CHGNet structure-invalid placeholders: 5; they remain zero and
  are never repaired or relaxed.
- Original evaluated counts: strict `0`, meta-like `3`, hull unknown `8`.
- The eight query records are fixed by source ordinal, attempt ID,
  reconstructed index, input structure hash, CHGNet relaxed composition and
  CHGNet energy per atom.
- The original job27631 files are read-only and hash-checked before the claim.

## Exact MP/S.U.N. method

The process runs on the A800 login node, not under Slurm.  It uses the pinned
sidecar Python with `mp-api 0.45.13`, `emmet-core 0.85.1` and
`pymatgen 2025.6.14`.  For each of the eight distinct chemical systems it calls
the official `MPRester.get_entries_in_chemsys` once with:

- `compatible_only=True`;
- thermo type `GGA_GGA+U`;
- no application-level retry or replacement.

It then reproduces the frozen A100 hull semantics:

1. construct `PhaseDiagram` from the compatible MP reference entries;
2. add a `PDEntry` using the already-frozen CHGNet relaxed composition and
   energy;
3. compute `e_above_hull` with `allow_negative=True`;
4. classify strict at `0.0 eV/atom` and meta-like at `0.1 eV/atom`.

The API key is accepted only from the process environment.  It is never
printed, serialized, included in a command artifact, or written into an audit.
No separate database-version request is made, keeping the explicit scientific
query scope at exactly eight records.

## Decision and stop rules

The immutable all-22 decision is:

- `PASS` when strict is at least `2/22` and meta-like at least `11/22`;
- `FAIL` when even counting every residual structured query error as stable
  cannot reach either threshold;
- otherwise `INCONCLUSIVE_MP_COVERAGE`.

Every query gets one cache record.  A structured query error remains unknown;
it is not silently retried.  The stage creates a new claim, query cache,
completed all-22 attempt ledger, report and terminal acceptance.  It never
modifies job27631 outputs, runs CHGNet, generates a structure, starts training,
or submits a GPU/CPU job.

## Operational sequence

1. Validate the one-use authorization and all frozen input hashes.
2. Build one cumulative, deterministic source archive.
3. Transfer that exact archive once to starteam5090 and once from starteam5090
   to A800 staging (port 7001, no private-key flag).
4. Atomically install the cumulative patch and verify the installed record.
5. Confirm the pinned sidecar and ephemeral `MP_API_KEY` are present.
6. Invoke `run_once.sh` once on the A800 login node.
7. Verify all append-only output hashes and seal a terminal audit.

Any identity mismatch, pre-existing claim/output, missing credential, package
drift, query error, or acceptance failure is preserved as evidence.  No Slurm,
query retry, sample replacement, or automatic training is allowed.

## Terminal outcome

The single login-node execution completed on 2026-07-26 without Slurm, GPU,
CHGNet reruns, geometry changes, generation, training, or query retries.
Seven of the eight frozen queries resolved.  The remaining `BaYb3O11` record
was preserved as a structured `ValueError` because the phase diagram could not
obtain a decomposition for the frozen CHGNet `PDEntry`; it was not retried.

The unchanged all-22 result is:

- strict S.U.N.: `2/22` (frozen minimum `2/22`);
- meta-like S.U.N.: `6/22` (frozen minimum `11/22`);
- remaining hull unknown: `1/22`;
- optimistic upper bound: strict `3/22`, meta-like `7/22`.

The pre-registered terminal decision is therefore **FAIL**: the residual
unknown cannot raise meta-like S.U.N. from `6/22` to `11/22`.  This stops
escalation of the current composition-projection mechanism and does not rewrite
the earlier formal survival failure.  No automatic training or new generation
was started.

Canonical remote terminal SHA256:
`34e82fd2c574c599b95a9ec261f1e41a74ece6666ff0a08f7ac2dca19558368a`.

Local terminal audit:
`runs/remote_audit/20260725_wq_existing22_mp_completion_v1/terminal_audit.json`.
