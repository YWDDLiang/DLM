# Official-MP clean S.U.N. re-evaluation V1

This immutable repair changes only the stability evaluator. It never reruns or
modifies Planner sampling, R03/B3 generation, reconstruction, novelty,
uniqueness, diffusion refinement, or CHGNet relaxation.

The repair contract is deliberately stricter than the contaminated August
cache path:

- start from an empty cache;
- call `MPRester.get_entries_in_chemsys()` directly;
- pin `thermo_types=["GGA_GGA+U"]` for historical CrysLLMGen parity;
- accept the compatible entries returned by MP without locally selecting,
  combining, or reprocessing variants;
- require a unary reference for every element before constructing a phase
  diagram;
- preserve a compressed full MSON snapshot, a query audit ledger, and a slim
  evaluation cache;
- permit only bounded, recorded transport retries (never scientific-sample
  retry or replacement), while rate-limiting all HTTP requests process-wide;
- keep relaxation failures separate from hull-query failures;
- compare old and clean `E_hull` values attempt by attempt.

The official query is pinned to the project's pre-existing, previously
audited `mp-api==0.45.13` / `emmet-core==0.85.1` Python sidecar. That client
returns compatible entries from the official thermo endpoint and defaults to
the historical `GGA_GGA+U` thermotype. Input collection and downstream
phase-diagram replay remain pinned to the registered `diff_meets_diff`
interpreter. No package is installed or modified by this repair.

V1 covers the 12 V4 all-attempt cells (R03/B3, three repeats, pre/post
model-494) and the four frozen historical R03 refined-256 repeats. Native-1000
and H1-A2 are added only after this diagnostic repair passes its scientific
gate.
