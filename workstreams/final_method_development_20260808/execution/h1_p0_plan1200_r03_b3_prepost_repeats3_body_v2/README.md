# P0 Plan1200 R03/B3 pre/post body evaluation — V2

This package consumes the three frozen, independently sampled P0 cohorts from
`20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_import_repair_v2`.  Within each repeat the
same 1,000 planner records and the same deterministic body/refiner seed ledger
are used by R03 and B3.  The two arms are submitted as separate concurrent GPU
Slurm arrays, each with repeat tasks 0–2.

The paired seed namespace remains the preregistered V1 namespace so the V2
engineering repair cannot silently change body or refiner randomization.

Every body result is evaluated before model_494 and again after the frozen
800-step model_494 refinement.  At both stages the byte-frozen R03 Direct
implementation (GCD before composition validity) and the byte-frozen R03E
S.U.N. implementation are run.  The complete native reports and per-attempt
ledgers are retained.  S.U.N. headline rates use its exact legacy
reconstructed-structure denominator; all-1,000-attempt rates are reported as a
conservative secondary view, and evaluated/stable denominators are diagnostic.

B3 receives the exact historical R5-C model-visible `plan_state` JSON prompt.
The raw rich seven-line plan was collator-sidecar-only during B3 training and
is not forwarded.  Canonical `charge_bucket` remains model-visible.

There is no retry, replacement, repair, filter, rerank, checkpoint selection,
training, promotion, or RL.  Any engineering failure is retained fail-closed.

Execution is deliberately split into irreversible, auditable gates:

1. `install_body_source_once.sh` installs one immutable source snapshot only
   after the three planner cohorts are terminal.
2. `mp_cache.py audit` computes the exact three-cohort chemical-system union
   against the frozen base cache plus the previously completed V4 cache.
3. `mp_cache.py complete` resolves only genuinely missing systems on the A800
   login node; credentials, when needed, arrive through a one-time file that
   the frozen completion module destroys.  No credential reaches Slurm.
4. `prepare_and_submit_body_once.sh` runs preflight and submits distinct R03
   and B3 `gpu` arrays (`0-2%3`) plus one `normal` afterany assembly job.
5. `assemble_panel.py` emits `terminal_report.json` and
   `RESULTS_COMPLETE.md`, including full native reports, per-repeat exact
   McNemar tests, 50k all-attempt hierarchical paired bootstraps, and separate
   50k reconstructed-denominator S.U.N. ratio bootstraps.
