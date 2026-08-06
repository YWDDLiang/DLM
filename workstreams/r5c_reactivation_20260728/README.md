# Active Shared-Plan + R5-C/DLM Workstream

This directory is the mutable work area for the 2026-07-28 R5-C reactivation.

`baseline/` is copied from the frozen `legacy_dlm_r5c/` snapshot.  The original
snapshot remains the provenance source and must not be edited.  New source,
launchers, configs, and outputs will be added outside `baseline/` or as an
explicitly versioned patch.

Portable restore artifacts:

```text
r5c_frozen_baseline_20260728.tar.gz
  SHA256 ad1b7f5b9ee0df0c06396ef1d3865f7a5e7b2e4d3f4b46216445288e04be8325
r5c_reactivation_bundle_20260728.tar.gz
  SHA256 is recorded in the adjacent .sha256 sidecar
```

The bundle contains the frozen baseline archive, recovered experiment index,
reactivation plan, and the first 256-attempt anchor config.  Final bundle
validation is intentionally stored beside, rather than inside, the archive so
that its digest is not self-referential.

The active ICLR execution plan and complete result review are:

```text
docs/experiment_program/20260728_iclr_plan_dlm_relaunch.md
workstreams/r5c_reactivation_20260728/ALL_POST_R5C_RESULTS_REVIEW.md
```

The recovered history is complete through H1-A4 and now also incorporates the
later `llm_plan_diff` Shared-Plan, minimal-Plan, terminal-bridge, PlanBridge and
PlanV2 results.  The selected de novo trunk is H1-A2 epoch 2 plus the frozen
R5-C exact-length body.  The strongest later continuous result, shared-Plan
S2, is retained only for a no-training algebraic-null repair and conditional
mechanism gate before it is connected to H1-A2.

The first prepared config is:

```text
configs/plan_dlm_shared_plan_null_repair256_v1.json
```

It is deliberately marked `prepared_not_authorized_for_submission`.  It does
not train, evaluate S.U.N., or authorize an automatic downstream job.

No historical run directory may be overwritten.  Every new A800 job is limited
to one A800 and at most eight CPU cores unless the user changes that rule.
