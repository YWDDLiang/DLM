# P0 Plan1200 × R03/B3 pre/post-refine repeats3

This execution package implements the frozen sampling stage for a three-repeat,
paired R03-versus-B3 evaluation.

This is the immutable V3 execution root.  It preserves the V1 scientific and
paired-seed contracts, includes the repository `scripts` package initializer,
and performs a login-node import preflight.  V1 is sealed after its missing
package initializer; V2 is sealed after a pre-prepare `0644` direct-exec denial.
Neither predecessor is modified or reused.  The V3 prepare entry is archived
as executable and is invoked explicitly through `/usr/bin/bash` exactly once.

Each repeat independently samples 1,200 raw plans from the frozen P0 planner.
The cohort is the first 1,000 parse-successful plans in planner ordinal order.
The remaining candidates are retained as planner reserve evidence and are never
selected in response to either body model's outcome.  R03 and B3 therefore see
the same 1,000 plans within each repeat, while the three repeats use different
planner seeds.

The downstream protocol is pre-registered in `CONFIG.json`: two independent
Slurm arrays (one R03 and one B3) consume the frozen cohorts concurrently.  Each
arm is evaluated before and after the byte-frozen model_494 refinement with the
full CrysLLMGen and S.U.N. outputs.  The planner stage does not submit those jobs;
they are submitted only after all cohort identities and the required MP cache
coverage have been audited.

The B3 training dataset kept the historical R5-C model-visible prompt/answer
bytes.  The rich seven-line plan was collator-only sidecar metadata.  Therefore
both arms receive the historical `plan_state` JSON body prompt; raw seven-line
planner text is never forwarded to either body model.  In particular, the model
visible prompt contains the canonical `charge_bucket` key rather than a raw
`charge:` line.
