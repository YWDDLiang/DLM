# D3PO sealed late-guidance job 38233 engineering failure

Date: 2026-08-30  
Scientific classification: **none; failure before model loading or sampling**

## Frozen facts

- Job `38233` was the one authorized class-M late-only guidance launch.
- The sealed cohort SHA remained
  `1b7f7111f75ed5b26cb235274f1b8da70393676898732722c3a63a2f61a35ce0`.
- The wrapper created the conservative global burn marker immediately before
  starting the six shell workers.
- Every worker failed while expanding an empty Bash array under `set -u`:
  `guidance_args[@]: unbound variable`.
- No Python sampler started, no model loaded, and no body, graph, refinement, or
  scientific outcome file exists.  The run contains only the contract, input
  hashes, `_FAILED`, and `ENGINEERING_FAILURE.tsv`.
- The failure was reported at the parent `wait` because the six workers ran in
  the background; the Slurm stdout contains the actual child-shell error.

## Root cause and correction

The base arms constructed an empty optional `guidance_args` array.  On the
cluster's Bash version, expanding that array with `set -u` is an unbound-variable
error.  Local and remote `bash -n` cannot detect this runtime shell-version
behavior.

The wrapper now constructs a non-empty `sample_command` array and appends the
three guidance arguments only for guided arms.  This correction is retained for
future code quality and independent prospective cohorts.

## No-repeat decision

Although the failure was pre-science, the sealed cohort's conservative burn
marker already exists and the active sprint contract forbids another scientific
fallback launch after that marker.  Job `38233` is therefore a terminal negative
engineering artifact, not a D3PO sampling result.  No stability inference is
drawn and the sealed cohort is not reused.
