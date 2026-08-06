# H1 CR-Plan CR-0 runtime diagnosis

Date: 2026-08-04

## Scope

This note diagnoses why immutable CR-0 job `30325` did not finish inside its
40-minute Slurm limit. It separates runtime/packaging failures from CR-Plan
scientific logic and records the bounded repairs used to continue the
experiment.

No Planner weight, prompt, tokenizer asset, formula FSM, oxidation-state table,
charge-reachability DP, logits processor, sampling parameter, seed, raw-attempt
denominator, evaluator, or scientific gate was changed.

## Job 30325: observed terminal evidence

- Run:
  `runs/20260804_h1_crplan_r0_paired32_v1`
- Slurm top-level state: `TIMEOUT`
- Elapsed: `00:40:12`
- Batch step: `CANCELLED 0:15`
- Total CPU: `00:01.440`
- Maximum RSS: `359292K`
- Maximum disk read: `273.06M`
- Focused unit tests: `26/26`, pass, `0.397s`
- Main CR-0 terminal report: absent
- `_SUCCESS`: absent
- paired-32 submitted: false
- Preserved terminal observation SHA256:
  `39b7aa279f29851ccbfbf704035bc624c6af87abe1e103bd26320727ee07067b`

The combination of 40 minutes wall time, 1.44 seconds CPU, slowly increasing
RSS, and hundreds of megabytes read excludes a compute-bound DP loop. The
process was dominated by dependency and tokenizer-stack cold loading from the
shared filesystem.

## Code-level cause

The V1 audit imported `transformers.AutoTokenizer` and later imported full
PyTorch solely to compare softmax before and after applying an
all-vocabulary support mask.

The actual generation path has different semantics:

1. while inside the formula value, the frozen logits processor sets illegal
   token logits to `-inf`, renormalizes once, and fails closed on empty support;
2. after the formula newline, it returns the original score tensor unchanged.

Therefore the audit's explicit PyTorch import was redundant. Equality under a
full-vocabulary mask is verified exactly by support identity plus a stable
binary64 softmax comparison; no tensor framework is required.

Phase instrumentation in successor job `30346` localized the dominant delay
exactly:

- `from transformers import AutoTokenizer`: `2301.394764s` (`38m21.395s`);
- frozen tokenizer load after the import: `2.259488s`;
- frozen SMACT table load: `2.689369s`;
- token-fragment vocabulary construction: `0.830469s`.

Thus the original 40-minute request left only seconds for the evaluator
alignment, DP/brute-force parity, tokenizer transition audit, gates, and
atomic report write. The 40:12 terminal time is explained by the scheduler
envelope, not by a deadlock in those scientific checks.

Transformers 4.54 reads its `USE_TORCH`, `USE_TF`, and `USE_FLAX` controls
during `import_utils` initialization. With their default `AUTO` values it
discovers installed frameworks and optional packages, producing very low CPU
time and hundreds of megabytes of shared-filesystem reads. A separate V4
contingency therefore sets `USE_TORCH=0`, `USE_TF=0`, `USE_FLAX=0`, and
`USE_TORCH_XLA=0` for CR-0 only; `AutoTokenizer` requires none of those
frameworks. The paired GPU sampling script deliberately does not set them.
The initial backend-only V4 draft passed 28/28 local focused tests and was not
deployed while immutable V3 was still running.

## Bounded repair history

### V2 / job 30342

V2:

- replaced the explicit PyTorch probability sub-audit with stable Python
  binary64 full-vocabulary parity;
- added timestamped shell and Python phases;
- expanded CR-0 walltime from 40 minutes to 2 hours;
- added a regression test that raises if the probability audit imports
  PyTorch.

Job `30342` failed before the main audit after `00:01:55` because the isolated
bundle contained only the V2 execution directory while one test still imported
the byte-identical evaluator through the V1 module path.

- State: `FAILED 1:0`
- Unit-test stderr SHA256:
  `ac413170d86a46faad12ab3364d68487a7aa415c7d7fa3a254f5af82dccd52a0`
- Job stdout SHA256:
  `ab6c0e95dfb20d0bad33e980bec8442feb6a7c26096b5982b44ffa34ecfed6ae`
- Job stderr SHA256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Submission record SHA256:
  `dac5378394be64976ece6a183fa21b4fecb934b7d7c1b158391e2ec94914078f`
- Main audit started: false
- paired-32 submitted: false

This was an isolated-source test-assembly error, not a scientific or runtime
audit result.

### V3 / job 30346

V3 changes only the test import to the evaluator shipped inside the same
immutable V3 bundle. That evaluator is byte-identical to V1:

`1a92bddf6322e86a499c0944f40dbe4360a8bc9665db7de3c2ccc6d8369c0445`

The isolated staged source passed 27/27 tests before deployment.

- Archive SHA256:
  `475a400531c755374bc720aac613d6f313390b6570486b4c22224648f32611f2`
- Source-manifest SHA256:
  `e2890c8e2fa98ea0ea6d9d91691790abe05237e7d297076761bdf0ae16c3a5a2`
- Remote job: `30346`
- Remote 27/27 unit tests: pass
- Terminal state: `FAILED 2:0`
- Elapsed: `00:45:22`
- Terminal report SHA256:
  `453291851e900ad8db7551f066c241a164c3786b0901464d0aab7ef56580194e`
- `_SUCCESS`: absent
- paired-32 submitted: false

The complete audit failed two gates:

1. `exhaustive_frozen_evaluator_alignment`
2. `pauling_non_hard_constraint_fixture_missing`

The first failure was an audit-side Direct-alignment bug. All 16 mismatches
were formulas with a nontrivial count GCD, such as `Li2O2`, `Na3Cl3`, and
`Fe2O2`. The CR-0 fixture passed raw counts to `classify_smact_validity`;
the frozen Direct evaluator first GCD-reduces composition counts. No Planner
mask or oxidation table mismatch was involved.

The second failure meant the registered panel lacked a witness for a policy
that the protocol intentionally preserves: Pauling is measured but not a hard
decoding constraint. A bounded frozen-table search found `HSi`, with the
table-relative uniform witness H(+1)/Si(-1); the frozen Direct classifier
returns `pauling_fail_or_ratio_rejected`.

V4 applies only the following bounded repairs:

- disable unused Transformers framework probes in CPU CR-0 only;
- apply the exact Direct GCD reduction in the alignment audit;
- register the fixed `HSi` Pauling counterexample;
- correct the table-source label from
  `smact.Element.oxidation_states_default` to the attribute actually read,
  `smact.Element.oxidation_states` (table contents unchanged).

After these changes, 29/29 local focused tests pass. Planner legal support,
paired evaluator bytes, model/tokenizer assets, sampling parameters, seeds,
denominators, and all downstream gates remain unchanged.

### V4 / job 30348

V4 was frozen as a new immutable source instead of altering any prior run.

- Source-manifest SHA256:
  `73ceb62807cf5903f60fe8918069c237c02bcc988eac685e84bee504e542b0ca`
- Archive SHA256:
  `d2c6bc0adbaddab9e4c69825ab0e0856b7e60c59179422f4ede378fa9de60d2e`
- Remote job: `30348`
- State: `COMPLETED 0:0`
- Elapsed: `00:12:36`
- Remote focused tests: pass
- CR-0 terminal status: `pass`
- CR-0 failures: `[]`
- Frozen-Direct evaluator alignment: all aligned
- Registered Pauling-measurement-only witness count: `1`
- Terminal-report SHA256:
  `dac9ffc3aa8fa699158236aea6056b144a724d47fdec3258564f2aea60c154e3`
- Job-stdout SHA256:
  `3eaeb733dbae0fbc5db1a5eabe32e72d2a73f749af46e029cdf16dfb914d9c32`
- Job-stderr SHA256:
  `2d056a402d36c0559e8176f3048ad5de7f1c8cc6a83cd0416e91573be6e2ac50`

The non-empty stderr contains only the expected Transformers tokenizer-only
notice that PyTorch, TensorFlow, and Flax are unavailable under the CR-0
backend contract. It contains no traceback or scientific warning.

Compared with the measured `38m21.395s` V3 Transformers import, the V4
tokenizer-only backend contract reduced the entire Slurm job to `12m36s`,
including source verification, environment activation, focused tests, the
complete audit, report writing, and the success marker. This confirms the
40m12s V1 failure was a dependency-probe/shared-filesystem startup pathology,
not CR-Plan charge-reachability compute.

Only after observing this clean terminal did the frozen one-shot release
script submit paired-32 job `30353`. No paired-64/256, four-arm, training,
promotion, or other downstream job was released.

### V4 paired-32 / job 30353

Job `30353` passed source-manifest verification and the one-A800 identity
check, then failed before model loading or any raw attempt:

- State: `FAILED 1:0`
- Elapsed: `00:30:35`
- Failure:
  `ModuleNotFoundError: No module named 'scripts.sample_llada_dynamic_crystals'`
- Control raw attempts: `0`
- Candidate raw attempts: `0`
- Paired terminal report: absent
- `_SUCCESS`: absent
- Job-stdout SHA256:
  `126f5539a380d417a29f79cc1e6f6690f4db6b49f5126515dac3513b4fb50111`
- Job-stderr SHA256:
  `e1e9f91174a57fe4ad0f7dfdaf20abe048cab3758fcec678220034b81b5bbdf1`
- Submission-record SHA256:
  `bc8511a3ded9c32973fb053e32e6698b719e4830eadb32ecc771fdba53933096`

The repository already contained `scripts/__init__.py`, specifically to keep
an installed regular package named `scripts` from shadowing repository-local
runner imports. The V4 manually frozen source-member list included both
sampler files but omitted that existing package marker. Source verification
therefore passed an incomplete *declared* bundle, and Python resolved the
wrong parent package before it could import the local sibling module.

This is a source-bundle completeness bug, not a Planner, CR-Plan, GPU, model,
or endpoint result.

### V5 source-bundle repair

V5 is a new immutable run identity. It:

- adds the unchanged repository `scripts/__init__.py` to the frozen source;
- adds shell fail-close checks for that member in both CR-0 and paired-32;
- adds an isolated-source resolution test that requires
  `scripts.sample_llada_dynamic_crystals` to resolve inside the frozen source;
- records job `30353` as immutable predecessor evidence.

No sampler byte, Planner/adapter/tokenizer asset, CR-Plan token support,
sampling parameter, seed, denominator, evaluator, or scientific gate changes.

- Focused local and fresh-extraction tests: `30/30`, pass
- Source-manifest SHA256:
  `7ff37f45c20e48fd1a43ef7fbf58a8a0ecb97070176c633c209dc6569ef236b6`
- Archive SHA256:
  `e697118d51c5db3f3f7765395481930374dbc4fb443df4517ddf8da7e5177fa5`
- Frozen package-marker SHA256:
  `692bec4a014d8c2be09296210037872ead6ea1c397bb01ae662ca58d3167c270`

### V5 CR-0 and paired-32 terminal

V5 CR-0 job `30356` and paired-32 job `30358` are the first clean terminal
execution of this route.

- CR-0 job: `30356`, `COMPLETED 0:0`, elapsed `00:05:21`
- CR-0 terminal status: `pass`, failures `[]`
- CR-0 terminal SHA256:
  `4265fc6f910d41d68e9f70a1439e0fde69534a8a757f646e4f4d79ab37ccc94a`
- Paired-32 job: `30358`, `COMPLETED 0:0`, elapsed `02:23:48`
- Paired-32 terminal status: `pass`, all registered gates true
- Paired-32 terminal SHA256:
  `a1a34a663d691480c6aefb9beb4e9174c9483a95e81669cb6b4888f96d598a80`
- Both arms contain exactly 32 raw attempts with the frozen ordinal, seed,
  prompt, and input-ID pairing; the control exactly matches historical P0
  first-32 scientific output.
- Control: parse `31/32`, completion `32/32`, composition-valid `17/32`,
  primary charge-valid `9/32`, unique formulae `31`, element coverage `45`.
- Candidate: parse `32/32`, completion `32/32`, composition-valid `18/32`,
  primary charge-valid `10/32`, unique formulae `32`, element coverage `45`.
- Paired composition-valid discordance: baseline-only `0`, candidate-only
  `1`, exact McNemar two-sided `p=1.0`.
- Candidate charge-applicable terminal failures, identity failures,
  dead ends, silent fallbacks, and invalid DP telemetry: all zero.
- No retry, replacement, repair, filter, rerank, endpoint selection,
  S.U.N. use, promotion, or automatic downstream action occurred.

The screen is an engineering pass and a one-sample positive composition
direction, not evidence of a prefix-reachability gain. The frozen SMACT
oxidation-state table has no states for `He`, `Ne`, `Ar`, `Pm`, `At`, `Rn`,
`Fr`, and `Ra`. Under the paired-32 terminal policy, a table-missing element
is an allowed non-applicable suffix. Consequently, every unfinished prefix
with atom budget remaining can append such an element; full-prefix
reachability structurally degenerates toward terminal-only. The terminal
therefore correctly records
`allow_prepare_four_arm_only_after_missing_policy_review` and forbids
prefix-control attribution.

The paired run also establishes that the current implementation cannot be
carried unchanged into the registered 512 mechanism panel:

- control median/p95 Planner latency: `2.835 / 3.022 s`
- candidate median/p95 Planner latency: `112.813 / 215.895 s`
- candidate maximum per-attempt DP states created: `1,714,193`
- candidate maximum cache entries: `298,571`

These exceed the prospective 512 gates of at most `1.5x / 2x` control
latency and at most `100,000` DP states per attempt. Before any 512
submission, the route must freeze a shared terminal/full table-missing
policy and demonstrate semantics-preserving support computation with bounded
state and latency. No 512, Body, refiner, Direct, S.U.N., training, or other
downstream job has yet been submitted.
