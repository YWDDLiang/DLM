# Current state

Updated: 2026-08-10 (Asia/Shanghai)

Overall status: `PLANNER_AND_B3_TERMINAL_FOUR_CELL_DIRECT_SUN_FREEZING`

## Active edge: complete current-run four-cell Direct/S.U.N.

The user-authorized V14 diagnostic now has exactly 256 all-attempt rows for
both P0 and SFT-v2. The local-only exact SMACT4 audit is complete and
SHA-bound. V18's normal-CPU SMACT3.1 assembly computed the complete formal
legacy result before an old-P0-schema identity gate labeled the job failed:
P0 is 128/256 (50.0%) and SFT-v2 is 195/256 (76.171875%), a gain of 67/256.
The same ledger shows exact-SMACT4 valid 135 versus 122, but uniform-primary
56 versus 101 and all-metal shortcuts 72 versus 10. Thus SFT-v2 strongly
changes the chemistry mode while not improving broad exact validity.

V20 job `31318` failed closed because the local audit manifest correctly
binds the raw-generation source inventory `4d8e7bde...`, while the repaired
assembly source inventory is `319b7591...`. No science bytes changed. V21 and
V22 then stopped before SBatch on two generator-anchor checks. V23 also
stopped before SBatch because its new root retained `_v20_` and therefore
triggered the launcher's exact stale-marker guard. None was rerun or created
an assembly job. V24 kept the same raw/assembly source separation and exact
guard, but used a clean immutable root without a parent-version marker.
Normal-CPU assembly job `31329` completed `0:0`; its SFT terminal-report SHA
is `cf51e406...` and stage-summary SHA is `445d58c2...`. It preserves the
complete V18 science above, emits the scientific-stop marker, and launches no
downstream or RL work. No A800 SMACT4 execution occurred.

DLM B0-v4 job `31308` produced all 1,773 synthetic states, 2,208 actual
rollout states, and 64 actual attempts, then failed a rescore identity gate.
The cause is BF16 batch geometry: rollout production used registered batches
up to eight, while the rescore regrouped states differently. B0-v5 kept
producer rollout bytes/batching unchanged, replays each serialized state in
its exact producer batch under the original `5e-4` gate, and uses the
historical fixed-panel batch size one for B0/B3 scientific scores. Its frozen
archive SHA is `ce94d793...`; gpu job `31323` completed `0:0` on node99. The
producer replay passed with maximum/mean/p95 absolute delta all zero, and the
frozen panel-manifest SHA is `6cc3d810...`. B3-v2 then stopped before SBatch
because its strict adapter expected the old run-root once in each sbatch file,
whereas the frozen train and scorer files each contain it three times. V3
fixed only those counts and passed every source identity, then stopped before
SBatch because A800's tar lacks `--sort=name`. V4 commit `3b5d775` reuses the
already successful B0 portable `tar -czf` pattern. Training job `31330`
completed `0:0` in `00:56:59` at exactly 1,696 updates, and dependent scorer
`31331` completed `0:0` in `00:10:42`. The terminal adapter SHA is
`ab4f3b82...`; training and score terminal SHAs are `1f9ab27d...` and
`755b5b86...`. B3 improves token-weighted NLL on IID (-0.276611), D1
(-0.337032), and synthetic safe-axis (-0.363534), but worsens the actual
protected-B0 rollout (+0.161890). The required two-panel transfer condition
is false, so B3 is not promoted and no ratio sweep was launched. No automatic
downstream, S.U.N., or RL job was submitted.

The active work is now the user-mandated complete current-run matrix:
P0+B0, SFT-v2+B0, P0+B3, and SFT-v2+B3, each using the common 256-ordinal
seed ledger, D2 safe-axis body generation, model_494 refine800, Direct, and
frozen-cache S.U.N. Historical R03 evidence is shell/reference context only;
it cannot replace a new cell. The immutable package is being frozen for a
four-element `gpu` array with at most two concurrent A800 tasks and an
`afterany` normal-CPU terminal assembler. All failures remain in denominator;
there is no retry, repair, filter, rerank, automatic promotion, or RL.

The Evidence-First workstream is active on branch
`codex/evidence-first-sun-msun`. The V8 optimizer audit repair passed both
two-update smoke arms, and V10 completed both fixed-endpoint 4,505-update
training arms plus the common-ledger P0/SFT-v2/SFT-v2-C raw64 generation.
V12 then completed the requested formal SMACT 3.1 recomputation. On the same
64-attempt ledger, P0 is 34/64 legacy composition-valid, while SFT-v2 and
SFT-v2-C are both 52/64: an absolute gain of 18/64 (+28.125 pp). SFT-v2's
paired exact McNemar p-value is 0.000912234.

Neither candidate is formally promoted. SFT-v2 misses the registered element
coverage, mean-N drift, and no-new-failure-class gates; SFT-v2-C additionally
misses parse and unique-formula gates and has larger distribution drift. The
V12 evaluator also exposes a schema-asymmetric P0 embedded-validator identity
check; this does not alter the recomputed SMACT 3.1 counts. Per the user's
explicit override, SFT-v2 alone proceeds through a diagnostic raw256 and
protected B0+safe-axis+model_494 Direct/S.U.N. chain. Independently of its
outcome, the mandatory DLM B0/B1/B2 inventory and B3 route now proceed. No RL
is authorized.

V13 preparation passed but its one-time submit command stopped before the
lock/SBatch boundary because it checked relative manifest entries outside the
run root; no V13 job or generation output exists. Immutable V14 changes only
that working directory. Its diagnostic raw256 array `31236_[0-1]%2`
completed on `gpu` for P0 and SFT-v2 (256 raw attempts per arm, common ledger
SHA `d5a3ac87458969816a0b27313fd9deecae47d2ddb10289ec08b9d93c5db48669`).
V24 completed formal assembly; downstream remains deliberately manual.

The mandatory DLM artifact inventory is also complete. Protected B0 is bound
to its frozen checkpoint/SHA; historical B1/B2 are bound to their terminal
1,696-update checkpoints and fixed-panel/dependency evidence. B2 remains a
non-revivable scientific stop because its dependency margin did not exceed
B1. The four B0-v5 state panels are now frozen, and B3 V4 training plus its
dependent scorer were submitted only after those identities were sealed.

The user has additionally required complete same-pipeline comparisons, not
candidate-only diagnostics. The frozen matrix is P0+B0 (protected control),
SFT-v2+B0 (Planner main effect), P0+B3 (DLM main effect), and SFT-v2+B3
(interaction), all using safe-axis, model_494 refine800, Direct, and S.U.N.
Historical summary rows remain context only; every requested cell must produce
its own current-run raw-denominator evidence. Scientific gate failures remain
visible labels, while engineering failures still fail closed.

The DLM state-panel contract and minimal B3 grouping implementation are now
frozen locally before reading any B3 result. Synthetic IID/D1/safe-axis panels
use validation ordinals 0..99; the real B0+safe-axis trace uses 0..63. Actual
wrong commitments remain visible and are counted, while NLL targets only the
frozen ground-truth tokens still masked in the active group. B3 will be scored
on these exact frozen state bytes. The new `d2_safe_axis` training policy is
composition, lattice, every X group, every Y group, then every Z group, with
zero mixed-axis groups and no Z-before-XY. B0-v5 panel job `31323`, B3
training `31330`, and dependent scorer `31331` all completed. Their mixed
terminal evidence is frozen without changing the contract; the complete
four-cell evaluation now consumes B0 and B3 read-only.

The B3 execution package is also frozen before any state-panel or B3 result.
It is a one-arm reuse of the successful historical two-A800, 1,696-update
training shell: B0 initialization, the same R5-C bytes/order/seeds, LR 5e-5,
and terminal checkpoint only. The sole scientific change is
`d2_safe_axis` at IID:planned 2:1. Its dependent job scores B3 on the exact
B0-frozen panel bytes; neither job can submit body64, a ratio sweep, S.U.N.,
or downstream work. The B0 terminal and manifest are frozen. V4 training and
its scorer are terminal; B3 remains an unpromoted diagnostic checkpoint used
only in the explicitly authorized four-cell comparison.

## Connection and read-only audit

- 5090 is reachable only through port 2213 and the configured private key.
- Existing A800 tmux sessions `ssha800` and `ssha800_2` were present with
  `pane_dead=0` and `pane_current_command=ssh` at the last audit.
- A800 access remains restricted to those existing sessions. Neither may be
  recreated or reconnected; if both fail the workstream stops.
- Local-to-5090 transfers are unrestricted. Only 5090-to-A800 SCP attempts are
  rate-limited, with at least ten minutes between attempts.
- A800 had no user jobs at the initial audit. Every submission still requires
  a fresh `sinfo`/`squeue` snapshot.
- Audit marker: `__EF_AUDIT_DONE__` at 2026-08-08T12:59:52+08:00.

## Frozen evidence already obtained

- Protected P0/B0/model_494 identities remain unchanged.
- Transfer v1 failed before source creation because archived executable files
  contained CRLF. The immutable failure evidence is retained.
- Corrected archive from commit `ef82ffc` has SHA
  `79d1e6e60b06e61e0654ebdafcfad828cb86888b7f17fff9bcbfeae2a97e42b9`.
- The corrected A800 source bootstrap passed under run
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_v1`:
  source inventory SHA
  `f429f63ef42ead9162149ddce135bb35da7fc2ad94d3e86c8135506063f6a801`,
  source archive SHA
  `555fdcc59901aad4bd4ceae685b28827ab3d5b312434706ce5a319d2c815de25`,
  source manifest SHA
  `b0fab254166928df999ffeb81199aeca699904508895fb98f73645687e8faf5d`,
  590 files.
- The attempted A800 portable SMACT4 runtime failed before project tests or
  science because `contourpy-1.3.3` required a newer manylinux tag. Its partial
  build and logs remain immutable; it is not repaired or reused.
- Two independent engineering reviewers converged on the same safe remedy:
  exact SMACT4 must be machine-separated, complete, SHA-bound, and joined
  one-to-one; stale, missing, duplicate, extra or substituted rows must fail.

## User-frozen evaluator split

A800 now uses only its existing Python 3.10/SMACT 3.1 environment. Exact
SMACT 4.0 must never be installed or executed on A800. The earlier foreground
zip-overlay probe was already active when this rule was frozen; it is ignored,
is not a gate, and must not be interrupted under the tmux safety contract.

The replacement identity is `h1_chemistry_first_sft_v2_smact_split_v2`:

1. A800 exports the complete immutable MP20 legacy snapshot using SMACT 3.1.
2. The local machine runs only the explicitly authorized exact-SMACT4 witness
   builder from the same frozen source inventory and returns a sealed
   one-row-per-source ledger.
3. A800 verifies parent/source/ordinal/material/formula/witness hashes and uses
   the exact POS intersection to build data, smoke, train, and generate raw64.
4. Raw64 generation stops. The local machine produces complete exact-SMACT4
   per-arm audit reports; A800 verifies every report against its current raw
   bytes before SMACT3 gate assembly.
5. Raw256 repeats the same generation/local-audit/assembly split only for
   raw64-passing candidates.

No A800 submission contains `SMACT4_PYTHON` or an exact-SMACT4 executable
path. The local exception is limited to witness and raw secondary-audit
production; all model tests, tokenizer checks, data assembly, training,
generation, and primary evaluation remain on A800.
The replacement source also excludes the retired portable-runtime builders,
requirements, and launcher, so the A800 snapshot contains no dormant SMACT4
runtime entry point. The immutable failed runtime run remains preserved as
historical evidence outside this source.

## Split-source audit terminal and minimal repair

The first split-source run reached an engineering terminal before Slurm:

- run root:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2`;
- source tests `48/48`, isolated tests `48/48`, and preflight-focused tests
  `35/35` all passed;
- the only failed check was `legacy_evaluator_source_sha`;
- archived `composition_validity.py` had CRLF SHA `c078c1ca...`, while its LF
  normalization exactly matched the frozen evaluator SHA `ca1c94f5...`;
- terminal SHA:
  `1fc5776c66c3ba34f9afc991200cec81734a35168092e14fd5da2b633956765d`;
- no Slurm, GPU, training, generation, SMACT4-on-A800, or science occurred.

The allowed repair is packaging-only. It writes the Git archive directly to
its output file, requires LF for source text classes, checks the evaluator
member SHA before extraction, and uses new transfer/staging/freeze/run paths.
The new run root is
`20260808_h1_chemistry_first_sft_v2_smact_split_v2_packaging_repair_v3`.
Following the user's audit-budget instruction, its source gate runs one
protocol test, one isolated inventory/SHA check, and one focused preflight;
the duplicate broad source/isolated test pass is not repeated.

## Packaging repair v3 and evaluator-split inputs

The packaging-only v3 repair passed its reduced A800 source gate and is now
the immutable active run root:

- run root:
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_packaging_repair_v3`;
- source-input archive SHA:
  `d64bf8790fe9a2df4926e25e045883f0f6077d2b8c1d8c86d73b3b3bb7c8f0cf`;
- source inventory SHA:
  `410fb34d2543f620fff012fde574a75a935bcaf54e41449fb529c12eb913c20c`;
- frozen source archive SHA:
  `3c5d93f697e97e29ee233bcefab41c8636599711e6decaf971eeb383f8137559`;
- source manifest SHA:
  `50f9e541a8fcafe13a4d953b1907f949259be4564bffd576379db72fcdbcf89a`;
- archived legacy evaluator member SHA:
  `ca1c94f583e0c97a172b5c9b7ba96505257fd74dedfc618b584c34486ac1f178`;
- the one protocol test, isolated inventory/evaluator identity check, and
  focused 35-test preflight all passed; A800 used Python 3.10.18 and
  SMACT 3.1.0, and did not import or execute SMACT 4.

Legacy snapshot job `31025` completed `0:0` in `00:03:38`. Its sealed bundle
SHA is `d65a682c8e29820938a8e4637963dd56f18f2f6d893abd77add02043fb6a13bf`
and report SHA is
`351a13f8c462a9a0ee377c1d07ebaafb426a0b64cffc2c9c3ad950725e3deefb`.
It contains exactly 27,136 train, 9,047 validation, and 9,046 test rows.

The explicitly authorized local-only exact-SMACT4 witness build also passed.
Its persistent isolated runtime terminal SHA is
`0853bbea0c714f8a3150b08f3c94bdf1ec03b9cd49585d3a16533a9f88ea67d6`.
The witness manifest SHA is
`d21698e29664c607541d7ab644250e93e18cfb2a0cd03d1687a270b42c8ccd32`,
ledger SHA is
`ab687c5f16dc64887de2446c3aae20c0e15021a3bfa4ffe71dd6cfe09b482c93`,
and deterministic archive SHA is
`b896516351a76b869cc10d8c95a321ee53bb1cf25e3ed38d207d1dd40b7322c3`.
The train census is 7,079 legacy-primary and 4,451 exact uniform-primary;
validation is 2,264 and 1,406 respectively. Official/witness parity is true.
The sealed ledger was transferred and frozen as data bytes only; SMACT 4 was
not run on A800.

Data job `31035` completed `0:0` in `00:15:47`. Both candidates contain
36,038 train and 9,047 validation records. The record multiset is identical;
base and curriculum order SHAs differ as registered, the 3,603-record
curriculum prefix alternates correctly, and all data/tokenizer audits pass.
There are zero invalid unconditional formula targets, generated charge-field
leaks, token-weight failures, or truncation failures. Frozen optimizer
geometry is 4,505 updates, 135 warmup steps, and six microbatches in the final
accumulation group. Audit-report SHA is
`1c9a5e2cba51a1258acf107255ca308c7c9f7122d50f5ae3a09ed97d2681612a`;
order-ledger SHA is
`c31df9dd44bbe1ea75a99131899d4e6ea7131d1f230b39ecfb25f730d5406a43`.

Minimal smoke array `31036_[0-1]` reached `FAILED 1:0` for both tasks after
`00:13:08`. It loaded the base model but stopped before the first model forward:
all 448 candidate/reference tensor keys matched and no noncandidate parameter
was trainable, but PEFT 0.16 had loaded the trainable candidate copy in FP32
and the frozen reference copy in the BF16 base dtype. The resulting maximum
absolute difference was `6.103515625e-05` for both candidates. No optimizer
step, formal training, scientific generation, or SMACT4-on-A800 occurred.

The frozen V4 repair loads both copies through the same protected-P0 FP32 PEFT
path, then freezes every reference parameter before forward or optimizer
construction. It changes no model payload, data, task order, prompt, seed,
optimizer, ledger, evaluator, or gate. The V3 data/legacy snapshot will be
reused byte-for-byte under a new manifest; a fresh dual-arm GPU smoke remains
mandatory before training.

V4 source freezing itself completed with source inventory
`2e7997cfc9894db5ba099d4fc3bfa18440b10d0c29e36927768dc35eaef03968`,
archive `193816fb1aa6919f0bed755f5fae9459f7a0bd63afc797b579dc741f43cb8b73`,
and manifest `02fcb50da21bd54e314aeaa0e09cf2b9cbb71300f24a9dddeaca009f01beffc1`.
Before executing its A800 source gate, a static check found that the isolated
archive path was still bound to the existing immutable V3 directory. The gate
was not executed, no pass marker was written, and no job was submitted.

Two independent propose/red-team reviews agreed that deleting or reusing V3,
hand-writing the pass marker, or changing only one path would be indefensible.
The active repair is therefore a new path-only V5 identity:
`20260808_h1_chemistry_first_sft_v2_smact_split_v2_source_gate_path_repair_v5`.
Every active script/SBatch/config path moves to V5 and isolated extraction is
scoped under `${RUN_ROOT}/isolated_archive_test`. V4's PEFT repair, all model
and science code, data, prompts, seeds, optimizers, ledgers, evaluators, and
gates remain unchanged. SMACT4 remains forbidden on A800.

V5 source inventory `7cef386be864eef760088fe8bb7c7073b7d1e908ca92e33f7d8a9a951ffebc91`,
archive `e37b073b34f1d55f55e96683534bcdeb56ed33f52b60baa1d7fa579e849e1958`,
and manifest `10ddf8a7be3678b3440842ba66f4f97a7943862d4983be32ed62b7012a41829a`
are frozen. The reduced A800 source gate passed: 16 protocol tests, isolated
inventory/evaluator identity and 35 focused preflight tests all passed under
Python 3.10.18 and SMACT 3.1.0; SMACT4 was not executed. V3 data reuse also
passed with reused-tree manifest `951aa186b37fd821d35f6a9fe63919abfc8753c4bd4ffce2d96968a039457981`.
Fresh dual-arm smoke array `31064_[0-1]` also reached `FAILED 1:0` for both
tasks after `00:15:02`. Both arms stopped at the protected-P0 identity gate
before the first forward or optimizer construction. Their identity reports
are byte-identical to V3 (SHA
`ac04b54094136dddb3fe5f6bbe9b10b369ec76b0f736345d00b858ae3e29889c`):
all 448 keys match, all 448 values differ, and maximum absolute difference is
`6.103515625e-05`. The V4 loader-flag change therefore did not alter the
runtime tensors. The immutable V5 failure report SHA is
`3f161979bef1de77351ba9178aa59cbbaa794cfcb29e9f9bb7b11884022d9be8`.
At that historical V5 boundary, formal training and scientific generation
remained unsubmitted; later V10/V14/V24 evidence supersedes that status.

## Immediate critical path

1. Preserve V3/V4/V5 source, job, logs, identity reports, and terminal evidence
   unchanged.
2. Complete two independent propose/red-team reviews of the repeated PEFT
   identity failure.
3. Run one minimal A800 runtime probe that compares candidate, reference, and
   protected source tensors after each load/freeze/device/activation step. It
   must not perform model forward, optimizer construction, or training.
4. Freeze a new immutable repair and rerun the dual-arm smoke only if the
   reviews and probe support one exact-identity implementation. The exact gate
   may not be relaxed.
5. Submit fixed-endpoint training and raw64 only after both repaired smoke
   tasks complete `0:0`; do not queue assembly before the local audit.
6. Continue C0/C1 and the B3 portfolio after this source family reaches its
   registered terminal.

No Planner or DLM RL is authorized.
